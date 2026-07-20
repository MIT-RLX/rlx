// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! WideEP Phase 0/1: expert-parallel MoE dispatch / combine.
//!
//! [`moe_dispatch`] packs tokens by destination rank and exchanges them with
//! variable-size [`ProcessGroup::all_to_all_v`](rlx_driver::ProcessGroup::all_to_all_v)
//! (Phase 1), writing into a static `[P, H+4]` buffer (`P = world × max_tokens`)
//! for shape-stable [`Op::GroupedMatMul`](rlx_ir::Op::GroupedMatMul).
//! [`moe_combine`] returns outputs the same way and applies the gate.
//!
//! Inference-first: VJP is intentionally empty (non-differentiable).

use crate::lookup_group;
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef, register_cpu_kernel};
use rlx_ir::infer::GraphExt;
use rlx_ir::op::{BinaryOp, ReduceOp};
use rlx_ir::op_registry::{OpExtension, register_op};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use std::sync::Arc;

/// Registry name for MoE expert-parallel dispatch (variable A2A, static out).
pub const MOE_DISPATCH: &str = "collective.moe_dispatch";
/// Registry name for MoE expert-parallel combine (variable A2A, static out).
pub const MOE_COMBINE: &str = "collective.moe_combine";

/// Packed dispatch row layout: `token[H] | src_rank | src_row | expert_local | valid`.
pub const DISPATCH_META: usize = 4;

/// Static EP MoE configuration baked into op attrs (shape inference).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeEpConfig {
    pub group_id: u64,
    pub world_size: u32,
    pub rank: u32,
    pub num_experts: u32,
    pub hidden: u32,
    /// Must be `1` in Phase 0/1.
    pub top_k: u32,
    /// Tokens per rank (`M`). Local GMM capacity is `world × max_tokens`
    /// (static IR shape); wire traffic uses variable-size all-to-all.
    pub max_tokens: u32,
    /// Expert slots across the EP group (`≥ num_experts`, divisible by
    /// `world_size`). Local `GroupedMatMul` width is `num_slots / world_size`.
    /// Extra slots hold hot-expert replicas (EPLB).
    pub num_slots: u32,
    /// `placement[e] = rank` that owns expert `e`. Empty → `e % world_size`.
    pub placement: Vec<u32>,
}

impl MoeEpConfig {
    /// Default placement `e % world_size`, `num_slots = num_experts`.
    pub fn new(
        group_id: u64,
        world_size: u32,
        rank: u32,
        num_experts: u32,
        hidden: u32,
        max_tokens: u32,
    ) -> Self {
        assert!(world_size >= 1, "world_size must be >= 1");
        assert!(rank < world_size, "rank must be < world_size");
        assert!(num_experts >= 1, "num_experts must be >= 1");
        assert!(hidden >= 1, "hidden must be >= 1");
        assert!(max_tokens >= 1, "max_tokens must be >= 1");
        Self {
            group_id,
            world_size,
            rank,
            num_experts,
            hidden,
            top_k: 1,
            max_tokens,
            num_slots: num_experts,
            placement: Vec::new(),
        }
    }

    /// Set EPLB slot count (`≥ num_experts`, divisible by `world_size`).
    pub fn with_num_slots(mut self, num_slots: u32) -> Self {
        assert!(
            num_slots >= self.num_experts,
            "num_slots must be >= num_experts"
        );
        assert!(
            num_slots.is_multiple_of(self.world_size),
            "num_slots must be divisible by world_size"
        );
        self.num_slots = num_slots;
        self
    }

    /// Explicit expert→rank map (`len == num_experts`).
    pub fn with_placement(mut self, placement: Vec<u32>) -> Self {
        assert_eq!(
            placement.len(),
            self.num_experts as usize,
            "placement len must equal num_experts"
        );
        for (e, &r) in placement.iter().enumerate() {
            assert!(
                r < self.world_size,
                "placement[{e}]={r} >= world_size {}",
                self.world_size
            );
        }
        self.placement = placement;
        self
    }

    /// Local buffer capacity after dispatch (static IR / GMM token rows).
    pub fn local_capacity(&self) -> usize {
        self.world_size as usize * self.max_tokens as usize
    }

    /// Slots (experts + replicas) on each rank.
    pub fn slots_per_rank(&self) -> u32 {
        self.num_slots / self.world_size
    }

    /// Experts owned by `self.rank`, in ascending global id order.
    pub fn local_expert_globals(&self) -> Vec<u32> {
        crate::eplb::experts_on_rank(self, self.rank)
    }

    /// Number of local weight rows for `GroupedMatMul` (`num_slots / world`).
    pub fn local_num_experts(&self) -> u32 {
        self.slots_per_rank()
    }
}

// ── attrs codec ───────────────────────────────────────────────────
//
// [group_id:u64][world:u32][rank:u32][num_experts:u32][hidden:u32]
// [top_k:u32][max_tokens:u32][num_slots:u32][placement: u32 × num_experts]?

const ATTR_HEADER: usize = 8 + 4 * 7;

pub(crate) fn encode_moe_ep_attrs(cfg: &MoeEpConfig) -> Vec<u8> {
    let mut v = Vec::with_capacity(ATTR_HEADER + cfg.placement.len() * 4);
    v.extend_from_slice(&cfg.group_id.to_le_bytes());
    v.extend_from_slice(&cfg.world_size.to_le_bytes());
    v.extend_from_slice(&cfg.rank.to_le_bytes());
    v.extend_from_slice(&cfg.num_experts.to_le_bytes());
    v.extend_from_slice(&cfg.hidden.to_le_bytes());
    v.extend_from_slice(&cfg.top_k.to_le_bytes());
    v.extend_from_slice(&cfg.max_tokens.to_le_bytes());
    v.extend_from_slice(&cfg.num_slots.to_le_bytes());
    for &p in &cfg.placement {
        v.extend_from_slice(&p.to_le_bytes());
    }
    v
}

pub(crate) fn decode_moe_ep_attrs(attrs: &[u8]) -> Result<MoeEpConfig, String> {
    if attrs.len() < ATTR_HEADER {
        return Err(format!(
            "moe_ep attrs: need at least {ATTR_HEADER} bytes, got {}",
            attrs.len()
        ));
    }
    let group_id = u64::from_le_bytes(attrs[0..8].try_into().unwrap());
    let world_size = u32::from_le_bytes(attrs[8..12].try_into().unwrap());
    let rank = u32::from_le_bytes(attrs[12..16].try_into().unwrap());
    let num_experts = u32::from_le_bytes(attrs[16..20].try_into().unwrap());
    let hidden = u32::from_le_bytes(attrs[20..24].try_into().unwrap());
    let top_k = u32::from_le_bytes(attrs[24..28].try_into().unwrap());
    let max_tokens = u32::from_le_bytes(attrs[28..32].try_into().unwrap());
    let num_slots = u32::from_le_bytes(attrs[32..36].try_into().unwrap());
    let mut cfg = MoeEpConfig {
        group_id,
        world_size,
        rank,
        num_experts,
        hidden,
        top_k,
        max_tokens,
        num_slots: if num_slots == 0 {
            num_experts
        } else {
            num_slots
        },
        placement: Vec::new(),
    };
    let rest = &attrs[ATTR_HEADER..];
    if !rest.is_empty() {
        if rest.len() != num_experts as usize * 4 {
            return Err(format!(
                "moe_ep attrs: placement needs {} bytes, got {}",
                num_experts * 4,
                rest.len()
            ));
        }
        let mut placement = Vec::with_capacity(num_experts as usize);
        for i in 0..num_experts as usize {
            placement.push(u32::from_le_bytes(rest[i * 4..i * 4 + 4].try_into().unwrap()));
        }
        cfg = cfg.with_placement(placement);
    }
    if cfg.top_k != 1 {
        return Err(format!(
            "moe_ep Phase 0 requires top_k=1, got {}",
            cfg.top_k
        ));
    }
    if cfg.num_slots < cfg.num_experts {
        return Err(format!(
            "moe_ep: num_slots {} < num_experts {}",
            cfg.num_slots, cfg.num_experts
        ));
    }
    if !cfg.num_slots.is_multiple_of(cfg.world_size) {
        return Err(format!(
            "moe_ep: num_slots {} not divisible by world {}",
            cfg.num_slots, cfg.world_size
        ));
    }
    Ok(cfg)
}

// ── op extensions + CPU kernels ───────────────────────────────────

fn static_dims(shape: &Shape) -> Vec<usize> {
    shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect()
}

struct MoeDispatchExt;
impl OpExtension for MoeDispatchExt {
    fn name(&self) -> &str {
        MOE_DISPATCH
    }
    fn num_inputs(&self) -> usize {
        2 // tokens [M,H], expert_idx [M]
    }
    fn infer_shape(&self, inputs: &[&Shape], attrs: &[u8]) -> Shape {
        let cfg = decode_moe_ep_attrs(attrs).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(inputs.len(), 2, "moe_dispatch: two inputs");
        let h = cfg.hidden as usize;
        let m = cfg.max_tokens as usize;
        assert_eq!(
            static_dims(inputs[0]),
            vec![m, h],
            "moe_dispatch: tokens shape must be [max_tokens, hidden]"
        );
        assert_eq!(
            static_dims(inputs[1]),
            vec![m],
            "moe_dispatch: expert_idx shape must be [max_tokens]"
        );
        let p = cfg.local_capacity();
        Shape::new(&[p, h + DISPATCH_META], DType::F32)
    }
}

struct MoeCombineExt;
impl OpExtension for MoeCombineExt {
    fn name(&self) -> &str {
        MOE_COMBINE
    }
    fn num_inputs(&self) -> usize {
        3 // local_out [P,H], packed [P,H+4], gate [M]
    }
    fn infer_shape(&self, inputs: &[&Shape], attrs: &[u8]) -> Shape {
        let cfg = decode_moe_ep_attrs(attrs).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(inputs.len(), 3, "moe_combine: three inputs");
        let h = cfg.hidden as usize;
        let m = cfg.max_tokens as usize;
        let p = cfg.local_capacity();
        assert_eq!(
            static_dims(inputs[0]),
            vec![p, h],
            "moe_combine: local_out shape must be [P, hidden]"
        );
        assert_eq!(
            static_dims(inputs[1]),
            vec![p, h + DISPATCH_META],
            "moe_combine: packed shape must be [P, hidden+4]"
        );
        assert_eq!(
            static_dims(inputs[2]),
            vec![m],
            "moe_combine: gate shape must be [max_tokens]"
        );
        Shape::new(&[m, h], DType::F32)
    }
}

struct MoeDispatchCpu;
impl CpuKernel for MoeDispatchCpu {
    fn name(&self) -> &str {
        MOE_DISPATCH
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let cfg = decode_moe_ep_attrs(attrs)?;
        let h = cfg.hidden as usize;
        let m = cfg.max_tokens as usize;
        let w = cfg.world_size as usize;
        let row = h + DISPATCH_META;
        let tokens = inputs[0].expect_f32("moe_dispatch tokens")?;
        let expert_idx = inputs[1].expect_f32("moe_dispatch expert_idx")?;
        let out = output.expect_f32_mut("moe_dispatch output")?;
        if tokens.len() != m * h {
            return Err(format!(
                "moe_dispatch: tokens len {} != {}",
                tokens.len(),
                m * h
            ));
        }
        if expert_idx.len() != m {
            return Err(format!(
                "moe_dispatch: expert_idx len {} != {m}",
                expert_idx.len()
            ));
        }
        let p = w * m;
        if out.len() != p * row {
            return Err(format!(
                "moe_dispatch: output len {} != {}",
                out.len(),
                p * row
            ));
        }

        // Bucket live rows per destination (no pad on the wire).
        let mut buckets: Vec<Vec<f32>> = (0..w).map(|_| Vec::new()).collect();
        for i in 0..m {
            let e = expert_idx[i] as u32;
            if e >= cfg.num_experts {
                return Err(format!(
                    "moe_dispatch: expert_idx[{i}]={e} >= num_experts {}",
                    cfg.num_experts
                ));
            }
            let dest = crate::eplb::pick_dispatch_rank(&cfg, e, i as u32) as usize;
            let b = &mut buckets[dest];
            if b.len() / row >= m {
                return Err(format!(
                    "moe_dispatch: overflow sending to rank {dest} (>{m} tokens)"
                ));
            }
            b.extend_from_slice(&tokens[i * h..(i + 1) * h]);
            b.push(cfg.rank as f32);
            b.push(i as f32);
            b.push(crate::eplb::local_id_on_rank(&cfg, e, dest as u32) as f32);
            b.push(1.0);
        }

        let send_counts: Vec<usize> = buckets.iter().map(|b| b.len()).collect();
        let mut send = Vec::with_capacity(send_counts.iter().sum());
        for b in &buckets {
            send.extend_from_slice(b);
        }

        let group = lookup_group(cfg.group_id).ok_or_else(|| {
            format!(
                "collective.moe_dispatch: group id {} not registered",
                cfg.group_id
            )
        })?;
        let (recv, recv_counts) = group
            .all_to_all_v(&send, &send_counts)
            .map_err(|e| e.to_string())?;

        // Scatter into static [P, H+4]: one slot block per source rank.
        out.fill(0.0);
        let mut off = 0usize;
        for src in 0..w {
            let nbytes = recv_counts[src];
            if nbytes % row != 0 {
                return Err(format!(
                    "moe_dispatch: recv from {src} len {nbytes} not multiple of row {row}"
                ));
            }
            let n_rows = nbytes / row;
            if n_rows > m {
                return Err(format!(
                    "moe_dispatch: recv {n_rows} rows from {src} > max_tokens {m}"
                ));
            }
            let dst_base = src * m * row;
            out[dst_base..dst_base + nbytes].copy_from_slice(&recv[off..off + nbytes]);
            off += nbytes;
        }
        Ok(())
    }
}

struct MoeCombineCpu;
impl CpuKernel for MoeCombineCpu {
    fn name(&self) -> &str {
        MOE_COMBINE
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let cfg = decode_moe_ep_attrs(attrs)?;
        let h = cfg.hidden as usize;
        let m = cfg.max_tokens as usize;
        let w = cfg.world_size as usize;
        let meta_row = h + DISPATCH_META;
        let local_out = inputs[0].expect_f32("moe_combine local_out")?;
        let packed = inputs[1].expect_f32("moe_combine packed")?;
        let gate = inputs[2].expect_f32("moe_combine gate")?;
        let out = output.expect_f32_mut("moe_combine output")?;
        let p = w * m;
        if local_out.len() != p * h {
            return Err(format!(
                "moe_combine: local_out len {} != {}",
                local_out.len(),
                p * h
            ));
        }
        if packed.len() != p * meta_row {
            return Err(format!(
                "moe_combine: packed len {} != {}",
                packed.len(),
                p * meta_row
            ));
        }
        if gate.len() != m {
            return Err(format!("moe_combine: gate len {} != {m}", gate.len()));
        }
        if out.len() != m * h {
            return Err(format!("moe_combine: output len {} != {}", out.len(), m * h));
        }

        // Return payload: token[H] | src_row  (presence ⇒ valid).
        let ret_row = h + 1;
        let mut buckets: Vec<Vec<f32>> = (0..w).map(|_| Vec::new()).collect();
        for j in 0..p {
            let pb = j * meta_row;
            if packed[pb + h + 3] < 0.5 {
                continue;
            }
            let dest = packed[pb + h] as usize; // src_rank
            let src_row = packed[pb + h + 1];
            if dest >= w {
                return Err(format!("moe_combine: bad src_rank {dest}"));
            }
            let b = &mut buckets[dest];
            if b.len() / ret_row >= m {
                return Err(format!(
                    "moe_combine: overflow returning to rank {dest} (>{m} tokens)"
                ));
            }
            b.extend_from_slice(&local_out[j * h..(j + 1) * h]);
            b.push(src_row);
        }

        let send_counts: Vec<usize> = buckets.iter().map(|b| b.len()).collect();
        let mut send = Vec::with_capacity(send_counts.iter().sum());
        for b in &buckets {
            send.extend_from_slice(b);
        }

        let group = lookup_group(cfg.group_id).ok_or_else(|| {
            format!(
                "collective.moe_combine: group id {} not registered",
                cfg.group_id
            )
        })?;
        let (recv, recv_counts) = group
            .all_to_all_v(&send, &send_counts)
            .map_err(|e| e.to_string())?;

        out.fill(0.0);
        let mut off = 0usize;
        for src in 0..w {
            let nbytes = recv_counts[src];
            if nbytes % ret_row != 0 {
                return Err(format!(
                    "moe_combine: recv from {src} len {nbytes} not multiple of {ret_row}"
                ));
            }
            let n_rows = nbytes / ret_row;
            for slot in 0..n_rows {
                let base = off + slot * ret_row;
                let src_row = recv[base + h] as usize;
                if src_row >= m {
                    return Err(format!("moe_combine: bad src_row {src_row}"));
                }
                let g = gate[src_row];
                for d in 0..h {
                    out[src_row * h + d] += recv[base + d] * g;
                }
            }
            off += nbytes;
        }
        Ok(())
    }
}

/// Install MoE EP dispatch/combine IR extensions + CPU kernels.
pub fn register_moe_ep() {
    register_op(Arc::new(MoeDispatchExt));
    register_op(Arc::new(MoeCombineExt));
    register_cpu_kernel(Arc::new(MoeDispatchCpu));
    register_cpu_kernel(Arc::new(MoeCombineCpu));
}

/// Dispatch tokens to expert-owning ranks. Output `[P, H+4]` packed rows.
pub fn moe_dispatch(
    g: &mut Graph,
    tokens: NodeId,
    expert_idx: NodeId,
    cfg: &MoeEpConfig,
) -> NodeId {
    g.custom_op(
        MOE_DISPATCH,
        encode_moe_ep_attrs(cfg),
        vec![tokens, expert_idx],
    )
}

/// Combine expert outputs back to originating ranks and apply `gate`.
pub fn moe_combine(
    g: &mut Graph,
    local_out: NodeId,
    packed: NodeId,
    gate: NodeId,
    cfg: &MoeEpConfig,
) -> NodeId {
    g.custom_op(
        MOE_COMBINE,
        encode_moe_ep_attrs(cfg),
        vec![local_out, packed, gate],
    )
}

/// Build a k=1 EP MoE FFN: gate → TopK → dispatch → GroupedMatMul → combine.
///
/// `expert_w_local` must be `[E_local, H, H]` for this rank's shard.
pub fn moe_ep_ffn(
    g: &mut Graph,
    tokens: NodeId,
    gate_w: NodeId,
    expert_w_local: NodeId,
    cfg: &MoeEpConfig,
) -> NodeId {
    assert_eq!(cfg.top_k, 1, "moe_ep_ffn requires top_k=1");
    let m = cfg.max_tokens as usize;
    let h = cfg.hidden as usize;
    let e = cfg.num_experts as usize;
    let e_local = cfg.local_num_experts() as usize;
    assert!(
        e_local >= 1,
        "moe_ep_ffn: rank {} owns no experts",
        cfg.rank
    );
    let p = cfg.local_capacity();
    let f = DType::F32;

    let logits = g.matmul(tokens, gate_w, Shape::new(&[m, e], f));
    let probs = g.add_node(
        Op::Softmax { axis: -1 },
        vec![logits],
        Shape::new(&[m, e], f),
    );
    let top_idx_2d = g.add_node(Op::TopK { k: 1 }, vec![probs], Shape::new(&[m, 1], f));
    let top_idx = g.reshape_(top_idx_2d, vec![m as i64]);
    let gate_val = g.add_node(
        Op::Reduce {
            op: ReduceOp::Max,
            axes: vec![1],
            keep_dim: false,
        },
        vec![probs],
        Shape::new(&[m], f),
    );

    let packed = moe_dispatch(g, tokens, top_idx, cfg);
    let local_tok = g.narrow_(packed, 1, 0, h);
    let expert_col = g.narrow_(packed, 1, h + 2, 1);
    let local_expert_idx = g.reshape_(expert_col, vec![p as i64]);

    let local_out = g.add_node(
        Op::GroupedMatMul,
        vec![local_tok, expert_w_local, local_expert_idx],
        Shape::new(&[p, h], f),
    );
    // Zero contributions from pad rows (valid=0) so combine only sees live work.
    let valid_col = g.narrow_(packed, 1, h + 3, 1); // [P, 1]
    let valid_b = g.add_node(
        Op::Expand {
            target_shape: vec![p as i64, h as i64],
        },
        vec![valid_col],
        Shape::new(&[p, h], f),
    );
    let local_out_masked = g.binary(BinaryOp::Mul, local_out, valid_b, Shape::new(&[p, h], f));

    moe_combine(g, local_out_masked, packed, gate_val, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{register, register_group, unregister_group};
    use rlx_driver::{NetTransport, ProcessGroup};
    use rlx_runtime::{Device, Session};
    use std::net::{Ipv4Addr, SocketAddr, TcpListener};
    use std::thread;

    #[test]
    fn attrs_roundtrip_default_placement() {
        let cfg = MoeEpConfig::new(42, 2, 1, 4, 8, 16);
        let bytes = encode_moe_ep_attrs(&cfg);
        let back = decode_moe_ep_attrs(&bytes).unwrap();
        assert_eq!(back.group_id, 42);
        assert_eq!(back.world_size, 2);
        assert_eq!(back.rank, 1);
        assert_eq!(back.num_experts, 4);
        assert_eq!(back.hidden, 8);
        assert_eq!(back.max_tokens, 16);
        assert_eq!(back.num_slots, 4);
        assert!(back.placement.is_empty());
        assert_eq!(crate::eplb::owner_of(&back, 3), 1);
        // Rank 0 owns experts 0,2 → local ids 0,1; rank 1 owns 1,3.
        assert_eq!(crate::eplb::local_id_on_owner(&back, 2), 1);
        assert_eq!(crate::eplb::local_id_on_owner(&back, 3), 1);
    }

    #[test]
    fn attrs_roundtrip_explicit_placement() {
        let cfg = MoeEpConfig::new(7, 2, 0, 2, 4, 4).with_placement(vec![0, 1]);
        let back = decode_moe_ep_attrs(&encode_moe_ep_attrs(&cfg)).unwrap();
        assert_eq!(back.placement, vec![0, 1]);
        assert_eq!(back.local_expert_globals(), vec![0]);
        assert_eq!(back.local_num_experts(), 1);
    }

    #[test]
    fn single_rank_matches_dense_moe() {
        register();
        let m = 4usize;
        let h = 8usize;
        let e = 2usize;
        let det = |seed: usize, n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| (((i + seed) * 7 + 11) % 17) as f32 / 17.0 * scale - scale * 0.5)
                .collect()
        };
        let x_data = det(0, m * h, 0.5);
        let gate_w = det(1, h * e, 0.3);
        let expert_w = det(2, e * h * h, 0.2);

        // Dense reference (moe_demo path).
        let mut reference = vec![0f32; m * h];
        for i in 0..m {
            let mut logits = vec![0f32; e];
            for ei in 0..e {
                for k in 0..h {
                    logits[ei] += x_data[i * h + k] * gate_w[k * e + ei];
                }
            }
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|&v| v / sum).collect();
            let mut best_e = 0usize;
            let mut best_p = probs[0];
            for ei in 1..e {
                if probs[ei] > best_p {
                    best_p = probs[ei];
                    best_e = ei;
                }
            }
            for j in 0..h {
                let mut acc = 0f32;
                for k in 0..h {
                    acc += x_data[i * h + k] * expert_w[(best_e * h + k) * h + j];
                }
                reference[i * h + j] = acc * best_p;
            }
        }

        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let t = NetTransport::from_listener(0, 1, listener, vec![addr], 1 << 20).unwrap();
        let gid = 9100u64;
        register_group(gid, Arc::new(ProcessGroup::new(Arc::new(t))));

        let cfg = MoeEpConfig::new(gid, 1, 0, e as u32, h as u32, m as u32);
        let mut g = Graph::new("moe_ep1");
        let x = g.input("x", Shape::new(&[m, h], DType::F32));
        let gw = g.param("gate_w", Shape::new(&[h, e], DType::F32));
        let ew = g.param("expert_w", Shape::new(&[e, h, h], DType::F32));
        let out = moe_ep_ffn(&mut g, x, gw, ew, &cfg);
        g.set_outputs(vec![out]);

        let mut c = Session::new(Device::Cpu).compile(g);
        c.set_param("gate_w", &gate_w);
        c.set_param("expert_w", &expert_w);
        let got = c.run(&[("x", &x_data)]);
        unregister_group(gid);

        let err = got[0]
            .iter()
            .zip(reference.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(err < 1e-5, "single-rank EP vs dense max_err={err}");
    }

    #[test]
    fn two_rank_ep_matches_dense_reference() {
        register();
        let world = 2u32;
        let m = 4usize;
        let h = 8usize;
        let e = 2usize;
        let det = |seed: usize, n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| (((i + seed) * 7 + 11) % 17) as f32 / 17.0 * scale - scale * 0.5)
                .collect()
        };
        let x_data = det(0, m * h, 0.5);
        let gate_w = det(1, h * e, 0.3);
        let expert_w = det(2, e * h * h, 0.2);

        let mut reference = vec![0f32; m * h];
        for i in 0..m {
            let mut logits = vec![0f32; e];
            for ei in 0..e {
                for k in 0..h {
                    logits[ei] += x_data[i * h + k] * gate_w[k * e + ei];
                }
            }
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|&v| v / sum).collect();
            let mut best_e = 0usize;
            let mut best_p = probs[0];
            for ei in 1..e {
                if probs[ei] > best_p {
                    best_p = probs[ei];
                    best_e = ei;
                }
            }
            for j in 0..h {
                let mut acc = 0f32;
                for k in 0..h {
                    acc += x_data[i * h + k] * expert_w[(best_e * h + k) * h + j];
                }
                reference[i * h + j] = acc * best_p;
            }
        }

        let listeners: Vec<TcpListener> = (0..world)
            .map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap())
            .collect();
        let addrs: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
        let gid_base = 9200u64;

        let handles: Vec<_> = listeners
            .into_iter()
            .enumerate()
            .map(|(rank, listener)| {
                let addrs = addrs.clone();
                let x_data = x_data.clone();
                let gate_w = gate_w.clone();
                let expert_w = expert_w.clone();
                thread::spawn(move || {
                    let rank = rank as u32;
                    let t =
                        NetTransport::from_listener(rank, world, listener, addrs, 1 << 20).unwrap();
                    let gid = gid_base + rank as u64;
                    register_group(gid, Arc::new(ProcessGroup::new(Arc::new(t))));

                    let cfg = MoeEpConfig::new(gid, world, rank, e as u32, h as u32, m as u32)
                        .with_placement(vec![0, 1]);
                    // Shard: rank r owns expert r → [1, H, H]
                    let mut w_local = vec![0f32; h * h];
                    let e_off = rank as usize * h * h;
                    w_local.copy_from_slice(&expert_w[e_off..e_off + h * h]);

                    let mut g = Graph::new("wide_ep");
                    let x = g.input("x", Shape::new(&[m, h], DType::F32));
                    let gw = g.param("gate_w", Shape::new(&[h, e], DType::F32));
                    let ew = g.param("expert_w_local", Shape::new(&[1, h, h], DType::F32));
                    let out = moe_ep_ffn(&mut g, x, gw, ew, &cfg);
                    g.set_outputs(vec![out]);

                    let mut c = Session::new(Device::Cpu).compile(g);
                    c.set_param("gate_w", &gate_w);
                    c.set_param("expert_w_local", &w_local);
                    let res = c.run(&[("x", &x_data)]);
                    unregister_group(gid);
                    res
                })
            })
            .collect();

        let outs: Vec<Vec<Vec<f32>>> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        for (r, ro) in outs.iter().enumerate() {
            let err = ro[0]
                .iter()
                .zip(reference.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                err < 1e-5,
                "rank {r} EP vs dense max_err={err} (got {:?})",
                &ro[0][..8.min(ro[0].len())]
            );
        }
    }

    #[test]
    fn eplb_rebalance_ep_still_matches_dense() {
        use crate::eplb::{
            rebalance_placement, register_ep_placement, shard_expert_weights,
            unregister_ep_placement,
        };
        use crate::{register, register_group, unregister_group};
        use rlx_ir::{DType, Graph, Shape};
        use rlx_runtime::{Device, Session};

        register();
        let world = 2u32;
        let m = 4usize;
        let h = 4usize;
        let e = 4usize;
        let det = |seed: usize, n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| (((i + seed) * 7 + 11) % 17) as f32 / 17.0 * scale - scale * 0.5)
                .collect()
        };
        let x_data = det(0, m * h, 0.5);
        let gate_w = det(1, h * e, 0.3);
        let expert_w = det(2, e * h * h, 0.2);

        // Skewed hits: experts 0 and 2 dominate (same default rank under e%2).
        let hits = vec![50u64, 2, 40, 3];
        let placement = rebalance_placement(&hits, world).unwrap();
        assert_ne!(placement[0], placement[2]);

        let mut reference = vec![0f32; m * h];
        for i in 0..m {
            let mut logits = vec![0f32; e];
            for ei in 0..e {
                for k in 0..h {
                    logits[ei] += x_data[i * h + k] * gate_w[k * e + ei];
                }
            }
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|&v| v / sum).collect();
            let mut best_e = 0usize;
            let mut best_p = probs[0];
            for ei in 1..e {
                if probs[ei] > best_p {
                    best_p = probs[ei];
                    best_e = ei;
                }
            }
            for j in 0..h {
                let mut acc = 0f32;
                for k in 0..h {
                    acc += x_data[i * h + k] * expert_w[(best_e * h + k) * h + j];
                }
                reference[i * h + j] = acc * best_p;
            }
        }

        let listeners: Vec<TcpListener> = (0..world)
            .map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap())
            .collect();
        let addrs: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
        let gid_base = 9400u64;
        let placement_c = placement.clone();

        let handles: Vec<_> = listeners
            .into_iter()
            .enumerate()
            .map(|(rank, listener)| {
                let addrs = addrs.clone();
                let x_data = x_data.clone();
                let gate_w = gate_w.clone();
                let expert_w = expert_w.clone();
                let placement = placement_c.clone();
                thread::spawn(move || {
                    let rank = rank as u32;
                    let t =
                        NetTransport::from_listener(rank, world, listener, addrs, 1 << 20).unwrap();
                    let gid = gid_base + rank as u64;
                    register_group(gid, Arc::new(ProcessGroup::new(Arc::new(t))));
                    // Same logical placement on every rank; keyed by this rank's group id
                    // (each rank has its own ProcessGroup id in the toy harness).
                    register_ep_placement(gid, placement.clone());

                    let cfg = MoeEpConfig::new(gid, world, rank, e as u32, h as u32, m as u32);
                    let w_local =
                        shard_expert_weights(&expert_w, e, h, h, &placement, rank);
                    assert_eq!(w_local.len(), 2 * h * h); // 2 experts per rank

                    let mut g = Graph::new("eplb_ep");
                    let x = g.input("x", Shape::new(&[m, h], DType::F32));
                    let gw = g.param("gate_w", Shape::new(&[h, e], DType::F32));
                    let ew = g.param("expert_w_local", Shape::new(&[2, h, h], DType::F32));
                    let out = moe_ep_ffn(&mut g, x, gw, ew, &cfg);
                    g.set_outputs(vec![out]);

                    let mut c = Session::new(Device::Cpu).compile(g);
                    c.set_param("gate_w", &gate_w);
                    c.set_param("expert_w_local", &w_local);
                    let res = c.run(&[("x", &x_data)]);
                    unregister_ep_placement(gid);
                    unregister_group(gid);
                    res
                })
            })
            .collect();

        for (r, h) in handles.into_iter().enumerate() {
            let ro = h.join().unwrap();
            let err = ro[0]
                .iter()
                .zip(reference.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(err < 1e-5, "rank {r} EPLB EP max_err={err}");
        }
    }

    #[test]
    fn eplb_replicas_ep_still_matches_dense() {
        use crate::eplb::{
            rebalance_with_replicas, register_ep_replicas, shard_expert_weights_slots,
            unregister_ep_replicas,
        };
        use crate::{register, register_group, unregister_group};
        use rlx_ir::{DType, Graph, Shape};
        use rlx_runtime::{Device, Session};

        register();
        let world = 2u32;
        let m = 4usize;
        let h = 4usize;
        let e = 4usize;
        let num_slots = 6u32; // 3 slots/rank, 2 hot replicas
        let det = |seed: usize, n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| (((i + seed) * 7 + 11) % 17) as f32 / 17.0 * scale - scale * 0.5)
                .collect()
        };
        let x_data = det(0, m * h, 0.5);
        let gate_w = det(1, h * e, 0.3);
        let expert_w = det(2, e * h * h, 0.2);

        let hits = vec![100u64, 1, 90, 1];
        let map = rebalance_with_replicas(&hits, world, num_slots).unwrap();
        assert!(map.owners.iter().any(|o| o.len() > 1));

        let mut reference = vec![0f32; m * h];
        for i in 0..m {
            let mut logits = vec![0f32; e];
            for ei in 0..e {
                for k in 0..h {
                    logits[ei] += x_data[i * h + k] * gate_w[k * e + ei];
                }
            }
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|&v| v / sum).collect();
            let mut best_e = 0usize;
            let mut best_p = probs[0];
            for ei in 1..e {
                if probs[ei] > best_p {
                    best_p = probs[ei];
                    best_e = ei;
                }
            }
            for j in 0..h {
                let mut acc = 0f32;
                for k in 0..h {
                    acc += x_data[i * h + k] * expert_w[(best_e * h + k) * h + j];
                }
                reference[i * h + j] = acc * best_p;
            }
        }

        let listeners: Vec<TcpListener> = (0..world)
            .map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap())
            .collect();
        let addrs: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
        let gid_base = 9500u64;
        let map_c = map.clone();

        let handles: Vec<_> = listeners
            .into_iter()
            .enumerate()
            .map(|(rank, listener)| {
                let addrs = addrs.clone();
                let x_data = x_data.clone();
                let gate_w = gate_w.clone();
                let expert_w = expert_w.clone();
                let map = map_c.clone();
                thread::spawn(move || {
                    let rank = rank as u32;
                    let t =
                        NetTransport::from_listener(rank, world, listener, addrs, 1 << 20).unwrap();
                    let gid = gid_base + rank as u64;
                    register_group(gid, Arc::new(ProcessGroup::new(Arc::new(t))));
                    register_ep_replicas(gid, map.clone());

                    let cfg = MoeEpConfig::new(gid, world, rank, e as u32, h as u32, m as u32)
                        .with_num_slots(num_slots);
                    let slots = &map.local_slots[rank as usize];
                    let w_local = shard_expert_weights_slots(&expert_w, e, h, h, slots);
                    assert_eq!(w_local.len(), 3 * h * h);

                    let mut g = Graph::new("eplb_rep");
                    let x = g.input("x", Shape::new(&[m, h], DType::F32));
                    let gw = g.param("gate_w", Shape::new(&[h, e], DType::F32));
                    let ew = g.param("expert_w_local", Shape::new(&[3, h, h], DType::F32));
                    let out = moe_ep_ffn(&mut g, x, gw, ew, &cfg);
                    g.set_outputs(vec![out]);

                    let mut c = Session::new(Device::Cpu).compile(g);
                    c.set_param("gate_w", &gate_w);
                    c.set_param("expert_w_local", &w_local);
                    let res = c.run(&[("x", &x_data)]);
                    unregister_ep_replicas(gid);
                    unregister_group(gid);
                    res
                })
            })
            .collect();

        for (r, h) in handles.into_iter().enumerate() {
            let ro = h.join().unwrap();
            let err = ro[0]
                .iter()
                .zip(reference.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(err < 1e-5, "rank {r} replica EP max_err={err}");
        }
    }

    #[test]
    fn migrate_then_ep_parity() {
        use crate::eplb::{
            migrate_to_placement, register_ep_placement, shard_expert_weights,
            unregister_ep_placement,
        };
        use crate::{register, register_group, unregister_group};
        use rlx_ir::{DType, Graph, Shape};
        use rlx_runtime::{Device, Session};

        register();
        let world = 2u32;
        let m = 4usize;
        let h = 4usize;
        let e = 2usize;
        let det = |seed: usize, n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| (((i + seed) * 7 + 11) % 17) as f32 / 17.0 * scale - scale * 0.5)
                .collect()
        };
        let x_data = det(0, m * h, 0.5);
        let gate_w = det(1, h * e, 0.3);
        let expert_w = det(2, e * h * h, 0.2);

        let old_placement = vec![0u32, 1];
        let new_placement = vec![1u32, 0]; // swap

        let mut reference = vec![0f32; m * h];
        for i in 0..m {
            let mut logits = vec![0f32; e];
            for ei in 0..e {
                for k in 0..h {
                    logits[ei] += x_data[i * h + k] * gate_w[k * e + ei];
                }
            }
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|&v| v / sum).collect();
            let mut best_e = 0usize;
            let mut best_p = probs[0];
            for ei in 1..e {
                if probs[ei] > best_p {
                    best_p = probs[ei];
                    best_e = ei;
                }
            }
            for j in 0..h {
                let mut acc = 0f32;
                for k in 0..h {
                    acc += x_data[i * h + k] * expert_w[(best_e * h + k) * h + j];
                }
                reference[i * h + j] = acc * best_p;
            }
        }

        let listeners: Vec<TcpListener> = (0..world)
            .map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap())
            .collect();
        let addrs: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
        let gid_base = 9600u64;

        let handles: Vec<_> = listeners
            .into_iter()
            .enumerate()
            .map(|(rank, listener)| {
                let addrs = addrs.clone();
                let x_data = x_data.clone();
                let gate_w = gate_w.clone();
                let expert_w = expert_w.clone();
                let old_placement = old_placement.clone();
                let new_placement = new_placement.clone();
                thread::spawn(move || {
                    let rank = rank as u32;
                    let t =
                        NetTransport::from_listener(rank, world, listener, addrs, 1 << 20).unwrap();
                    let group = Arc::new(ProcessGroup::new(Arc::new(t)));
                    let gid = gid_base + rank as u64;
                    register_group(gid, group.clone());

                    let old_w =
                        shard_expert_weights(&expert_w, e, h, h, &old_placement, rank);
                    let new_w = migrate_to_placement(
                        &group,
                        &old_placement,
                        &old_w,
                        &new_placement,
                        h,
                        h,
                    )
                    .unwrap();
                    register_ep_placement(gid, new_placement.clone());

                    let cfg = MoeEpConfig::new(gid, world, rank, e as u32, h as u32, m as u32);
                    let mut g = Graph::new("mig_ep");
                    let x = g.input("x", Shape::new(&[m, h], DType::F32));
                    let gw = g.param("gate_w", Shape::new(&[h, e], DType::F32));
                    let ew = g.param("expert_w_local", Shape::new(&[1, h, h], DType::F32));
                    let out = moe_ep_ffn(&mut g, x, gw, ew, &cfg);
                    g.set_outputs(vec![out]);

                    let mut c = Session::new(Device::Cpu).compile(g);
                    c.set_param("gate_w", &gate_w);
                    c.set_param("expert_w_local", &new_w);
                    let res = c.run(&[("x", &x_data)]);
                    unregister_ep_placement(gid);
                    unregister_group(gid);
                    res
                })
            })
            .collect();

        for (r, h) in handles.into_iter().enumerate() {
            let ro = h.join().unwrap();
            let err = ro[0]
                .iter()
                .zip(reference.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(err < 1e-5, "rank {r} migrate+EP max_err={err}");
        }
    }
}
