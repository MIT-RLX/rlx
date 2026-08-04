// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tier 3 — mine the *workload* rather than the values. Per op we estimate
//! FLOPs, bytes moved, and arithmetic intensity (FLOP/byte), which gives a
//! roofline classification (compute- vs memory-bound → drives fusion strategy)
//! and a histogram of the actual GEMM shapes that occur (→ autotune/specialize
//! kernels for the hot shapes, generate a dispatch table). Pure static analysis
//! over any [`Graph`] — no execution.

use rlx_ir::{Graph, NodeId, Op};
use std::collections::HashMap;

const F32: u64 = 4; // bytes/elem

/// Per-op cost estimate. `bytes` is the **external** (DRAM) traffic — inputs read
/// + output written. `internal_bytes` is the **on-chip** intermediate traffic a
/// *fused* op materializes-and-consumes internally (0 for plain ops); a non-fused
/// equivalent would spill these to DRAM, so it's the fusion's IO saving.
#[derive(Clone, Debug)]
pub struct OpCost {
    pub id: u32,
    pub op: String,
    /// GEMM dims when applicable (else 0).
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub flops: u64,
    pub bytes: u64,
    /// On-chip intermediate traffic kept off DRAM by fusion (0 for plain ops).
    pub internal_bytes: u64,
    /// True for composite/fused ops (Attention, GroupedMatMul, Fused*).
    pub fused: bool,
}

impl OpCost {
    /// Arithmetic intensity vs **external** (DRAM) bytes — what the memory system
    /// actually sees for a fused op (its intermediates never hit DRAM).
    pub fn intensity(&self) -> f64 {
        if self.bytes == 0 {
            0.0
        } else {
            self.flops as f64 / self.bytes as f64
        }
    }
    /// Arithmetic intensity if the op were NOT fused (intermediates spill to DRAM).
    /// The gap to [`intensity`] is what fusion buys on the roofline.
    pub fn intensity_unfused(&self) -> f64 {
        let denom = self.bytes + self.internal_bytes;
        if denom == 0 {
            0.0
        } else {
            self.flops as f64 / denom as f64
        }
    }
    pub fn is_gemm(&self) -> bool {
        self.k != 0
    }
}

fn numel(g: &Graph, id: NodeId) -> u64 {
    g.shape(id).num_elements().unwrap_or(0) as u64
}

fn dim_static(g: &Graph, id: NodeId, i: usize) -> u64 {
    let s = g.shape(id);
    if i < s.rank() {
        s.dim(i).unwrap_static() as u64
    } else {
        1
    }
}

/// Is this op a composite/fused kernel (its cost model here accounts for internal,
/// on-chip intermediate traffic that a decomposed version would spill to DRAM)?
pub fn is_fused_op(op: &Op) -> bool {
    matches!(
        op,
        Op::Attention { .. }
            | Op::GroupedMatMul
            | Op::FusedSwiGLU { .. }
            | Op::FusedMatMulBiasAct { .. }
            | Op::FusedConvBiasAct { .. }
            | Op::FusedResidualLN { .. }
            | Op::FusedResidualRmsNorm { .. }
            | Op::FusedAttentionBlock { .. }
            | Op::FusedTransformerLayer { .. }
    )
}

/// Estimate cost of every compute node — fusion-aware (real internal FLOPs + the
/// external/internal IO split for composite ops).
pub fn op_costs(g: &Graph) -> Vec<OpCost> {
    let mut out = Vec::new();
    for node in g.nodes() {
        let outn = numel(g, node.id);
        let insum: u64 = node.inputs.iter().map(|&i| numel(g, i)).sum();
        let ext = (insum + outn) * F32; // default external DRAM traffic
        let (op, m, k, n, flops, bytes, internal, fused) = match &node.op {
            Op::MatMul | Op::GroupedMatMul => {
                // lhs [M,K], out [M,N]  →  2·M·K·N MACs.
                let lhs = g.shape(node.inputs[0]);
                let outs = &node.shape;
                let (m, kk) = (
                    lhs.dim(0).unwrap_static(),
                    lhs.dim(lhs.rank() - 1).unwrap_static(),
                );
                let n = outs.dim(outs.rank() - 1).unwrap_static();
                let grouped = matches!(node.op, Op::GroupedMatMul);
                let name = if grouped { "GroupedMatMul" } else { "MatMul" };
                let flops = 2 * (m as u64) * (kk as u64) * (n as u64);
                (name.to_string(), m, kk, n, flops, ext, 0, grouped)
            }
            // Fused SDPA: QKᵀ (2·bh·s²·dh) + softmax (~5·bh·s²) + scores·V (2·bh·s²·dhv).
            // The [bh,s,s] scores matrix is materialized+consumed ON-CHIP — that's the
            // fusion's whole point (a decomposed attention spills it to DRAM). So it's
            // INTERNAL traffic, and the fused op's DRAM intensity stays high while it
            // grows O(s²) in compute — the flash-attention insight, made measurable.
            Op::Attention {
                num_heads,
                head_dim,
                v_head_dim,
                ..
            } => {
                let dh = *head_dim as u64;
                let dhv = v_head_dim.unwrap_or(*head_dim) as u64;
                let os = &node.shape;
                let s = if os.rank() >= 2 {
                    os.dim(os.rank() - 2).unwrap_static() as u64
                } else {
                    1
                };
                let bh = if s > 0 && dhv > 0 {
                    outn / (s * dhv)
                } else {
                    *num_heads as u64
                };
                let flops = 2 * bh * s * s * dh + 5 * bh * s * s + 2 * bh * s * s * dhv;
                let internal = bh * s * s * F32 * 2; // scores: write + read on-chip
                ("Attention".into(), 0, 0, 0, flops, ext, internal, true)
            }
            // Fused attention BLOCK: qkv proj + SDPA + out proj, all resident.
            Op::FusedAttentionBlock {
                num_heads,
                head_dim,
                ..
            } => {
                let d = dim_static(g, node.inputs[0], g.shape(node.inputs[0]).rank() - 1);
                let bs = numel(g, node.inputs[0]).checked_div(d).unwrap_or(0);
                let s = dim_static(
                    g,
                    node.inputs[0],
                    g.shape(node.inputs[0]).rank().saturating_sub(2),
                );
                let (h, dh) = (*num_heads as u64, *head_dim as u64);
                let b = bs.checked_div(s).unwrap_or(0);
                let flops = 2 * bs * d * 3 * d
                    + 2 * bs * d * d
                    + 4 * b * h * s * s * dh
                    + 5 * b * h * s * s;
                let internal = (bs * 3 * d + b * h * s * s + bs * d) * F32; // qkv + scores + attn_out
                (
                    "FusedAttentionBlock".into(),
                    0,
                    0,
                    0,
                    flops,
                    ext,
                    internal,
                    true,
                )
            }
            // matmul + bias + activation; the pre-epilogue matmul output stays on-chip.
            Op::FusedMatMulBiasAct { .. } => {
                let lhs = g.shape(node.inputs[0]);
                let kk = lhs.dim(lhs.rank() - 1).unwrap_static() as u64;
                let m = numel(g, node.inputs[0]).checked_div(kk).unwrap_or(0);
                let nn = node.shape.dim(node.shape.rank() - 1).unwrap_static() as u64;
                let flops = 2 * m * kk * nn + 2 * outn;
                (
                    "FusedMatMulBiasAct".into(),
                    m as usize,
                    kk as usize,
                    nn as usize,
                    flops,
                    ext,
                    outn * F32,
                    true,
                )
            }
            // silu(gate)·up on a [.,2N] concat → [.,N]; the silu temp stays on-chip.
            Op::FusedSwiGLU { .. } => (
                "FusedSwiGLU".into(),
                0,
                0,
                0,
                4 * outn,
                ext,
                outn * F32,
                true,
            ),
            // residual add + norm; the residual-sum intermediate stays on-chip.
            Op::FusedResidualRmsNorm { .. } => (
                "FusedResidualRmsNorm".into(),
                0,
                0,
                0,
                6 * outn,
                ext,
                outn * F32,
                true,
            ),
            Op::FusedResidualLN { .. } => (
                "FusedResidualLN".into(),
                0,
                0,
                0,
                8 * outn,
                ext,
                outn * F32,
                true,
            ),
            Op::Softmax { .. } => ("Softmax".into(), 0, 0, 0, 5 * outn, ext, 0, false),
            Op::TopK { .. } => ("TopK".into(), 0, 0, 0, insum, ext, 0, false),
            Op::Activation(_) | Op::Binary(_) | Op::Compare(_) => (
                format!("{:?}", node.op.kind()),
                0,
                0,
                0,
                outn,
                ext,
                0,
                false,
            ),
            Op::Input { .. } | Op::Param { .. } | Op::Constant { .. } => continue,
            other => {
                let fused = is_fused_op(other);
                // Fused op we don't model precisely: still flag it + estimate its
                // internal traffic as the output (a lower bound on what it keeps resident).
                let internal = if fused { outn * F32 } else { 0 };
                (
                    format!("{:?}", other.kind()),
                    0,
                    0,
                    0,
                    outn,
                    ext,
                    internal,
                    fused,
                )
            }
        };
        out.push(OpCost {
            id: node.id.0,
            op,
            m,
            k,
            n,
            flops,
            bytes,
            internal_bytes: internal,
            fused,
        });
    }
    out
}

/// Per-fused-op-kind IO roll-up: FLOPs, external (DRAM) vs internal (on-chip,
/// fusion-saved) bytes, and the two arithmetic intensities (fused vs if-unfused).
/// The gap between them is what fusion buys on the roofline — e.g. attention stays
/// compute-bound because its O(s²) scores never hit DRAM.
#[derive(Clone, Debug)]
pub struct FusedIo {
    pub op: String,
    pub count: usize,
    pub flops: u64,
    pub external_bytes: u64,
    pub internal_bytes: u64,
    pub intensity_fused: f64,
    pub intensity_unfused: f64,
}

/// Aggregate the fused ops of a graph by kind (ranked by FLOPs).
pub fn fused_io_report(g: &Graph) -> Vec<FusedIo> {
    let mut h: HashMap<String, (usize, u64, u64, u64)> = HashMap::new();
    for c in op_costs(g).iter().filter(|c| c.fused) {
        let e = h.entry(c.op.clone()).or_default();
        e.0 += 1;
        e.1 += c.flops;
        e.2 += c.bytes;
        e.3 += c.internal_bytes;
    }
    let mut v: Vec<FusedIo> = h
        .into_iter()
        .map(|(op, (count, flops, ext, int))| FusedIo {
            op,
            count,
            flops,
            external_bytes: ext,
            internal_bytes: int,
            intensity_fused: if ext > 0 {
                flops as f64 / ext as f64
            } else {
                0.0
            },
            intensity_unfused: if ext + int > 0 {
                flops as f64 / (ext + int) as f64
            } else {
                0.0
            },
        })
        .collect();
    v.sort_by_key(|b| std::cmp::Reverse(b.flops));
    v
}

/// Roofline ridge point (FLOP/byte). Below → memory-bound, above → compute-
/// bound. Default ~10 is a reasonable generic CPU/GPU ridge; pass the real one.
pub const DEFAULT_RIDGE: f64 = 10.0;

pub fn roofline_class(cost: &OpCost, ridge: f64) -> &'static str {
    if cost.flops == 0 {
        "trivial"
    } else if cost.intensity() < ridge {
        "memory-bound"
    } else {
        "compute-bound"
    }
}

/// Per-op-kind cost roll-up: `(kind, count, total_flops, total_bytes)` ranked by
/// bytes (the memory-bound axis) descending — the analytical companion to a
/// measured per-region time profile (`crate::timing`): compare each kind's byte
/// share to its measured time share to confirm/refute "time follows bytes".
pub fn cost_by_kind(costs: &[OpCost]) -> Vec<(String, usize, u64, u64)> {
    let mut h: HashMap<String, (usize, u64, u64)> = HashMap::new();
    for c in costs {
        let e = h.entry(c.op.clone()).or_default();
        e.0 += 1;
        e.1 += c.flops;
        e.2 += c.bytes;
    }
    let mut v: Vec<_> = h.into_iter().map(|(k, (n, f, b))| (k, n, f, b)).collect();
    v.sort_by_key(|b| std::cmp::Reverse(b.3));
    v
}

/// Where a graph's analytical memory traffic goes, split by *reducibility* — the
/// IO story the raw byte total hides. `weight` is quant's lever (fewer bytes per
/// weight), `fusible` is fusion's lever (intermediates that never need to hit
/// DRAM), `io` is irreducible (graph inputs/outputs must cross the boundary).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TrafficSplit {
    /// Param/Constant reads, summed over each consuming op (weights re-read per use).
    pub weight: u64,
    /// Graph inputs + outputs — must cross the DRAM/PCIe boundary.
    pub io: u64,
    /// Intermediate activations (producer write + consumer reads) — fusion can keep
    /// these on-chip, so this is the fusion bandwidth headroom.
    pub fusible: u64,
}

impl TrafficSplit {
    pub fn total(&self) -> u64 {
        self.weight + self.io + self.fusible
    }
    /// Bytes that MUST hit DRAM regardless of fusion (weights + graph I/O).
    pub fn irreducible(&self) -> u64 {
        self.weight + self.io
    }
    /// Fraction of traffic fusion could remove (0..1).
    pub fn fusible_frac(&self) -> f64 {
        let t = self.total();
        if t == 0 {
            0.0
        } else {
            self.fusible as f64 / t as f64
        }
    }
}

/// Split a graph's memory traffic into weight / io / fusible-intermediate. This
/// separates the two IO levers numerically: how much traffic is weight-stream
/// (quant-addressable) vs intermediate (fusion-addressable) vs irreducible I/O.
pub fn traffic_split(g: &Graph) -> TrafficSplit {
    let mut consumers: HashMap<NodeId, u32> = HashMap::new();
    for node in g.nodes() {
        for &inp in &node.inputs {
            *consumers.entry(inp).or_default() += 1;
        }
    }
    let outs: std::collections::HashSet<NodeId> = g.outputs.iter().copied().collect();
    let mut s = TrafficSplit::default();
    for node in g.nodes() {
        let bytes = numel(g, node.id) * F32;
        let nc = *consumers.get(&node.id).unwrap_or(&0) as u64;
        match &node.op {
            // Weights: streamed once per consuming op (a matmul reads its weight once).
            Op::Param { .. } | Op::Constant { .. } => s.weight += bytes * nc.max(1),
            // Graph inputs cross the boundary once (consumer reads hit cache).
            Op::Input { .. } => s.io += bytes,
            _ if outs.contains(&node.id) => s.io += bytes, // graph output written out
            // Intermediate activation: written once + read by each consumer. All of
            // this is removable if the producer fuses into its consumer(s).
            _ => s.fusible += bytes * (1 + nc),
        }
    }
    s
}

/// GEMM-shape histogram: `(M,K,N) → (count, total_flops)`, ranked by total FLOPs
/// (the hot shapes worth a specialized/autotuned kernel).
pub fn gemm_shape_histogram(costs: &[OpCost]) -> Vec<((usize, usize, usize), (usize, u64))> {
    let mut h: HashMap<(usize, usize, usize), (usize, u64)> = HashMap::new();
    for c in costs.iter().filter(|c| c.is_gemm()) {
        let e = h.entry((c.m, c.k, c.n)).or_default();
        e.0 += 1;
        e.1 += c.flops;
    }
    let mut v: Vec<_> = h.into_iter().collect();
    v.sort_by_key(|b| std::cmp::Reverse(b.1.1));
    v
}

#[cfg(test)]
mod traffic_tests {
    use super::*;
    use rlx_ir::{DType, Shape, op::BinaryOp};

    #[test]
    fn traffic_split_separates_weight_io_fusible() {
        // x[4,384] @ w[384,8] = mm[4,8] (intermediate); + bias b[8] = out[4,8].
        let mut g = Graph::new("t");
        let x = g.input("x", Shape::new(&[4, 384], DType::F32));
        let w = g.param("w", Shape::new(&[384, 8], DType::F32));
        let mm = g.matmul(x, w, Shape::new(&[4, 8], DType::F32));
        let b = g.param("b", Shape::new(&[8], DType::F32));
        let out = g.binary(BinaryOp::Add, mm, b, Shape::new(&[4, 8], DType::F32));
        g.set_outputs(vec![out]);

        let s = traffic_split(&g);
        assert_eq!(s.weight, 384 * 8 * 4 + 8 * 4); // w + b, one consumer each
        assert_eq!(s.io, 4 * 384 * 4 + 4 * 8 * 4); // input x + output out
        assert_eq!(s.fusible, 4 * 8 * 4 * 2); // mm: 1 write + 1 read
        assert_eq!(s.total(), s.weight + s.io + s.fusible);
        // On this weight-heavy graph, fusible traffic is a sliver.
        assert!(s.fusible_frac() < 0.05);
    }

    #[test]
    fn attention_cost_is_o_s2_with_internal_scores() {
        use rlx_ir::op::MaskKind;
        let mut g = Graph::new("attn");
        let sh = Shape::new(&[1, 4, 16], DType::F32); // B=1, S=4, H·Dh=16 (H=2, Dh=8)
        let q = g.input("q", sh.clone());
        let k = g.input("k", sh.clone());
        let v = g.input("v", sh.clone());
        let attn = g.add_node(
            Op::Attention {
                num_heads: 2,
                head_dim: 8,
                v_head_dim: None,
                mask_kind: MaskKind::Causal,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q, k, v],
            sh.clone(),
        );
        g.set_outputs(vec![attn]);

        let costs = op_costs(&g);
        let a = costs.iter().find(|c| c.op == "Attention").unwrap();
        // bh=2, s=4, dh=dhv=8: QK 2·2·16·8 + softmax 5·2·16 + AV 2·2·16·8 = 512+160+512.
        assert_eq!(a.flops, 1184);
        assert_eq!(a.internal_bytes, 2 * 16 * 4 * 2); // scores [bh,s,s]=32 elems, write+read
        assert!(a.fused);
        // Fusion keeps the scores off DRAM → higher DRAM intensity than if unfused.
        assert!(a.intensity() > a.intensity_unfused());

        let rep = fused_io_report(&g);
        assert_eq!(rep.len(), 1);
        assert_eq!(rep[0].op, "Attention");
        assert_eq!(rep[0].internal_bytes, 256);
        assert!(rep[0].intensity_fused > rep[0].intensity_unfused);
    }
}
