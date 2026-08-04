// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fuse the three sibling `Op::AttentionBackward` nodes (dQ / dK / dV, which
//! share `q/k/v/dy`) into one `Op::AttentionBackwardAll` so the score+softmax
//! is recomputed **once** instead of three times.
//!
//! The fused op produces a packed `[3B, …]` output (dQ‖dK‖dV stacked on the
//! outermost axis); three axis-0 `Narrow`s recover the individual gradients.
//! Doing this at the IR level (before memory planning) is what makes it correct
//! — the planner sees one node with one live packed buffer, and the `Narrow`s
//! are pure views into it. (A thunk-level version fails: the planner reclaims
//! the absorbed siblings' slots.)
//!
//! Self-attention only (`q_seq == k_seq`, so all three gradients share a shape);
//! cross-attention groups are left untouched. Backend-gated in the pipeline on
//! `OpKind::AttentionBackwardAll`, so only backends that lower it run this pass.

use crate::pass::Pass;
use rlx_ir::op::*;
use rlx_ir::*;
use std::collections::{HashMap, HashSet};

pub struct FuseAttentionBackwardAll;

struct FuseInfo {
    num_heads: usize,
    head_dim: usize,
    mask_kind: MaskKind,
    q_id: NodeId,
    k_id: NodeId,
    v_id: NodeId,
}

impl Pass for FuseAttentionBackwardAll {
    fn name(&self) -> &str {
        "fuse_attention_backward_all"
    }

    fn run(&self, graph: Graph) -> Graph {
        // OPT-IN (default OFF), enable with RLX_CPU_ATTN_BWD_FUSE=1. Backend status:
        //   • Metal — CORRECT (bit-exact vs the per-`wrt` path; verified in
        //     rlx-tinystories `--diag-grads`). The `Op::AttentionBackwardAll` arm in
        //     rlx-metal `compile.rs` lowers the packed output to `encode_attention_bwd_all`,
        //     and `scratch_bytes` sizes the scores/dp/ds scratch for it. Note Metal
        //     also fuses by default via its own thunk-level path (RLX_METAL_ATTN_BWD_FUSE),
        //     so this IR pass is the alternative, not the only, route there.
        //   • CPU — still mis-plans: the packed-output/view-alias interaction with the
        //     memory planner leaves the q/k/v inputs reading zero at backward time (the
        //     forward reads a packed strided QKV while the backward reads separate,
        //     unwritten buffers). Under investigation; until fixed the correct non-fused
        //     CPU path (parallelize + BLAS, ~75× over the original naive kernel) is default.
        if std::env::var("RLX_CPU_ATTN_BWD_FUSE").as_deref() != Ok("1") {
            return graph;
        }
        // Group the sibling AttentionBackward nodes by their shared (q,k,v,dy).
        let mut groups: HashMap<(NodeId, NodeId, NodeId, NodeId), Vec<NodeId>> = HashMap::new();
        for n in graph.nodes() {
            if matches!(n.op, Op::AttentionBackward { .. }) && n.inputs.len() >= 4 {
                groups
                    .entry((n.inputs[0], n.inputs[1], n.inputs[2], n.inputs[3]))
                    .or_default()
                    .push(n.id);
            }
        }

        // Plan: emit-point (min id) → group info; absorbed = the other two.
        let mut emit: HashMap<NodeId, FuseInfo> = HashMap::new();
        let mut absorbed: HashSet<NodeId> = HashSet::new();
        for ids in groups.values() {
            if ids.len() != 3 {
                continue;
            }
            let (mut q_id, mut k_id, mut v_id) = (None, None, None);
            let (mut num_heads, mut head_dim, mut mask_kind) = (0usize, 0usize, MaskKind::None);
            for &id in ids {
                if let Op::AttentionBackward {
                    num_heads: nh,
                    head_dim: hd,
                    mask_kind: mk,
                    wrt,
                } = &graph.node(id).op
                {
                    num_heads = *nh;
                    head_dim = *hd;
                    mask_kind = *mk;
                    match wrt {
                        AttentionBwdWrt::Query => q_id = Some(id),
                        AttentionBwdWrt::Key => k_id = Some(id),
                        AttentionBwdWrt::Value => v_id = Some(id),
                    }
                }
            }
            let (Some(q_id), Some(k_id), Some(v_id)) = (q_id, k_id, v_id) else {
                continue;
            };
            // Self-attention gate: q and k must share a shape so dQ/dK/dV pack
            // uniformly. `inputs[0]`=q, `inputs[1]`=k on any sibling.
            let anchor = graph.node(q_id);
            let q_shape = &graph.node(anchor.inputs[0]).shape;
            let k_shape = &graph.node(anchor.inputs[1]).shape;
            if q_shape.dims() != k_shape.dims() {
                continue;
            }
            let emit_id = *ids.iter().min().unwrap();
            emit.insert(
                emit_id,
                FuseInfo {
                    num_heads,
                    head_dim,
                    mask_kind,
                    q_id,
                    k_id,
                    v_id,
                },
            );
            for &id in ids {
                if id != emit_id {
                    absorbed.insert(id);
                }
            }
        }
        if emit.is_empty() {
            return graph;
        }

        // Rebuild, remapping ids. The emit-point (min id, processed first) emits
        // the fused op + three narrows and maps ALL three sibling ids to their
        // narrows; the other two siblings are skipped (already mapped).
        let mut out = Graph::new(graph.name.clone());
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
        for node in graph.nodes() {
            if absorbed.contains(&node.id) {
                continue;
            }
            let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
            if let Some(info) = emit.get(&node.id) {
                let q_shape = graph.node(node.inputs[0]).shape.clone();
                let b = q_shape.dim(0).unwrap_static();
                let mut packed_dims = q_shape.dims().to_vec();
                packed_dims[0] = Dim::Static(3 * b);
                let packed = Shape::from_dims(&packed_dims, q_shape.dtype());
                let all = out.add_node(
                    Op::AttentionBackwardAll {
                        num_heads: info.num_heads,
                        head_dim: info.head_dim,
                        mask_kind: info.mask_kind,
                    },
                    new_inputs,
                    packed,
                );
                let dq = out.add_node(
                    Op::Narrow {
                        axis: 0,
                        start: 0,
                        len: b,
                    },
                    vec![all],
                    q_shape.clone(),
                );
                let dk = out.add_node(
                    Op::Narrow {
                        axis: 0,
                        start: b,
                        len: b,
                    },
                    vec![all],
                    q_shape.clone(),
                );
                let dv = out.add_node(
                    Op::Narrow {
                        axis: 0,
                        start: 2 * b,
                        len: b,
                    },
                    vec![all],
                    q_shape.clone(),
                );
                id_map.insert(info.q_id, dq);
                id_map.insert(info.k_id, dk);
                id_map.insert(info.v_id, dv);
            } else {
                let new_id = out.add_node(node.op.clone(), new_inputs, node.shape.clone());
                id_map.insert(node.id, new_id);
            }
        }
        out.set_outputs(graph.outputs.iter().map(|i| id_map[i]).collect());
        out
    }
}
