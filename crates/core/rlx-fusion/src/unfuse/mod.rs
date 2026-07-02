// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Decompose tier-2 fused MIR ops into primitives for autodiff and backends.

use rlx_ir::op::*;
use rlx_ir::shape::Shape as IrShape;
use rlx_ir::{Dim, Graph, NodeId, Op};
use std::collections::HashMap;

mod fused;
mod rnn;

use fused::*;
use rnn::*;

/// Expand fused blocks so per-op VJP rules apply.
pub fn unfuse_fused_for_autodiff(g: Graph) -> Graph {
    // Walk the input graph, copy node-by-node into a new graph,
    // expanding each fused op into the primitive chain inline.

    let mut out = Graph::new(g.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

    // Snapshot inputs so we don't double-borrow during iteration.
    let original_outputs = g.outputs.clone();
    let nodes: Vec<rlx_ir::Node> = g.nodes().to_vec();

    for node in &nodes {
        let new_inputs: Vec<NodeId> = node
            .inputs
            .iter()
            .map(|i| {
                *id_map.get(i).unwrap_or_else(|| {
                    panic!(
                        "unfuse_fused_for_autodiff: node {:?} ({}) references input {i:?} \
                 which has not been mapped — graph is not in strict topological \
                 order at the start of this pass. Run \
                 `legalize_multi_axis_reduce` before this pass if the input came \
                 from a user-built multi-axis `Op::Reduce`, or check for an \
                 upstream rewriter that left a dangling NodeId.",
                        node.id, node.op,
                    )
                })
            })
            .collect();
        let new_id = match &node.op {
            Op::FusedMatMulBiasAct { .. } => unfuse_fused_mat_mul_bias_act(node, new_inputs, &mut out),
            Op::FusedResidualLN { .. } => unfuse_fused_residual_l_n(node, new_inputs, &mut out),
            Op::FusedResidualRmsNorm { .. } => unfuse_fused_residual_rms_norm(node, new_inputs, &mut out),
            Op::FusedAttentionBlock { .. } => unfuse_fused_attention_block(node, new_inputs, &mut out),
            Op::FusedTransformerLayer { .. } => unfuse_fused_transformer_layer(node, new_inputs, &mut out),
            Op::FusedSwiGLU { .. } => unfuse_fused_swi_g_l_u(node, new_inputs, &mut out),
            Op::LoraMatMul { .. } => unfuse_lora_mat_mul(node, new_inputs, &mut out),
            Op::GatedDeltaNet { .. } => unfuse_gated_delta_net(node, new_inputs, &mut out),
            Op::Lstm { .. } => unfuse_lstm(node, new_inputs, &mut out),
            Op::Gru { .. } => unfuse_gru(node, new_inputs, &mut out),
            Op::Rnn { .. } => unfuse_rnn(node, new_inputs, &mut out),
            Op::Mamba2 { .. } => unfuse_mamba2(node, new_inputs, &mut out),
            Op::SelectiveScan { .. } => unfuse_selective_scan(node, new_inputs, &mut out),
            _ => {
                // Pass through unchanged.
                out.add_node(node.op.clone(), new_inputs, node.shape.clone())
            }
        };
        id_map.insert(node.id, new_id);
    }

    // Re-pin outputs.
    let new_outputs: Vec<NodeId> = original_outputs.iter().map(|i| id_map[i]).collect();
    out.set_outputs(new_outputs);
    out
}


/// Decompose a single `Op::FusedAttentionBlock` into its primitive chain,
/// appending the nodes to `out` and returning the NodeId of the
/// output-projection result:
///
/// `MatMul` → \[bias\] → `Narrow`×3 → `Reshape`+`Transpose` (→ `[B,H,S,D]`)
/// → \[`Rope` on Q/K\] → `Attention`(custom mask) → `Transpose`+`Reshape`
/// → `MatMul` → \[bias\].
///
/// `new_inputs` are the FAB's inputs already remapped into `out`, in IR
/// order: `hidden, qkv_w, out_w, mask, [qkv_b, out_b], [rope_cos, rope_sin]`.
///
/// Shared by [`unfuse_fused_for_autodiff`] (autodiff / whole-graph unfuse)
/// and [`unfuse_attention_block`] (the FAB-only backend lowering pass), so
/// the decomposition has a single source of truth.
pub fn expand_attention_block(
    out: &mut Graph,
    new_inputs: &[NodeId],
    num_heads: usize,
    head_dim: usize,
    has_bias: bool,
    has_rope: bool,
) -> NodeId {
    let nh = num_heads;
    let dh = head_dim;
    let hd = nh * dh;
    let in_hidden = new_inputs[0];
    let in_qkv_w = new_inputs[1];
    let in_out_w = new_inputs[2];
    let in_mask = new_inputs[3];
    let mut next_idx = 4;
    let (in_qkv_b, in_out_b) = if has_bias {
        let qb = new_inputs[next_idx];
        let ob = new_inputs[next_idx + 1];
        next_idx += 2;
        (Some(qb), Some(ob))
    } else {
        (None, None)
    };
    let (in_rope_cos, in_rope_sin) = if has_rope {
        let c = new_inputs[next_idx];
        let s = new_inputs[next_idx + 1];
        let _ = next_idx + 2;
        (Some(c), Some(s))
    } else {
        (None, None)
    };
    let _ = next_idx;

    let h_shape = out.node(in_hidden).shape.clone();
    let dtype = h_shape.dtype();
    let b = h_shape.dim(0);
    let s = h_shape.dim(1);

    // qkv = hidden @ qkv_w   shape [B, S, 3*H*D]
    let qkv_shape = IrShape::from_dims(&[b, s, Dim::Static(3 * hd)], dtype);
    let mut qkv = out.matmul(in_hidden, in_qkv_w, qkv_shape.clone());
    if let Some(qb) = in_qkv_b {
        let qb_b = out.add_node(
            Op::Expand {
                target_shape: qkv_shape
                    .dims()
                    .iter()
                    .map(|d| match d {
                        Dim::Static(n) => *n as i64,
                        _ => -1,
                    })
                    .collect(),
            },
            vec![qb],
            qkv_shape.clone(),
        );
        qkv = out.binary(BinaryOp::Add, qkv, qb_b, qkv_shape);
    }

    // Narrow into Q/K/V each shape [B, S, H*D].
    let qkv_part_shape = IrShape::from_dims(&[b, s, Dim::Static(hd)], dtype);
    let q = out.add_node(
        Op::Narrow {
            axis: 2,
            start: 0,
            len: hd,
        },
        vec![qkv],
        qkv_part_shape.clone(),
    );
    let k = out.add_node(
        Op::Narrow {
            axis: 2,
            start: hd,
            len: hd,
        },
        vec![qkv],
        qkv_part_shape.clone(),
    );
    let v = out.add_node(
        Op::Narrow {
            axis: 2,
            start: 2 * hd,
            len: hd,
        },
        vec![qkv],
        qkv_part_shape,
    );

    // Reshape to [B, S, H, D], transpose to [B, H, S, D].
    let r4_shape = IrShape::from_dims(&[b, s, Dim::Static(nh), Dim::Static(dh)], dtype);
    let bhsd_shape = IrShape::from_dims(&[b, Dim::Static(nh), s, Dim::Static(dh)], dtype);

    let s_static = match s {
        Dim::Static(n) => n,
        _ => panic!("FAB unfuse: dyn S"),
    };
    let b_static = match b {
        Dim::Static(n) => n,
        _ => panic!("FAB unfuse: dyn B"),
    };
    let r4_dims_i64 = vec![b_static as i64, s_static as i64, nh as i64, dh as i64];

    let q_4d = out.reshape(q, r4_dims_i64.clone(), r4_shape.clone());
    let k_4d = out.reshape(k, r4_dims_i64.clone(), r4_shape.clone());
    let v_4d = out.reshape(v, r4_dims_i64, r4_shape);

    let q_h = out.add_node(
        Op::Transpose {
            perm: vec![0, 2, 1, 3],
        },
        vec![q_4d],
        bhsd_shape.clone(),
    );
    let k_h = out.add_node(
        Op::Transpose {
            perm: vec![0, 2, 1, 3],
        },
        vec![k_4d],
        bhsd_shape.clone(),
    );
    let v_h = out.add_node(
        Op::Transpose {
            perm: vec![0, 2, 1, 3],
        },
        vec![v_4d],
        bhsd_shape.clone(),
    );

    let (q_h, k_h) = if let (Some(rc), Some(rs)) = (in_rope_cos, in_rope_sin) {
        let q_rot = out.add_node(
            Op::Rope {
                head_dim: dh,
                n_rot: dh,
                // Unfusing a fused attention block — preserve the
                // pre-fusion default rope style (NeoX was the only
                // style; thread the source style here once fused ops
                // carry it).
                style: RopeStyle::NeoX,
            },
            vec![q_h, rc, rs],
            bhsd_shape.clone(),
        );
        let k_rot = out.add_node(
            Op::Rope {
                head_dim: dh,
                n_rot: dh,
                style: RopeStyle::NeoX,
            },
            vec![k_h, rc, rs],
            bhsd_shape.clone(),
        );
        (q_rot, k_rot)
    } else {
        (q_h, k_h)
    };

    // Attention with custom mask (4-input form).
    let attn_h = out.attention(q_h, k_h, v_h, in_mask, nh, dh, bhsd_shape);

    // Transpose back to [B, S, H, D] and reshape to [B, S, H*D].
    let bshd_shape = IrShape::from_dims(&[b, s, Dim::Static(nh), Dim::Static(dh)], dtype);
    let attn_back = out.add_node(
        Op::Transpose {
            perm: vec![0, 2, 1, 3],
        },
        vec![attn_h],
        bshd_shape,
    );
    let bsh_shape = IrShape::from_dims(&[b, s, Dim::Static(hd)], dtype);
    let attn_2d = out.reshape(
        attn_back,
        vec![b_static as i64, s_static as i64, hd as i64],
        bsh_shape.clone(),
    );

    // Output projection.
    let mut out_node = out.matmul(attn_2d, in_out_w, bsh_shape.clone());
    if let Some(ob) = in_out_b {
        let ob_b = out.add_node(
            Op::Expand {
                target_shape: bsh_shape
                    .dims()
                    .iter()
                    .map(|d| match d {
                        Dim::Static(n) => *n as i64,
                        _ => -1,
                    })
                    .collect(),
            },
            vec![ob],
            bsh_shape.clone(),
        );
        out_node = out.binary(BinaryOp::Add, out_node, ob_b, bsh_shape);
    }
    out_node
}


/// Decompose **only** `Op::FusedAttentionBlock` nodes into primitives,
/// leaving every other op untouched (including other fused ops a backend
/// may lower natively, e.g. `FusedMatMulBiasAct` / `FusedResidualLN`).
///
/// This is the backend-facing counterpart to [`unfuse_fused_for_autodiff`]:
/// backends that *declare* `OpKind::FusedAttentionBlock` (so the
/// `FuseAttentionBlock` pass fires) but have no monolithic fused-attention
/// kernel run this to lower the block down to the primitive chain they do
/// implement. It is idempotent and returns `g` unchanged (no rebuild) when
/// no FAB node is present, so it is cheap to call unconditionally in a
/// backend's compile path.
pub fn unfuse_attention_block(g: Graph) -> Graph {
    if !g
        .nodes()
        .iter()
        .any(|n| matches!(n.op, Op::FusedAttentionBlock { .. }))
    {
        return g;
    }

    let mut out = Graph::new(g.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    let original_outputs = g.outputs.clone();
    let nodes: Vec<rlx_ir::Node> = g.nodes().to_vec();

    for node in &nodes {
        let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let new_id = match &node.op {
            Op::FusedAttentionBlock {
                num_heads,
                head_dim,
                has_bias,
                has_rope,
            } => expand_attention_block(
                &mut out,
                &new_inputs,
                *num_heads,
                *head_dim,
                *has_bias,
                *has_rope,
            ),
            other => out.add_node(other.clone(), new_inputs, node.shape.clone()),
        };
        id_map.insert(node.id, new_id);
    }

    let new_outputs: Vec<NodeId> = original_outputs.iter().map(|i| id_map[i]).collect();
    out.set_outputs(new_outputs);
    out
}

