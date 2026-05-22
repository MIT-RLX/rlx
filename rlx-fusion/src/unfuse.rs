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
        let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let new_id = match &node.op {
            Op::FusedMatMulBiasAct { activation } => {
                // Inputs: [input, weight, bias]. Decomposes to:
                //   y0 = MatMul(input, weight)
                //   y1 = y0 + bias_expanded
                //   y2 = activation(y1)   [if Some(act)]
                let in_x = new_inputs[0];
                let in_w = new_inputs[1];
                let in_b = new_inputs[2];
                let y_shape = node.shape.clone();
                let y0 = out.matmul(in_x, in_w, y_shape.clone());
                let bias_b = out.add_node(
                    Op::Expand {
                        target_shape: y_shape
                            .dims()
                            .iter()
                            .map(|d| match d {
                                Dim::Static(n) => *n as i64,
                                _ => -1,
                            })
                            .collect(),
                    },
                    vec![in_b],
                    y_shape.clone(),
                );
                let y1 = out.binary(BinaryOp::Add, y0, bias_b, y_shape.clone());
                if let Some(act) = activation {
                    out.activation(*act, y1, y_shape)
                } else {
                    y1
                }
            }
            Op::FusedResidualLN { has_bias, eps } => {
                // Inputs: [x, residual, [bias], gamma, beta]
                // Decomposes to:
                //   r = x + residual
                //   r' = r + bias_expanded   [if has_bias]
                //   y = LayerNorm(r', gamma, beta, axis=-1, eps)
                let in_x = new_inputs[0];
                let in_res = new_inputs[1];
                let (in_bias, in_gamma, in_beta) = if *has_bias {
                    (Some(new_inputs[2]), new_inputs[3], new_inputs[4])
                } else {
                    (None, new_inputs[2], new_inputs[3])
                };
                let y_shape = node.shape.clone();
                let r0 = out.binary(BinaryOp::Add, in_x, in_res, y_shape.clone());
                let r1 = if let Some(b) = in_bias {
                    let bias_b = out.add_node(
                        Op::Expand {
                            target_shape: y_shape
                                .dims()
                                .iter()
                                .map(|d| match d {
                                    Dim::Static(n) => *n as i64,
                                    _ => -1,
                                })
                                .collect(),
                        },
                        vec![b],
                        y_shape.clone(),
                    );
                    out.binary(BinaryOp::Add, r0, bias_b, y_shape.clone())
                } else {
                    r0
                };
                out.layer_norm(r1, in_gamma, in_beta, -1, *eps, y_shape)
            }
            Op::FusedResidualRmsNorm { has_bias, eps } => {
                let in_x = new_inputs[0];
                let in_res = new_inputs[1];
                let (in_bias, in_gamma, in_beta) = if *has_bias {
                    (Some(new_inputs[2]), new_inputs[3], new_inputs[4])
                } else {
                    (None, new_inputs[2], new_inputs[3])
                };
                let y_shape = node.shape.clone();
                let r0 = out.binary(BinaryOp::Add, in_x, in_res, y_shape.clone());
                let r1 = if let Some(b) = in_bias {
                    let bias_b = out.add_node(
                        Op::Expand {
                            target_shape: y_shape
                                .dims()
                                .iter()
                                .map(|d| match d {
                                    Dim::Static(n) => *n as i64,
                                    _ => -1,
                                })
                                .collect(),
                        },
                        vec![b],
                        y_shape.clone(),
                    );
                    out.binary(BinaryOp::Add, r0, bias_b, y_shape.clone())
                } else {
                    r0
                };
                use rlx_ir::infer::GraphExt;
                out.rms_norm(r1, in_gamma, in_beta, *eps)
            }
            Op::FusedAttentionBlock {
                num_heads,
                head_dim,
                has_bias,
                has_rope,
            } => {
                // Inputs (in order):
                //   hidden, qkv_w, out_w, mask,
                //   [qkv_b, out_b]      if has_bias,
                //   [rope_cos, rope_sin] if has_rope
                // Decomposition:
                //   qkv  = hidden @ qkv_w [+ qkv_b]
                //   q,k,v = Narrow(qkv) ×3 → [B,S,H*D] each
                //   q_h,k_h,v_h = reshape+transpose to [B,H,S,D]
                //   [if has_rope] q_h = Rope(q_h, cos, sin),
                //                 k_h = Rope(k_h, cos, sin)
                //   attn_h = Attention(q_h, k_h, v_h, mask, Custom)
                //   attn   = transpose+reshape back to [B,S,H*D]
                //   out    = attn @ out_w [+ out_b]
                let nh = *num_heads;
                let dh = *head_dim;
                let hd = nh * dh;
                let in_hidden = new_inputs[0];
                let in_qkv_w = new_inputs[1];
                let in_out_w = new_inputs[2];
                let in_mask = new_inputs[3];
                let mut next_idx = 4;
                let (in_qkv_b, in_out_b) = if *has_bias {
                    let qb = new_inputs[next_idx];
                    let ob = new_inputs[next_idx + 1];
                    next_idx += 2;
                    (Some(qb), Some(ob))
                } else {
                    (None, None)
                };
                let (in_rope_cos, in_rope_sin) = if *has_rope {
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
                let bhsd_shape =
                    IrShape::from_dims(&[b, Dim::Static(nh), s, Dim::Static(dh)], dtype);

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
                        },
                        vec![q_h, rc, rs],
                        bhsd_shape.clone(),
                    );
                    let k_rot = out.add_node(
                        Op::Rope {
                            head_dim: dh,
                            n_rot: dh,
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
                let bshd_shape =
                    IrShape::from_dims(&[b, s, Dim::Static(nh), Dim::Static(dh)], dtype);
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
            Op::FusedTransformerLayer {
                num_heads,
                head_dim,
                intermediate_size,
                eps1,
                eps2,
                activation,
                has_bias,
            } => {
                // BERT-style post-norm transformer layer. Decomposes
                // to primitive ops (matmul, add, narrow, attention,
                // layer_norm, activation) so every step has a VJP
                // rule. Output shape == hidden shape.
                //
                // Inputs (with bias, 14 entries):
                //   0 hidden, 1 qkv_w, 2 qkv_b, 3 out_w, 4 out_b,
                //   5 ln1_g, 6 ln1_b, 7 fc1_w, 8 fc1_b,
                //   9 fc2_w, 10 fc2_b, 11 ln2_g, 12 ln2_b, 13 mask
                // Without bias (8 entries):
                //   0 hidden, 1 qkv_w, 2 out_w, 3 ln1_g, 4 fc1_w,
                //   5 fc2_w, 6 ln2_g, 7 mask
                let nh = *num_heads;
                let dh = *head_dim;
                let inner = nh * dh;
                let inter = *intermediate_size;
                let h_shape = node.shape.clone();
                let dtype = h_shape.dtype();
                let b = h_shape.dim(0);
                let s = h_shape.dim(1);
                let h_dim = match h_shape.dim(2) {
                    Dim::Static(n) => n,
                    _ => panic!("FTL unfuse: dynamic hidden dim"),
                };

                let (
                    in_hidden,
                    in_qkv_w,
                    in_qkv_b,
                    in_out_w,
                    in_out_b,
                    in_ln1_g,
                    in_ln1_b,
                    in_fc1_w,
                    in_fc1_b,
                    in_fc2_w,
                    in_fc2_b,
                    in_ln2_g,
                    in_ln2_b,
                    in_mask,
                ) = if *has_bias {
                    (
                        new_inputs[0],
                        new_inputs[1],
                        Some(new_inputs[2]),
                        new_inputs[3],
                        Some(new_inputs[4]),
                        new_inputs[5],
                        new_inputs[6],
                        new_inputs[7],
                        Some(new_inputs[8]),
                        new_inputs[9],
                        Some(new_inputs[10]),
                        new_inputs[11],
                        new_inputs[12],
                        new_inputs[13],
                    )
                } else {
                    // Synthesize zero beta vectors for the two
                    // LayerNorms so we can always emit Op::LayerNorm
                    // (which takes a beta input). Shape [H_dim].
                    let zero_bytes = vec![0u8; h_dim * 4];
                    let zero_beta_shape = IrShape::from_dims(&[Dim::Static(h_dim)], dtype);
                    let zero_beta =
                        out.add_node(Op::Constant { data: zero_bytes }, vec![], zero_beta_shape);
                    (
                        new_inputs[0],
                        new_inputs[1],
                        None,
                        new_inputs[2],
                        None,
                        new_inputs[3],
                        zero_beta,
                        new_inputs[4],
                        None,
                        new_inputs[5],
                        None,
                        new_inputs[6],
                        zero_beta,
                        new_inputs[7],
                    )
                };

                // 1) qkv projection.
                let qkv_shape = IrShape::from_dims(&[b, s, Dim::Static(3 * inner)], dtype);
                let mut qkv = out.matmul(in_hidden, in_qkv_w, qkv_shape.clone());
                if let Some(qb) = in_qkv_b {
                    let qb_e = out.add_node(
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
                    qkv = out.binary(BinaryOp::Add, qkv, qb_e, qkv_shape);
                }

                // 2) Narrow into Q/K/V, each [B, S, H*D].
                let proj_shape = IrShape::from_dims(&[b, s, Dim::Static(inner)], dtype);
                let q = out.add_node(
                    Op::Narrow {
                        axis: 2,
                        start: 0,
                        len: inner,
                    },
                    vec![qkv],
                    proj_shape.clone(),
                );
                let k = out.add_node(
                    Op::Narrow {
                        axis: 2,
                        start: inner,
                        len: inner,
                    },
                    vec![qkv],
                    proj_shape.clone(),
                );
                let v = out.add_node(
                    Op::Narrow {
                        axis: 2,
                        start: 2 * inner,
                        len: inner,
                    },
                    vec![qkv],
                    proj_shape.clone(),
                );

                // 3) Attention. The autodiff Attention VJP assumes
                // rank-4 [B, H, S, D] layout, so reshape Q/K/V from
                // [B, S, H*D] → [B, S, H, D] → transpose → [B, H, S, D],
                // run attention, then transpose+reshape back to
                // [B, S, H*D].
                let r4_shape = IrShape::from_dims(&[b, s, Dim::Static(nh), Dim::Static(dh)], dtype);
                let bhsd_shape =
                    IrShape::from_dims(&[b, Dim::Static(nh), s, Dim::Static(dh)], dtype);
                let s_static = match s {
                    Dim::Static(n) => n,
                    _ => panic!("FTL unfuse: dyn S"),
                };
                let b_static = match b {
                    Dim::Static(n) => n,
                    _ => panic!("FTL unfuse: dyn B"),
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
                let attn_h = out.attention(q_h, k_h, v_h, in_mask, nh, dh, bhsd_shape);
                let bshd_shape =
                    IrShape::from_dims(&[b, s, Dim::Static(nh), Dim::Static(dh)], dtype);
                let attn_back = out.add_node(
                    Op::Transpose {
                        perm: vec![0, 2, 1, 3],
                    },
                    vec![attn_h],
                    bshd_shape,
                );
                let attn = out.reshape(
                    attn_back,
                    vec![b_static as i64, s_static as i64, inner as i64],
                    proj_shape.clone(),
                );

                // 4) Output projection.
                let mut attn_out = out.matmul(attn, in_out_w, h_shape.clone());
                if let Some(ob) = in_out_b {
                    let ob_e = out.add_node(
                        Op::Expand {
                            target_shape: h_shape
                                .dims()
                                .iter()
                                .map(|d| match d {
                                    Dim::Static(n) => *n as i64,
                                    _ => -1,
                                })
                                .collect(),
                        },
                        vec![ob],
                        h_shape.clone(),
                    );
                    attn_out = out.binary(BinaryOp::Add, attn_out, ob_e, h_shape.clone());
                }

                // 5) Residual + LayerNorm 1.
                let r1 = out.binary(BinaryOp::Add, attn_out, in_hidden, h_shape.clone());
                let h1 = out.layer_norm(r1, in_ln1_g, in_ln1_b, -1, *eps1, h_shape.clone());

                // 6) FFN: act(h1 @ fc1_w + fc1_b) @ fc2_w + fc2_b.
                let inter_shape = IrShape::from_dims(&[b, s, Dim::Static(inter)], dtype);
                let mut fc1 = out.matmul(h1, in_fc1_w, inter_shape.clone());
                if let Some(fb) = in_fc1_b {
                    let fb_e = out.add_node(
                        Op::Expand {
                            target_shape: inter_shape
                                .dims()
                                .iter()
                                .map(|d| match d {
                                    Dim::Static(n) => *n as i64,
                                    _ => -1,
                                })
                                .collect(),
                        },
                        vec![fb],
                        inter_shape.clone(),
                    );
                    fc1 = out.binary(BinaryOp::Add, fc1, fb_e, inter_shape.clone());
                }
                let fc1_act = out.activation(*activation, fc1, inter_shape.clone());

                let mut ffn_out = out.matmul(fc1_act, in_fc2_w, h_shape.clone());
                if let Some(fb) = in_fc2_b {
                    let fb_e = out.add_node(
                        Op::Expand {
                            target_shape: h_shape
                                .dims()
                                .iter()
                                .map(|d| match d {
                                    Dim::Static(n) => *n as i64,
                                    _ => -1,
                                })
                                .collect(),
                        },
                        vec![fb],
                        h_shape.clone(),
                    );
                    ffn_out = out.binary(BinaryOp::Add, ffn_out, fb_e, h_shape.clone());
                }

                // 7) Residual + LayerNorm 2.
                let r2 = out.binary(BinaryOp::Add, ffn_out, h1, h_shape.clone());
                out.layer_norm(r2, in_ln2_g, in_ln2_b, -1, *eps2, h_shape)
            }
            Op::FusedSwiGLU { cast_to, .. } => {
                // Inputs: [packed]. Forward splits the last axis
                // into [up | gate] halves, computes
                //   out = silu(gate) * up
                // Optionally cast at the end.
                let in_packed = new_inputs[0];
                let in_shape = out.node(in_packed).shape.clone();
                let dtype = in_shape.dtype();
                let rank = in_shape.rank();
                let last = rank - 1;
                let total = match in_shape.dim(last) {
                    Dim::Static(n) => n,
                    _ => panic!("FusedSwiGLU unfuse: dynamic last dim"),
                };
                let half = total / 2;
                let mut half_dims: Vec<Dim> = in_shape.dims().to_vec();
                half_dims[last] = Dim::Static(half);
                let half_shape = IrShape::from_dims(&half_dims, dtype);

                let up = out.add_node(
                    Op::Narrow {
                        axis: last,
                        start: 0,
                        len: half,
                    },
                    vec![in_packed],
                    half_shape.clone(),
                );
                let gate = out.add_node(
                    Op::Narrow {
                        axis: last,
                        start: half,
                        len: half,
                    },
                    vec![in_packed],
                    half_shape.clone(),
                );
                let gate_silu = out.activation(Activation::Silu, gate, half_shape.clone());
                let prod = out.binary(BinaryOp::Mul, gate_silu, up, half_shape.clone());
                if let Some(target) = cast_to {
                    let cast_shape = IrShape::from_dims(&half_dims, *target);
                    out.add_node(Op::Cast { to: *target }, vec![prod], cast_shape)
                } else {
                    prod
                }
            }
            Op::LoraMatMul { scale } => {
                // Inputs: [x, w, a, b]. Decomposes to:
                //   y_main = x @ w
                //   inter  = x @ a
                //   lora   = (inter @ b) * scale
                //   y      = y_main + lora
                let in_x = new_inputs[0];
                let in_w = new_inputs[1];
                let in_a = new_inputs[2];
                let in_b = new_inputs[3];
                let y_shape = node.shape.clone();

                let y_main = out.matmul(in_x, in_w, y_shape.clone());

                // inter shape: replace last dim of x with `r`.
                let x_shape = out.node(in_x).shape.clone();
                let a_shape = out.node(in_a).shape.clone();
                let r = a_shape.dim(a_shape.rank() - 1);
                let mut inter_dims: Vec<Dim> = x_shape.dims().to_vec();
                *inter_dims.last_mut().unwrap() = r;
                let inter_shape = IrShape::from_dims(&inter_dims, x_shape.dtype());
                let inter = out.matmul(in_x, in_a, inter_shape);

                let lora_unscaled = out.matmul(inter, in_b, y_shape.clone());
                let scale_bytes = scale.to_le_bytes().to_vec();
                let scale_scalar = out.add_node(
                    Op::Constant { data: scale_bytes },
                    vec![],
                    IrShape::from_dims(&[Dim::Static(1)], x_shape.dtype()),
                );
                let scale_b = out.add_node(
                    Op::Expand {
                        target_shape: y_shape
                            .dims()
                            .iter()
                            .map(|d| match d {
                                Dim::Static(n) => *n as i64,
                                _ => -1,
                            })
                            .collect(),
                    },
                    vec![scale_scalar],
                    y_shape.clone(),
                );
                let lora = out.binary(BinaryOp::Mul, lora_unscaled, scale_b, y_shape.clone());

                out.binary(BinaryOp::Add, y_main, lora, y_shape)
            }
            Op::GatedDeltaNet {
                state_size,
                carry_state,
            } => {
                // Gated DeltaNet linear-attention scan. Decomposes by
                // unrolling the time loop so every step is MatMul /
                // Mul / Add / Sub / Exp / Concat / Narrow / Reshape —
                // the gradient walk reaches them via existing VJPs
                // (mirrors rlx-mlx `lower_gated_delta_net` and CPU
                // `execute_gated_delta_net_f32`).
                //
                // Per timestep t (per batch row b, head h):
                //   S *= exp(g[t,h])
                //   sk = k[t] @ S            (row-vector × matrix)
                //   sk = (v[t] - sk) * beta[t]
                //   S += outer(k[t], sk)
                //   out[t] = (q[t] @ S) / sqrt(n)
                //
                // Inputs: q,k,v [B,S,H,N]; g,beta [B,S,H];
                // optional state [B,H,N,N] when carry_state.
                let n = *state_size;
                let in_q = new_inputs[0];
                let in_k = new_inputs[1];
                let in_v = new_inputs[2];
                let in_g = new_inputs[3];
                let in_beta = new_inputs[4];

                let q_shape = out.node(in_q).shape.clone();
                let dtype = q_shape.dtype();
                let b_dim = match q_shape.dim(0) {
                    Dim::Static(v) => v,
                    _ => panic!("GatedDeltaNet unfuse: dynamic B"),
                };
                let s_dim = match q_shape.dim(1) {
                    Dim::Static(v) => v,
                    _ => panic!("GatedDeltaNet unfuse: dynamic S"),
                };
                let h_dim = match q_shape.dim(2) {
                    Dim::Static(v) => v,
                    _ => panic!("GatedDeltaNet unfuse: dynamic H"),
                };
                if q_shape.dim(3) != Dim::Static(n) {
                    panic!("GatedDeltaNet unfuse: q last dim != state_size");
                }

                let bh = b_dim * h_dim;
                let bhnn = IrShape::from_dims(
                    &[Dim::Static(bh), Dim::Static(n), Dim::Static(n)],
                    dtype,
                );
                let bh1n = IrShape::from_dims(
                    &[Dim::Static(bh), Dim::Static(1), Dim::Static(n)],
                    dtype,
                );
                let bh11 = IrShape::from_dims(
                    &[Dim::Static(bh), Dim::Static(1), Dim::Static(1)],
                    dtype,
                );
                let bh_n1 = IrShape::from_dims(
                    &[Dim::Static(bh), Dim::Static(n), Dim::Static(1)],
                    dtype,
                );
                let bhnn_i64 = vec![bh as i64, n as i64, n as i64];
                let bh1n_i64 = vec![bh as i64, 1, n as i64];

                let bhn = IrShape::from_dims(
                    &[Dim::Static(b_dim), Dim::Static(h_dim), Dim::Static(n)],
                    dtype,
                );
                let b1h = IrShape::from_dims(
                    &[Dim::Static(b_dim), Dim::Static(1), Dim::Static(h_dim)],
                    dtype,
                );
                let b1hn = IrShape::from_dims(
                    &[Dim::Static(b_dim), Dim::Static(1), Dim::Static(h_dim), Dim::Static(n)],
                    dtype,
                );
                let bhnn4 = IrShape::from_dims(
                    &[
                        Dim::Static(b_dim),
                        Dim::Static(h_dim),
                        Dim::Static(n),
                        Dim::Static(n),
                    ],
                    dtype,
                );

                let mut state = if *carry_state {
                    new_inputs[5]
                } else {
                    let zero_bytes = vec![0u8; b_dim * h_dim * n * n * 4];
                    out.add_node(Op::Constant { data: zero_bytes }, vec![], bhnn4.clone())
                };

                let scale_val = (1.0f32 / (n as f32).sqrt()).to_le_bytes().to_vec();
                let scale_scalar = out.add_node(
                    Op::Constant { data: scale_val },
                    vec![],
                    IrShape::from_dims(&[Dim::Static(1)], dtype),
                );
                let scale_111 = out.reshape(
                    scale_scalar,
                    vec![1, 1, 1],
                    IrShape::from_dims(&[Dim::Static(1), Dim::Static(1), Dim::Static(1)], dtype),
                );
                let scale_bh1n = out.add_node(
                    Op::Expand {
                        target_shape: bh1n_i64.clone(),
                    },
                    vec![scale_111],
                    bh1n.clone(),
                );

                let mut ys: Vec<NodeId> = Vec::with_capacity(s_dim);

                for t in 0..s_dim {
                    let qt_b1hn = out.add_node(
                        Op::Narrow {
                            axis: 1,
                            start: t,
                            len: 1,
                        },
                        vec![in_q],
                        b1hn.clone(),
                    );
                    let kt_b1hn = out.add_node(
                        Op::Narrow {
                            axis: 1,
                            start: t,
                            len: 1,
                        },
                        vec![in_k],
                        b1hn.clone(),
                    );
                    let vt_b1hn = out.add_node(
                        Op::Narrow {
                            axis: 1,
                            start: t,
                            len: 1,
                        },
                        vec![in_v],
                        b1hn.clone(),
                    );
                    let gt_b1h = out.add_node(
                        Op::Narrow {
                            axis: 1,
                            start: t,
                            len: 1,
                        },
                        vec![in_g],
                        b1h.clone(),
                    );
                    let beta_b1h = out.add_node(
                        Op::Narrow {
                            axis: 1,
                            start: t,
                            len: 1,
                        },
                        vec![in_beta],
                        b1h.clone(),
                    );

                    let gt_bhn = out.reshape(
                        gt_b1h,
                        vec![b_dim as i64, h_dim as i64, 1],
                        IrShape::from_dims(
                            &[Dim::Static(b_dim), Dim::Static(h_dim), Dim::Static(1)],
                            dtype,
                        ),
                    );
                    let gt_bh11 = out.reshape(
                        gt_bhn,
                        vec![bh as i64, 1, 1],
                        bh11.clone(),
                    );
                    let gt_bhnn = out.add_node(
                        Op::Expand {
                            target_shape: bhnn_i64.clone(),
                        },
                        vec![gt_bh11],
                        bhnn.clone(),
                    );
                    let exp_g = out.activation(Activation::Exp, gt_bhnn, bhnn.clone());

                    let state_bhnn = out.reshape(
                        state,
                        vec![bh as i64, n as i64, n as i64],
                        bhnn.clone(),
                    );
                    let damped = out.binary(BinaryOp::Mul, exp_g, state_bhnn, bhnn.clone());
                    state = out.reshape(
                        damped,
                        vec![b_dim as i64, h_dim as i64, n as i64, n as i64],
                        bhnn4.clone(),
                    );

                    let kt_bh1n = out.reshape(
                        kt_b1hn,
                        vec![bh as i64, 1, n as i64],
                        bh1n.clone(),
                    );
                    let vt_bh1n = out.reshape(
                        vt_b1hn,
                        vec![bh as i64, 1, n as i64],
                        bh1n.clone(),
                    );
                    let state_bhnn = out.reshape(
                        state,
                        vec![bh as i64, n as i64, n as i64],
                        bhnn.clone(),
                    );

                    let mut sk = out.matmul(kt_bh1n, state_bhnn, bh1n.clone());
                    sk = out.binary(BinaryOp::Sub, vt_bh1n, sk, bh1n.clone());

                    let beta_bhn = out.reshape(
                        beta_b1h,
                        vec![b_dim as i64, h_dim as i64, 1],
                        IrShape::from_dims(
                            &[Dim::Static(b_dim), Dim::Static(h_dim), Dim::Static(1)],
                            dtype,
                        ),
                    );
                    let beta_bh11 = out.reshape(beta_bhn, vec![bh as i64, 1, 1], bh11.clone());
                    let beta_bh1n = out.add_node(
                        Op::Expand {
                            target_shape: bh1n_i64.clone(),
                        },
                        vec![beta_bh11],
                        bh1n.clone(),
                    );
                    sk = out.binary(BinaryOp::Mul, sk, beta_bh1n, bh1n.clone());

                    let kt_bhn = out.reshape(
                        kt_b1hn,
                        vec![b_dim as i64, h_dim as i64, n as i64],
                        bhn.clone(),
                    );
                    let kt_bhn1 = out.reshape(kt_bhn, vec![bh as i64, n as i64, 1], bh_n1.clone());
                    let sk_bh1 = out.reshape(sk, vec![bh as i64, 1, n as i64], bh1n.clone());
                    let outer = out.binary(BinaryOp::Mul, kt_bhn1, sk_bh1, bhnn.clone());
                    let state_bhnn = out.reshape(
                        state,
                        vec![bh as i64, n as i64, n as i64],
                        bhnn.clone(),
                    );
                    state = out.binary(
                        BinaryOp::Add,
                        state_bhnn,
                        outer,
                        bhnn.clone(),
                    );
                    state = out.reshape(
                        state,
                        vec![b_dim as i64, h_dim as i64, n as i64, n as i64],
                        bhnn4.clone(),
                    );

                    let qt_bh1n = out.reshape(
                        qt_b1hn,
                        vec![bh as i64, 1, n as i64],
                        bh1n.clone(),
                    );
                    let state_bhnn = out.reshape(
                        state,
                        vec![bh as i64, n as i64, n as i64],
                        bhnn.clone(),
                    );
                    let mut out_t = out.matmul(qt_bh1n, state_bhnn, bh1n.clone());
                    out_t = out.binary(BinaryOp::Mul, out_t, scale_bh1n, bh1n.clone());
                    let out_b1hn = out.reshape(
                        out_t,
                        vec![b_dim as i64, 1, h_dim as i64, n as i64],
                        b1hn.clone(),
                    );
                    ys.push(out_b1hn);
                }

                if ys.len() == 1 {
                    ys.pop().unwrap()
                } else {
                    out.add_node(Op::Concat { axis: 1 }, ys, node.shape.clone())
                }
            }
            Op::SelectiveScan { state_size } => {
                // Mamba SSM step. Decomposes by unrolling the time
                // loop (which makes every primitive a normal IR op
                // and the gradient walk reaches it via Mul / Add /
                // Activation::Exp / Reduce::Sum / Concat / Narrow /
                // Reshape / Expand VJPs — no special backward op).
                //
                // Recurrence per t:
                //   state_t = exp(δ_t * A) * state_{t-1} + δ_t * B_t * x_t
                //   y_t     = sum_n( C_t * state_t )
                //
                // Inputs: x [B,S,H], delta [B,S,H], a [H,N],
                //         b [B,S,N], c [B,S,N]
                // Output: y [B,S,H]
                //
                // Mirrors the rlx-mlx lowering structure (which also
                // unrolls the time loop because MLX has no native
                // scan primitive); this version emits IR nodes
                // instead of MLX arrays.
                let n = *state_size;
                let in_x = new_inputs[0];
                let in_delta = new_inputs[1];
                let in_a = new_inputs[2];
                let in_b = new_inputs[3];
                let in_c = new_inputs[4];

                let x_shape = out.node(in_x).shape.clone();
                let dtype = x_shape.dtype();
                let b_dim = match x_shape.dim(0) {
                    Dim::Static(v) => v,
                    _ => panic!("SelectiveScan unfuse: dynamic B"),
                };
                let s_dim = match x_shape.dim(1) {
                    Dim::Static(v) => v,
                    _ => panic!("SelectiveScan unfuse: dynamic S"),
                };
                let h_dim = match x_shape.dim(2) {
                    Dim::Static(v) => v,
                    _ => panic!("SelectiveScan unfuse: dynamic H"),
                };

                // Pre-build common shapes.
                let bhn = IrShape::from_dims(
                    &[Dim::Static(b_dim), Dim::Static(h_dim), Dim::Static(n)],
                    dtype,
                );
                let bh1 = IrShape::from_dims(
                    &[Dim::Static(b_dim), Dim::Static(h_dim), Dim::Static(1)],
                    dtype,
                );
                let b1n = IrShape::from_dims(
                    &[Dim::Static(b_dim), Dim::Static(1), Dim::Static(n)],
                    dtype,
                );
                let bh = IrShape::from_dims(&[Dim::Static(b_dim), Dim::Static(h_dim)], dtype);
                let b1h = IrShape::from_dims(
                    &[Dim::Static(b_dim), Dim::Static(1), Dim::Static(h_dim)],
                    dtype,
                );
                let bs1h = IrShape::from_dims(
                    &[Dim::Static(b_dim), Dim::Static(s_dim), Dim::Static(h_dim)],
                    dtype,
                );
                let _ = bs1h;

                let bhn_i64 = vec![b_dim as i64, h_dim as i64, n as i64];

                // Initial state: zero [B, H, N].
                let zero_bytes = vec![0u8; b_dim * h_dim * n * 4];
                let mut state =
                    out.add_node(Op::Constant { data: zero_bytes }, vec![], bhn.clone());

                // a: [H, N] → reshape [1, H, N] → expand [B, H, N].
                let a_1hn = out.reshape(
                    in_a,
                    vec![1, h_dim as i64, n as i64],
                    IrShape::from_dims(
                        &[Dim::Static(1), Dim::Static(h_dim), Dim::Static(n)],
                        dtype,
                    ),
                );
                let a_bhn = out.add_node(
                    Op::Expand {
                        target_shape: bhn_i64.clone(),
                    },
                    vec![a_1hn],
                    bhn.clone(),
                );

                // Per-time-step output collector.
                let mut ys: Vec<NodeId> = Vec::with_capacity(s_dim);

                for t in 0..s_dim {
                    // Narrow x[:, t, :] -> [B, 1, H], reshape to [B, H, 1].
                    let xt_b1h = out.add_node(
                        Op::Narrow {
                            axis: 1,
                            start: t,
                            len: 1,
                        },
                        vec![in_x],
                        b1h.clone(),
                    );
                    let xt_bh1 =
                        out.reshape(xt_b1h, vec![b_dim as i64, h_dim as i64, 1], bh1.clone());

                    // Narrow delta[:, t, :] -> [B, 1, H] → [B, H, 1].
                    let dt_b1h = out.add_node(
                        Op::Narrow {
                            axis: 1,
                            start: t,
                            len: 1,
                        },
                        vec![in_delta],
                        b1h.clone(),
                    );
                    let dt_bh1 =
                        out.reshape(dt_b1h, vec![b_dim as i64, h_dim as i64, 1], bh1.clone());

                    // Narrow b[:, t, :] -> [B, 1, N].
                    let bt_b1n = out.add_node(
                        Op::Narrow {
                            axis: 1,
                            start: t,
                            len: 1,
                        },
                        vec![in_b],
                        b1n.clone(),
                    );
                    // Narrow c[:, t, :] -> [B, 1, N].
                    let ct_b1n = out.add_node(
                        Op::Narrow {
                            axis: 1,
                            start: t,
                            len: 1,
                        },
                        vec![in_c],
                        b1n.clone(),
                    );

                    // Broadcast helpers to [B, H, N]:
                    //   dt: [B, H, 1] → expand [B, H, N]
                    //   xt: [B, H, 1] → expand [B, H, N]
                    //   bt: [B, 1, N] → expand [B, H, N]
                    //   ct: [B, 1, N] → expand [B, H, N]
                    let dt_bhn = out.add_node(
                        Op::Expand {
                            target_shape: bhn_i64.clone(),
                        },
                        vec![dt_bh1],
                        bhn.clone(),
                    );
                    let xt_bhn = out.add_node(
                        Op::Expand {
                            target_shape: bhn_i64.clone(),
                        },
                        vec![xt_bh1],
                        bhn.clone(),
                    );
                    let bt_bhn = out.add_node(
                        Op::Expand {
                            target_shape: bhn_i64.clone(),
                        },
                        vec![bt_b1n],
                        bhn.clone(),
                    );
                    let ct_bhn = out.add_node(
                        Op::Expand {
                            target_shape: bhn_i64.clone(),
                        },
                        vec![ct_b1n],
                        bhn.clone(),
                    );

                    // delta_a = dt * a, then exp.
                    let delta_a = out.binary(BinaryOp::Mul, dt_bhn, a_bhn, bhn.clone());
                    let exp_da = out.activation(Activation::Exp, delta_a, bhn.clone());

                    // delta_bx = (dt * bt) * xt.
                    let dtb = out.binary(BinaryOp::Mul, dt_bhn, bt_bhn, bhn.clone());
                    let delta_bx = out.binary(BinaryOp::Mul, dtb, xt_bhn, bhn.clone());

                    // state = exp(δA) * state + δ B x.
                    let damped = out.binary(BinaryOp::Mul, exp_da, state, bhn.clone());
                    state = out.binary(BinaryOp::Add, damped, delta_bx, bhn.clone());

                    // y_t = sum_n(c * state) → [B, H], reshape to [B,1,H].
                    let cstate = out.binary(BinaryOp::Mul, ct_bhn, state, bhn.clone());
                    let yt_bh = out.add_node(
                        Op::Reduce {
                            op: ReduceOp::Sum,
                            axes: vec![2],
                            keep_dim: false,
                        },
                        vec![cstate],
                        bh.clone(),
                    );
                    let yt_b1h =
                        out.reshape(yt_bh, vec![b_dim as i64, 1, h_dim as i64], b1h.clone());
                    ys.push(yt_b1h);
                }

                // Concat along seq axis. S==1 short-circuits.
                if ys.len() == 1 {
                    ys.pop().unwrap()
                } else {
                    out.add_node(Op::Concat { axis: 1 }, ys, node.shape.clone())
                }
            }
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
