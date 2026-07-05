// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! `fused` — extracted from the `unfuse` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use rlx_ir::op::*;
use rlx_ir::shape::Shape as IrShape;
use rlx_ir::{Dim, Graph, NodeId, Op};
use std::collections::HashMap;

use super::*;

pub(super) fn unfuse_fused_mat_mul_bias_act(
    node: &rlx_ir::Node,
    new_inputs: Vec<NodeId>,
    out: &mut Graph,
) -> NodeId {
    let Op::FusedMatMulBiasAct { activation } = &node.op else {
        unreachable!()
    };
    {
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
}

pub(super) fn unfuse_fused_residual_l_n(
    node: &rlx_ir::Node,
    new_inputs: Vec<NodeId>,
    out: &mut Graph,
) -> NodeId {
    let Op::FusedResidualLN { has_bias, eps } = &node.op else {
        unreachable!()
    };
    {
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
}

pub(super) fn unfuse_fused_residual_rms_norm(
    node: &rlx_ir::Node,
    new_inputs: Vec<NodeId>,
    out: &mut Graph,
) -> NodeId {
    let Op::FusedResidualRmsNorm { has_bias, eps } = &node.op else {
        unreachable!()
    };
    {
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
}

pub(super) fn unfuse_fused_attention_block(
    node: &rlx_ir::Node,
    new_inputs: Vec<NodeId>,
    out: &mut Graph,
) -> NodeId {
    let Op::FusedAttentionBlock {
        num_heads,
        head_dim,
        has_bias,
        has_rope,
    } = &node.op
    else {
        unreachable!()
    };
    expand_attention_block(
        &mut *out,
        &new_inputs,
        *num_heads,
        *head_dim,
        *has_bias,
        *has_rope,
    )
}

pub(super) fn unfuse_fused_transformer_layer(
    node: &rlx_ir::Node,
    new_inputs: Vec<NodeId>,
    out: &mut Graph,
) -> NodeId {
    let Op::FusedTransformerLayer {
        num_heads,
        head_dim,
        intermediate_size,
        eps1,
        eps2,
        activation,
        has_bias,
    } = &node.op
    else {
        unreachable!()
    };
    {
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
        let bhsd_shape = IrShape::from_dims(&[b, Dim::Static(nh), s, Dim::Static(dh)], dtype);
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
        let bshd_shape = IrShape::from_dims(&[b, s, Dim::Static(nh), Dim::Static(dh)], dtype);
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
}

pub(super) fn unfuse_fused_swi_g_l_u(
    node: &rlx_ir::Node,
    new_inputs: Vec<NodeId>,
    out: &mut Graph,
) -> NodeId {
    let Op::FusedSwiGLU { cast_to, .. } = &node.op else {
        unreachable!()
    };
    {
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
}

pub(super) fn unfuse_lora_mat_mul(
    node: &rlx_ir::Node,
    new_inputs: Vec<NodeId>,
    out: &mut Graph,
) -> NodeId {
    let Op::LoraMatMul { scale } = &node.op else {
        unreachable!()
    };
    {
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
}

/// Decompose [`Op::PartitionedConv`] into the batched-GEMM frequency-domain
/// path (`rfft → complex matmul over partitions → irfft`). Reuses the validated
/// [`rlx_ir::Graph::partitioned_conv1d_gemm`] builder, so the whole thing lowers
/// as primitive `Op::Fft` / `Op::MatMul` / `Op::Reverse` / elementwise nodes.
pub(super) fn unfuse_partitioned_conv(
    node: &rlx_ir::Node,
    new_inputs: Vec<NodeId>,
    out: &mut Graph,
) -> NodeId {
    let Op::PartitionedConv { block } = &node.op else {
        unreachable!()
    };
    out.partitioned_conv1d_gemm(new_inputs[0], new_inputs[1], *block)
}
