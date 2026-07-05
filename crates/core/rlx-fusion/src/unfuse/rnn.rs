// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! `rnn` — extracted from the `unfuse` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use rlx_ir::op::*;
use rlx_ir::shape::Shape as IrShape;
use rlx_ir::{Dim, Graph, NodeId, Op};
use std::collections::HashMap;

use super::*;

pub(super) fn unfuse_gated_delta_net(
    node: &rlx_ir::Node,
    new_inputs: Vec<NodeId>,
    out: &mut Graph,
) -> NodeId {
    let Op::GatedDeltaNet {
        state_size,
        carry_state,
    } = &node.op
    else {
        unreachable!()
    };
    {
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
        let bhnn = IrShape::from_dims(&[Dim::Static(bh), Dim::Static(n), Dim::Static(n)], dtype);
        let bh1n = IrShape::from_dims(&[Dim::Static(bh), Dim::Static(1), Dim::Static(n)], dtype);
        let bh11 = IrShape::from_dims(&[Dim::Static(bh), Dim::Static(1), Dim::Static(1)], dtype);
        let bh_n1 = IrShape::from_dims(&[Dim::Static(bh), Dim::Static(n), Dim::Static(1)], dtype);
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
            &[
                Dim::Static(b_dim),
                Dim::Static(1),
                Dim::Static(h_dim),
                Dim::Static(n),
            ],
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
            let gt_bh11 = out.reshape(gt_bhn, vec![bh as i64, 1, 1], bh11.clone());
            let gt_bhnn = out.add_node(
                Op::Expand {
                    target_shape: bhnn_i64.clone(),
                },
                vec![gt_bh11],
                bhnn.clone(),
            );
            let exp_g = out.activation(Activation::Exp, gt_bhnn, bhnn.clone());

            let state_bhnn = out.reshape(state, vec![bh as i64, n as i64, n as i64], bhnn.clone());
            let damped = out.binary(BinaryOp::Mul, exp_g, state_bhnn, bhnn.clone());
            state = out.reshape(
                damped,
                vec![b_dim as i64, h_dim as i64, n as i64, n as i64],
                bhnn4.clone(),
            );

            let kt_bh1n = out.reshape(kt_b1hn, vec![bh as i64, 1, n as i64], bh1n.clone());
            let vt_bh1n = out.reshape(vt_b1hn, vec![bh as i64, 1, n as i64], bh1n.clone());
            let state_bhnn = out.reshape(state, vec![bh as i64, n as i64, n as i64], bhnn.clone());

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
            let state_bhnn = out.reshape(state, vec![bh as i64, n as i64, n as i64], bhnn.clone());
            state = out.binary(BinaryOp::Add, state_bhnn, outer, bhnn.clone());
            state = out.reshape(
                state,
                vec![b_dim as i64, h_dim as i64, n as i64, n as i64],
                bhnn4.clone(),
            );

            let qt_bh1n = out.reshape(qt_b1hn, vec![bh as i64, 1, n as i64], bh1n.clone());
            let state_bhnn = out.reshape(state, vec![bh as i64, n as i64, n as i64], bhnn.clone());
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
}

pub(super) fn unfuse_lstm(node: &rlx_ir::Node, new_inputs: Vec<NodeId>, out: &mut Graph) -> NodeId {
    let Op::Lstm {
        hidden_size,
        num_layers,
        bidirectional,
        carry,
    } = &node.op
    else {
        unreachable!()
    };
    {
        // Multi-layer (optionally bidirectional, optional decode
        // carry) LSTM. Unrolls every (layer, direction, timestep)
        // into MatMul / Add / Mul / Sigmoid / Tanh / Narrow /
        // Reshape / Transpose / Expand / Concat, so the gradient
        // walk reaches it via existing VJPs and backends without a
        // native LSTM lower it via these primitives. Gate order
        // i, f, g, o. Mirrors `rlx_cpu::thunk::execute_lstm_f32`.
        // Carry seeds h/c from h0/c0 but (like GatedDeltaNet) does
        // not thread the final state back out — decode that needs
        // hn/cn uses the native/host path.
        let hidden = *hidden_size;
        let four_h = 4 * hidden;
        let l_count = *num_layers;
        let dirs = if *bidirectional { 2 } else { 1 };
        let in_x = new_inputs[0];
        let in_wih = new_inputs[1];
        let in_whh = new_inputs[2];
        let in_bias = new_inputs[3];

        let x_shape = out.node(in_x).shape.clone();
        let dtype = x_shape.dtype();
        let b_dim = match x_shape.dim(0) {
            Dim::Static(v) => v,
            _ => panic!("Lstm unfuse: dynamic B"),
        };
        let s_dim = match x_shape.dim(1) {
            Dim::Static(v) => v,
            _ => panic!("Lstm unfuse: dynamic S"),
        };
        let in_dim = match x_shape.dim(2) {
            Dim::Static(v) => v,
            _ => panic!("Lstm unfuse: dynamic input size"),
        };

        let sh = |dims: &[usize]| {
            IrShape::from_dims(
                &dims.iter().map(|&d| Dim::Static(d)).collect::<Vec<_>>(),
                dtype,
            )
        };

        // Flatten packed weights so per-(layer,direction) blocks can
        // be sliced by element offset (block widths vary by layer).
        let mut total_ih = dirs * four_h * in_dim;
        for _ in 1..l_count {
            total_ih += dirs * four_h * (dirs * hidden);
        }
        let total_hh = l_count * dirs * four_h * hidden;
        let total_bias = l_count * dirs * four_h;
        let wih_flat = out.reshape(in_wih, vec![total_ih as i64], sh(&[total_ih]));
        let whh_flat = out.reshape(in_whh, vec![total_hh as i64], sh(&[total_hh]));
        let bias_flat = out.reshape(in_bias, vec![total_bias as i64], sh(&[total_bias]));

        let mut layer_in = in_x;
        let mut in_l = in_dim;
        let mut wih_cursor = 0usize;

        for l in 0..l_count {
            let out_width = dirs * hidden;
            let wih_block = four_h * in_l;
            let mut dir_outs: Vec<NodeId> = Vec::with_capacity(dirs);

            for dir in 0..dirs {
                let ld = l * dirs + dir;
                let wih_b = out.add_node(
                    Op::Narrow {
                        axis: 0,
                        start: wih_cursor + dir * wih_block,
                        len: wih_block,
                    },
                    vec![wih_flat],
                    sh(&[wih_block]),
                );
                let wih_2d =
                    out.reshape(wih_b, vec![four_h as i64, in_l as i64], sh(&[four_h, in_l]));
                let wih_t = out.add_node(
                    Op::Transpose { perm: vec![1, 0] },
                    vec![wih_2d],
                    sh(&[in_l, four_h]),
                );
                let whh_b = out.add_node(
                    Op::Narrow {
                        axis: 0,
                        start: ld * four_h * hidden,
                        len: four_h * hidden,
                    },
                    vec![whh_flat],
                    sh(&[four_h * hidden]),
                );
                let whh_2d = out.reshape(
                    whh_b,
                    vec![four_h as i64, hidden as i64],
                    sh(&[four_h, hidden]),
                );
                let whh_t = out.add_node(
                    Op::Transpose { perm: vec![1, 0] },
                    vec![whh_2d],
                    sh(&[hidden, four_h]),
                );
                let bias_b = out.add_node(
                    Op::Narrow {
                        axis: 0,
                        start: ld * four_h,
                        len: four_h,
                    },
                    vec![bias_flat],
                    sh(&[four_h]),
                );
                let bias_1 = out.reshape(bias_b, vec![1, four_h as i64], sh(&[1, four_h]));
                let bias_bcast = out.add_node(
                    Op::Expand {
                        target_shape: vec![b_dim as i64, four_h as i64],
                    },
                    vec![bias_1],
                    sh(&[b_dim, four_h]),
                );

                let li_flat = out.reshape(
                    layer_in,
                    vec![(b_dim * s_dim) as i64, in_l as i64],
                    sh(&[b_dim * s_dim, in_l]),
                );
                let xw_flat = out.matmul(li_flat, wih_t, sh(&[b_dim * s_dim, four_h]));
                let xw = out.reshape(
                    xw_flat,
                    vec![b_dim as i64, s_dim as i64, four_h as i64],
                    sh(&[b_dim, s_dim, four_h]),
                );

                let bh = sh(&[b_dim, hidden]);
                let b4h = sh(&[b_dim, four_h]);

                let (mut h_prev, mut c_prev) = if *carry {
                    let in_h0 = new_inputs[4];
                    let in_c0 = new_inputs[5];
                    let h_sl = out.add_node(
                        Op::Narrow {
                            axis: 0,
                            start: ld,
                            len: 1,
                        },
                        vec![in_h0],
                        sh(&[1, b_dim, hidden]),
                    );
                    let h_bh = out.reshape(h_sl, vec![b_dim as i64, hidden as i64], bh.clone());
                    let c_sl = out.add_node(
                        Op::Narrow {
                            axis: 0,
                            start: ld,
                            len: 1,
                        },
                        vec![in_c0],
                        sh(&[1, b_dim, hidden]),
                    );
                    let c_bh = out.reshape(c_sl, vec![b_dim as i64, hidden as i64], bh.clone());
                    (h_bh, c_bh)
                } else {
                    let z0 = out.add_node(
                        Op::Constant {
                            data: vec![0u8; b_dim * hidden * 4],
                        },
                        vec![],
                        bh.clone(),
                    );
                    let z1 = out.add_node(
                        Op::Constant {
                            data: vec![0u8; b_dim * hidden * 4],
                        },
                        vec![],
                        bh.clone(),
                    );
                    (z0, z1)
                };

                // Per-direction outputs collected in time order.
                let mut ys: Vec<NodeId> = vec![h_prev; s_dim];
                for step in 0..s_dim {
                    let t = if dir == 0 { step } else { s_dim - 1 - step };
                    let xw_t3 = out.add_node(
                        Op::Narrow {
                            axis: 1,
                            start: t,
                            len: 1,
                        },
                        vec![xw],
                        sh(&[b_dim, 1, four_h]),
                    );
                    let xw_t = out.reshape(xw_t3, vec![b_dim as i64, four_h as i64], b4h.clone());
                    let hw = out.matmul(h_prev, whh_t, b4h.clone());
                    let z_0 = out.binary(BinaryOp::Add, xw_t, hw, b4h.clone());
                    let z = out.binary(BinaryOp::Add, z_0, bias_bcast, b4h.clone());

                    let i_pre = out.add_node(
                        Op::Narrow {
                            axis: 1,
                            start: 0,
                            len: hidden,
                        },
                        vec![z],
                        bh.clone(),
                    );
                    let f_pre = out.add_node(
                        Op::Narrow {
                            axis: 1,
                            start: hidden,
                            len: hidden,
                        },
                        vec![z],
                        bh.clone(),
                    );
                    let g_pre = out.add_node(
                        Op::Narrow {
                            axis: 1,
                            start: 2 * hidden,
                            len: hidden,
                        },
                        vec![z],
                        bh.clone(),
                    );
                    let o_pre = out.add_node(
                        Op::Narrow {
                            axis: 1,
                            start: 3 * hidden,
                            len: hidden,
                        },
                        vec![z],
                        bh.clone(),
                    );
                    let i_g = out.activation(Activation::Sigmoid, i_pre, bh.clone());
                    let f_g = out.activation(Activation::Sigmoid, f_pre, bh.clone());
                    let g_g = out.activation(Activation::Tanh, g_pre, bh.clone());
                    let o_g = out.activation(Activation::Sigmoid, o_pre, bh.clone());
                    let fc = out.binary(BinaryOp::Mul, f_g, c_prev, bh.clone());
                    let ig = out.binary(BinaryOp::Mul, i_g, g_g, bh.clone());
                    let c = out.binary(BinaryOp::Add, fc, ig, bh.clone());
                    let tc = out.activation(Activation::Tanh, c, bh.clone());
                    let h_new = out.binary(BinaryOp::Mul, o_g, tc, bh.clone());

                    ys[t] = out.reshape(
                        h_new,
                        vec![b_dim as i64, 1, hidden as i64],
                        sh(&[b_dim, 1, hidden]),
                    );
                    c_prev = c;
                    h_prev = h_new;
                }

                let dir_out = if s_dim == 1 {
                    ys[0]
                } else {
                    out.add_node(Op::Concat { axis: 1 }, ys, sh(&[b_dim, s_dim, hidden]))
                };
                dir_outs.push(dir_out);
            }

            layer_in = if dirs == 1 {
                dir_outs[0]
            } else {
                out.add_node(
                    Op::Concat { axis: 2 },
                    dir_outs,
                    sh(&[b_dim, s_dim, out_width]),
                )
            };
            wih_cursor += dirs * wih_block;
            in_l = out_width;
        }

        layer_in
    }
}

pub(super) fn unfuse_gru(node: &rlx_ir::Node, new_inputs: Vec<NodeId>, out: &mut Graph) -> NodeId {
    let Op::Gru {
        hidden_size,
        num_layers,
        bidirectional,
        carry,
    } = &node.op
    else {
        unreachable!()
    };
    {
        // Multi-layer (optionally bidirectional) GRU, gate order
        // r, z, n. Unrolls to MatMul / Add / Sub / Mul / Sigmoid /
        // Tanh / Narrow / Reshape / Transpose / Expand / Concat (see
        // Op::Gru). Carry seeds h0 but does not thread hn back out
        // (matches Lstm).
        let hidden = *hidden_size;
        let three_h = 3 * hidden;
        let l_count = *num_layers;
        let dirs = if *bidirectional { 2 } else { 1 };
        let in_x = new_inputs[0];
        let in_wih = new_inputs[1];
        let in_whh = new_inputs[2];
        let in_bih = new_inputs[3];
        let in_bhh = new_inputs[4];

        let x_shape = out.node(in_x).shape.clone();
        let dtype = x_shape.dtype();
        let b_dim = match x_shape.dim(0) {
            Dim::Static(v) => v,
            _ => panic!("Gru unfuse: dynamic B"),
        };
        let s_dim = match x_shape.dim(1) {
            Dim::Static(v) => v,
            _ => panic!("Gru unfuse: dynamic S"),
        };
        let in_dim = match x_shape.dim(2) {
            Dim::Static(v) => v,
            _ => panic!("Gru unfuse: dynamic input size"),
        };
        let sh = |dims: &[usize]| {
            IrShape::from_dims(
                &dims.iter().map(|&d| Dim::Static(d)).collect::<Vec<_>>(),
                dtype,
            )
        };

        let mut total_ih = dirs * three_h * in_dim;
        for _ in 1..l_count {
            total_ih += dirs * three_h * (dirs * hidden);
        }
        let total_hh = l_count * dirs * three_h * hidden;
        let total_b = l_count * dirs * three_h;
        let wih_flat = out.reshape(in_wih, vec![total_ih as i64], sh(&[total_ih]));
        let whh_flat = out.reshape(in_whh, vec![total_hh as i64], sh(&[total_hh]));
        let bih_flat = out.reshape(in_bih, vec![total_b as i64], sh(&[total_b]));
        let bhh_flat = out.reshape(in_bhh, vec![total_b as i64], sh(&[total_b]));

        let ones_bytes: Vec<u8> = std::iter::repeat_n(1.0f32.to_le_bytes(), b_dim * hidden)
            .flatten()
            .collect();
        let ones = out.add_node(
            Op::Constant { data: ones_bytes },
            vec![],
            sh(&[b_dim, hidden]),
        );

        let mut layer_in = in_x;
        let mut in_l = in_dim;
        let mut wih_cursor = 0usize;
        for l in 0..l_count {
            let out_width = dirs * hidden;
            let wih_block = three_h * in_l;
            let mut dir_outs: Vec<NodeId> = Vec::with_capacity(dirs);
            for dir in 0..dirs {
                let ld = l * dirs + dir;
                let wih_b = out.add_node(
                    Op::Narrow {
                        axis: 0,
                        start: wih_cursor + dir * wih_block,
                        len: wih_block,
                    },
                    vec![wih_flat],
                    sh(&[wih_block]),
                );
                let wih_2d = out.reshape(
                    wih_b,
                    vec![three_h as i64, in_l as i64],
                    sh(&[three_h, in_l]),
                );
                let wih_t = out.add_node(
                    Op::Transpose { perm: vec![1, 0] },
                    vec![wih_2d],
                    sh(&[in_l, three_h]),
                );
                let whh_b = out.add_node(
                    Op::Narrow {
                        axis: 0,
                        start: ld * three_h * hidden,
                        len: three_h * hidden,
                    },
                    vec![whh_flat],
                    sh(&[three_h * hidden]),
                );
                let whh_2d = out.reshape(
                    whh_b,
                    vec![three_h as i64, hidden as i64],
                    sh(&[three_h, hidden]),
                );
                let whh_t = out.add_node(
                    Op::Transpose { perm: vec![1, 0] },
                    vec![whh_2d],
                    sh(&[hidden, three_h]),
                );
                let bih_b = out.add_node(
                    Op::Narrow {
                        axis: 0,
                        start: ld * three_h,
                        len: three_h,
                    },
                    vec![bih_flat],
                    sh(&[three_h]),
                );
                let bih_1 = out.reshape(bih_b, vec![1, three_h as i64], sh(&[1, three_h]));
                let bih_bc = out.add_node(
                    Op::Expand {
                        target_shape: vec![b_dim as i64, three_h as i64],
                    },
                    vec![bih_1],
                    sh(&[b_dim, three_h]),
                );
                let bhh_b = out.add_node(
                    Op::Narrow {
                        axis: 0,
                        start: ld * three_h,
                        len: three_h,
                    },
                    vec![bhh_flat],
                    sh(&[three_h]),
                );
                let bhh_1 = out.reshape(bhh_b, vec![1, three_h as i64], sh(&[1, three_h]));
                let bhh_bc = out.add_node(
                    Op::Expand {
                        target_shape: vec![b_dim as i64, three_h as i64],
                    },
                    vec![bhh_1],
                    sh(&[b_dim, three_h]),
                );

                let li_flat = out.reshape(
                    layer_in,
                    vec![(b_dim * s_dim) as i64, in_l as i64],
                    sh(&[b_dim * s_dim, in_l]),
                );
                let xw_flat = out.matmul(li_flat, wih_t, sh(&[b_dim * s_dim, three_h]));
                let xw = out.reshape(
                    xw_flat,
                    vec![b_dim as i64, s_dim as i64, three_h as i64],
                    sh(&[b_dim, s_dim, three_h]),
                );

                let bh = sh(&[b_dim, hidden]);
                let b3h = sh(&[b_dim, three_h]);

                let mut h_prev = if *carry {
                    let in_h0 = new_inputs[5];
                    let h_sl = out.add_node(
                        Op::Narrow {
                            axis: 0,
                            start: ld,
                            len: 1,
                        },
                        vec![in_h0],
                        sh(&[1, b_dim, hidden]),
                    );
                    out.reshape(h_sl, vec![b_dim as i64, hidden as i64], bh.clone())
                } else {
                    out.add_node(
                        Op::Constant {
                            data: vec![0u8; b_dim * hidden * 4],
                        },
                        vec![],
                        bh.clone(),
                    )
                };

                let mut ys: Vec<NodeId> = vec![h_prev; s_dim];
                for step in 0..s_dim {
                    let t = if dir == 0 { step } else { s_dim - 1 - step };
                    let xw_t3 = out.add_node(
                        Op::Narrow {
                            axis: 1,
                            start: t,
                            len: 1,
                        },
                        vec![xw],
                        sh(&[b_dim, 1, three_h]),
                    );
                    let xw_t = out.reshape(xw_t3, vec![b_dim as i64, three_h as i64], b3h.clone());
                    let xih = out.binary(BinaryOp::Add, xw_t, bih_bc, b3h.clone());
                    let hw = out.matmul(h_prev, whh_t, b3h.clone());
                    let hhh = out.binary(BinaryOp::Add, hw, bhh_bc, b3h.clone());
                    let xr = out.add_node(
                        Op::Narrow {
                            axis: 1,
                            start: 0,
                            len: hidden,
                        },
                        vec![xih],
                        bh.clone(),
                    );
                    let xz = out.add_node(
                        Op::Narrow {
                            axis: 1,
                            start: hidden,
                            len: hidden,
                        },
                        vec![xih],
                        bh.clone(),
                    );
                    let xn = out.add_node(
                        Op::Narrow {
                            axis: 1,
                            start: 2 * hidden,
                            len: hidden,
                        },
                        vec![xih],
                        bh.clone(),
                    );
                    let hr = out.add_node(
                        Op::Narrow {
                            axis: 1,
                            start: 0,
                            len: hidden,
                        },
                        vec![hhh],
                        bh.clone(),
                    );
                    let hz = out.add_node(
                        Op::Narrow {
                            axis: 1,
                            start: hidden,
                            len: hidden,
                        },
                        vec![hhh],
                        bh.clone(),
                    );
                    let hn = out.add_node(
                        Op::Narrow {
                            axis: 1,
                            start: 2 * hidden,
                            len: hidden,
                        },
                        vec![hhh],
                        bh.clone(),
                    );
                    let r_sum = out.binary(BinaryOp::Add, xr, hr, bh.clone());
                    let r = out.activation(Activation::Sigmoid, r_sum, bh.clone());
                    let z_sum = out.binary(BinaryOp::Add, xz, hz, bh.clone());
                    let z = out.activation(Activation::Sigmoid, z_sum, bh.clone());
                    let rhn = out.binary(BinaryOp::Mul, r, hn, bh.clone());
                    let n_sum = out.binary(BinaryOp::Add, xn, rhn, bh.clone());
                    let n = out.activation(Activation::Tanh, n_sum, bh.clone());
                    let one_minus_z = out.binary(BinaryOp::Sub, ones, z, bh.clone());
                    let term1 = out.binary(BinaryOp::Mul, one_minus_z, n, bh.clone());
                    let term2 = out.binary(BinaryOp::Mul, z, h_prev, bh.clone());
                    let h_new = out.binary(BinaryOp::Add, term1, term2, bh.clone());
                    ys[t] = out.reshape(
                        h_new,
                        vec![b_dim as i64, 1, hidden as i64],
                        sh(&[b_dim, 1, hidden]),
                    );
                    h_prev = h_new;
                }
                let dir_out = if s_dim == 1 {
                    ys[0]
                } else {
                    out.add_node(Op::Concat { axis: 1 }, ys, sh(&[b_dim, s_dim, hidden]))
                };
                dir_outs.push(dir_out);
            }
            layer_in = if dirs == 1 {
                dir_outs[0]
            } else {
                out.add_node(
                    Op::Concat { axis: 2 },
                    dir_outs,
                    sh(&[b_dim, s_dim, out_width]),
                )
            };
            wih_cursor += dirs * wih_block;
            in_l = out_width;
        }
        layer_in
    }
}

pub(super) fn unfuse_rnn(node: &rlx_ir::Node, new_inputs: Vec<NodeId>, out: &mut Graph) -> NodeId {
    let Op::Rnn {
        hidden_size,
        num_layers,
        bidirectional,
        carry,
        relu,
    } = &node.op
    else {
        unreachable!()
    };
    {
        // Multi-layer (optionally bidirectional) Elman RNN. Unrolls
        // to MatMul / Add / Tanh|Relu / Narrow / Reshape / Transpose
        // / Expand / Concat (see Op::Rnn).
        let hidden = *hidden_size;
        let l_count = *num_layers;
        let dirs = if *bidirectional { 2 } else { 1 };
        let act = if *relu {
            Activation::Relu
        } else {
            Activation::Tanh
        };
        let in_x = new_inputs[0];
        let in_wih = new_inputs[1];
        let in_whh = new_inputs[2];
        let in_bias = new_inputs[3];

        let x_shape = out.node(in_x).shape.clone();
        let dtype = x_shape.dtype();
        let b_dim = match x_shape.dim(0) {
            Dim::Static(v) => v,
            _ => panic!("Rnn unfuse: dynamic B"),
        };
        let s_dim = match x_shape.dim(1) {
            Dim::Static(v) => v,
            _ => panic!("Rnn unfuse: dynamic S"),
        };
        let in_dim = match x_shape.dim(2) {
            Dim::Static(v) => v,
            _ => panic!("Rnn unfuse: dynamic input size"),
        };
        let sh = |dims: &[usize]| {
            IrShape::from_dims(
                &dims.iter().map(|&d| Dim::Static(d)).collect::<Vec<_>>(),
                dtype,
            )
        };

        let mut total_ih = dirs * hidden * in_dim;
        for _ in 1..l_count {
            total_ih += dirs * hidden * (dirs * hidden);
        }
        let total_hh = l_count * dirs * hidden * hidden;
        let total_b = l_count * dirs * hidden;
        let wih_flat = out.reshape(in_wih, vec![total_ih as i64], sh(&[total_ih]));
        let whh_flat = out.reshape(in_whh, vec![total_hh as i64], sh(&[total_hh]));
        let bias_flat = out.reshape(in_bias, vec![total_b as i64], sh(&[total_b]));

        let mut layer_in = in_x;
        let mut in_l = in_dim;
        let mut wih_cursor = 0usize;
        for l in 0..l_count {
            let out_width = dirs * hidden;
            let wih_block = hidden * in_l;
            let mut dir_outs: Vec<NodeId> = Vec::with_capacity(dirs);
            for dir in 0..dirs {
                let ld = l * dirs + dir;
                let wih_b = out.add_node(
                    Op::Narrow {
                        axis: 0,
                        start: wih_cursor + dir * wih_block,
                        len: wih_block,
                    },
                    vec![wih_flat],
                    sh(&[wih_block]),
                );
                let wih_2d =
                    out.reshape(wih_b, vec![hidden as i64, in_l as i64], sh(&[hidden, in_l]));
                let wih_t = out.add_node(
                    Op::Transpose { perm: vec![1, 0] },
                    vec![wih_2d],
                    sh(&[in_l, hidden]),
                );
                let whh_b = out.add_node(
                    Op::Narrow {
                        axis: 0,
                        start: ld * hidden * hidden,
                        len: hidden * hidden,
                    },
                    vec![whh_flat],
                    sh(&[hidden * hidden]),
                );
                let whh_2d = out.reshape(
                    whh_b,
                    vec![hidden as i64, hidden as i64],
                    sh(&[hidden, hidden]),
                );
                let whh_t = out.add_node(
                    Op::Transpose { perm: vec![1, 0] },
                    vec![whh_2d],
                    sh(&[hidden, hidden]),
                );
                let bias_b = out.add_node(
                    Op::Narrow {
                        axis: 0,
                        start: ld * hidden,
                        len: hidden,
                    },
                    vec![bias_flat],
                    sh(&[hidden]),
                );
                let bias_1 = out.reshape(bias_b, vec![1, hidden as i64], sh(&[1, hidden]));
                let bias_bc = out.add_node(
                    Op::Expand {
                        target_shape: vec![b_dim as i64, hidden as i64],
                    },
                    vec![bias_1],
                    sh(&[b_dim, hidden]),
                );

                let li_flat = out.reshape(
                    layer_in,
                    vec![(b_dim * s_dim) as i64, in_l as i64],
                    sh(&[b_dim * s_dim, in_l]),
                );
                let xw_flat = out.matmul(li_flat, wih_t, sh(&[b_dim * s_dim, hidden]));
                let xw = out.reshape(
                    xw_flat,
                    vec![b_dim as i64, s_dim as i64, hidden as i64],
                    sh(&[b_dim, s_dim, hidden]),
                );

                let bh = sh(&[b_dim, hidden]);

                let mut h_prev = if *carry {
                    let in_h0 = new_inputs[4];
                    let h_sl = out.add_node(
                        Op::Narrow {
                            axis: 0,
                            start: ld,
                            len: 1,
                        },
                        vec![in_h0],
                        sh(&[1, b_dim, hidden]),
                    );
                    out.reshape(h_sl, vec![b_dim as i64, hidden as i64], bh.clone())
                } else {
                    out.add_node(
                        Op::Constant {
                            data: vec![0u8; b_dim * hidden * 4],
                        },
                        vec![],
                        bh.clone(),
                    )
                };

                let mut ys: Vec<NodeId> = vec![h_prev; s_dim];
                for step in 0..s_dim {
                    let t = if dir == 0 { step } else { s_dim - 1 - step };
                    let xw_t3 = out.add_node(
                        Op::Narrow {
                            axis: 1,
                            start: t,
                            len: 1,
                        },
                        vec![xw],
                        sh(&[b_dim, 1, hidden]),
                    );
                    let xw_t = out.reshape(xw_t3, vec![b_dim as i64, hidden as i64], bh.clone());
                    let pre0 = out.binary(BinaryOp::Add, xw_t, bias_bc, bh.clone());
                    let hw = out.matmul(h_prev, whh_t, bh.clone());
                    let pre = out.binary(BinaryOp::Add, pre0, hw, bh.clone());
                    let h_new = out.activation(act, pre, bh.clone());
                    ys[t] = out.reshape(
                        h_new,
                        vec![b_dim as i64, 1, hidden as i64],
                        sh(&[b_dim, 1, hidden]),
                    );
                    h_prev = h_new;
                }
                let dir_out = if s_dim == 1 {
                    ys[0]
                } else {
                    out.add_node(Op::Concat { axis: 1 }, ys, sh(&[b_dim, s_dim, hidden]))
                };
                dir_outs.push(dir_out);
            }
            layer_in = if dirs == 1 {
                dir_outs[0]
            } else {
                out.add_node(
                    Op::Concat { axis: 2 },
                    dir_outs,
                    sh(&[b_dim, s_dim, out_width]),
                )
            };
            wih_cursor += dirs * wih_block;
            in_l = out_width;
        }
        layer_in
    }
}

pub(super) fn unfuse_mamba2(
    node: &rlx_ir::Node,
    new_inputs: Vec<NodeId>,
    out: &mut Graph,
) -> NodeId {
    let Op::Mamba2 {
        head_dim: _,
        state_size,
    } = &node.op
    else {
        unreachable!()
    };
    {
        // Mamba-2 / SSD scalar-decay SSM. Unrolls the scan into
        // Narrow / Reshape / Mul / Add / Exp / Reduce(Sum) (with
        // NumPy broadcasting) so the gradient walk reaches it via
        // existing VJPs and backends without a native kernel lower
        // it via these primitives. State S [B,H,P,N] zero-init.
        let n = *state_size;
        let in_x = new_inputs[0];
        let in_dt = new_inputs[1];
        let in_a = new_inputs[2];
        let in_b = new_inputs[3];
        let in_c = new_inputs[4];

        let x_shape = out.node(in_x).shape.clone();
        let dtype = x_shape.dtype();
        let dim = |i: usize| match x_shape.dim(i) {
            Dim::Static(v) => v,
            _ => panic!("Mamba2 unfuse: dynamic dim {i}"),
        };
        let (b_dim, s_dim, h_dim, p_dim) = (dim(0), dim(1), dim(2), dim(3));
        let sh = |dims: &[usize]| {
            IrShape::from_dims(
                &dims.iter().map(|&d| Dim::Static(d)).collect::<Vec<_>>(),
                dtype,
            )
        };

        let bh = sh(&[b_dim, h_dim]);
        let bhp = sh(&[b_dim, h_dim, p_dim]);
        let bhn = sh(&[b_dim, h_dim, n]);
        let bhpn = sh(&[b_dim, h_dim, p_dim, n]);

        // S_0 = 0  [B,H,P,N]
        let mut state = out.add_node(
            Op::Constant {
                data: vec![0u8; b_dim * h_dim * p_dim * n * 4],
            },
            vec![],
            bhpn.clone(),
        );

        let mut ys: Vec<NodeId> = Vec::with_capacity(s_dim);
        for t in 0..s_dim {
            // Slice timestep t.
            let dt_t = out.add_node(
                Op::Narrow {
                    axis: 1,
                    start: t,
                    len: 1,
                },
                vec![in_dt],
                sh(&[b_dim, 1, h_dim]),
            );
            let dt_bh = out.reshape(dt_t, vec![b_dim as i64, h_dim as i64], bh.clone());
            let x_t = out.add_node(
                Op::Narrow {
                    axis: 1,
                    start: t,
                    len: 1,
                },
                vec![in_x],
                sh(&[b_dim, 1, h_dim, p_dim]),
            );
            let x_bhp = out.reshape(
                x_t,
                vec![b_dim as i64, h_dim as i64, p_dim as i64],
                bhp.clone(),
            );
            let b_t = out.add_node(
                Op::Narrow {
                    axis: 1,
                    start: t,
                    len: 1,
                },
                vec![in_b],
                sh(&[b_dim, 1, h_dim, n]),
            );
            let b_bhn = out.reshape(b_t, vec![b_dim as i64, h_dim as i64, n as i64], bhn.clone());
            let c_t = out.add_node(
                Op::Narrow {
                    axis: 1,
                    start: t,
                    len: 1,
                },
                vec![in_c],
                sh(&[b_dim, 1, h_dim, n]),
            );
            let c_bhn = out.reshape(c_t, vec![b_dim as i64, h_dim as i64, n as i64], bhn.clone());

            // dA = exp(dt · a)   [B,H]   (a [H] broadcasts)
            let dta = out.binary(BinaryOp::Mul, dt_bh, in_a, bh.clone());
            let d_a = out.activation(Activation::Exp, dta, bh.clone());

            // (dt · x) ⊗ b  →  outer [B,H,P,N]
            let dt_bh1 = out.reshape(
                dt_bh,
                vec![b_dim as i64, h_dim as i64, 1],
                sh(&[b_dim, h_dim, 1]),
            );
            let dtx = out.binary(BinaryOp::Mul, dt_bh1, x_bhp, bhp.clone());
            let dtx_4 = out.reshape(
                dtx,
                vec![b_dim as i64, h_dim as i64, p_dim as i64, 1],
                sh(&[b_dim, h_dim, p_dim, 1]),
            );
            let b_4 = out.reshape(
                b_bhn,
                vec![b_dim as i64, h_dim as i64, 1, n as i64],
                sh(&[b_dim, h_dim, 1, n]),
            );
            let outer = out.binary(BinaryOp::Mul, dtx_4, b_4, bhpn.clone());

            // S = dA · S + outer
            let da_4 = out.reshape(
                d_a,
                vec![b_dim as i64, h_dim as i64, 1, 1],
                sh(&[b_dim, h_dim, 1, 1]),
            );
            let decayed = out.binary(BinaryOp::Mul, da_4, state, bhpn.clone());
            state = out.binary(BinaryOp::Add, decayed, outer, bhpn.clone());

            // y = Σ_n S · c   →  [B,H,P]
            let c_4 = out.reshape(
                c_bhn,
                vec![b_dim as i64, h_dim as i64, 1, n as i64],
                sh(&[b_dim, h_dim, 1, n]),
            );
            let sc = out.binary(BinaryOp::Mul, state, c_4, bhpn.clone());
            let y_bhp = out.add_node(
                Op::Reduce {
                    op: ReduceOp::Sum,
                    axes: vec![3],
                    keep_dim: false,
                },
                vec![sc],
                bhp.clone(),
            );
            let y_t = out.reshape(
                y_bhp,
                vec![b_dim as i64, 1, h_dim as i64, p_dim as i64],
                sh(&[b_dim, 1, h_dim, p_dim]),
            );
            ys.push(y_t);
        }

        if ys.len() == 1 {
            ys.pop().unwrap()
        } else {
            out.add_node(Op::Concat { axis: 1 }, ys, node.shape.clone())
        }
    }
}

pub(super) fn unfuse_selective_scan(
    node: &rlx_ir::Node,
    new_inputs: Vec<NodeId>,
    out: &mut Graph,
) -> NodeId {
    let Op::SelectiveScan { state_size } = &node.op else {
        unreachable!()
    };
    {
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
        let b1n = IrShape::from_dims(&[Dim::Static(b_dim), Dim::Static(1), Dim::Static(n)], dtype);
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
        let mut state = out.add_node(Op::Constant { data: zero_bytes }, vec![], bhn.clone());

        // a: [H, N] → reshape [1, H, N] → expand [B, H, N].
        let a_1hn = out.reshape(
            in_a,
            vec![1, h_dim as i64, n as i64],
            IrShape::from_dims(&[Dim::Static(1), Dim::Static(h_dim), Dim::Static(n)], dtype),
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
            let xt_bh1 = out.reshape(xt_b1h, vec![b_dim as i64, h_dim as i64, 1], bh1.clone());

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
            let dt_bh1 = out.reshape(dt_b1h, vec![b_dim as i64, h_dim as i64, 1], bh1.clone());

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
            let yt_b1h = out.reshape(yt_bh, vec![b_dim as i64, 1, h_dim as i64], b1h.clone());
            ys.push(yt_b1h);
        }

        // Concat along seq axis. S==1 short-circuits.
        if ys.len() == 1 {
            ys.pop().unwrap()
        } else {
            out.add_node(Op::Concat { axis: 1 }, ys, node.shape.clone())
        }
    }
}
