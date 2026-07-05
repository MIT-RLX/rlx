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

//
// Primitive compositions for training `*Backward` ops (higher-order AD).

//! `conv_pool` — extracted from the `decompose_backward_kernels` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use rlx_ir::infer::GraphExt;
use rlx_ir::op::{AttentionBwdWrt, CmpOp, MaskKind, SteKind};
use rlx_ir::shape;
use rlx_ir::shape::Dim;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

use super::*;

/// `Conv2dBackwardWeight` via static im2col + matmul (static NCHW).
pub fn compose_conv2d_backward_weight(
    g: &mut Graph,
    x: NodeId,
    dy: NodeId,
    dw_shape: &Shape,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
) -> NodeId {
    assert!(groups >= 1, "compose_conv2d_backward_weight: groups >= 1");
    let [n, c_in, _h, _w_in] = static_dim4(&g.node(x).shape).expect("static NCHW x");
    let [n2, c_out, _h_out, _w_out] = static_dim4(&g.node(dy).shape).expect("static NCHW dy");
    assert_eq!(n, n2, "conv2d_backward_weight: batch mismatch");
    let [dw_co, dw_ci, kh, kw] = static_dim4(dw_shape).expect("static dw");
    assert_eq!((kernel_size[0], kernel_size[1]), (kh, kw));
    assert_eq!(dw_co, c_out);
    assert_eq!(
        dw_ci * groups,
        c_in,
        "conv2d_backward_weight: c_in/groups mismatch"
    );

    if groups == 1 {
        return compose_conv2d_backward_weight_group(
            g,
            x,
            dy,
            dw_shape,
            kernel_size,
            stride,
            padding,
            dilation,
        );
    }

    assert_eq!(
        c_in % groups,
        0,
        "compose_conv2d_backward_weight: c_in divisible by groups"
    );
    assert_eq!(
        c_out % groups,
        0,
        "compose_conv2d_backward_weight: c_out divisible by groups"
    );
    let c_in_pg = c_in / groups;
    let c_out_pg = c_out / groups;
    let dt = dw_shape.dtype();
    let mut dw_groups: Vec<NodeId> = Vec::with_capacity(groups);
    for gi in 0..groups {
        let x_g = g.narrow_(x, 1, gi * c_in_pg, c_in_pg);
        let dy_g = g.narrow_(dy, 1, gi * c_out_pg, c_out_pg);
        let dw_g_shape = Shape::new(&[c_out_pg, c_in_pg, kh, kw], dt);
        dw_groups.push(compose_conv2d_backward_weight_group(
            g,
            x_g,
            dy_g,
            &dw_g_shape,
            kernel_size,
            stride,
            padding,
            dilation,
        ));
    }
    g.concat_(dw_groups, 0)
}

/// `Conv2dBackwardWeight` via runtime `Op::Im2Col` (dynamic batch NCHW).
pub fn compose_conv2d_backward_weight_im2col(
    g: &mut Graph,
    x: NodeId,
    dy: NodeId,
    dw_shape: &Shape,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
) -> NodeId {
    assert!(groups >= 1);
    let c_in = g.node(x).shape.dim(1).unwrap_static();
    let c_out = g.node(dy).shape.dim(1).unwrap_static();
    let [dw_co, dw_ci, kh, kw] = static_dim4(dw_shape).expect("static dw");
    assert_eq!(
        (dw_co, dw_ci, kh, kw),
        (c_out, c_in / groups, kernel_size[0], kernel_size[1])
    );

    if groups == 1 {
        return compose_conv2d_backward_weight_im2col_group(
            g,
            x,
            dy,
            dw_shape,
            kernel_size,
            stride,
            padding,
            dilation,
        );
    }
    assert_eq!(c_in % groups, 0);
    assert_eq!(c_out % groups, 0);
    let c_in_pg = c_in / groups;
    let c_out_pg = c_out / groups;
    let dt = dw_shape.dtype();
    let mut dw_groups: Vec<NodeId> = Vec::with_capacity(groups);
    for gi in 0..groups {
        let x_g = g.narrow_(x, 1, gi * c_in_pg, c_in_pg);
        let dy_g = g.narrow_(dy, 1, gi * c_out_pg, c_out_pg);
        let dw_g_shape = Shape::new(&[c_out_pg, c_in_pg, kh, kw], dt);
        dw_groups.push(compose_conv2d_backward_weight_im2col_group(
            g,
            x_g,
            dy_g,
            &dw_g_shape,
            kernel_size,
            stride,
            padding,
            dilation,
        ));
    }
    g.concat_(dw_groups, 0)
}

pub(super) fn compose_conv2d_backward_weight_im2col_group(
    g: &mut Graph,
    x: NodeId,
    dy: NodeId,
    dw_shape: &Shape,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> NodeId {
    let c_in = g.node(x).shape.dim(1).unwrap_static();
    let c_out = g.node(dy).shape.dim(1).unwrap_static();
    let [dw_co, dw_ci, kh, kw] = static_dim4(dw_shape).expect("static dw");
    assert_eq!((dw_co, dw_ci), (c_out, c_in));
    assert_eq!((kernel_size[0], kernel_size[1]), (kh, kw));

    let dt = dw_shape.dtype();
    let x_col = g.im2col(x, kernel_size, stride, padding, dilation);
    let m_dim = g.node(x_col).shape.dim(0);
    let k = g.node(x_col).shape.dim(1).unwrap_static();
    let dy_r_shape = Shape::from_dims(&[Dim::Static(c_out), m_dim], dt);
    let dy_r = g.add_node(
        Op::Reshape {
            new_shape: vec![c_out as i64, -1],
        },
        vec![dy],
        dy_r_shape,
    );
    let dw_mat_shape = Shape::new(&[c_out, k], dt);
    let prod = g.matmul(dy_r, x_col, dw_mat_shape);
    g.reshape_(prod, vec![c_out as i64, c_in as i64, kh as i64, kw as i64])
}

/// Single-group (or groups=1) im2col + matmul for `Conv2dBackwardWeight`.
pub(super) fn compose_conv2d_backward_weight_group(
    g: &mut Graph,
    x: NodeId,
    dy: NodeId,
    dw_shape: &Shape,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> NodeId {
    let [n, c_in, h, w_in] = static_dim4(&g.node(x).shape).expect("static NCHW x");
    let [n2, c_out, h_out, w_out] = static_dim4(&g.node(dy).shape).expect("static NCHW dy");
    assert_eq!(n, n2);
    let [dw_co, dw_ci, kh, kw] = static_dim4(dw_shape).expect("static dw");
    assert_eq!((dw_co, dw_ci), (c_out, c_in));
    assert_eq!((kernel_size[0], kernel_size[1]), (kh, kw));

    let (sh, sw) = (stride[0], stride[1]);
    let (ph, pw) = (padding[0], padding[1]);
    let (dh, dw_d) = (dilation[0], dilation[1]);

    if w_in == 1 && kw == 1 {
        return compose_conv2d_backward_weight_w1_h(
            g, x, dy, c_in, c_out, h, h_out, kh, sh, ph, dh, dw_shape,
        );
    }

    let m = n * h_out * w_out;
    let k = c_in * kh * kw;

    let flat_n = n * c_in * h * w_in;
    let flat_x = g.reshape_(x, vec![flat_n as i64]);
    let dt = dw_shape.dtype();
    let zero = f32_tensor_const(vec![0.0], Shape::new(&[1], dt), g);

    let dy_r = g.reshape_(dy, vec![c_out as i64, m as i64]);
    let dw_mat_shape = Shape::new(&[c_out, k], DType::F32);

    let matmul_dw = |g: &mut Graph, dy_slice: NodeId, x_col: NodeId| -> NodeId {
        let prod = g.matmul(dy_slice, x_col, dw_mat_shape.clone());
        g.reshape_(prod, vec![c_out as i64, c_in as i64, kh as i64, kw as i64])
    };

    if m * k <= IM2COL_MAX_MKL {
        let x_col = build_im2col_rows(
            g, flat_x, zero, n, c_in, h, w_in, h_out, w_out, kh, kw, sh, sw, ph, pw, dh, dw_d, k,
            0, m,
        );
        return matmul_dw(g, dy_r, x_col);
    }

    let m_chunk = (IM2COL_MAX_MKL / k.max(1)).max(1);
    let zero_dw = f32_tensor_const(vec![0.0; c_out * k], dw_mat_shape.clone(), g);
    let mut accum = zero_dw;
    for m0 in (0..m).step_by(m_chunk) {
        let m_len = (m - m0).min(m_chunk);
        let x_col = build_im2col_rows(
            g,
            flat_x,
            zero,
            n,
            c_in,
            h,
            w_in,
            h_out,
            w_out,
            kh,
            kw,
            sh,
            sw,
            ph,
            pw,
            dh,
            dw_d,
            k,
            m0,
            m0 + m_len,
        );
        let dy_chunk = g.narrow_(dy_r, 1, m0, m_len);
        let partial = g.matmul(dy_chunk, x_col, dw_mat_shape.clone());
        accum = g.add(accum, partial);
    }
    g.reshape_(accum, vec![c_out as i64, c_in as i64, kh as i64, kw as i64])
}

/// Fast path for K×1 conv (codec conv1d with time in H): O(h_out × kh) matmuls.
pub(super) fn compose_conv2d_backward_weight_w1_h(
    g: &mut Graph,
    x: NodeId,
    dy: NodeId,
    c_in: usize,
    c_out: usize,
    h_in: usize,
    h_out: usize,
    kh: usize,
    stride_h: usize,
    pad_h: usize,
    dilation_h: usize,
    dw_shape: &Shape,
) -> NodeId {
    let dt = dw_shape.dtype();
    let zero = f32_tensor_const(vec![0.0], Shape::new(&[1], dt), g);
    let mm_shape = Shape::new(&[c_out, c_in], dt);
    let mut slices = Vec::with_capacity(kh);
    for ki in 0..kh {
        let mut acc: Option<NodeId> = None;
        for ho in 0..h_out {
            let hi = ho * stride_h + ki * dilation_h;
            if hi < pad_h || hi - pad_h >= h_in {
                continue;
            }
            let hi_idx = hi - pad_h;
            let x_sl = g.narrow_(x, 2, hi_idx, 1);
            let dy_sl = g.narrow_(dy, 2, ho, 1);
            let x2 = g.reshape_(x_sl, vec![c_in as i64, 1]);
            let x2t = g.transpose_(x2, vec![1, 0]);
            let dy2 = g.reshape_(dy_sl, vec![c_out as i64, 1]);
            let term = g.matmul(dy2, x2t, mm_shape.clone());
            acc = Some(match acc {
                Some(prev) => g.add(prev, term),
                None => term,
            });
        }
        let slice = match acc {
            Some(v) => g.reshape_(v, vec![c_out as i64, c_in as i64, 1, 1]),
            None => {
                let dy2 = g.reshape_(dy, vec![c_out as i64, 1]);
                let xz = g.mul(x, zero);
                let x2t = g.reshape_(xz, vec![1, c_in as i64]);
                let z = g.matmul(dy2, x2t, mm_shape.clone());
                g.reshape_(z, vec![c_out as i64, c_in as i64, 1, 1])
            }
        };
        slices.push(slice);
    }
    g.concat_(slices, 2)
}

/// `MaxPool2dBackward` via runtime argmax + dy scatter (static NCHW).
pub fn compose_max_pool2d_backward(
    g: &mut Graph,
    x: NodeId,
    dy: NodeId,
    out_shape: &Shape,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
) -> NodeId {
    let [n, c, h, w_in] = static_dim4(&g.node(x).shape).expect("static NCHW x");
    let [n2, c2, h_out, w_out] = static_dim4(&g.node(dy).shape).expect("static NCHW dy");
    assert_eq!((n, c), (n2, c2));
    let (kh, kw) = (kernel_size[0], kernel_size[1]);
    let (sh, sw) = (stride[0], stride[1]);
    let (ph, pw) = (padding[0], padding[1]);
    let dt = out_shape.dtype();

    // Non-overlapping pools (stride == kernel, no padding) — the common CNN
    // case — decompose in O(input) instead of the quadratic dense scatter:
    //   pooled = maxpool(x);  dx = (x == upsample(pooled)) ? upsample(dy) : 0
    // Upsample is nearest by the kernel factor. Inputs outside the covered
    // region (odd dims) get zero gradient (zero-pad via concat). Ties route dy
    // to all maxima — measure-zero for real inputs, matches frameworks that
    // split tie gradients. This is what makes GPU/decompose-path training work
    // at real sizes (the old dense path capped out at 4096 elements).
    if sh == kh && sw == kw && ph == 0 && pw == 0 {
        let (ch, cw) = (h_out * kh, w_out * kw);
        let pooled = g.add_node(
            Op::Pool {
                kind: rlx_ir::op::ReduceOp::Max,
                kernel_size: vec![kh, kw],
                stride: vec![sh, sw],
                padding: vec![0, 0],
            },
            vec![x],
            g.node(dy).shape.clone(),
        );
        let pooled_up = nn_upsample_nchw(g, pooled, n, c, h_out, w_out, kh, kw, dt);
        let dy_up = nn_upsample_nchw(g, dy, n, c, h_out, w_out, kh, kw, dt);
        let mut x_crop = x;
        if ch != h {
            x_crop = g.narrow_(x_crop, 2, 0, ch);
        }
        if cw != w_in {
            x_crop = g.narrow_(x_crop, 3, 0, cw);
        }
        let eq = compare_eq(g, x_crop, pooled_up);
        let zero = f32_tensor_const(vec![0.0], Shape::scalar(dt), g);
        let mut dx = where_select(g, eq, dy_up, zero); // [n, c, ch, cw]
        if ch != h {
            let pad = h - ch;
            let z = f32_tensor_const(
                vec![0.0; n * c * pad * cw],
                Shape::new(&[n, c, pad, cw], dt),
                g,
            );
            dx = g.concat_(vec![dx, z], 2);
        }
        if cw != w_in {
            let pad = w_in - cw;
            let z = f32_tensor_const(
                vec![0.0; n * c * h * pad],
                Shape::new(&[n, c, h, pad], dt),
                g,
            );
            dx = g.concat_(vec![dx, z], 3);
        }
        return dx;
    }

    // Fallback (overlapping / padded pools): dense argmax + scatter. Still
    // capped — only toy sizes use this path now.
    let flat_n = n * c * h * w_in;
    let num_windows = n * c * h_out * w_out;
    assert!(
        flat_n.saturating_mul(num_windows) <= 4096,
        "compose_max_pool2d_backward: dense scatter too large ({flat_n}x{num_windows}); \
         only non-overlapping pools (stride==kernel, no pad) have the O(input) decomposition"
    );

    let flat_x = g.reshape_(x, vec![flat_n as i64]);
    let flat_dy = g.reshape_(dy, vec![num_windows as i64]);
    let zero = f32_tensor_const(vec![0.0], Shape::scalar(dt), g);

    let mut elems: Vec<NodeId> = Vec::with_capacity(flat_n);
    for j in 0..flat_n {
        let j_const = f32_tensor_const(vec![j as f32], Shape::new(&[1], dt), g);
        let mut acc = zero;
        let mut win = 0usize;
        for ni in 0..n {
            for ci in 0..c {
                for ho in 0..h_out {
                    for wo in 0..w_out {
                        let argmax = argmax_window_flat(
                            g, flat_x, n, c, h, w_in, ni, ci, ho, wo, kh, kw, sh, sw, ph, pw, dt,
                        );
                        let eq = compare_eq(g, argmax, j_const);
                        let hit = cast_f32(g, eq);
                        let dy_w = gather_flat_f32(g, flat_dy, win, dt);
                        let term = g.mul(hit, dy_w);
                        acc = g.add(acc, term);
                        win += 1;
                    }
                }
            }
        }
        elems.push(acc);
    }
    let flat_dx = g.concat_(elems, 0);
    g.reshape_(flat_dx, vec![n as i64, c as i64, h as i64, w_in as i64])
}

/// `Conv2dBackwardInput` → forward `Conv` (same as autodiff VJP).
pub fn compose_conv2d_backward_input(
    g: &mut Graph,
    dy: NodeId,
    w: NodeId,
    out_shape: &Shape,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
) -> NodeId {
    // dx = conv_transpose(dy, W). The transposed convolution IS the adjoint of the
    // forward (cross-correlation) conv — i.e. the input gradient. (A plain forward
    // `conv` here was a long-standing bug: the wrong gradient on every backend, and
    // CoreML rejected its channel layout.) The forward weight `[Cout, Cin, kH, kW]`
    // is already in conv_transpose layout — dim 0 (Cout) is the transpose's input
    // channels, matching `dy`. Solve `output_padding` so the transposed output
    // recovers the original input spatial size (conv's floor division can drop up
    // to stride-1 pixels): from `out = (in-1)·s + op + d·(k-1) - 2p + 1`.
    let dy_shape = g.node(dy).shape.clone();
    let out_pad = |axis: usize| -> usize {
        let in_sz = dy_shape.dim(axis + 2).unwrap_static() as i64;
        let out = out_shape.dim(axis + 2).unwrap_static() as i64;
        let base = (in_sz - 1) * stride[axis] as i64
            + dilation[axis] as i64 * (kernel_size[axis] as i64 - 1)
            + 1
            - 2 * padding[axis] as i64;
        (out - base).max(0) as usize
    };
    g.add_node(
        Op::ConvTranspose2d {
            kernel_size: kernel_size.to_vec(),
            stride: stride.to_vec(),
            padding: padding.to_vec(),
            dilation: dilation.to_vec(),
            output_padding: vec![out_pad(0), out_pad(1)],
            groups,
        },
        vec![dy, w],
        out_shape.clone(),
    )
}
