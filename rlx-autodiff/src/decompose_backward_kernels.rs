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

use rlx_ir::infer::GraphExt;
use rlx_ir::op::{AttentionBwdWrt, CmpOp, MaskKind, SteKind};
use rlx_ir::shape;
use rlx_ir::shape::Dim;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

/// Matches `RuntimeConfig::attn_mask_neg_inf` default on CPU.
const ATTN_MASK_NEG_INF: f32 = -1e9;
const MASK_BINARY_THRESHOLD: f32 = 0.5;

/// Max scan length for decomposed `ScanBackward` / `ScanBackwardXs` unrolls.
pub const SCAN_DECOMPOSE_MAX_LENGTH: u32 = 256;

use crate::activation_deriv::scalar_const;
use crate::autodiff::grad_with_loss;
use crate::compose::{broadcast_scalar, merge_subgraph};
use crate::prepare_ad::prepare_graph_for_ad;

fn axis_pos(axis: i32, rank: usize) -> usize {
    if axis < 0 {
        (rank as i32 + axis) as usize
    } else {
        axis as usize
    }
}

fn broadcast_eps(g: &mut Graph, eps: f32, like: &Shape) -> NodeId {
    let eps_s = scalar_const(eps as f64, &Shape::scalar(like.dtype()), g);
    broadcast_scalar(g, eps_s, like)
}

/// LayerNorm backward w.r.t. input (axis = -1), matching CPU kernel.
pub fn compose_layer_norm_backward_input(
    g: &mut Graph,
    x: NodeId,
    gamma: NodeId,
    dy: NodeId,
    axis: i32,
    eps: f32,
    out_shape: &Shape,
) -> NodeId {
    assert_eq!(axis, -1, "compose_layer_norm_backward_input: only axis=-1");
    let rank = out_shape.rank();
    let ax = axis_pos(axis, rank);
    let axes = vec![ax];

    // Pin activations — decomposed backward re-reads x/dy many times; memory
    // planning may alias input slots with temporaries in large grad graphs.
    let ones = broadcast_eps(g, 1.0, out_shape);
    let x = g.mul(x, ones);
    let dy = g.mul(dy, ones);

    let mean = g.mean(x, axes.clone(), true);
    let mean_b = broadcast_scalar(g, mean, out_shape);
    let xc = g.sub(x, mean_b);
    let xc2 = g.mul(xc, xc);
    let var = g.mean(xc2, axes.clone(), true);
    let var_shape = g.node(var).shape.clone();
    let eps_b = broadcast_eps(g, eps, &var_shape);
    let var_eps = g.add(var, eps_b);
    let inv_std = g.add_node(
        Op::Activation(rlx_ir::op::Activation::Rsqrt),
        vec![var_eps],
        g.node(var_eps).shape.clone(),
    );

    let inv_std_b = broadcast_scalar(g, inv_std, out_shape);
    let x_hat = g.mul(xc, inv_std_b);
    let g_b = broadcast_scalar(g, gamma, out_shape);
    let sy = g.mul(dy, g_b);
    let m_sy = g.mean(sy, axes.clone(), true);
    let m_sy_b = broadcast_scalar(g, m_sy, out_shape);
    let sy_xh = g.mul(sy, x_hat);
    let m_sxh = g.mean(sy_xh, axes, true);
    let m_sxh_b = broadcast_scalar(g, m_sxh, out_shape);
    let t1 = g.sub(sy, m_sy_b);
    let t2 = g.mul(x_hat, m_sxh_b);
    let t3 = g.sub(t1, t2);
    g.mul(inv_std_b, t3)
}

/// RMSNorm backward w.r.t. input (axis = -1), matching `training_bwd::rms_norm_backward_row`.
pub fn compose_rms_norm_backward_input(
    g: &mut Graph,
    x: NodeId,
    gamma: NodeId,
    _beta: NodeId,
    dy: NodeId,
    axis: i32,
    eps: f32,
    out_shape: &Shape,
) -> NodeId {
    assert_eq!(axis, -1, "compose_rms_norm_backward_input: only axis=-1");
    let rank = out_shape.rank();
    let ax = axis_pos(axis, rank);
    let axes = vec![ax];

    // Pin activations — decomposed backward re-reads x/dy many times; memory
    // planning may alias input slots with temporaries in large grad graphs.
    let ones = broadcast_eps(g, 1.0, out_shape);
    let x = g.mul(x, ones);
    let dy = g.mul(dy, ones);

    let x2 = g.mul(x, x);
    let mean_x2 = g.mean(x2, axes.clone(), true);
    let mean_x2_shape = g.node(mean_x2).shape.clone();
    let eps_b = broadcast_eps(g, eps, &mean_x2_shape);
    let var_eps = g.add(mean_x2, eps_b);
    let inv_r = g.add_node(
        Op::Activation(rlx_ir::op::Activation::Rsqrt),
        vec![var_eps],
        g.node(var_eps).shape.clone(),
    );
    let inv_r2 = g.mul(inv_r, inv_r);
    let inv_r3 = g.mul(inv_r2, inv_r);
    let inv_r_b = broadcast_scalar(g, inv_r, out_shape);
    let inv_r3_full = broadcast_scalar(g, inv_r3, out_shape);

    let g_b = broadcast_scalar(g, gamma, out_shape);
    let dy_g = g.mul(dy, g_b);
    let x_dy_g = g.mul(x, dy_g);
    let dot = g.mean(x_dy_g, axes, true);
    let dot_b = broadcast_scalar(g, dot, out_shape);

    let term1 = g.mul(g_b, dy);
    let x_dot = g.mul(x, dot_b);
    let term2 = g.mul(x_dot, inv_r3_full);
    let diff = g.sub(term1, term2);
    g.mul(diff, inv_r_b)
}

fn batch_reduce_axes(rank: usize, feature_axis: usize) -> Vec<usize> {
    (0..rank).filter(|&i| i != feature_axis).collect()
}

/// LayerNorm backward w.r.t. gamma (axis = -1): `sum(dy * x_hat)` over batch axes.
pub fn compose_layer_norm_backward_gamma(
    g: &mut Graph,
    x: NodeId,
    dy: NodeId,
    axis: i32,
    eps: f32,
    gamma_shape: &Shape,
) -> NodeId {
    assert_eq!(axis, -1, "compose_layer_norm_backward_gamma: only axis=-1");
    let x_shape = g.node(x).shape.clone();
    let rank = x_shape.rank();
    let ax = axis_pos(axis, rank);
    let axes = vec![ax];

    let mean = g.mean(x, axes.clone(), true);
    let xc = g.sub(x, mean);
    let xc2 = g.mul(xc, xc);
    let var = g.mean(xc2, axes.clone(), true);
    let var_shape = g.node(var).shape.clone();
    let eps_b = broadcast_eps(g, eps, &var_shape);
    let var_eps = g.add(var, eps_b);
    let inv_std = g.add_node(
        Op::Activation(rlx_ir::op::Activation::Rsqrt),
        vec![var_eps],
        var_shape,
    );
    let x_hat = g.mul(xc, inv_std);
    let prod = g.mul(dy, x_hat);
    g.reduce(
        prod,
        rlx_ir::op::ReduceOp::Sum,
        batch_reduce_axes(rank, ax),
        false,
        gamma_shape.clone(),
    )
}

/// LayerNorm backward w.r.t. beta: `sum(dy)` over batch axes.
pub fn compose_layer_norm_backward_beta(g: &mut Graph, dy: NodeId, beta_shape: &Shape) -> NodeId {
    let dy_shape = g.node(dy).shape.clone();
    let rank = dy_shape.rank();
    let ax = rank.saturating_sub(1);
    g.reduce(
        dy,
        rlx_ir::op::ReduceOp::Sum,
        batch_reduce_axes(rank, ax),
        false,
        beta_shape.clone(),
    )
}

/// RMSNorm backward w.r.t. gamma (axis = -1): `sum(dy * x * inv_r)`.
pub fn compose_rms_norm_backward_gamma(
    g: &mut Graph,
    x: NodeId,
    dy: NodeId,
    axis: i32,
    eps: f32,
    gamma_shape: &Shape,
) -> NodeId {
    assert_eq!(axis, -1, "compose_rms_norm_backward_gamma: only axis=-1");
    let x_shape = g.node(x).shape.clone();
    let rank = x_shape.rank();
    let ax = axis_pos(axis, rank);
    let axes = vec![ax];

    let x2 = g.mul(x, x);
    let mean_x2 = g.mean(x2, axes.clone(), true);
    let mean_x2_shape = g.node(mean_x2).shape.clone();
    let eps_b = broadcast_eps(g, eps, &mean_x2_shape);
    let mean_eps = g.add(mean_x2, eps_b);
    let inv_r = g.add_node(
        Op::Activation(rlx_ir::op::Activation::Rsqrt),
        vec![mean_eps],
        mean_x2_shape,
    );
    let x_scaled = g.mul(x, inv_r);
    let prod = g.mul(dy, x_scaled);
    g.reduce(
        prod,
        rlx_ir::op::ReduceOp::Sum,
        batch_reduce_axes(rank, ax),
        false,
        gamma_shape.clone(),
    )
}

/// RMSNorm backward w.r.t. beta: `sum(dy)` over batch axes.
pub fn compose_rms_norm_backward_beta(g: &mut Graph, dy: NodeId, beta_shape: &Shape) -> NodeId {
    compose_layer_norm_backward_beta(g, dy, beta_shape)
}

fn static_dim4(shape: &Shape) -> Option<[usize; 4]> {
    if shape.rank() != 4 {
        return None;
    }
    let mut out = [0usize; 4];
    for (i, d) in shape.dims().iter().enumerate() {
        out[i] = match d {
            Dim::Static(n) => *n,
            Dim::Dynamic(_) => return None,
        };
    }
    Some(out)
}

fn gather_flat_f32(g: &mut Graph, flat_x: NodeId, index: usize, dt: DType) -> NodeId {
    let idx = f32_tensor_const(vec![index as f32], Shape::new(&[1], dt), g);
    g.gather_(flat_x, idx, 0)
}

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

/// True when conv backward-input can decompose to forward `Op::Conv` with
/// dynamic batch (spatial + channel dims static; weights static).
pub fn conv_di_decompose_eligible(dy: &Shape, w: &Shape, dx: &Shape) -> bool {
    if dy.rank() != 4 || w.rank() != 4 || dx.rank() != 4 {
        return false;
    }
    if !w.is_static() {
        return false;
    }
    if dy.dim(0) != dx.dim(0) {
        return false;
    }
    (1..4).all(|axis| dy.dim(axis).is_static() && dx.dim(axis).is_static())
}

/// True when conv backward-weight can decompose via `Op::Im2Col` without
/// binding batch (spatial + channel dims static; batch may be dynamic).
pub fn conv_dw_im2col_eligible(x: &Shape, dy: &Shape, dw: &Shape) -> bool {
    if x.rank() != 4 || dy.rank() != 4 || dw.rank() != 4 {
        return false;
    }
    if !dw.is_static() {
        return false;
    }
    if x.dim(0) != dy.dim(0) {
        return false;
    }
    (1..4).all(|axis| x.dim(axis).is_static() && dy.dim(axis).is_static())
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

fn compose_conv2d_backward_weight_im2col_group(
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

fn static_dim2(shape: &Shape) -> Option<[usize; 2]> {
    if shape.rank() != 2 {
        return None;
    }
    Some([shape.dim(0).unwrap_static(), shape.dim(1).unwrap_static()])
}

fn dim_product(shape: &Shape, start: usize, end: usize) -> usize {
    shape.dims()[start..end]
        .iter()
        .map(|d| d.unwrap_static())
        .product()
}

fn perm_move_axis_last(rank: usize, axis: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..rank).collect();
    perm.remove(axis);
    perm.push(axis);
    perm
}

fn invert_perm(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0usize; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    inv
}

fn apply_perm_shape(shape: &Shape, perm: &[usize]) -> Shape {
    let dims: Vec<Dim> = perm.iter().map(|&i| shape.dim(i)).collect();
    Shape::from_dims(&dims, shape.dtype())
}

fn cumsum_backward_matrix(l: usize, exclusive: bool) -> Vec<f32> {
    let mut mat = vec![0f32; l * l];
    for j in 0..l {
        for k in 0..l {
            let inc = if exclusive { j > k } else { j >= k };
            if inc {
                mat[j * l + k] = 1.0;
            }
        }
    }
    mat
}

fn q_max_for_bits(bits: u8) -> f32 {
    match bits {
        8 => 127.0,
        4 => 7.0,
        2 => 1.0,
        n => panic!("compose_fake_quantize_backward: bad bits {n}"),
    }
}

/// `CumsumBackward` via batched matmul with a static lower-triangular ones matrix.
pub fn compose_cumsum_backward(
    g: &mut Graph,
    dy: NodeId,
    out_shape: &Shape,
    axis: i32,
    exclusive: bool,
) -> NodeId {
    let rank = out_shape.rank();
    assert!(rank >= 1, "compose_cumsum_backward: rank >= 1");
    let ax = axis_pos(axis, rank);
    let l = out_shape.dim(ax).unwrap_static();
    assert!(l <= 256, "compose_cumsum_backward: L={l} too large");

    let perm = perm_move_axis_last(rank, ax);
    let dy_work = if ax != rank - 1 {
        let dy_shape = g.node(dy).shape.clone();
        let permuted = apply_perm_shape(&dy_shape, &perm);
        g.add_node(Op::Transpose { perm: perm.clone() }, vec![dy], permuted)
    } else {
        dy
    };
    let work_shape = g.node(dy_work).shape.clone();
    let batch = work_shape.num_elements().expect("static cumsum bwd") / l;
    let dt = work_shape.dtype();
    let flat_dy = g.reshape_(dy_work, vec![batch as i64, l as i64]);
    let mat = cumsum_backward_matrix(l, exclusive);
    let m_node = f32_tensor_const(mat, Shape::new(&[l, l], dt), g);
    let dx_flat = g.matmul(flat_dy, m_node, Shape::new(&[batch, l], dt));
    let dims: Vec<i64> = work_shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static() as i64)
        .collect();
    let dx = g.reshape_(dx_flat, dims);
    if ax != rank - 1 {
        let inv = invert_perm(&perm);
        g.add_node(Op::Transpose { perm: inv }, vec![dx], out_shape.clone())
    } else {
        dx
    }
}

/// `GatherBackward` via `ScatterAdd` (axis 0) or flattened scatter (other axes).
pub fn compose_gather_backward(
    g: &mut Graph,
    dy: NodeId,
    indices: NodeId,
    table_shape: &Shape,
    axis: i32,
) -> NodeId {
    let rank = table_shape.rank();
    let ax = axis_pos(axis, rank);
    if ax == 0 {
        return g.add_node(Op::ScatterAdd, vec![dy, indices], table_shape.clone());
    }

    let _dy_shape = g.node(dy).shape.clone();
    let idx_shape = g.node(indices).shape.clone();
    let outer = dim_product(table_shape, 0, ax);
    let axis_dim = table_shape.dim(ax).unwrap_static();
    let trailing = dim_product(table_shape, ax + 1, rank);
    let num_idx = idx_shape.num_elements().expect("static gather idx");
    let dt = table_shape.dtype();

    let updates = g.reshape_(dy, vec![(outer * num_idx) as i64, trailing as i64]);

    let idx_rep = if outer == 1 {
        g.reshape_(indices, vec![num_idx as i64])
    } else {
        let mut parts: Vec<NodeId> = Vec::with_capacity(outer);
        for _ in 0..outer {
            parts.push(g.reshape_(indices, vec![num_idx as i64]));
        }
        g.concat_(parts, 0)
    };

    let mut outer_ids: Vec<f32> = Vec::with_capacity(outer * num_idx);
    for o in 0..outer {
        for _ in 0..num_idx {
            outer_ids.push(o as f32);
        }
    }
    let outer_node = f32_tensor_const(outer_ids, Shape::new(&[outer * num_idx], dt), g);
    let axis_s = scalar_const(axis_dim as f64, &Shape::scalar(dt), g);
    let axis_b = broadcast_scalar(g, axis_s, &Shape::new(&[outer * num_idx], dt));
    let base = g.mul(outer_node, axis_b);
    let flat_idx = g.add(base, idx_rep);
    let scattered = g.add_node(
        Op::ScatterAdd,
        vec![updates, flat_idx],
        Shape::new(&[outer * axis_dim, trailing], dt),
    );

    let dims: Vec<i64> = table_shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static() as i64)
        .collect();
    g.reshape_(scattered, dims)
}

/// `SoftmaxCrossEntropyBackward` via softmax + compare-built one-hot.
pub fn compose_softmax_cross_entropy_backward(
    g: &mut Graph,
    logits: NodeId,
    labels: NodeId,
    d_loss: NodeId,
    out_shape: &Shape,
) -> NodeId {
    let [n, c] = static_dim2(out_shape).expect("static [N,C] logits");
    let dt = out_shape.dtype();
    let sm = g.softmax(logits, -1, out_shape.clone());
    let labels_flat = if g.node(labels).shape.rank() == 1 {
        labels
    } else {
        g.reshape_(labels, vec![n as i64])
    };
    let mut cols: Vec<NodeId> = Vec::with_capacity(c);
    let labels_shape = g.node(labels_flat).shape.clone();
    let one = scalar_const(1.0, &Shape::scalar(dt), g);
    let zero = scalar_const(0.0, &Shape::scalar(dt), g);
    let one_b = broadcast_scalar(g, one, &labels_shape);
    let zero_b = broadcast_scalar(g, zero, &labels_shape);
    for ci in 0..c {
        let class = scalar_const(ci as f64, &Shape::scalar(dt), g);
        let class_b = broadcast_scalar(g, class, &labels_shape);
        let eq = compare_eq(g, labels_flat, class_b);
        let col = g.add_node(Op::Where, vec![eq, one_b, zero_b], labels_shape.clone());
        cols.push(col);
    }
    let one_hot_flat = g.concat_(cols, 0);
    let one_hot = g.reshape_(one_hot_flat, vec![n as i64, c as i64]);
    let diff = g.sub(sm, one_hot);
    let dl_b = broadcast_scalar(g, d_loss, out_shape);
    g.mul(diff, dl_b)
}

/// `FakeQuantizeBackward` STE paths (matches CPU `training_bwd` kernels).
pub fn compose_fake_quantize_backward(
    g: &mut Graph,
    x: NodeId,
    dy: NodeId,
    out_shape: &Shape,
    bits: u8,
    axis: Option<usize>,
    ste: SteKind,
) -> NodeId {
    let len = out_shape.num_elements().expect("static fake_quant");
    let dt = out_shape.dtype();
    let q_max = q_max_for_bits(bits);

    let (chan_dim, _inner) = match axis {
        None => (1usize, len),
        Some(ax) => {
            let rank = out_shape.rank();
            assert!(ax < rank, "fake_quant axis in range");
            let cd = out_shape.dim(ax).unwrap_static();
            let inner = dim_product(out_shape, ax + 1, rank);
            (cd, inner)
        }
    };

    if matches!(ste, SteKind::Identity) {
        return dy;
    }

    let abs_x = g.add_node(
        Op::Activation(rlx_ir::op::Activation::Abs),
        vec![x],
        out_shape.clone(),
    );

    let flat_abs = g.reshape_(abs_x, vec![len as i64]);
    let max_abs = if chan_dim == 1 {
        g.reduce(
            flat_abs,
            rlx_ir::op::ReduceOp::Max,
            vec![0],
            false,
            Shape::new(&[1], dt),
        )
    } else {
        assert!(axis.is_some(), "per-channel fake_quant needs axis");
        let ax = axis.unwrap();
        g.reduce(
            abs_x,
            rlx_ir::op::ReduceOp::Max,
            vec![ax],
            true,
            Shape::from_dims(
                &out_shape
                    .dims()
                    .iter()
                    .enumerate()
                    .filter_map(|(i, d)| if i == ax { None } else { Some(*d) })
                    .collect::<Vec<_>>(),
                dt,
            ),
        )
    };

    let q_s = scalar_const(q_max as f64, &Shape::scalar(dt), g);
    let max_shape = g.node(max_abs).shape.clone();
    let q_b = broadcast_scalar(g, q_s, &max_shape);
    let scale = g.div(max_abs, q_b);
    let scale_shape = g.node(scale).shape.clone();
    let zero_node = f32_tensor_const(vec![0.0], out_shape.clone(), g);

    match ste {
        SteKind::Identity => dy,
        SteKind::ClippedIdentity => {
            let q_b2 = broadcast_scalar(g, q_s, &scale_shape);
            let bound = g.mul(q_b2, scale);
            let bound_b = broadcast_scalar(g, bound, out_shape);
            let cmp = compare_ge(g, bound_b, abs_x);
            g.add_node(Op::Where, vec![cmp, dy, zero_node], out_shape.clone())
        }
        SteKind::Tanh => {
            let scale_b = broadcast_scalar(g, scale, out_shape);
            let z = g.div(x, scale_b);
            let t = g.add_node(
                Op::Activation(rlx_ir::op::Activation::Tanh),
                vec![z],
                out_shape.clone(),
            );
            let t2 = g.mul(t, t);
            let one_s = scalar_const(1.0, &Shape::scalar(dt), g);
            let one = broadcast_scalar(g, one_s, out_shape);
            let att = g.sub(one, t2);
            g.mul(dy, att)
        }
        SteKind::HardTanh => {
            let q_b2 = broadcast_scalar(g, q_s, &scale_shape);
            let bound = g.mul(q_b2, scale);
            let bound_b = broadcast_scalar(g, bound, out_shape);
            let ratio = g.div(abs_x, bound_b);
            let one_s = scalar_const(1.0, &Shape::scalar(dt), g);
            let one = broadcast_scalar(g, one_s, out_shape);
            let att_raw = g.sub(one, ratio);
            let zero = f32_tensor_const(vec![0.0], out_shape.clone(), g);
            let pos = compare_ge(g, att_raw, zero);
            let att = g.add_node(Op::Where, vec![pos, att_raw, zero], out_shape.clone());
            g.mul(dy, att)
        }
    }
}

fn scan_vjp_input_names(body_vjp: &Graph) -> (String, Vec<String>) {
    scan_primal_input_names(body_vjp, "d_output")
}

fn scan_primal_input_names(body: &Graph, skip: &str) -> (String, Vec<String>) {
    let mut names: Vec<(NodeId, String)> = body
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Input { name } if skip.is_empty() || name.as_str() != skip => {
                Some((n.id, name.clone()))
            }
            _ => None,
        })
        .collect();
    names.sort_by_key(|(id, _)| *id);
    let ordered: Vec<String> = names.into_iter().map(|(_, n)| n).collect();
    let carry = ordered.first().expect("scan body carry input").clone();
    let xs = ordered.into_iter().skip(1).collect();
    (carry, xs)
}

fn xs_step_shape(g: &Graph, xs_id: NodeId, dt: DType) -> Shape {
    let xs_shape = g.node(xs_id).shape.clone();
    let mut step_dims: Vec<usize> = xs_shape
        .dims()
        .iter()
        .skip(1)
        .map(|d| d.unwrap_static())
        .collect();
    if step_dims.is_empty() {
        step_dims.push(1);
    }
    Shape::new(&step_dims, dt)
}

fn checkpoint_t_for_k(k: usize, k_total: usize, n_steps: usize) -> usize {
    if k_total == n_steps {
        k
    } else {
        ((k + 1) * n_steps)
            .div_ceil(k_total)
            .saturating_sub(1)
            .min(n_steps - 1)
    }
}

fn carry_before_step(
    g: &mut Graph,
    init: NodeId,
    trajectory: NodeId,
    t: usize,
    step_shape: &Shape,
) -> NodeId {
    if t == 0 {
        init
    } else {
        narrow_step(g, trajectory, t - 1, step_shape)
    }
}

fn forward_step_carry(
    g: &mut Graph,
    forward_body: &Graph,
    carry_name: &str,
    xs_names: &[String],
    xs: &[NodeId],
    carry_in: NodeId,
    t: usize,
    dt: DType,
) -> NodeId {
    let mut bind = std::collections::HashMap::from([(carry_name.to_string(), carry_in)]);
    for (i, name) in xs_names.iter().enumerate() {
        let step_shape = xs_step_shape(g, xs[i], dt);
        bind.insert(name.clone(), narrow_step(g, xs[i], t, &step_shape));
    }
    let id_map = merge_subgraph(g, forward_body, &bind);
    id_map[&forward_body.outputs[0]]
}

fn run_scan_backward_steps(
    g: &mut Graph,
    init: NodeId,
    trajectory: NodeId,
    upstream: NodeId,
    xs: &[NodeId],
    body_vjp: &Graph,
    forward_body: Option<&Graph>,
    length: u32,
    save_trajectory: bool,
    num_checkpoints: u32,
    out_shape: &Shape,
    xs_out_idx: Option<usize>,
) -> (NodeId, Option<Vec<NodeId>>) {
    let l = length as usize;
    let k_total = if num_checkpoints == 0 || num_checkpoints == length {
        l
    } else {
        num_checkpoints as usize
    };
    let is_partial = num_checkpoints != 0 && num_checkpoints != length;
    if is_partial {
        assert!(
            forward_body.is_some(),
            "compose_scan_backward: forward_body required for partial checkpoints"
        );
    }

    let (carry_name, xs_names) = scan_vjp_input_names(body_vjp);
    assert_eq!(xs.len(), xs_names.len());
    let (fwd_carry_name, fwd_xs_names) = forward_body
        .map(|fb| scan_primal_input_names(fb, ""))
        .unwrap_or_default();

    let dt = out_shape.dtype();
    let zero_bytes = vec![0u8; out_shape.num_elements().expect("static carry") * dt.size_bytes()];
    let zero_carry = g.add_node(Op::Constant { data: zero_bytes }, vec![], out_shape.clone());

    let mut dcarry = zero_carry;
    let mut dx_steps = xs_out_idx.map(|_| vec![NodeId(0); l]);

    let mut process_t = |g: &mut Graph, t: usize, carry_t: NodeId| {
        let mut d_out = dcarry;
        if save_trajectory {
            let up_t = narrow_step(g, upstream, t, out_shape);
            d_out = g.add(d_out, up_t);
        }
        let mut bind = std::collections::HashMap::from([
            (carry_name.clone(), carry_t),
            ("d_output".to_string(), d_out),
        ]);
        for (i, name) in xs_names.iter().enumerate() {
            let step_shape = xs_step_shape(g, xs[i], dt);
            bind.insert(name.clone(), narrow_step(g, xs[i], t, &step_shape));
        }
        let id_map = merge_subgraph(g, body_vjp, &bind);
        dcarry = id_map[&body_vjp.outputs[0]];
        if let (Some(idx), Some(steps)) = (xs_out_idx, dx_steps.as_mut()) {
            steps[t] = id_map[&body_vjp.outputs[idx]];
        }
    };

    if is_partial {
        let fb = forward_body.expect("forward_body");
        for seg_k in (0..k_total).rev() {
            let seg_end = checkpoint_t_for_k(seg_k, k_total, l);
            let seg_start = if seg_k == 0 {
                0
            } else {
                checkpoint_t_for_k(seg_k - 1, k_total, l) + 1
            };
            let anchor = if seg_k == 0 {
                init
            } else {
                narrow_step(g, trajectory, seg_k - 1, out_shape)
            };
            let mut carry_before = std::collections::HashMap::new();
            let mut carry = anchor;
            for t in seg_start..=seg_end {
                carry_before.insert(t, carry);
                if t < seg_end {
                    carry =
                        forward_step_carry(g, fb, &fwd_carry_name, &fwd_xs_names, xs, carry, t, dt);
                }
            }
            for t in (seg_start..=seg_end).rev() {
                process_t(g, t, carry_before[&t]);
            }
        }
    } else {
        for t in (0..l).rev() {
            let carry_t = carry_before_step(g, init, trajectory, t, out_shape);
            process_t(g, t, carry_t);
        }
    }
    (dcarry, dx_steps)
}

fn narrow_step(g: &mut Graph, x: NodeId, t: usize, step_shape: &Shape) -> NodeId {
    let narrowed = g.narrow_(x, 0, t, 1);
    let narrowed_shape = g.node(narrowed).shape.clone();
    if narrowed_shape.dims() == step_shape.dims() {
        narrowed
    } else {
        let dims: Vec<i64> = step_shape
            .dims()
            .iter()
            .map(|d| d.unwrap_static() as i64)
            .collect();
        g.reshape_(narrowed, dims)
    }
}

/// Max im2col rows × patch cols per matmul tile (raise for larger static convs).
// Im2col chunk-size guard for `compose_conv2d_backward_weight`. The original
// 16384 forces a chunked loop whenever `m * k` exceeds it, which on a
// modest-sized spatiotemporal CNN (e.g. `kernel (1, 25)`, batch 8, time 500)
// expands to ~50 chunks per backward op — each adding ~5 nodes — and slows
// graph compilation to multi-minute timescales. Bumped to 4 M (≈16 MiB per
// im2col tensor at f32) so a typical conv backward stays as a single
// im2col + matmul, dramatically reducing post-decompose graph size at the
// cost of larger transient buffers. Tunable per call site if the buffer
// blow-up ever becomes the bottleneck instead.
const IM2COL_MAX_MKL: usize = 4_194_304;

/// Unrolled `ScanBackward` for static length (trajectory-cached scans).
pub fn compose_scan_backward(
    g: &mut Graph,
    init: NodeId,
    trajectory: NodeId,
    upstream: NodeId,
    xs: &[NodeId],
    body_vjp: &Graph,
    forward_body: Option<&Graph>,
    length: u32,
    save_trajectory: bool,
    num_checkpoints: u32,
    out_shape: &Shape,
) -> NodeId {
    let l = length as usize;
    assert!(
        l > 0 && l <= SCAN_DECOMPOSE_MAX_LENGTH as usize,
        "compose_scan_backward: length 1..={SCAN_DECOMPOSE_MAX_LENGTH}"
    );
    assert!(
        save_trajectory,
        "compose_scan_backward: save_trajectory=true only"
    );
    run_scan_backward_steps(
        g,
        init,
        trajectory,
        upstream,
        xs,
        body_vjp,
        forward_body,
        length,
        save_trajectory,
        num_checkpoints,
        out_shape,
        None,
    )
    .0
}

/// Unrolled `ScanBackwardXs` — stacks per-step `dxs` along axis 0.
pub fn compose_scan_backward_xs(
    g: &mut Graph,
    init: NodeId,
    trajectory: NodeId,
    upstream: NodeId,
    xs: &[NodeId],
    body_vjp: &Graph,
    forward_body: Option<&Graph>,
    length: u32,
    save_trajectory: bool,
    num_checkpoints: u32,
    xs_idx: u32,
    out_shape: &Shape,
) -> NodeId {
    let l = length as usize;
    assert!(
        l > 0 && l <= SCAN_DECOMPOSE_MAX_LENGTH as usize,
        "compose_scan_backward_xs: length 1..={SCAN_DECOMPOSE_MAX_LENGTH}"
    );
    assert!(
        save_trajectory,
        "compose_scan_backward_xs: save_trajectory=true only"
    );
    let out_idx = 1 + xs_idx as usize;
    assert!(
        out_idx < body_vjp.outputs.len(),
        "compose_scan_backward_xs: xs_idx out of range"
    );
    let carry_shape = g.node(trajectory).shape.clone();
    let mut carry_step_dims: Vec<usize> = carry_shape
        .dims()
        .iter()
        .skip(1)
        .map(|d| d.unwrap_static())
        .collect();
    if carry_step_dims.is_empty() {
        carry_step_dims.push(1);
    }
    let carry_step = Shape::new(&carry_step_dims, carry_shape.dtype());
    let (dcarry, dx_steps) = run_scan_backward_steps(
        g,
        init,
        trajectory,
        upstream,
        xs,
        body_vjp,
        forward_body,
        length,
        save_trajectory,
        num_checkpoints,
        &carry_step,
        Some(out_idx),
    );
    let mut dx_steps = dx_steps.expect("dx steps");
    dx_steps.reverse();
    let stacked = g.concat_(dx_steps, 0);
    let _ = (dcarry, out_shape);
    stacked
}

fn reshape_attn_rank4(
    g: &mut Graph,
    q: NodeId,
    k: NodeId,
    v: NodeId,
    dy: NodeId,
    num_heads: usize,
    head_dim: usize,
) -> (NodeId, NodeId, NodeId, NodeId, Shape) {
    let q_shape = g.node(q).shape.clone();
    if q_shape.rank() == 4 {
        return (q, k, v, dy, q_shape);
    }
    if q_shape.rank() == 3 {
        let b = q_shape.dim(0).unwrap_static();
        let s = q_shape.dim(1).unwrap_static();
        let e = q_shape.dim(2).unwrap_static();
        assert_eq!(
            e,
            num_heads * head_dim,
            "rank-3 attention: last dim must be num_heads * head_dim"
        );
        let r4 = Shape::new(&[b, num_heads, s, head_dim], q_shape.dtype());
        let q4 = g.reshape_(
            q,
            vec![b as i64, num_heads as i64, s as i64, head_dim as i64],
        );
        let k4 = g.reshape_(
            k,
            vec![b as i64, num_heads as i64, s as i64, head_dim as i64],
        );
        let v4 = g.reshape_(
            v,
            vec![b as i64, num_heads as i64, s as i64, head_dim as i64],
        );
        let dy4 = g.reshape_(
            dy,
            vec![b as i64, num_heads as i64, s as i64, head_dim as i64],
        );
        return (q4, k4, v4, dy4, r4);
    }
    panic!(
        "compose_attention_backward: rank-3/4 [B,H,S,D] or [B,S,H*D] only, got rank {}",
        q_shape.rank()
    );
}

/// Single-group (or groups=1) im2col + matmul for `Conv2dBackwardWeight`.
fn compose_conv2d_backward_weight_group(
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
fn compose_conv2d_backward_weight_w1_h(
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

fn build_im2col_rows(
    g: &mut Graph,
    flat_x: NodeId,
    zero: NodeId,
    n: usize,
    c_in: usize,
    h: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw_d: usize,
    k: usize,
    m_start: usize,
    m_end: usize,
) -> NodeId {
    let mut rows: Vec<NodeId> = Vec::with_capacity(m_end - m_start);
    let mut flat = 0usize;
    'outer: for ni in 0..n {
        for ho in 0..h_out {
            for wo in 0..w_out {
                if flat >= m_end {
                    break 'outer;
                }
                if flat >= m_start {
                    let mut patch: Vec<NodeId> = Vec::with_capacity(k);
                    for ci in 0..c_in {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let hi = ho * sh + ki * dh;
                                let wi = wo * sw + kj * dw_d;
                                let val = if hi < ph || wi < pw || hi - ph >= h || wi - pw >= w_in {
                                    zero
                                } else {
                                    let idx = ((ni * c_in + ci) * h + (hi - ph)) * w_in + (wi - pw);
                                    gather_flat_f32(g, flat_x, idx, g.node(flat_x).shape.dtype())
                                };
                                patch.push(val);
                            }
                        }
                    }
                    rows.push(g.concat_(patch, 0));
                }
                flat += 1;
            }
        }
    }
    g.concat_(rows, 0)
}

fn compare_eq(g: &mut Graph, lhs: NodeId, rhs: NodeId) -> NodeId {
    let s = shape::compare_shape(&g.node(lhs).shape, &g.node(rhs).shape).expect("compare eq");
    g.add_node(Op::Compare(CmpOp::Eq), vec![lhs, rhs], s)
}

fn compare_ge(g: &mut Graph, lhs: NodeId, rhs: NodeId) -> NodeId {
    let s = shape::compare_shape(&g.node(lhs).shape, &g.node(rhs).shape).expect("compare ge");
    g.add_node(Op::Compare(CmpOp::Ge), vec![lhs, rhs], s)
}

fn compare_gt(g: &mut Graph, lhs: NodeId, rhs: NodeId) -> NodeId {
    let s = shape::compare_shape(&g.node(lhs).shape, &g.node(rhs).shape).expect("compare gt");
    g.add_node(Op::Compare(CmpOp::Gt), vec![lhs, rhs], s)
}

fn cast_f32(g: &mut Graph, x: NodeId) -> NodeId {
    let s = g.node(x).shape.clone().with_dtype(DType::F32);
    g.add_node(Op::Cast { to: DType::F32 }, vec![x], s)
}

fn argmax_window_flat(
    g: &mut Graph,
    flat_x: NodeId,
    _n: usize,
    c: usize,
    h: usize,
    w: usize,
    ni: usize,
    ci: usize,
    ho: usize,
    wo: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dt: DType,
) -> NodeId {
    let in_chan = (ni * c + ci) * h * w;
    let mut best_v: Option<NodeId> = None;
    let mut best_i: Option<NodeId> = None;
    for ki in 0..kh {
        for kj in 0..kw {
            let hi = ho * sh + ki;
            let wi = wo * sw + kj;
            if hi < ph || wi < pw {
                continue;
            }
            let hi = hi - ph;
            let wi = wi - pw;
            if hi >= h || wi >= w {
                continue;
            }
            let idx = in_chan + hi * w + wi;
            let val = gather_flat_f32(g, flat_x, idx, dt);
            let idx_n = f32_tensor_const(vec![idx as f32], Shape::new(&[1], dt), g);
            match (best_v, best_i) {
                (None, None) => {
                    best_v = Some(val);
                    best_i = Some(idx_n);
                }
                (Some(bv), Some(bi)) => {
                    let cond = compare_gt(g, val, bv);
                    best_v = Some(where_select(g, cond, val, bv));
                    best_i = Some(where_select(g, cond, idx_n, bi));
                }
                _ => unreachable!("maxpool argmax partial state"),
            }
        }
    }
    best_i.expect("maxpool window has no in-bounds positions")
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
    let flat_n = n * c * h * w_in;
    let num_windows = n * c * h_out * w_out;
    assert!(
        flat_n.saturating_mul(num_windows) <= 4096,
        "compose_max_pool2d_backward: scatter too large ({flat_n}x{num_windows})"
    );

    let dt = out_shape.dtype();
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

/// GroupNorm backward w.r.t. input (NCHW, static dims).
pub fn compose_group_norm_backward_input(
    g: &mut Graph,
    x: NodeId,
    gamma: NodeId,
    _beta: NodeId,
    dy: NodeId,
    num_groups: usize,
    eps: f32,
    out_shape: &Shape,
) -> NodeId {
    let [n, c, h, w] = static_dim4(out_shape).expect("static NCHW out");
    let cpg = c / num_groups;
    // Pin activations — decomposed backward re-reads x/dy many times; memory
    // planning may alias input slots with temporaries in large grad graphs.
    let ones = broadcast_eps(g, 1.0, out_shape);
    let x = g.mul(x, ones);
    let dy = g.mul(dy, ones);
    let mut dx_groups: Vec<NodeId> = Vec::with_capacity(num_groups);
    for gi in 0..num_groups {
        let c0 = gi * cpg;
        let x_g = g.narrow_(x, 1, c0, cpg);
        let dy_g = g.narrow_(dy, 1, c0, cpg);
        let gamma_g = g.narrow_(gamma, 0, c0, cpg);
        let g_shape = g.node(x_g).shape.clone();
        // Align gamma with NCHW channel axis (axis 1), not the trailing width axis.
        let gamma_r = g.reshape_(gamma_g, vec![1, cpg as i64, 1, 1]);
        let gamma_b = broadcast_scalar(g, gamma_r, &g_shape);

        // Flatten C×H×W so every Reduce is last-axis (wgpu only wires axis -1).
        let elems = cpg * h * w;
        let flat_shape = Shape::new(&[n, elems], g_shape.dtype());
        let flat_x = g.reshape_(x_g, vec![n as i64, elems as i64]);
        let flat_dy = g.reshape_(dy_g, vec![n as i64, elems as i64]);
        let flat_gamma = g.reshape_(gamma_b, vec![n as i64, elems as i64]);

        let mean = g.mean(flat_x, vec![1], true);
        let mean_b = broadcast_scalar(g, mean, &flat_shape);
        let xc = g.sub(flat_x, mean_b);
        let xc2 = g.mul(xc, xc);
        let var = g.mean(xc2, vec![1], true);
        let var_shape = g.node(var).shape.clone();
        let eps_b = broadcast_eps(g, eps, &var_shape);
        let var_eps = g.add(var, eps_b);
        let inv_std = g.add_node(
            Op::Activation(rlx_ir::op::Activation::Rsqrt),
            vec![var_eps],
            var_shape,
        );
        let inv_std_b = broadcast_scalar(g, inv_std, &flat_shape);
        let x_hat = g.mul(xc, inv_std_b);
        let sy = g.mul(flat_dy, flat_gamma);
        let mean_sy = g.mean(sy, vec![1], true);
        let mean_sy_b = broadcast_scalar(g, mean_sy, &flat_shape);
        let sy_xhat = g.mul(sy, x_hat);
        let mean_sy_xhat = g.mean(sy_xhat, vec![1], true);
        let mean_sy_xhat_b = broadcast_scalar(g, mean_sy_xhat, &flat_shape);
        let t1 = g.sub(sy, mean_sy_b);
        let t2 = g.mul(x_hat, mean_sy_xhat_b);
        let term = g.sub(t1, t2);
        let flat_dx = g.mul(term, inv_std_b);
        dx_groups.push(g.reshape_(flat_dx, vec![n as i64, cpg as i64, h as i64, w as i64]));
    }
    g.concat_(dx_groups, 1)
}

/// GroupNorm backward w.r.t. gamma (NCHW).
pub fn compose_group_norm_backward_gamma(
    g: &mut Graph,
    x: NodeId,
    dy: NodeId,
    num_groups: usize,
    eps: f32,
    gamma_shape: &Shape,
) -> NodeId {
    let x_shape = g.node(x).shape.clone();
    let [n, c, h, w] = static_dim4(&x_shape).expect("static NCHW x");
    let cpg = c / num_groups;
    let dt = gamma_shape.dtype();
    let ones = broadcast_eps(g, 1.0, &x_shape);
    let x = g.mul(x, ones);
    let dy = g.mul(dy, ones);
    let mut dgamma: Vec<NodeId> = Vec::with_capacity(c);
    for gi in 0..num_groups {
        let c0 = gi * cpg;
        let x_g = g.narrow_(x, 1, c0, cpg);
        let dy_g = g.narrow_(dy, 1, c0, cpg);
        let elems = cpg * h * w;
        let flat_shape = Shape::new(&[n, elems], dt);
        let flat_x = g.reshape_(x_g, vec![n as i64, elems as i64]);
        let flat_dy = g.reshape_(dy_g, vec![n as i64, elems as i64]);
        let mean = g.mean(flat_x, vec![1], true);
        let mean_b = broadcast_scalar(g, mean, &flat_shape);
        let xc = g.sub(flat_x, mean_b);
        let xc2 = g.mul(xc, xc);
        let var = g.mean(xc2, vec![1], true);
        let var_shape = g.node(var).shape.clone();
        let eps_b = broadcast_eps(g, eps, &var_shape);
        let var_eps = g.add(var, eps_b);
        let inv_std = g.add_node(
            Op::Activation(rlx_ir::op::Activation::Rsqrt),
            vec![var_eps],
            var_shape,
        );
        let inv_std_b = broadcast_scalar(g, inv_std, &flat_shape);
        let x_hat = g.mul(xc, inv_std_b);
        let prod = g.mul(flat_dy, x_hat);
        let prod_g = g.reshape_(prod, vec![cpg as i64, (n * h * w) as i64]);
        let summed_g = g.reduce(
            prod_g,
            rlx_ir::op::ReduceOp::Sum,
            vec![1],
            false,
            Shape::new(&[cpg], dt),
        );
        dgamma.push(summed_g);
    }
    g.concat_(dgamma, 0)
}

/// GroupNorm backward w.r.t. beta: `sum(dy)` over N,H,W.
pub fn compose_group_norm_backward_beta(g: &mut Graph, dy: NodeId, beta_shape: &Shape) -> NodeId {
    let dy_shape = g.node(dy).shape.clone();
    let [n, c, h, w] = static_dim4(&dy_shape).expect("static NCHW dy");
    let flat = g.reshape_(dy, vec![c as i64, (n * h * w) as i64]);
    g.reduce(
        flat,
        rlx_ir::op::ReduceOp::Sum,
        vec![1],
        false,
        beta_shape.clone(),
    )
}

/// RoPE backward = forward RoPE with negated sin table (NeoX).
pub fn compose_rope_backward(
    g: &mut Graph,
    dy: NodeId,
    cos: NodeId,
    sin: NodeId,
    head_dim: usize,
    n_rot: usize,
) -> NodeId {
    let sin_shape = g.node(sin).shape.clone();
    let neg = scalar_const(-1.0, &sin_shape, g);
    let neg_sin = g.mul(sin, neg);
    g.rope_n(dy, cos, neg_sin, head_dim, n_rot)
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
    g.add_node(
        Op::Conv {
            kernel_size: kernel_size.to_vec(),
            stride: stride.to_vec(),
            padding: padding.to_vec(),
            dilation: dilation.to_vec(),
            groups,
        },
        vec![dy, w],
        out_shape.clone(),
    )
}

fn f32_tensor_const(data: Vec<f32>, shape: Shape, g: &mut Graph) -> NodeId {
    let bytes: Vec<u8> = data.into_iter().flat_map(f32::to_le_bytes).collect();
    g.add_node(Op::Constant { data: bytes }, vec![], shape)
}

fn where_select(g: &mut Graph, cond: NodeId, on_true: NodeId, on_false: NodeId) -> NodeId {
    let s = shape::binary_shape(&g.node(on_true).shape, &g.node(on_false).shape).expect("where");
    g.add_node(Op::Where, vec![cond, on_true, on_false], s)
}

/// Additive 0 / `-inf` mask for [`MaskKind::Causal`] and [`MaskKind::SlidingWindow`],
/// matching `apply_synthetic_mask` in `rlx-cpu`.
fn synthetic_additive_mask_data(
    bh: usize,
    q_seq: usize,
    k_seq: usize,
    mask_kind: MaskKind,
) -> Vec<f32> {
    let mut buf = vec![0.0f32; bh * q_seq * k_seq];
    let q_offset = k_seq.saturating_sub(q_seq);
    match mask_kind {
        MaskKind::Causal => {
            for bh_i in 0..bh {
                for qi in 0..q_seq {
                    let abs_q = q_offset + qi;
                    for ki in (abs_q + 1)..k_seq {
                        buf[bh_i * q_seq * k_seq + qi * k_seq + ki] = ATTN_MASK_NEG_INF;
                    }
                }
            }
        }
        MaskKind::SlidingWindow(w) => {
            for bh_i in 0..bh {
                for qi in 0..q_seq {
                    let abs_q = q_offset + qi;
                    let lo = abs_q.saturating_sub(w);
                    for ki in 0..k_seq {
                        if ki < lo || ki > abs_q {
                            buf[bh_i * q_seq * k_seq + qi * k_seq + ki] = ATTN_MASK_NEG_INF;
                        }
                    }
                }
            }
        }
        MaskKind::None | MaskKind::Custom | MaskKind::Bias => {}
    }
    buf
}

fn broadcast_mask_to_scores(
    g: &mut Graph,
    mask: NodeId,
    mask_shape: &Shape,
    scores_shape: &Shape,
    b: usize,
    h: usize,
    bh: usize,
    q_seq: usize,
    k_seq: usize,
    mask_kind: MaskKind,
) -> NodeId {
    let dtype = scores_shape.dtype();
    match mask_kind {
        MaskKind::Bias => {
            let full = Shape::new(&[b, h, q_seq, k_seq], dtype);
            let node = if mask_shape == &full {
                mask
            } else {
                broadcast_scalar(g, mask, &full)
            };
            g.reshape(
                node,
                vec![bh as i64, q_seq as i64, k_seq as i64],
                scores_shape.clone(),
            )
        }
        MaskKind::Custom => match mask_shape.rank() {
            2 => {
                let mid = Shape::new(&[b, 1, 1, k_seq], dtype);
                let reshaped = g.reshape(mask, vec![b as i64, 1, 1, k_seq as i64], mid.clone());
                let full = Shape::new(&[b, h, q_seq, k_seq], dtype);
                let expanded = broadcast_scalar(g, reshaped, &full);
                g.reshape(
                    expanded,
                    vec![bh as i64, q_seq as i64, k_seq as i64],
                    scores_shape.clone(),
                )
            }
            4 => g.reshape(
                mask,
                vec![bh as i64, q_seq as i64, k_seq as i64],
                scores_shape.clone(),
            ),
            _ => broadcast_scalar(g, mask, scores_shape),
        },
        _ => panic!("broadcast_mask_to_scores: Custom/Bias only"),
    }
}

fn apply_attn_score_mask(
    g: &mut Graph,
    scaled: NodeId,
    scores_shape: &Shape,
    mask_kind: MaskKind,
    mask: Option<NodeId>,
    mask_shape: Option<&Shape>,
    bh: usize,
    b: usize,
    h: usize,
    q_seq: usize,
    k_seq: usize,
) -> NodeId {
    let dtype = scores_shape.dtype();
    let mut scores = scaled;

    if matches!(mask_kind, MaskKind::Custom) {
        let mask_id = mask.expect("Custom attention requires mask input");
        let mask_s = mask_shape.expect("Custom attention requires mask shape");
        let mask_b = broadcast_mask_to_scores(
            g,
            mask_id,
            mask_s,
            scores_shape,
            b,
            h,
            bh,
            q_seq,
            k_seq,
            mask_kind,
        );
        let thr = scalar_const(MASK_BINARY_THRESHOLD as f64, &Shape::scalar(dtype), g);
        let mask_b_shape = g.node(mask_b).shape.clone();
        let thr_b = broadcast_scalar(g, thr, &mask_b_shape);
        let valid = compare_ge(g, mask_b, thr_b);
        let neg = scalar_const(ATTN_MASK_NEG_INF as f64, &Shape::scalar(dtype), g);
        let neg_b = broadcast_scalar(g, neg, scores_shape);
        scores = where_select(g, valid, scores, neg_b);
    }

    if matches!(mask_kind, MaskKind::Bias) {
        let mask_id = mask.expect("Bias attention requires mask input");
        let mask_s = mask_shape.expect("Bias attention requires mask shape");
        let mask_b = broadcast_mask_to_scores(
            g,
            mask_id,
            mask_s,
            scores_shape,
            b,
            h,
            bh,
            q_seq,
            k_seq,
            mask_kind,
        );
        scores = g.add(scores, mask_b);
    }

    if matches!(mask_kind, MaskKind::Causal | MaskKind::SlidingWindow(_)) {
        let data = synthetic_additive_mask_data(bh, q_seq, k_seq, mask_kind);
        let additive = f32_tensor_const(data, scores_shape.clone(), g);
        scores = g.add(scores, additive);
    }

    scores
}

/// Rank-4 `[B, H, S, D]` SDPA via matmul + mask + softmax + matmul.
/// Used before reverse-mode so higher-order AD never sees `Op::Attention`.
pub fn expand_attention_forward_primitives(
    g: &mut Graph,
    q: NodeId,
    k: NodeId,
    v: NodeId,
    num_heads: usize,
    head_dim: usize,
    out_shape: &Shape,
    q_seq: usize,
    k_seq: usize,
    mask_kind: MaskKind,
    mask: Option<NodeId>,
    mask_shape: Option<&Shape>,
) -> NodeId {
    assert_eq!(
        out_shape.rank(),
        4,
        "expand_attention_forward_primitives: rank-4 [B,H,S,D] only"
    );
    let dtype = out_shape.dtype();
    let b = out_shape.dim(0).unwrap_static();
    let h = out_shape.dim(1).unwrap_static();
    let s = out_shape.dim(2).unwrap_static();
    let d = out_shape.dim(3).unwrap_static();
    assert_eq!(h, num_heads, "num_heads mismatch");
    assert_eq!(d, head_dim, "head_dim mismatch");
    let bh = b * h;

    let flat3 = Shape::new(&[bh, s, d], dtype);
    let q_flat = g.reshape(q, vec![bh as i64, s as i64, d as i64], flat3.clone());
    let k_flat = g.reshape(k, vec![bh as i64, s as i64, d as i64], flat3.clone());
    let v_flat = g.reshape(v, vec![bh as i64, s as i64, d as i64], flat3);

    let k_t_shape = Shape::new(&[bh, d, s], dtype);
    let k_t = g.add_node(
        Op::Transpose {
            perm: vec![0, 2, 1],
        },
        vec![k_flat],
        k_t_shape.clone(),
    );

    let scores_shape =
        shape::matmul_shape(&g.node(q_flat).shape, &k_t_shape).expect("attn scores shape");
    let scores = g.matmul(q_flat, k_t, scores_shape.clone());

    let scale = (head_dim as f32).sqrt().recip();
    let scale_n = scalar_const(scale as f64, &Shape::scalar(dtype), g);
    let scale_b = broadcast_scalar(g, scale_n, &scores_shape);
    let scaled = g.mul(scores, scale_b);
    let masked = apply_attn_score_mask(
        g,
        scaled,
        &scores_shape,
        mask_kind,
        mask,
        mask_shape,
        bh,
        b,
        h,
        q_seq,
        k_seq,
    );

    let weights = g.softmax(masked, -1, scores_shape.clone());

    let out_flat_shape =
        shape::matmul_shape(&g.node(weights).shape, &g.node(v_flat).shape).expect("attn out shape");
    let out_flat = g.matmul(weights, v_flat, out_flat_shape);

    g.reshape(
        out_flat,
        vec![b as i64, h as i64, s as i64, d as i64],
        out_shape.clone(),
    )
}

/// Build attention backward via reverse-mode on `sum(Attention(q,k,v) ⊙ dy)`.
pub fn compose_attention_backward(
    wrt: AttentionBwdWrt,
    q_shape: &Shape,
    k_shape: &Shape,
    v_shape: &Shape,
    dy_shape: &Shape,
    num_heads: usize,
    head_dim: usize,
    mask_kind: MaskKind,
    mask_shape: Option<&Shape>,
) -> Graph {
    let mut sub = Graph::new("attn_bwd_decomp");
    let q = sub.input("q", q_shape.clone());
    let k = sub.input("k", k_shape.clone());
    let v = sub.input("v", v_shape.clone());
    let dy = sub.input("dy", dy_shape.clone());
    let mask_node = if matches!(mask_kind, MaskKind::Custom | MaskKind::Bias) {
        Some(
            sub.input(
                "mask",
                mask_shape
                    .cloned()
                    .expect("Custom/Bias attention decompose requires mask shape"),
            ),
        )
    } else {
        None
    };

    let (q, k, v, dy, q_shape) = reshape_attn_rank4(&mut sub, q, k, v, dy, num_heads, head_dim);
    let k_shape = sub.node(k).shape.clone();
    let dy_shape = sub.node(dy).shape.clone();
    let q_seq = q_shape.dim(2).unwrap_static();
    let k_seq = k_shape.dim(2).unwrap_static();

    let y = expand_attention_forward_primitives(
        &mut sub, q, k, v, num_heads, head_dim, &dy_shape, q_seq, k_seq, mask_kind, mask_node,
        mask_shape,
    );

    let prod = sub.mul(y, dy);
    let rank = dy_shape.rank();
    let loss = sub.sum(prod, (0..rank).collect(), false);
    sub.set_outputs(vec![loss]);

    let prep = prepare_graph_for_ad(sub);
    let wrt_id = match wrt {
        AttentionBwdWrt::Query => q,
        AttentionBwdWrt::Key => k,
        AttentionBwdWrt::Value => v,
    };
    let mut bwd = grad_with_loss(&prep, &[wrt_id]);
    crate::compose::internalize_d_output(&mut bwd);
    let grad_out = bwd.outputs[1];
    bwd.set_outputs(vec![grad_out]);
    bwd
}

/// Merge an attention-backward subgraph into `g` at the given bindings.
pub fn emit_attention_backward(
    g: &mut Graph,
    wrt: AttentionBwdWrt,
    inputs: &[NodeId],
    _out_shape: &Shape,
    num_heads: usize,
    head_dim: usize,
    mask_kind: MaskKind,
) -> NodeId {
    let (q, k, v, dy, mask_in) = match inputs {
        [q, k, v, dy] => (*q, *k, *v, *dy, None),
        [q, k, v, dy, mask] => (*q, *k, *v, *dy, Some(*mask)),
        _ => panic!("AttentionBackward expects [q, k, v, dy] or [q, k, v, dy, mask]"),
    };
    let mask_shape = mask_in.map(|m| g.node(m).shape.clone());
    let sub = compose_attention_backward(
        wrt,
        &g.node(q).shape.clone(),
        &g.node(k).shape.clone(),
        &g.node(v).shape.clone(),
        &g.node(dy).shape.clone(),
        num_heads,
        head_dim,
        mask_kind,
        mask_shape.as_ref(),
    );
    let mut bind = std::collections::HashMap::from([
        ("q".to_string(), q),
        ("k".to_string(), k),
        ("v".to_string(), v),
        ("dy".to_string(), dy),
    ]);
    if let Some(mask) = mask_in {
        bind.insert("mask".to_string(), mask);
    }
    let id_map = merge_subgraph(g, &sub, &bind);
    id_map[&sub.outputs[0]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::DType;

    #[test]
    fn composed_ln_bwd_matches_kernel_smoke() {
        let rows = 2usize;
        let h = 4usize;
        let eps = 1e-5f32;
        let mut g = Graph::new("ln");
        let x = g.input("x", Shape::new(&[rows, h], DType::F32));
        let gamma = g.input("gamma", Shape::new(&[h], DType::F32));
        let dy = g.input("dy", Shape::new(&[rows, h], DType::F32));
        let dx_k = g.layer_norm_backward_input(x, gamma, dy, -1, eps);
        let dx_c = compose_layer_norm_backward_input(
            &mut g,
            x,
            gamma,
            dy,
            -1,
            eps,
            &Shape::new(&[rows, h], DType::F32),
        );
        assert_eq!(g.node(dx_k).shape, g.node(dx_c).shape);
    }
}
