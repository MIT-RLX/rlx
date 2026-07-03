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

//! `norm` — extracted from the `decompose_backward_kernels` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use rlx_ir::infer::GraphExt;
use rlx_ir::op::{AttentionBwdWrt, CmpOp, MaskKind, SteKind};
use rlx_ir::shape;
use rlx_ir::shape::Dim;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

use super::*;

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
    let inv_r_b = broadcast_scalar(g, inv_r, out_shape);
    let inv_r2_full = broadcast_scalar(g, inv_r2, out_shape);

    let g_b = broadcast_scalar(g, gamma, out_shape);
    let dy_g = g.mul(dy, g_b);
    let x_dy_g = g.mul(x, dy_g);
    let dot = g.mean(x_dy_g, axes, true);
    let dot_b = broadcast_scalar(g, dot, out_shape);

    // dx = inv·(γ·dy − inv²·x·dot), where dot = mean(x·γ·dy) over the axis. The
    // cross term is inv²·x·dot and the whole thing is scaled by inv once — earlier
    // this used inv³·x·dot then ×inv, an extra 1/r factor (the documented
    // "rms_norm_backward 1/r bug" — wrong gradient on every backend).
    let term1 = g.mul(g_b, dy);
    let x_dot = g.mul(x, dot_b);
    let term2 = g.mul(x_dot, inv_r2_full);
    let diff = g.sub(term1, term2);
    g.mul(diff, inv_r_b)
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

