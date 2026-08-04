// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
use crate::thunk::*;

/// KAN learnable spline activation (Gaussian-RBF basis).
///
/// Each channel `c` has its own univariate function
/// `y = Σ_g coeff[c,g] · exp(-((x − center_g)·inv_h)²)`, with `num_basis`
/// centers uniform on `[grid_min, grid_max]` and `inv_h = (num_basis−1) /
/// (grid_max−grid_min)` (adjacent RBFs ~one width apart). The centers/width are
/// fixed hyperparameters; only `coeff` is learned — "weights as functions".
#[allow(clippy::too_many_arguments)]
pub fn spline_activation_f32(
    x: &[f32],       // [rows, channels]
    coeff: &[f32],   // [channels, num_basis]
    out: &mut [f32], // [rows, channels]
    rows: usize,
    channels: usize,
    num_basis: usize,
    grid_min: f32,
    grid_max: f32,
) {
    let step = if num_basis > 1 {
        (grid_max - grid_min) / (num_basis as f32 - 1.0)
    } else {
        1.0
    };
    let inv_h = 1.0 / step;
    for r in 0..rows {
        for c in 0..channels {
            let xv = x[r * channels + c];
            let cb = &coeff[c * num_basis..c * num_basis + num_basis];
            let mut acc = 0f32;
            for (g, &w) in cb.iter().enumerate() {
                let center = grid_min + g as f32 * step;
                let z = (xv - center) * inv_h;
                acc += w * (-(z * z)).exp();
            }
            out[r * channels + c] = acc;
        }
    }
}

/// Lower an `Op::SplineActivation` node to a `Thunk::SplineActivation`.
pub(crate) fn compile_spline_activation(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
) -> Thunk {
    let Op::SplineActivation {
        num_basis,
        grid_min,
        grid_max,
    } = &node.op
    else {
        unreachable!()
    };
    let x_shape = &graph.node(node.inputs[0]).shape;
    let channels = x_shape.dim(x_shape.rank() - 1).unwrap_static();
    let total = x_shape.num_elements().unwrap();
    let rows = total / channels.max(1);
    Thunk::SplineActivation {
        x: node_offset(arena, node.inputs[0]),
        coeff: node_offset(arena, node.inputs[1]),
        dst: node_offset(arena, node.id),
        rows: rows as u32,
        channels: channels as u32,
        num_basis: *num_basis,
        grid_min: *grid_min,
        grid_max: *grid_max,
    }
}

/// Interpreter-path execution of `Thunk::SplineActivation`.
#[inline(always)]
pub(crate) fn exec_spline_activation(t: &Thunk, base: *mut u8) {
    let Thunk::SplineActivation {
        x,
        coeff,
        dst,
        rows,
        channels,
        num_basis,
        grid_min,
        grid_max,
    } = t
    else {
        unreachable!()
    };
    let (rows, channels, nb) = (*rows as usize, *channels as usize, *num_basis as usize);
    unsafe {
        let xs = sl(*x, base, rows * channels);
        let cb = sl(*coeff, base, channels * nb);
        let out = sl_mut(*dst, base, rows * channels);
        spline_activation_f32(xs, cb, out, rows, channels, nb, *grid_min, *grid_max);
    }
}
