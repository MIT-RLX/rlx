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
//! Common IR baseline for [`crate::op::Op::GaussianSplatRender`] (primitive ops only).
//!
//! This is a **portable baseline**, not a drop-in replacement for native tile-based splat.
//! It alpha-weights scene colors and broadcasts a single RGBA tile across the framebuffer so
//! every backend executes the same elementwise/reduce schedule. Native `GaussianSplatRender`
//! thunks remain the production fast path when declared in `supported_ops`.
//!
//! Backward matches this baseline analytically (mean / broadcast / mul VJPs in MIR).

use crate::infer::GraphExt;
use crate::op::{BinaryOp, Op};
use crate::ops::splat::gaussian_splat_packed_grad_len;
use crate::shape;
use crate::{DType, Graph, NodeId, Shape};

fn expand_to(g: &mut Graph, x: NodeId, target: &[i64]) -> NodeId {
    let dtype = g.shape(x).dtype();
    let out = Shape::new(
        &target.iter().map(|&d| d as usize).collect::<Vec<_>>(),
        dtype,
    );
    g.add_node(
        Op::Expand {
            target_shape: target.to_vec(),
        },
        vec![x],
        out,
    )
}

fn scalar_const(g: &mut Graph, v: f32, dtype: DType) -> NodeId {
    g.add_node(
        Op::Constant {
            data: v.to_le_bytes().to_vec(),
        },
        vec![],
        Shape::new(&[1], dtype),
    )
}

fn zeros_like(g: &mut Graph, like: NodeId) -> NodeId {
    let n = g.shape(like).num_elements().unwrap_or(0);
    let z = scalar_const(g, 0.0, g.shape(like).dtype());
    expand_to(g, z, &[n.max(1) as i64])
}

/// Replace one `GaussianSplatRender` node with primitive MIR ops.
pub fn lower_gaussian_splat_render(
    g: &mut Graph,
    positions: NodeId,
    _scales: NodeId,
    _rotations: NodeId,
    opacities: NodeId,
    colors: NodeId,
    _sh_coeffs: NodeId,
    _meta: NodeId,
    width: u32,
    height: u32,
    _out_shape: Shape,
) -> NodeId {
    let _ = positions;
    let count = g.shape(opacities).num_elements().unwrap_or(0).max(1);

    let colors2 = g.reshape_(colors, vec![count as i64, 3]);
    let op_2d = g.reshape_(opacities, vec![count as i64, 1]);
    let op_3 = expand_to(g, op_2d, &[count as i64, 3]);
    let wshape = shape::binary_shape(g.shape(colors2), g.shape(op_3)).expect("splat common mul");
    let weighted = g.binary(BinaryOp::Mul, colors2, op_3, wshape);

    let rgb = g.mean(weighted, vec![0], false);
    let alpha = g.mean(op_2d, vec![0], false);
    let rgba = g.concat_(vec![rgb, alpha], 0);

    let pixels = (width as usize).saturating_mul(height as usize).max(1);
    let tile = expand_to(g, rgba, &[pixels as i64, 4]);
    g.reshape_(tile, vec![(pixels * 4) as i64])
}

/// Common backward for [`lower_gaussian_splat_render`] — packed scene grads via primitive ops.
pub fn lower_gaussian_splat_render_backward(
    g: &mut Graph,
    positions: NodeId,
    scales: NodeId,
    rotations: NodeId,
    opacities: NodeId,
    colors: NodeId,
    sh_coeffs: NodeId,
    _meta: NodeId,
    d_loss_rgba: NodeId,
    width: u32,
    height: u32,
    out_shape: Shape,
) -> NodeId {
    let dtype = out_shape.dtype();
    let count = g.shape(positions).num_elements().unwrap_or(0) / 3;
    let count = count.max(1);
    let sh_len = g.shape(sh_coeffs).num_elements().unwrap_or(0);
    let sh_coeff_count = if count == 0 {
        1
    } else {
        (sh_len / (count * 3)).max(1)
    };
    let pixels = (width as usize)
        .saturating_mul(height as usize)
        .max(1);

    // Forward intermediates (same as `lower_gaussian_splat_render`).
    let colors2 = g.reshape_(colors, vec![count as i64, 3]);
    let op_2d = g.reshape_(opacities, vec![count as i64, 1]);
    let op_3 = expand_to(g, op_2d, &[count as i64, 3]);

    // Upstream: d_loss [pixels*4] → sum over broadcast pixels → d_rgba [4].
    let dy = g.reshape_(d_loss_rgba, vec![pixels as i64, 4]);
    let d_rgba = g.sum(dy, vec![0], false);
    let d_rgb = g.narrow_(d_rgba, 0, 0, 3);
    let d_alpha = g.narrow_(d_rgba, 0, 3, 1);

    let inv_n = scalar_const(g, 1.0 / count as f32, dtype);

    // mean(weighted, axis=0) → d_weighted = broadcast(d_rgb) / N
    let d_rgb_bc = expand_to(g, d_rgb, &[count as i64, 3]);
    let inv_n_3 = expand_to(g, inv_n, &[count as i64, 3]);
    let d_weighted = g.mul(d_rgb_bc, inv_n_3);

    // mean(op_2d, axis=0) → d_op from alpha path
    let d_alpha_bc = expand_to(g, d_alpha, &[count as i64, 1]);
    let inv_n_1 = expand_to(g, inv_n, &[count as i64, 1]);
    let d_op_from_mean = g.mul(d_alpha_bc, inv_n_1);

    // mul: weighted = colors2 * op_3
    let d_colors2 = g.mul(d_weighted, op_3);
    let d_op_3 = g.mul(d_weighted, colors2);
    let d_op_from_mul = g.sum(d_op_3, vec![1], false);
    let d_op_2d = g.add(d_op_from_mean, d_op_from_mul);

    let d_colors = g.reshape_(d_colors2, vec![(count * 3) as i64]);
    let d_opacities = g.reshape_(d_op_2d, vec![count as i64]);

    let d_positions = zeros_like(g, positions);
    let d_scales = zeros_like(g, scales);
    let d_rotations = zeros_like(g, rotations);
    let d_sh = zeros_like(g, sh_coeffs);

    let packed_len = gaussian_splat_packed_grad_len(count, sh_coeff_count);
    debug_assert_eq!(packed_len, out_shape.num_elements().unwrap_or(packed_len));

    let packed = g.concat_(
        vec![
            d_positions,
            d_scales,
            d_rotations,
            d_opacities,
            d_colors,
            d_sh,
        ],
        0,
    );
    g.reshape_(packed, vec![packed_len as i64])
}
