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
//! Strict RLX IR splat pipeline — decomposed graph builders and common-IR baseline.

use rlx_ir::infer::GraphExt;
use rlx_ir::logical_kernel::splat_common;
use rlx_ir::ops::splat::{GaussianSplatInputs, GaussianSplatRenderParams};
use rlx_ir::{Graph, NodeId, Shape};

/// Build a framebuffer using the **common IR baseline** (primitive ops only).
pub fn gaussian_splat_render_common_ir(
    g: &mut Graph,
    inputs: GaussianSplatInputs,
    params: GaussianSplatRenderParams,
) -> NodeId {
    let out_elems = (params.width as usize) * (params.height as usize) * 4;
    let dtype = g.shape(inputs.positions).dtype();
    let out_shape = Shape::new(&[out_elems], dtype);
    splat_common::lower_gaussian_splat_render(
        g,
        inputs.positions,
        inputs.scales,
        inputs.rotations,
        inputs.opacities,
        inputs.colors,
        inputs.sh_coeffs,
        inputs.meta,
        params.width,
        params.height,
        out_shape,
    )
}

/// Strict IR: [`Op::GaussianSplatPrepare`] → [`Op::GaussianSplatRasterize`].
pub fn gaussian_splat_render_decomposed(
    g: &mut Graph,
    inputs: GaussianSplatInputs,
    params: GaussianSplatRenderParams,
) -> NodeId {
    g.gaussian_splat_render_decomposed(inputs, params)
}

/// Training backward for graphs built with [`gaussian_splat_render_decomposed`].
///
/// Autodiff ([`rlx_autodiff::prepare_graph_for_ad`]) fuses prepare+rasterize into
/// [`Op::GaussianSplatRender`] and applies the monolithic [`Op::GaussianSplatRenderBackward`] VJP.
/// This helper builds the explicit backward op directly (same as [`crate::graph::gaussian_splat_backward_scene`]).
pub fn gaussian_splat_backward_decomposed(
    g: &mut Graph,
    inputs: GaussianSplatInputs,
    d_loss_rgba: NodeId,
    backward: crate::GaussianSplatBackwardParams,
) -> GaussianSplatInputs {
    let positions = inputs.positions;
    let sh_coeffs = inputs.sh_coeffs;
    let count = g.shape(positions).num_elements().unwrap_or(0) / 3;
    let sh_len = g.shape(sh_coeffs).num_elements().unwrap_or(0);
    let packed = g.gaussian_splat_render_backward(inputs, d_loss_rgba, backward);
    let sh_coeff_count = if count == 0 {
        1
    } else {
        (sh_len / (count * 3)).max(1)
    };
    crate::unpack_gaussian_splat_packed_grads(g, packed, count, sh_coeff_count)
}

/// Full tile-based reference render (logical kernel — CPU executor via [`crate::register`]).
pub fn gaussian_splat_render_reference(
    g: &mut Graph,
    inputs: GaussianSplatInputs,
    params: GaussianSplatRenderParams,
) -> NodeId {
    g.gaussian_splat_render(inputs, params)
}
