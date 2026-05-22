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
//! Graph builders for [`rlx_ir::Op::GaussianSplatRender`].

pub use rlx_ir::ops::splat::{
    GaussianSplatBackwardParams, GaussianSplatInputs, GaussianSplatRenderParams,
    unpack_gaussian_splat_packed_grads,
};

use rlx_ir::{DType, Graph, NodeId, Shape};

use crate::core::{Camera, GaussianScene};
use crate::reference::RenderParams;

/// Scene tensor [`Op::Input`] nodes (see [`SCENE_INPUT_NAMES`]).
#[derive(Clone, Copy, Debug)]
pub struct SceneGraphInputs {
    pub positions: NodeId,
    pub scales: NodeId,
    pub rotations: NodeId,
    pub opacities: NodeId,
    pub colors: NodeId,
    pub sh_coeffs: NodeId,
}

pub const SCENE_INPUT_NAMES: [&str; 6] = [
    "positions",
    "scales",
    "rotations",
    "opacities",
    "colors",
    "sh_coeffs",
];

pub fn scene_graph_inputs(g: &mut Graph, scene: &GaussianScene) -> SceneGraphInputs {
    let count = scene.count();
    let sh_coeff_count = scene.sh_coeff_count;
    SceneGraphInputs {
        positions: g.input("positions", Shape::new(&[count * 3], DType::F32)),
        scales: g.input("scales", Shape::new(&[count * 3], DType::F32)),
        rotations: g.input("rotations", Shape::new(&[count * 4], DType::F32)),
        opacities: g.input("opacities", Shape::new(&[count], DType::F32)),
        colors: g.input("colors", Shape::new(&[count * 3], DType::F32)),
        sh_coeffs: g.input(
            "sh_coeffs",
            Shape::new(&[count * sh_coeff_count * 3], DType::F32),
        ),
    }
}

/// Build [`rlx_ir::Op::GaussianSplatRender`] from a scene and pinhole camera.
pub fn gaussian_splat_render_scene(
    g: &mut Graph,
    positions: NodeId,
    scales: NodeId,
    rotations: NodeId,
    opacities: NodeId,
    colors: NodeId,
    sh_coeffs: NodeId,
    camera: &Camera,
    background: [f32; 3],
    render: &RenderParams,
) -> NodeId {
    let params = GaussianSplatRenderParams {
        width: render.width,
        height: render.height,
        tile_size: render.tile_size,
        radius_scale: render.radius_scale,
        alpha_cutoff: render.alpha_cutoff,
        max_splat_steps: render.max_splat_steps,
        transmittance_threshold: render.transmittance_threshold,
        max_list_entries: render.max_list_entries,
    };
    let meta = g.gaussian_splat_render_meta(
        camera.position,
        camera.target,
        camera.up,
        camera.fov_y_degrees,
        camera.near,
        camera.far,
        background,
        params,
    );
    g.gaussian_splat_render(
        GaussianSplatInputs {
            positions,
            scales,
            rotations,
            opacities,
            colors,
            sh_coeffs,
            meta,
        },
        params,
    )
}

/// Build [`Op::GaussianSplatRenderBackward`] and return unpacked positions gradient.
pub fn gaussian_splat_backward_scene(
    g: &mut Graph,
    positions: NodeId,
    scales: NodeId,
    rotations: NodeId,
    opacities: NodeId,
    colors: NodeId,
    sh_coeffs: NodeId,
    camera: &Camera,
    background: [f32; 3],
    render: &RenderParams,
    d_loss_rgba: NodeId,
    backward: GaussianSplatBackwardParams,
) -> NodeId {
    let render_params = GaussianSplatRenderParams {
        width: render.width,
        height: render.height,
        tile_size: render.tile_size,
        radius_scale: render.radius_scale,
        alpha_cutoff: render.alpha_cutoff,
        max_splat_steps: render.max_splat_steps,
        transmittance_threshold: render.transmittance_threshold,
        max_list_entries: render.max_list_entries,
    };
    let meta = g.gaussian_splat_render_meta(
        camera.position,
        camera.target,
        camera.up,
        camera.fov_y_degrees,
        camera.near,
        camera.far,
        background,
        render_params,
    );
    let count = g.shape(positions).num_elements().unwrap_or(0) / 3;
    let sh_len = g.shape(sh_coeffs).num_elements().unwrap_or(0);
    let sh_coeff_count = if count == 0 {
        1
    } else {
        (sh_len / (count * 3)).max(1)
    };
    let packed = g.gaussian_splat_render_backward(
        GaussianSplatInputs {
            positions,
            scales,
            rotations,
            opacities,
            colors,
            sh_coeffs,
            meta,
        },
        d_loss_rgba,
        backward,
    );
    let grads = unpack_gaussian_splat_packed_grads(g, packed, count, sh_coeff_count);
    grads.positions
}
