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
//! Scene I/O and graph input builders.

pub use crate::io_format::*;

use crate::core::GaussianScene;
use crate::graph::scene_graph_inputs;
use crate::reference::RenderParams;
use rlx_ir::infer::GraphExt;
use rlx_ir::ops::splat::GaussianSplatInputs;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

fn f32_constant(g: &mut Graph, data: &[f32]) -> NodeId {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    g.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(&[data.len()], DType::F32),
    )
}

/// Build [`GaussianSplatInputs`] from host scene buffers and camera meta.
pub fn scene_host_inputs(g: &mut Graph, scene: &GaussianScene, meta: NodeId) -> GaussianSplatInputs {
    GaussianSplatInputs {
        positions: f32_constant(g, &scene.positions),
        scales: f32_constant(g, &scene.scales),
        rotations: f32_constant(g, &scene.rotations),
        opacities: f32_constant(g, &scene.opacities),
        colors: f32_constant(g, &scene.colors),
        sh_coeffs: f32_constant(g, &scene.sh_coeffs),
        meta,
    }
}

/// Load PLY + build render graph inputs and meta constant.
pub fn load_ply_render_graph(
    g: &mut Graph,
    path: &std::path::Path,
    camera_position: [f32; 3],
    camera_target: [f32; 3],
    camera_up: [f32; 3],
    fov_y_degrees: f32,
    near: f32,
    far: f32,
    background: [f32; 3],
    params: RenderParams,
) -> anyhow::Result<(GaussianSplatInputs, NodeId)> {
    let scene = crate::io_format::load_gaussian_ply(path)?;
    let render_params = rlx_ir::ops::splat::GaussianSplatRenderParams {
        width: params.width,
        height: params.height,
        tile_size: params.tile_size,
        radius_scale: params.radius_scale,
        alpha_cutoff: params.alpha_cutoff,
        max_splat_steps: params.max_splat_steps,
        transmittance_threshold: params.transmittance_threshold,
        max_list_entries: params.max_list_entries,
    };
    let meta = g.gaussian_splat_render_meta(
        camera_position,
        camera_target,
        camera_up,
        fov_y_degrees,
        near,
        far,
        background,
        render_params,
    );
    Ok((scene_host_inputs(g, &scene, meta), meta))
}
