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
//! Decomposed splat is fused to monolithic render before AD.

use rlx_autodiff::prepare_graph_for_ad;
use rlx_ir::infer::GraphExt;
use rlx_ir::ops::splat::{GaussianSplatInputs, GaussianSplatRenderParams};
use rlx_ir::{DType, Graph, Op, Shape};

#[test]
fn fuse_prepare_rasterize_to_render() {
    let mut g = Graph::new("fuse");
    let positions = g.input("positions", Shape::new(&[3], DType::F32));
    let scales = g.input("scales", Shape::new(&[3], DType::F32));
    let rotations = g.input("rotations", Shape::new(&[4], DType::F32));
    let opacities = g.input("opacities", Shape::new(&[1], DType::F32));
    let colors = g.input("colors", Shape::new(&[3], DType::F32));
    let sh_coeffs = g.input("sh_coeffs", Shape::new(&[3], DType::F32));
    let params = GaussianSplatRenderParams {
        width: 8,
        height: 8,
        tile_size: 4,
        radius_scale: 1.0,
        alpha_cutoff: 0.01,
        max_splat_steps: 8,
        transmittance_threshold: 0.05,
        max_list_entries: 64,
    };
    let meta = g.gaussian_splat_render_meta(
        [0.0; 3],
        [0.0; 3],
        [0.0, 1.0, 0.0],
        60.0,
        0.1,
        10.0,
        [0.0; 3],
        params,
    );
    let rgba = g.gaussian_splat_render_decomposed(
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
    );
    g.set_outputs(vec![rgba]);

    let fused = prepare_graph_for_ad(g);
    assert!(
        fused
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::GaussianSplatRender { .. })),
        "expected fused GaussianSplatRender"
    );
    assert!(
        !fused
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::GaussianSplatRasterize { .. })),
        "rasterize should be fused away"
    );
}
