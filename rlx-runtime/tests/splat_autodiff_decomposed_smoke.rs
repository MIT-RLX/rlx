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
//! Autodiff through decomposed prepare → rasterize (fused to monolithic render for VJP).

#[test]
fn gaussian_splat_decomposed_autodiff_smoke() {
    rlx_splat::register();

    use rlx_autodiff::grad;
    use rlx_ir::ops::splat::{GaussianSplatInputs, GaussianSplatRenderParams};
    use rlx_ir::{DType, Graph, Op, Shape};
    use rlx_splat::core::{Camera, make_parity_scene};
    use rlx_splat::pipeline::gaussian_splat_render_decomposed;

    let scene = make_parity_scene();
    let camera = Camera::look_at(
        [0.0, 0.0, 4.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        60.0,
        0.1,
        20.0,
    );
    let params = GaussianSplatRenderParams {
        width: 64,
        height: 64,
        tile_size: 16,
        radius_scale: 1.6,
        alpha_cutoff: 1.0 / 255.0,
        max_splat_steps: 32,
        transmittance_threshold: 0.01,
        max_list_entries: 18 * 32,
    };

    let count = scene.count();
    let sh_coeff_count = scene.sh_coeff_count;

    let mut g = Graph::new("splat_ad_decomposed");
    let positions = g.input("positions", Shape::new(&[count * 3], DType::F32));
    let scales = g.input("scales", Shape::new(&[count * 3], DType::F32));
    let rotations = g.input("rotations", Shape::new(&[count * 4], DType::F32));
    let opacities = g.input("opacities", Shape::new(&[count], DType::F32));
    let colors = g.input("colors", Shape::new(&[count * 3], DType::F32));
    let sh_coeffs = g.input(
        "sh_coeffs",
        Shape::new(&[count * sh_coeff_count * 3], DType::F32),
    );
    let meta = g.gaussian_splat_render_meta(
        camera.position,
        camera.target,
        camera.up,
        camera.fov_y_degrees,
        camera.near,
        camera.far,
        [0.1, 0.15, 0.2],
        params,
    );

    let rgba = gaussian_splat_render_decomposed(
        &mut g,
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

    assert!(
        g.nodes()
            .iter()
            .any(|n| matches!(n.op, Op::GaussianSplatPrepare { .. })),
        "forward graph should contain prepare"
    );

    let grad_graph = grad(&g, &[positions]);
    assert!(
        grad_graph
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::GaussianSplatRenderBackward { .. })),
        "fused AD should lower to render backward"
    );
}
