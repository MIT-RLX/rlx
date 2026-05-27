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
//! CPU backward + autodiff VJP smoke for `Op::GaussianSplatRender`.

#[test]
fn gaussian_splat_backward_direct_op() {
    rlx_splat::register();

    use rlx_ir::ops::splat::{
        GaussianSplatBackwardParams, GaussianSplatInputs, GaussianSplatRenderParams,
        unpack_gaussian_splat_packed_grads,
    };
    use rlx_ir::{DType, Graph, Shape};
    use rlx_runtime::{Device, Session};
    use rlx_splat::core::{Camera, make_parity_scene};
    use rlx_splat::reference::RenderParams;

    let scene = make_parity_scene();
    let camera = Camera::look_at(
        [0.0, 0.0, 4.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        60.0,
        0.1,
        20.0,
    );
    let render = RenderParams {
        width: 64,
        height: 64,
        tile_size: 16,
        radius_scale: 1.6,
        alpha_cutoff: 1.0 / 255.0,
        max_splat_steps: 32,
        transmittance_threshold: 0.01,
        max_list_entries: 18 * 32,
    };
    let rparams = GaussianSplatRenderParams {
        width: render.width,
        height: render.height,
        tile_size: render.tile_size,
        radius_scale: render.radius_scale,
        alpha_cutoff: render.alpha_cutoff,
        max_splat_steps: render.max_splat_steps,
        transmittance_threshold: render.transmittance_threshold,
        max_list_entries: render.max_list_entries,
    };

    let count = scene.count();
    let sh_coeff_count = scene.sh_coeff_count;

    let mut g = Graph::new("splat_bwd");
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
        rparams,
    );
    let inputs = GaussianSplatInputs {
        positions,
        scales,
        rotations,
        opacities,
        colors,
        sh_coeffs,
        meta,
    };
    let d_loss = g.input("d_loss", Shape::new(&[64 * 64 * 4], DType::F32));
    let packed = g.gaussian_splat_render_backward(
        inputs,
        d_loss,
        GaussianSplatBackwardParams {
            render: rparams,
            ..Default::default()
        },
    );
    let grads = unpack_gaussian_splat_packed_grads(&mut g, packed, count, sh_coeff_count);
    g.set_outputs(vec![grads.positions]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let ones = vec![1.0f32; 64 * 64 * 4];
    let out = compiled.run(&[
        ("positions", &scene.positions),
        ("scales", &scene.scales),
        ("rotations", &scene.rotations),
        ("opacities", &scene.opacities),
        ("colors", &scene.colors),
        ("sh_coeffs", &scene.sh_coeffs),
        ("d_loss", &ones),
    ]);
    assert!(out[0].iter().any(|v| *v != 0.0));
    assert!(out[0].iter().all(|v| v.is_finite()));
}
