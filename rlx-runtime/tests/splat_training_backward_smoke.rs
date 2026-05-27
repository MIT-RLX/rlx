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
//! Monolithic `GaussianSplatRenderBackward` through CPU Session (autodiff training path).

mod splat_common;
use splat_common::ParityFixture;

#[test]
fn training_backward_session_cpu_smoke() {
    rlx_splat::register();

    use rlx_ir::ops::splat::GaussianSplatBackwardParams;
    use rlx_ir::{DType, Graph, Shape};
    use rlx_runtime::{Device, Session};
    use rlx_splat::graph::gaussian_splat_backward_scene;

    let fixture = ParityFixture::tiny();
    let scene = &fixture.scene;
    let count = scene.count();

    let mut g = Graph::new("training_backward");
    let positions = g.input("positions", Shape::new(&[count * 3], DType::F32));
    let scales = g.input("scales", Shape::new(&[count * 3], DType::F32));
    let rotations = g.input("rotations", Shape::new(&[count * 4], DType::F32));
    let opacities = g.input("opacities", Shape::new(&[count], DType::F32));
    let colors = g.input("colors", Shape::new(&[count * 3], DType::F32));
    let sh_coeffs = g.input(
        "sh_coeffs",
        Shape::new(&[count * scene.sh_coeff_count * 3], DType::F32),
    );
    let d_loss = g.input("d_loss", Shape::new(&[64 * 64 * 4], DType::F32));

    let pos_grad = gaussian_splat_backward_scene(
        &mut g,
        positions,
        scales,
        rotations,
        opacities,
        colors,
        sh_coeffs,
        &fixture.camera,
        fixture.background,
        &fixture.render,
        d_loss,
        GaussianSplatBackwardParams {
            render: fixture.render_params(),
            ..GaussianSplatBackwardParams::default()
        },
    );
    g.set_outputs(vec![pos_grad]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let d_loss_data = vec![0.0f32; 64 * 64 * 4];
    let outs = compiled.run(&[
        ("positions", &scene.positions),
        ("scales", &scene.scales),
        ("rotations", &scene.rotations),
        ("opacities", &scene.opacities),
        ("colors", &scene.colors),
        ("sh_coeffs", &scene.sh_coeffs),
        ("d_loss", &d_loss_data),
    ]);
    assert_eq!(outs.len(), 1);
    assert_eq!(outs[0].len(), count * 3);
    assert!(outs[0].iter().all(|v| v.is_finite()));
}
