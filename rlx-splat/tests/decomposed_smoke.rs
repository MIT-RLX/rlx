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
//! Strict IR prepare → rasterize graph builds and legalizes on CPU.

mod common;

use rlx_ir::infer::GraphExt;
use rlx_ir::ops::splat::GaussianSplatInputs;
use rlx_ir::Op;
use rlx_splat::{gaussian_splat_render_decomposed, register};

#[test]
fn decomposed_graph_contains_prepare_and_rasterize() {
    register();
    let fixture = common::ParityFixture::tiny();
    let scene = &fixture.scene;
    let count = scene.count();

    let mut g = rlx_ir::Graph::new("decomposed_splat");
    let positions = g.input("positions", rlx_ir::Shape::new(&[count * 3], rlx_ir::DType::F32));
    let scales = g.input("scales", rlx_ir::Shape::new(&[count * 3], rlx_ir::DType::F32));
    let rotations = g.input("rotations", rlx_ir::Shape::new(&[count * 4], rlx_ir::DType::F32));
    let opacities = g.input("opacities", rlx_ir::Shape::new(&[count], rlx_ir::DType::F32));
    let colors = g.input("colors", rlx_ir::Shape::new(&[count * 3], rlx_ir::DType::F32));
    let sh_coeffs = g.input(
        "sh_coeffs",
        rlx_ir::Shape::new(&[count * scene.sh_coeff_count * 3], rlx_ir::DType::F32),
    );
    let meta = g.gaussian_splat_render_meta(
        fixture.camera.position,
        fixture.camera.target,
        fixture.camera.up,
        fixture.camera.fov_y_degrees,
        fixture.camera.near,
        fixture.camera.far,
        fixture.background,
        fixture.render_params(),
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
        fixture.render_params(),
    );
    g.set_outputs(vec![rgba]);

    assert!(
        g.nodes()
            .iter()
            .any(|n| matches!(n.op, Op::GaussianSplatPrepare { .. }))
    );
    assert!(
        g.nodes()
            .iter()
            .any(|n| matches!(n.op, Op::GaussianSplatRasterize { .. }))
    );

    rlx_runtime::legalize_graph_for_device(g, rlx_runtime::Device::Cpu).expect("CPU legalize");
}
