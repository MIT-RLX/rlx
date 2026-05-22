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
//! Example: compile Gaussian splat through the **common IR** path (no native splat thunk).
//!
//! ```bash
//! cargo run -p rlx-splat --example common_splat_session --features cpu,core
//! ```

use rlx_runtime::{CompileOptions, Device, Session};
use rlx_splat::logical_kernel::{splat_common_only_config, PRIMITIVE_SPLAT_SUPPORTED_OPS};
use rlx_splat::core::{make_parity_scene, Camera};
use rlx_splat::reference::RenderParams;

fn main() {
    let scene = make_parity_scene();
    let camera = Camera::look_at([0.0, 0.0, 4.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 60.0, 0.1, 20.0);
    let render = RenderParams {
        width: 64,
        height: 64,
        ..Default::default()
    };

    let mut g = rlx_ir::Graph::new("common_splat_example");
    let count = scene.count();
    let sh = scene.sh_coeff_count;
    let positions = g.input("positions", rlx_ir::Shape::new(&[count * 3], rlx_ir::DType::F32));
    let scales = g.input("scales", rlx_ir::Shape::new(&[count * 3], rlx_ir::DType::F32));
    let rotations = g.input("rotations", rlx_ir::Shape::new(&[count * 4], rlx_ir::DType::F32));
    let opacities = g.input("opacities", rlx_ir::Shape::new(&[count], rlx_ir::DType::F32));
    let colors = g.input("colors", rlx_ir::Shape::new(&[count * 3], rlx_ir::DType::F32));
    let sh_coeffs = g.input(
        "sh_coeffs",
        rlx_ir::Shape::new(&[count * sh * 3], rlx_ir::DType::F32),
    );
    let rgba = rlx_splat::gaussian_splat_render_scene(
        &mut g,
        positions,
        scales,
        rotations,
        opacities,
        colors,
        sh_coeffs,
        &camera,
        [0.1, 0.15, 0.2],
        &render,
    );
    g.set_outputs(vec![rgba]);

    let opts = CompileOptions::new()
        .supported_ops(PRIMITIVE_SPLAT_SUPPORTED_OPS)
        .kernel_dispatch_config(splat_common_only_config());
    let mut compiled = Session::new(Device::Cpu).compile_with(g, &opts);
    let out = compiled.run(&[
        ("positions", &scene.positions),
        ("scales", &scene.scales),
        ("rotations", &scene.rotations),
        ("opacities", &scene.opacities),
        ("colors", &scene.colors),
        ("sh_coeffs", &scene.sh_coeffs),
    ]);
    let n = out[0].len();
    let finite = out[0].iter().filter(|v| v.is_finite()).count();
    println!("common IR splat: {n} floats, {finite} finite");
}
