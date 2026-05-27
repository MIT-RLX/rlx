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
//! Decomposed splat: `GaussianSplatPrepare` → `GaussianSplatRasterize` via CPU Session.
//!
//! ```bash
//! cargo run -p rlx-runtime --example splat_decomposed_session --features cpu
//! ```

fn main() {
    rlx_splat::register();

    use rlx_ir::ops::splat::{GaussianSplatInputs, GaussianSplatRenderParams};
    use rlx_ir::{DType, Graph, Shape};
    use rlx_runtime::{Device, Session};
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
    let params = GaussianSplatRenderParams::default();
    let count = scene.count();

    let mut g = Graph::new("decomposed_example");
    let positions = g.input("positions", Shape::new(&[count * 3], DType::F32));
    let scales = g.input("scales", Shape::new(&[count * 3], DType::F32));
    let rotations = g.input("rotations", Shape::new(&[count * 4], DType::F32));
    let opacities = g.input("opacities", Shape::new(&[count], DType::F32));
    let colors = g.input("colors", Shape::new(&[count * 3], DType::F32));
    let sh_coeffs = g.input(
        "sh_coeffs",
        Shape::new(&[count * scene.sh_coeff_count * 3], DType::F32),
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
    let out = gaussian_splat_render_decomposed(
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
    g.set_outputs(vec![out]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let results = compiled.run(&[
        ("positions", &scene.positions),
        ("scales", &scene.scales),
        ("rotations", &scene.rotations),
        ("opacities", &scene.opacities),
        ("colors", &scene.colors),
        ("sh_coeffs", &scene.sh_coeffs),
    ]);

    println!(
        "decomposed render: {} RGBA floats, first pixel {:?}",
        results[0].len(),
        &results[0][..4.min(results[0].len())]
    );
}
