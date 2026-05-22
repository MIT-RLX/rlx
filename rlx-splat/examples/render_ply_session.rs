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
//! Load a 3D Gaussian PLY, build an RLX render graph, and run one forward pass on CPU.
//!
//! ```bash
//! cargo run -p rlx-splat --example render_ply_session --features io,test-support -- \
//!   /path/to/scene.ply
//! ```
//!
//! For parity/CLI workflows see `slang-splat-cli` in the sibling `slang-splat-rs` repo.

use std::env;
use std::path::PathBuf;

use rlx_runtime::{Device, Session};
use rlx_splat::{gaussian_splat_render_scene, load_gaussian_ply, scene_graph_inputs};
use rlx_splat::core::Camera;
use rlx_splat::reference::RenderParams;

fn main() -> anyhow::Result<()> {
    let ply = env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("usage: render_ply_session <scene.ply>"))?;

    let scene = load_gaussian_ply(&ply)?;
    let camera = Camera::look_at([0.0, 0.0, 4.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 60.0, 0.1, 20.0);
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

    let mut g = rlx_ir::Graph::new("ply_render");
    let inputs = scene_graph_inputs(&mut g, &scene);
    let rgba = gaussian_splat_render_scene(
        &mut g,
        inputs.positions,
        inputs.scales,
        inputs.rotations,
        inputs.opacities,
        inputs.colors,
        inputs.sh_coeffs,
        &camera,
        [0.0, 0.0, 0.0],
        &render,
    );
    g.set_outputs(vec![rgba]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let outs = compiled.run(&inputs.run_args(&scene));
    let n = (render.width * render.height * 4) as usize;
    assert_eq!(outs[0].len(), n);
    println!(
        "rendered {} splats → {}×{} RGBA ({} floats)",
        scene.count(),
        render.width,
        render.height,
        outs[0].len()
    );
    Ok(())
}
