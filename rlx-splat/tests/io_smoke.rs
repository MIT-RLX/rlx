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
//! PLY round-trip + graph input smoke (feature `io`).

#![cfg(feature = "io")]

#[test]
fn ply_roundtrip_and_graph_inputs() -> anyhow::Result<()> {
    use rlx_splat::core::make_parity_scene;
    use rlx_splat::{SavePlyOptions, save_gaussian_ply, scene_graph_inputs};

    let scene = make_parity_scene();
    let dir = std::env::temp_dir().join("rlx_splat_io_smoke");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("parity.ply");
    save_gaussian_ply(&path, &scene, SavePlyOptions::default())?;
    let loaded = rlx_splat::load_gaussian_ply(&path)?;
    assert_eq!(loaded.count(), scene.count());

    let mut g = rlx_ir::Graph::new("io");
    let inputs = scene_graph_inputs(&mut g, &loaded);
    let args = inputs.run_args(&loaded);
    assert_eq!(args.len(), 6);
    let _ = args;
    let _ = std::fs::remove_file(&path);
    Ok(())
}
