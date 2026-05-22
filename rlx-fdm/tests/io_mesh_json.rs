#![cfg(feature = "io")]
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

use std::path::PathBuf;

use rlx_fdm::{from_json_str, mesh_from_json_str, merge_mesh, to_json_str, Network};

#[test]
fn quad_mesh_json_roundtrip_and_solve() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/quad_mesh.json");
    let s = std::fs::read_to_string(&path).expect("read");
    let net = from_json_str(&s).expect("parse");
    assert!(net.faces.as_ref().is_some_and(|f| f.len() == 2));
    let out = to_json_str(&net).expect("serialize");
    let net2 = from_json_str(&out).expect("roundtrip");
    assert_eq!(net2.faces, net.faces);
}

#[test]
fn mesh_sidecar_merge() {
    let mesh_json = r#"{
        "faces": [[0,1,2],[0,2,3]],
        "faces_load": [[0,0,-2],[0,0,-2]],
        "faces_load_local": false
    }"#;
    let mesh = mesh_from_json_str(mesh_json).expect("mesh");
    let mut net = Network::from_polyline(
        &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        -1.0,
    );
    net.anchor_nodes(&[0, 2]);
    merge_mesh(&mut net, &mesh);
    assert_eq!(net.faces.as_ref().map(|f| f.len()), Some(2));
}
