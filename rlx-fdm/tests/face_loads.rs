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
use rlx_fdm::{fdm_with_options, mesh::edges_from_faces, Network, FdmOptions};

#[test]
fn quad_mesh_face_load_sags() {
    // Non-degenerate quad in the xy plane; supported at corners 0 and 2.
    let faces = vec![vec![0, 1, 2], vec![0, 2, 3]];
    let edges = edges_from_faces(&faces);
    let ne = edges.len();
    let xyz = vec![
        0.0, 0.0, 0.0, //
        1.0, 0.0, 0.0, //
        1.0, 1.0, 0.0, //
        0.0, 1.0, 0.0, //
    ];
    let net = Network {
        xyz,
        is_support: vec![true, false, true, false],
        loads: vec![0.0; 12],
        edges,
        q: vec![-1.0; ne],
        edges_load: None,
        faces: Some(faces),
        faces_load: Some(vec![[0.0, 0.0, -10.0], [0.0, 0.0, -10.0]]),
        faces_load_local: false,
    };
    let opts = FdmOptions::nonlinear(30, 1e-5, false);
    let eq = fdm_with_options(&net, &opts).expect("fdm");
    assert!(
        eq.xyz[3 * 1 + 2] < -0.001,
        "interior node 1 should sag under face load, z={}",
        eq.xyz[3 * 1 + 2]
    );
}

#[test]
fn quad_mesh_local_face_load_sags() {
    let faces = vec![vec![0, 1, 2], vec![0, 2, 3]];
    let edges = edges_from_faces(&faces);
    let ne = edges.len();
    let net = Network {
        xyz: vec![
            0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            1.0, 1.0, 0.0, //
            0.0, 1.0, 0.0, //
        ],
        is_support: vec![true, false, true, false],
        loads: vec![0.0; 12],
        edges,
        q: vec![-1.0; ne],
        edges_load: None,
        faces: Some(faces),
        faces_load: Some(vec![[0.0, 0.0, -10.0], [0.0, 0.0, -10.0]]),
        faces_load_local: true,
    };
    let mut opts = FdmOptions::nonlinear(5, 1e-4, false);
    opts.iterative.tmax = 5;
    let eq = fdm_with_options(&net, &opts).expect("fdm");
    assert!(
        eq.xyz[3 * 1 + 2] < -0.001,
        "local LCS face load should sag interior (short iter), z={}",
        eq.xyz[3 * 1 + 2]
    );
}
