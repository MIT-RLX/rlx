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
use rlx_fdm::{
    Goal, Network, fdm,
    goals::{
        mesh_laplacian_energy, mesh_mean_face_rectangular, mesh_mean_planarity, mesh_total_area,
    },
    goals_loss_with_structure,
    mesh::edges_from_faces,
};

#[test]
fn mesh_area_and_planarity_goals_finite() {
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
        faces_load: None,
        faces_load_local: false,
    };
    let mesh = net.mesh_structure().expect("mesh");
    let s = rlx_fdm::Structure::from_network(&net);
    let area = mesh_total_area(&mesh, &net.xyz);
    let plan = mesh_mean_planarity(&mesh, &net.xyz);
    assert!(area > 0.9 && area < 1.1, "quad area ~1, got {area}");
    let eq = fdm(&net).expect("fdm");
    assert!(
        plan >= 0.0 && plan < 0.5,
        "planarity should be modest: {plan}"
    );

    let rect = mesh_mean_face_rectangular(&mesh, &net.xyz);
    let lap = mesh_laplacian_energy(&mesh, &net.xyz);
    assert!(rect >= 0.0 && lap >= 0.0);

    let goals = vec![
        Goal::mesh_area(area, 0.1),
        Goal::mesh_planarity(0.0, 0.1),
        Goal::mesh_face_rectangular(0.0, 0.1),
        Goal::mesh_laplacian(lap, 0.01),
    ];
    let loss = goals_loss_with_structure(&goals, &eq, &s, &net.is_support, Some(&mesh));
    assert!(loss.is_finite() && loss >= 0.0);
}
