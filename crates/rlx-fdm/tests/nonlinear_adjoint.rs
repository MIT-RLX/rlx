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
    EquilibriumModel, FdmOptions, IterativeConfig, Network, goals, grad_loss_wrt_q,
    grad_loss_wrt_q_fd, grad_loss_wrt_q_fixedpoint,
};

#[test]
fn fixedpoint_adjoint_matches_fd_on_edge_loads() {
    let mut net = Network::arch_chain(4.0, 8, -1.0, 0.0);
    net.loads_on_free([0.0, 0.0, 0.0]);
    net.edges_load_uniform([0.0, 0.0, -0.04]);

    let s = rlx_fdm::Structure::from_network(&net);
    let load_state = net.load_state();
    let na = s.num_fixed();
    let mut xf = vec![0.0; na * 3];
    for (j, &node) in s.indices_fixed.iter().enumerate() {
        for c in 0..3 {
            xf[j * 3 + c] = net.xyz[node * 3 + c];
        }
    }
    let mut opts = FdmOptions::nonlinear(40, 1e-6, false);
    opts.iterative.tmax = 40;
    let eq = rlx_fdm::fdm_with_options(&net, &opts).expect("eq");
    let nf = s.num_free();
    let mut loss_grad = vec![0.0; nf * 3];
    loss_grad[2] = 1.0;
    let _xyz_free = EquilibriumModel::nodes_free_positions(&net.q, &xf, &net.loads, &s).expect("x");
    let _ = goals::mean_edge_length(&eq);

    let analytic = grad_loss_wrt_q_fixedpoint(
        &net.q,
        &xf,
        &load_state,
        &s,
        &net.edges,
        &net.xyz,
        &opts.iterative,
        None,
        &loss_grad,
    )
    .expect("fp");
    let fd = grad_loss_wrt_q_fd(
        &net.q,
        &xf,
        &load_state,
        &s,
        &net.edges,
        &net.xyz,
        &opts.iterative,
        None,
        &loss_grad,
        1e-7,
    )
    .expect("fd");
    let mut max_err: f64 = 0.0;
    for (a, f) in analytic.dq.iter().zip(fd.dq.iter()) {
        max_err = max_err.max((a - f).abs());
    }
    assert!(
        max_err < 0.08,
        "fixed-point adjoint vs FD max_err={max_err}"
    );
}

#[test]
fn grad_loss_wrt_q_uses_fixedpoint_for_edge_loads() {
    let mut net = Network::arch_chain(3.0, 6, -1.0, 0.0);
    net.edges_load_uniform([0.0, 0.0, -0.02]);
    let s = rlx_fdm::Structure::from_network(&net);
    let load_state = net.load_state();
    let xf = vec![0.0; s.num_fixed() * 3];
    let nf = s.num_free();
    let loss_grad = vec![0.0; nf * 3];
    let cfg = IterativeConfig {
        tmax: 30,
        ..Default::default()
    };
    let xyz = EquilibriumModel::nodes_free_positions(&net.q, &xf, &net.loads, &s).unwrap();
    let auto = grad_loss_wrt_q(
        &net.q,
        &xf,
        &load_state,
        &s,
        &net.edges,
        &net.xyz,
        &cfg,
        None,
        &xyz,
        &loss_grad,
        1e-7,
    )
    .unwrap();
    let fp = grad_loss_wrt_q_fixedpoint(
        &net.q,
        &xf,
        &load_state,
        &s,
        &net.edges,
        &net.xyz,
        &cfg,
        None,
        &loss_grad,
    )
    .unwrap();
    let mut max_err: f64 = 0.0;
    for (a, b) in auto.dq.iter().zip(fp.dq.iter()) {
        max_err = max_err.max((a - b).abs());
    }
    assert!(
        max_err < 1e-10,
        "router should select fixedpoint, err={max_err}"
    );
}

#[test]
fn fixedpoint_adjoint_matches_fd_on_global_face_loads() {
    use rlx_fdm::mesh::edges_from_faces;

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
        faces_load: Some(vec![[0.0, 0.0, -5.0], [0.0, 0.0, -5.0]]),
        faces_load_local: false,
    };
    let s = rlx_fdm::Structure::from_network(&net);
    let mesh = net.mesh_structure().expect("mesh");
    let load_state = net.load_state();
    let na = s.num_fixed();
    let mut xf = vec![0.0; na * 3];
    for (j, &node) in s.indices_fixed.iter().enumerate() {
        for c in 0..3 {
            xf[j * 3 + c] = net.xyz[node * 3 + c];
        }
    }
    let mut opts = FdmOptions::nonlinear(25, 1e-5, false);
    opts.iterative.tmax = 25;
    let nf = s.num_free();
    let mut loss_grad = vec![0.0; nf * 3];
    loss_grad[2] = 1.0;
    let _ = rlx_fdm::equilibrium_iterative(
        &net.q,
        &xf,
        &load_state,
        &s,
        &net.edges,
        &net.xyz,
        &opts.iterative,
        Some(&mesh),
    )
    .expect("eq");

    let analytic = grad_loss_wrt_q_fixedpoint(
        &net.q,
        &xf,
        &load_state,
        &s,
        &net.edges,
        &net.xyz,
        &opts.iterative,
        Some(&mesh),
        &loss_grad,
    )
    .expect("fp");
    let fd = grad_loss_wrt_q_fd(
        &net.q,
        &xf,
        &load_state,
        &s,
        &net.edges,
        &net.xyz,
        &opts.iterative,
        Some(&mesh),
        &loss_grad,
        1e-7,
    )
    .expect("fd");
    let mut max_err: f64 = 0.0;
    for (a, f) in analytic.dq.iter().zip(fd.dq.iter()) {
        max_err = max_err.max((a - f).abs());
    }
    assert!(
        max_err < 0.08,
        "global face-load fixed-point adjoint vs FD max_err={max_err}"
    );
}

#[test]
fn fixedpoint_adjoint_matches_fd_on_local_face_loads() {
    use rlx_fdm::mesh::edges_from_faces;

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
        faces_load: Some(vec![[0.0, 0.0, -5.0], [0.0, 0.0, -5.0]]),
        faces_load_local: true,
    };
    let s = rlx_fdm::Structure::from_network(&net);
    let mesh = net.mesh_structure().expect("mesh");
    let load_state = net.load_state();
    let na = s.num_fixed();
    let mut xf = vec![0.0; na * 3];
    for (j, &node) in s.indices_fixed.iter().enumerate() {
        for c in 0..3 {
            xf[j * 3 + c] = net.xyz[node * 3 + c];
        }
    }
    let mut opts = FdmOptions::nonlinear(30, 1e-5, false);
    opts.iterative.tmax = 30;
    let nf = s.num_free();
    let mut loss_grad = vec![0.0; nf * 3];
    loss_grad[2] = 1.0;

    let analytic = grad_loss_wrt_q_fixedpoint(
        &net.q,
        &xf,
        &load_state,
        &s,
        &net.edges,
        &net.xyz,
        &opts.iterative,
        Some(&mesh),
        &loss_grad,
    )
    .expect("fp");
    let fd = grad_loss_wrt_q_fd(
        &net.q,
        &xf,
        &load_state,
        &s,
        &net.edges,
        &net.xyz,
        &opts.iterative,
        Some(&mesh),
        &loss_grad,
        1e-7,
    )
    .expect("fd");
    let mut max_err: f64 = 0.0;
    for (a, f) in analytic.dq.iter().zip(fd.dq.iter()) {
        max_err = max_err.max((a - f).abs());
    }
    assert!(
        max_err < 0.12,
        "local face-load fixed-point adjoint vs FD max_err={max_err}"
    );
}

#[test]
fn transpose_face_jacobian_matches_fd() {
    use rlx_fdm::{
        mesh::edges_from_faces, transpose_face_loads_jacobian, transpose_face_loads_jacobian_fd,
    };

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
        xyz: xyz.clone(),
        is_support: vec![true, false, true, false],
        loads: vec![0.0; 12],
        edges,
        q: vec![-1.0; ne],
        edges_load: None,
        faces: Some(faces),
        faces_load: Some(vec![[0.0, 0.0, -5.0], [0.0, 0.0, -5.0]]),
        faces_load_local: true,
    };
    let s = rlx_fdm::Structure::from_network(&net);
    let mesh = net.mesh_structure().expect("mesh");
    let face_loads = net.faces_load.clone().expect("loads");
    let nf = s.num_free();
    let lambda: Vec<f64> = (0..nf * 3).map(|i| (i as f64 + 1.0) * 0.01).collect();

    let ana =
        transpose_face_loads_jacobian(&xyz, &face_loads, &mesh, &s, &net.edges, &lambda, true);
    let fd =
        transpose_face_loads_jacobian_fd(&xyz, &face_loads, &mesh, &s, &net.edges, &lambda, true);
    let mut max_err: f64 = 0.0;
    for (a, b) in ana.iter().zip(fd.iter()) {
        max_err = max_err.max((a - b).abs());
    }
    assert!(
        max_err < 0.12,
        "local face transpose vs FD max_err={max_err}"
    );

    let ana_g =
        transpose_face_loads_jacobian(&xyz, &face_loads, &mesh, &s, &net.edges, &lambda, false);
    let fd_g =
        transpose_face_loads_jacobian_fd(&xyz, &face_loads, &mesh, &s, &net.edges, &lambda, false);
    let mut max_g = 0.0_f64;
    for (a, b) in ana_g.iter().zip(fd_g.iter()) {
        max_g = max_g.max((a - b).abs());
    }
    assert!(max_g < 0.12, "global face transpose vs FD max_err={max_g}");
}
