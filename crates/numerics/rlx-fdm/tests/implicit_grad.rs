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
    EquilibriumModel, FdmOptions, IterativeConfig, Network, Structure, fdm_with_options, goals,
    grad_loss_wrt_q, grad_loss_wrt_q_fd, grad_loss_wrt_q_linear,
};

#[test]
fn grad_wrt_q_nonzero_for_sag_objective() {
    let net = Network::arch_chain(5.0, 6, -1.0, -0.2);
    let opts = FdmOptions::default();
    let eq = fdm_with_options(&net, &opts).expect("eq");
    let s = Structure::from_network(&net);
    let load_state = net.load_state();
    let nf = s.num_free();
    let mut loss_grad = vec![0.0; nf * 3];
    for a in 0..nf {
        loss_grad[a * 3 + 2] = 1.0;
    }
    let na = s.num_fixed();
    let mut xf = vec![0.0; na * 3];
    for (j, &node) in s.indices_fixed.iter().enumerate() {
        for c in 0..3 {
            xf[j * 3 + c] = net.xyz[node * 3 + c];
        }
    }
    let xyz_free = EquilibriumModel::nodes_free_positions(&net.q, &xf, &net.loads, &s).expect("x");
    let g = grad_loss_wrt_q(
        &net.q,
        &xf,
        &load_state,
        &s,
        &net.edges,
        &net.xyz,
        &IterativeConfig::linear(),
        None,
        &xyz_free,
        &loss_grad,
        1e-7,
    )
    .expect("grad");
    assert!(g.dq.iter().any(|&x| x.abs() > 1e-12), "dq={:?}", g.dq);
    let loss = goals::mean_edge_length(&eq);
    assert!(loss > 0.0);
}

#[test]
fn analytic_matches_finite_diff_on_linear_arch() {
    let net = Network::arch_chain(4.0, 8, -1.0, -0.15);
    let s = Structure::from_network(&net);
    let load_state = net.load_state();
    let na = s.num_fixed();
    let mut xf = vec![0.0; na * 3];
    for (j, &node) in s.indices_fixed.iter().enumerate() {
        for c in 0..3 {
            xf[j * 3 + c] = net.xyz[node * 3 + c];
        }
    }
    let _eq = fdm_with_options(&net, &FdmOptions::default()).expect("eq");
    let nf = s.num_free();
    // Objective: z coordinate of the first free node (linear functional of x_f).
    let mut loss_grad = vec![0.0; nf * 3];
    loss_grad[2] = 1.0;
    let xyz_free = EquilibriumModel::nodes_free_positions(&net.q, &xf, &net.loads, &s).expect("x");

    let analytic = grad_loss_wrt_q_linear(&net.q, &xf, &net.loads, &s, &xyz_free, &loss_grad)
        .expect("analytic");
    for eps in [1e-6, 1e-7, 1e-8] {
        let fd = grad_loss_wrt_q_fd(
            &net.q,
            &xf,
            &load_state,
            &s,
            &net.edges,
            &net.xyz,
            &IterativeConfig::linear(),
            None,
            &loss_grad,
            eps,
        )
        .expect("fd");
        let mut max_err = 0.0f64;
        for (a, f) in analytic.dq.iter().zip(fd.dq.iter()) {
            max_err = max_err.max((a - f).abs());
        }
        if max_err < 1e-4 {
            return;
        }
    }
    panic!("analytic vs FD did not match at any eps");
}

#[test]
fn dense_solve_single_rhs_matches_three_rhs_z_column() {
    let net = Network::arch_chain(4.0, 6, -1.0, -0.15);
    let s = Structure::from_network(&net);
    let na = s.num_fixed();
    let mut xf = vec![0.0; na * 3];
    for (j, &node) in s.indices_fixed.iter().enumerate() {
        for c in 0..3 {
            xf[j * 3 + c] = net.xyz[node * 3 + c];
        }
    }
    let k = EquilibriumModel::stiffness_matrix(&net.q, &s);
    let p = EquilibriumModel::load_matrix(&net.q, &xf, &net.loads, &s);
    let xyz = EquilibriumModel::nodes_free_positions(&net.q, &xf, &net.loads, &s).expect("xyz");
    let nf = s.num_free();
    let lu_xyz = rlx_fdm::solve::solve_columns_dense(&k, &p, nf, 3).expect("solve");
    for i in 0..xyz.len() {
        assert!(
            (xyz[i] - lu_xyz[i]).abs() < 1e-8,
            "packed xyz mismatch at {i}: direct={} lu={}",
            xyz[i],
            lu_xyz[i]
        );
    }
    let sol: Vec<f64> = (0..nf).map(|a| lu_xyz[a * 3 + 2]).collect();
    let xz: Vec<f64> = (0..nf).map(|a| xyz[a * 3 + 2]).collect();
    for a in 0..nf {
        assert!(
            (sol[a] - xz[a]).abs() < 1e-8,
            "z solve mismatch at {a}: sol={} xz={}",
            sol[a],
            xz[a]
        );
    }
}

#[test]
fn analytic_matches_fd_for_equilibrium_z_coordinate() {
    let net = Network::arch_chain(4.0, 6, -1.0, -0.15);
    let s = Structure::from_network(&net);
    let load_state = net.load_state();
    let na = s.num_fixed();
    let mut xf = vec![0.0; na * 3];
    for (j, &node) in s.indices_fixed.iter().enumerate() {
        for c in 0..3 {
            xf[j * 3 + c] = net.xyz[node * 3 + c];
        }
    }
    let nf = s.num_free();
    let mut loss_grad = vec![0.0; nf * 3];
    loss_grad[2] = 1.0;
    let xyz_free = EquilibriumModel::nodes_free_positions(&net.q, &xf, &net.loads, &s).expect("x");
    let analytic = grad_loss_wrt_q(
        &net.q,
        &xf,
        &load_state,
        &s,
        &net.edges,
        &net.xyz,
        &IterativeConfig::linear(),
        None,
        &xyz_free,
        &loss_grad,
        1e-7,
    )
    .expect("analytic");
    let fd = grad_loss_wrt_q_fd(
        &net.q,
        &xf,
        &load_state,
        &s,
        &net.edges,
        &net.xyz,
        &IterativeConfig::linear(),
        None,
        &loss_grad,
        1e-7,
    )
    .expect("fd");
    let mut max_err = 0.0f64;
    for (a, f) in analytic.dq.iter().zip(fd.dq.iter()) {
        max_err = max_err.max((a - f).abs());
    }
    assert!(
        max_err < 1e-4,
        "max_err={max_err}\nanalytic={:?}\nfd={:?}",
        analytic.dq,
        fd.dq
    );
}
