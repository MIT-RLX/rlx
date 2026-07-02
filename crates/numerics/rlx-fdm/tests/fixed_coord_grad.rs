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
//! Gradients w.r.t. support (fixed) coordinates.

use rlx_fdm::{
    AdjointSolveConfig, EquilibriumModel, IterativeConfig, Network, Structure, grad_loss_wrt_q_fd,
    grad_loss_wrt_xyz_fixed_linear,
};

#[test]
fn grad_wrt_support_z_matches_fd() {
    let net = Network::arch_chain(5.0, 8, -1.0, -0.2);
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

    let analytic = grad_loss_wrt_xyz_fixed_linear(
        &net.q,
        &s,
        &xyz_free,
        &loss_grad,
        &AdjointSolveConfig::default(),
    )
    .expect("dxf");

    let eps = 1e-7;
    let mut fd = vec![0.0; na * 3];
    for j in 0..na {
        for d in 0..3 {
            let mut xp = xf.clone();
            let mut xm = xf.clone();
            xp[j * 3 + d] += eps;
            xm[j * 3 + d] -= eps;
            let pp = EquilibriumModel::nodes_free_positions(&net.q, &xp, &net.loads, &s).unwrap();
            let pm = EquilibriumModel::nodes_free_positions(&net.q, &xm, &net.loads, &s).unwrap();
            let mut g = 0.0;
            for a in 0..nf {
                g += loss_grad[a * 3 + 2] * (pp[a * 3 + 2] - pm[a * 3 + 2]) / (2.0 * eps);
            }
            fd[j * 3 + d] = g;
        }
    }

    let mut max_err = 0.0_f64;
    for (a, b) in analytic.dxf.iter().zip(fd.iter()) {
        max_err = max_err.max((a - b).abs());
    }
    assert!(
        max_err < 1e-4,
        "dxf analytic vs FD max_err={max_err} dxf={:?} fd={fd:?}",
        analytic.dxf
    );

    let _ = grad_loss_wrt_q_fd(
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
    );
}
