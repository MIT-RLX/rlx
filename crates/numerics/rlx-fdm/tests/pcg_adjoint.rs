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
//! PCG adjoint parity with dense LU (`solve_adjoint_columns`).

use rlx_fdm::{
    AdjointSolveConfig, EquilibriumModel, Network, Structure, grad_loss_wrt_q_linear,
    grad_loss_wrt_q_linear_with_solver,
};

fn arch_grad(num_segments: usize) -> (Network, Structure, Vec<f64>, Vec<f64>, Vec<f64>) {
    let net = Network::arch_chain(8.0, num_segments, -1.0, -0.2);
    let s = Structure::from_network(&net);
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
    (net, s, xf, loss_grad, xyz_free)
}

#[test]
fn pcg_adjoint_matches_dense_on_large_arch() {
    let (net, s, xf, loss_grad, xyz_free) = arch_grad(40);
    assert!(s.num_free() >= 32, "need sparse threshold");

    let dense =
        grad_loss_wrt_q_linear(&net.q, &xf, &net.loads, &s, &xyz_free, &loss_grad).expect("dense");

    let pcg = grad_loss_wrt_q_linear_with_solver(
        &net.q,
        &xf,
        &s,
        &xyz_free,
        &loss_grad,
        &AdjointSolveConfig {
            use_sparse: true,
            pcg_max_iter: 8000,
            pcg_tol: 1e-10,
            sparse_min_free: 32,
        },
    )
    .expect("pcg");

    let mut max_err = 0.0_f64;
    for (a, b) in dense.dq.iter().zip(pcg.dq.iter()) {
        max_err = max_err.max((a - b).abs());
    }
    assert!(max_err < 1e-5, "PCG adjoint vs dense LU max_err={max_err}");
}
