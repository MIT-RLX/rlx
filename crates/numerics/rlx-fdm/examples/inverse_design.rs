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
//! Inverse form-finding: adjust force densities `q` so a target edge length is met.
//!
//! ```bash
//! cargo run -p rlx-fdm --example inverse_design
//! ```

use rlx_fdm::{
    EquilibriumModel, IterativeConfig, Network, Structure, apply_equilibrium, edge_length_error,
    fdm, grad_edge_length_error_wrt_xyz_free, grad_loss_wrt_q,
};

fn main() {
    let num_segments = 10;
    let target_edge = num_segments / 2;
    let target_length = 1.05;
    let mut net = Network::arch_chain(5.0, num_segments, -1.0, -0.2);

    let s = Structure::from_network(&net);
    let na = s.num_fixed();
    let mut xf = vec![0.0; na * 3];
    for (j, &node) in s.indices_fixed.iter().enumerate() {
        for c in 0..3 {
            xf[j * 3 + c] = net.xyz[node * 3 + c];
        }
    }

    let lr = 0.05;
    let mut loss_hist = Vec::new();

    println!("inverse design: edge {target_edge} → length {target_length:.3}");
    println!("iter   loss      q_mid     z_mid");

    for iter in 0..80 {
        let eq = fdm(&net).expect("equilibrium");
        let loss = edge_length_error(&eq, target_edge, target_length);
        loss_hist.push(loss);

        let loss_grad =
            grad_edge_length_error_wrt_xyz_free(&eq, &s, &net.edges, target_edge, target_length);
        let xyz_free =
            EquilibriumModel::nodes_free_positions(&net.q, &xf, &net.loads, &s).expect("solve");
        let gq = grad_loss_wrt_q(
            &net.q,
            &xf,
            &net.load_state(),
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

        if iter % 10 == 0 || iter == 79 {
            let mid_node = target_edge + 1;
            println!(
                "{iter:4}   {loss:8.5}   {:8.4}   {:8.5}",
                net.q[target_edge],
                eq.xyz[3 * mid_node + 2]
            );
        }

        for (i, dq) in gq.dq.iter().enumerate() {
            net.q[i] -= lr * dq;
            if net.q[i].abs() < 1e-3 {
                net.q[i] = net.q[i].signum() * 1e-3;
            }
        }

        if loss < 1e-6 {
            println!("converged at iter {iter}");
            break;
        }
    }

    let eq = fdm(&net).expect("final");
    apply_equilibrium(&mut net, &eq);
    let final_len = eq.lengths[target_edge];
    let err = (final_len - target_length).abs();
    println!(
        "\nfinal edge length {:.4} (target {target_length:.4}, |err| {err:.6})",
        final_len
    );
    assert!(
        err < 0.02,
        "inverse design did not reach target length: {final_len}"
    );
    assert!(
        loss_hist.last().copied().unwrap_or(f64::INFINITY)
            < loss_hist.first().copied().unwrap_or(0.0),
        "loss should decrease"
    );
}
