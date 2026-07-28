// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
use rlx_fdm::{Goal, Network, fdm, goals_loss_with_structure};

#[test]
fn extended_goals_compile_and_loss_finite() {
    let net = Network::arch_chain(4.0, 6, -1.0, -0.1);
    let eq = fdm(&net).expect("fdm");
    let s = rlx_fdm::Structure::from_network(&net);
    let goals = vec![
        Goal::mean_edge_length(0.5, 0.1),
        Goal::edge_force(2, eq.forces[2], 0.5),
        Goal::node_z(3, eq.xyz[3 * 3 + 2], 1.0),
        Goal::residual(0.01),
    ];
    let loss = goals_loss_with_structure(&goals, &eq, &s, &net.is_support, None);
    assert!(loss.is_finite() && loss >= 0.0);
}
