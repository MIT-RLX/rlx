// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Residual goal gradient through equilibrium adjoint.

use rlx_fdm::{
    EquilibriumModel, Goal, Network, Structure, goals_grad_xyz_free, grad_residual_wrt_xyz_free,
};

#[test]
fn residual_goal_grad_nonzero_on_arch() {
    let net = Network::arch_chain(4.0, 6, -1.0, -0.12);
    let structure = Structure::from_network(&net);
    let eq = rlx_fdm::fdm(&net).expect("eq");
    let goals = vec![Goal::residual(1.0)];
    let g_goals = goals_grad_xyz_free(&goals, &eq, &structure, &net.edges, &net.is_support, None);
    let g_res = grad_residual_wrt_xyz_free(&eq, &structure);
    let pred = goals[0].prediction_with_structure(&eq, &structure, &net.is_support, None);
    let scale = 2.0 * goals[0].weight() * (pred - goals[0].target());
    let nf = structure.num_free();
    for i in 0..nf * 3 {
        assert!(
            (g_goals[i] - g_res[i] * scale).abs() < 1e-10,
            "Goal::residual grad mismatch at {i}"
        );
    }
    let mut eq2 = eq.clone();
    for r in eq2.residuals.iter_mut() {
        *r += 0.01;
    }
    let g2 = grad_residual_wrt_xyz_free(&eq2, &structure);
    assert!(g2.iter().any(|x| x.abs() > 1e-12));
    let _ = EquilibriumModel::pack_xyz_free(&eq.xyz, &structure);
}
