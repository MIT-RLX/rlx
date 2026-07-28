// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
use rlx_fdm::{
    Constraint, Goal, Network, OptimizeConfig, OptimizerKind, constrained_fdm, fdm,
    network_loadpath,
};

#[test]
fn lbfgs_reduces_loadpath_like_gd() {
    let net = Network::arch_chain(5.0, 10, -1.0, -0.2);
    let eq0 = fdm(&net).expect("fdm");
    let lp0 = network_loadpath(&eq0);
    let target = lp0 * 0.9;
    let goals = vec![Goal::network_loadpath(target, 1.0)];
    let constraints = vec![Constraint::all_edge_q(-50.0, -0.5, 0.0)];

    let config = OptimizeConfig {
        optimizer: OptimizerKind::Lbfgs,
        max_iter: 60,
        q_l2_weight: 1e-4,
        ..OptimizeConfig::default()
    };
    let res = constrained_fdm(&net, &goals, &constraints, &config).expect("lbfgs");
    assert!(
        res.goal_loss < (lp0 - target).powi(2) * 0.95,
        "L-BFGS should improve goal loss, got {}",
        res.goal_loss
    );
}
