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
//! Constrained form-finding via [`rlx_fdm::constrained_fdm`].

use rlx_fdm::{
    Constraint, Goal, Network, OptimizeConfig, OptimizerKind, constrained_fdm, edge_length_error,
    fdm, network_loadpath,
};

#[test]
fn constrained_fdm_reduces_loadpath_goal() {
    let num_segments = 10;
    let net = Network::arch_chain(5.0, num_segments, -1.0, -0.2);
    let eq0 = fdm(&net).expect("fdm");
    let lp0 = network_loadpath(&eq0);
    let target = lp0 * 0.92;
    let initial_goal_loss = (lp0 - target).powi(2);

    let goals = vec![Goal::network_loadpath(target, 1.0)];
    let constraints = vec![Constraint::all_edge_q(-50.0, -0.5, 0.0)];

    let config = OptimizeConfig {
        learning_rate: 0.05,
        max_iter: 100,
        ..OptimizeConfig::default()
    };

    let res = constrained_fdm(&net, &goals, &constraints, &config).expect("optimize");
    let lp1 = network_loadpath(&res.equilibrium);

    assert!(
        res.goal_loss < initial_goal_loss * 0.95,
        "goal loss {:.6} should improve from initial {:.6}",
        res.goal_loss,
        initial_goal_loss
    );
    assert!(
        *res.loss_history.last().unwrap_or(&f64::INFINITY)
            <= res.loss_history.first().copied().unwrap_or(0.0) + 1e-6,
        "total loss should not increase over the run"
    );
    assert!(
        lp1 <= lp0 + 0.05,
        "loadpath should not blow up: {lp0} -> {lp1}"
    );
    for &qi in &res.network.q {
        assert!(qi >= -50.0 && qi <= -0.5, "q bounds violated: {qi}");
    }
}

#[test]
fn edge_length_goal_via_constrained_fdm() {
    let num_segments = 10;
    let edge = num_segments / 2;
    let net = Network::arch_chain(5.0, num_segments, -1.0, -0.2);
    let eq0 = fdm(&net).expect("fdm");
    let target = eq0.lengths[edge] * 1.05;
    let err0 = edge_length_error(&eq0, edge, target);

    let goals = vec![Goal::edge_length(edge, target, 1.0)];
    let constraints = vec![Constraint::all_edge_q(-50.0, -0.5, 0.0)];

    let config = OptimizeConfig {
        learning_rate: 0.05,
        max_iter: 80,
        ..OptimizeConfig::default()
    };

    let res = constrained_fdm(&net, &goals, &constraints, &config).expect("optimize");
    let err = edge_length_error(&res.equilibrium, edge, target);
    assert!(
        err < err0 * 0.5 + 1e-4,
        "edge length error should improve: {err0} -> {err}"
    );
    assert!(err < 0.02, "edge length goal not met: err={err}");
}

#[test]
fn arch_form_finding_preset_converges() {
    let net = Network::arch_chain(5.0, 10, -1.0, -0.2);
    let eq0 = fdm(&net).expect("fdm");
    let lp0 = network_loadpath(&eq0);
    let goals = vec![Goal::network_loadpath(lp0 * 0.93, 1.0)];
    let constraints = vec![Constraint::all_edge_q(-50.0, -0.5, 0.0)];

    let mut config = OptimizeConfig::arch_form_finding();
    config.max_iter = 60;
    config.verbose = false;

    let res = constrained_fdm(&net, &goals, &constraints, &config).expect("optimize");
    assert_eq!(config.optimizer, OptimizerKind::Lbfgs);
    assert!(
        res.goal_loss < (lp0 - lp0 * 0.93).powi(2),
        "L-BFGS preset should reduce loadpath goal"
    );
}
