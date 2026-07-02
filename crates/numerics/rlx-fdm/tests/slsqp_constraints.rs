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
//! SLSQP-style penalty optimizer with nonlinear angle constraints.

use rlx_fdm::{Constraint, Goal, Network, OptimizeConfig, OptimizerKind, constrained_fdm, fdm};

#[test]
fn slsqp_respects_edge_angle_band() {
    let net = Network::arch_chain(3.0, 6, -1.0, -0.1);
    let eq = fdm(&net).expect("eq");
    let e = 2usize;
    let v = [
        eq.vectors[e * 3],
        eq.vectors[e * 3 + 1],
        eq.vectors[e * 3 + 2],
    ];
    let goals = vec![Goal::min_free_z(-0.2, 0.5)];
    let constraints = vec![Constraint::edge_angle(e, v, -0.05, 0.05, 10.0)];

    let config = OptimizeConfig {
        optimizer: OptimizerKind::Slsqp,
        learning_rate: 1.0,
        max_iter: 30,
        slsqp_penalty_weight: 80.0,
        ..OptimizeConfig::default()
    };

    let res = constrained_fdm(&net, &goals, &constraints, &config).expect("opt");
    let ineq = rlx_fdm::nonlinear_ineq_values(
        &constraints,
        &res.equilibrium,
        net.mesh_structure().as_ref(),
        &res.network.edges,
    );
    let max_v = ineq.iter().copied().fold(0.0f64, f64::max);
    assert!(
        max_v < 0.15,
        "nonlinear ineq should be near feasible: max={max_v:?}"
    );
}
