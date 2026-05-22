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
//! Constrained arch form-finding (jax_fdm `constrained_fdm` + loadpath goal).
//!
//! ```bash
//! cargo run -p rlx-fdm --example constrained_arch
//! ```

use rlx_fdm::{
    constrained_fdm, Constraint, Goal, Network, OptimizeConfig,
};

fn main() {
    let num_segments = 10;
    let mid_edge = num_segments / 2;
    let net = Network::arch_chain(5.0, num_segments, -1.0, -0.2);

    let eq0 = rlx_fdm::fdm(&net).expect("initial equilibrium");
    let loadpath0 = rlx_fdm::network_loadpath(&eq0);
    let target_loadpath = loadpath0 * 0.92;

    let goals = vec![
        Goal::network_loadpath(target_loadpath, 1.0),
        Goal::edge_length(mid_edge, eq0.lengths[mid_edge], 0.1),
    ];
    let constraints = vec![
        Constraint::all_edge_q(-50.0, -0.5, 0.01),
        Constraint::edge_length_max(mid_edge, eq0.lengths[mid_edge] * 1.02, 2.0),
    ];

    let mut config = OptimizeConfig::arch_form_finding();
    config.learning_rate = 0.03;
    config.verbose = true;

    println!(
        "constrained arch: loadpath {loadpath0:.4} → target {target_loadpath:.4}, mid edge cap {:.4}",
        eq0.lengths[mid_edge] * 1.02
    );

    let result = constrained_fdm(&net, &goals, &constraints, &config).expect("optimize");

    for r in &result.goal_reports {
        println!(
            "  goal {}: pred={:.4} target={:.4} loss={:.6}",
            r.index.0, r.prediction, r.target, r.loss
        );
    }
    println!(
        "done: iters={} loss={:.6} (goals={:.6} pen={:.6})",
        result.iterations, result.loss, result.goal_loss, result.penalty
    );
    println!(
        "  loadpath {:.4}  mid length {:.4}",
        rlx_fdm::network_loadpath(&result.equilibrium),
        result.equilibrium.lengths[mid_edge]
    );

    assert!(
        result.loss <= loadpath0,
        "total loss should not exceed initial loadpath proxy"
    );
    assert!(
        result.equilibrium.lengths[mid_edge] <= eq0.lengths[mid_edge] * 1.03,
        "mid edge length constraint violated"
    );
}
