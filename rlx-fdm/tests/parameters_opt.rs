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
//! Multi-parameter design vector (support Z + q).

use rlx_fdm::{DesignParam, Goal, Network, OptimizeConfig, OptimizerKind, constrained_fdm, fdm};

#[test]
fn support_z_parameter_moves_arch() {
    let net = Network::arch_chain(4.0, 8, -1.0, -0.15);
    let eq0 = fdm(&net).expect("eq");
    let z_mid = eq0.xyz[4 * 3 + 2];

    let anchor = 0usize;
    let z_anchor = net.xyz[anchor * 3 + 2];
    let goals = vec![Goal::min_free_z(z_mid * 0.98, 1.0)];
    let mut config = OptimizeConfig {
        optimizer: OptimizerKind::Gd,
        learning_rate: 0.02,
        max_iter: 40,
        parameters: vec![
            DesignParam::all_edge_q(-20.0, -0.3),
            DesignParam::SupportCoord {
                node: anchor,
                axis: 2,
            },
        ],
        ..OptimizeConfig::default()
    };
    config.fdm.sparse = false;

    let res = constrained_fdm(&net, &goals, &[], &config).expect("opt");
    let z1 = res.network.xyz[anchor * 3 + 2];
    assert!(
        (z1 - z_anchor).abs() > 1e-6 || res.goal_loss < goals[0].target().powi(2),
        "support Z or goal should change: z {z_anchor} -> {z1}"
    );
}
