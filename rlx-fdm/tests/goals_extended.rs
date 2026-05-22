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
use rlx_fdm::{fdm, goals_loss_with_structure, Goal, Network};

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
