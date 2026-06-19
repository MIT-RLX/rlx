#![cfg(feature = "nonlinear")]
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

use rlx_fdm::{IterativeConfig, Network, equilibrium_iterative};

#[test]
fn anderson_accelerated_edge_load_solve_converges() {
    let mut net = Network::arch_chain(4.0, 10, -1.0, 0.0);
    net.edges_load_uniform([0.0, 0.0, -0.04]);

    let cfg = IterativeConfig {
        tmax: 30,
        eta: 1e-6,
        anderson_depth: 4,
        ..Default::default()
    };

    let s = rlx_fdm::Structure::from_network(&net);
    let load_state = net.load_state();
    let na = s.num_fixed();
    let mut xf = vec![0.0; na * 3];
    for (j, &node) in s.indices_fixed.iter().enumerate() {
        for c in 0..3 {
            xf[j * 3 + c] = net.xyz[node * 3 + c];
        }
    }

    let xyz_free = equilibrium_iterative(
        &net.q,
        &xf,
        &load_state,
        &s,
        &net.edges,
        &net.xyz,
        &cfg,
        None,
    )
    .expect("anderson solve");
    assert!(
        xyz_free[2] < -0.02,
        "expected sag under edge load, z={}",
        xyz_free[2]
    );
}
