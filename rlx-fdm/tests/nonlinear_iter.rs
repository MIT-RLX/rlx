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
use rlx_fdm::{FdmOptions, Network, fdm, fdm_with_options};

#[test]
fn edge_follower_load_changes_equilibrium() {
    let mut net = Network::arch_chain(5.0, 10, -1.0, 0.0);
    net.loads_on_free([0.0, 0.0, 0.0]);
    net.edges_load_uniform([0.0, 0.0, -0.05]);

    let linear = fdm(&net).expect("linear");
    let mut opts = FdmOptions::nonlinear(50, 1e-6, false);
    opts.iterative.tmax = 50;
    let nonlinear = fdm_with_options(&net, &opts).expect("nonlinear");

    let z_lin = linear.xyz[3 * 5 + 2];
    let z_non = nonlinear.xyz[3 * 5 + 2];
    assert!(z_non < -0.01, "expected sag from edge load, z={z_non}");
    assert!(
        (z_non - z_lin).abs() > 1e-4,
        "edge loads should move geometry: linear z={z_lin} nonlinear z={z_non}"
    );
}

#[test]
fn nonlinear_converges_in_one_step_without_edge_loads() {
    let net = Network::arch_chain(5.0, 10, -1.0, -0.2);
    let once = fdm(&net).expect("once");
    let opts = FdmOptions::nonlinear(20, 1e-8, false);
    let many = fdm_with_options(&net, &opts).expect("many");
    let dz = (once.xyz[3 * 5 + 2] - many.xyz[3 * 5 + 2]).abs();
    assert!(dz < 1e-10, "nodal-only loads: extra iters noop, dz={dz}");
}
