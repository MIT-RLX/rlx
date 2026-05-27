#![cfg(all(feature = "ir", feature = "rlx-sparse"))]
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

//! Fused MIR equilibrium path matches host optimization on arch loadpath.

use rlx_fdm::{
    Constraint, Goal, Network, OptimizeConfig, constrained_fdm, fdm, network_loadpath,
    try_constrained_fdm_fused,
};

#[test]
fn fused_runner_matches_host_equilibrium() {
    let net = Network::arch_chain(5.0, 40, -1.0, -0.2);
    let mut cfg = OptimizeConfig::arch_form_finding();
    cfg.fuse_mir = true;
    cfg.max_iter = 1;

    let host = {
        let mut c = cfg.clone();
        c.fuse_mir = false;
        constrained_fdm(&net, &[], &[], &c).expect("host")
    };
    let fused = constrained_fdm(&net, &[], &[], &cfg).expect("fused");
    let mut max_diff = 0.0f64;
    for (a, b) in host
        .equilibrium
        .xyz
        .iter()
        .zip(fused.equilibrium.xyz.iter())
    {
        max_diff = max_diff.max((a - b).abs());
    }
    assert!(
        max_diff < 1e-4,
        "fused vs host equilibrium max |Δx| = {max_diff}"
    );
}

#[test]
fn fused_constrained_reduces_loadpath_like_host() {
    let net = Network::arch_chain(5.0, 40, -1.0, -0.2);
    let eq0 = fdm(&net).expect("fdm");
    let lp0 = network_loadpath(&eq0);
    let goals = vec![Goal::network_loadpath(lp0 * 0.93, 1.0)];
    let constraints = vec![Constraint::all_edge_q(-50.0, -0.5, 0.0)];

    let mut cfg = OptimizeConfig::arch_form_finding();
    cfg.max_iter = 40;
    cfg.fuse_mir = true;

    let res = try_constrained_fdm_fused(&net, &goals, &constraints, &cfg)
        .expect("try")
        .expect("should fuse large arch");
    let lp1 = network_loadpath(&res.equilibrium);
    assert!(lp1 < lp0 + 0.05, "loadpath should improve: {lp0} -> {lp1}");
}
