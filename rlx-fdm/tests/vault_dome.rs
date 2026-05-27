#![cfg(feature = "io")]
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

//! Regression: jax_fdm arch JSON (vault / dome style meshes in-repo).

use std::path::PathBuf;

use rlx_fdm::{
    ErrorKind, Goal, Loss, Network, OptimizeConfig, OptimizerKind, constrained_fdm, fdm,
    io::from_json_str, losses_total,
};

fn arch_network() -> Network {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/arch.json");
    let s = std::fs::read_to_string(&path).expect("arch.json");
    let mut net = from_json_str(&s).expect("parse");
    net.anchor_nodes(&[0, 10]);
    net.q.fill(-1.0);
    net.loads_on_free([0.0, 0.0, -0.2]);
    net
}

#[test]
fn arch_json_equilibrium_small_residual() {
    let net = arch_network();
    let eq = fdm(&net).expect("fdm");
    assert!(eq.max_free_residual_norm(&net.is_support) < 1e-7);
}

#[test]
fn arch_json_constrained_opt_with_loss_api() {
    let net = arch_network();
    let eq0 = fdm(&net).expect("eq");
    let lp = rlx_fdm::network_loadpath(&eq0);
    let goals = vec![Goal::network_loadpath(lp * 0.95, 1.0)];
    let losses = vec![Loss::new(goals).with_error(ErrorKind::Squared)];
    let constraints = vec![rlx_fdm::Constraint::all_edge_q(-50.0, -0.5, 0.0)];

    let mut config = OptimizeConfig::arch_form_finding();
    config.max_iter = 25;
    config.losses = losses;
    config.optimizer = OptimizerKind::Lbfgs;
    config.fuse_mir = false;

    let res = constrained_fdm(&net, &[], &constraints, &config).expect("opt");
    let mesh = res.network.mesh_structure();
    let l = losses_total(
        &config.losses,
        &res.equilibrium,
        &rlx_fdm::Structure::from_network(&res.network),
        &res.network.is_support,
        mesh.as_ref(),
    );
    assert!(l < (lp - lp * 0.95).powi(2) * 1.2);
}
