#![cfg(feature = "io")]
// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::PathBuf;

use rlx_fdm::{Network, fdm, io::from_json_str};

#[test]
fn load_arch_json_and_solve() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/arch.json");
    let s = std::fs::read_to_string(&path).expect("arch.json");
    let mut net = from_json_str(&s).expect("parse");
    net.anchor_nodes(&[0, 10]);
    net.q.fill(-1.0);
    net.loads_on_free([0.0, 0.0, -0.2]);
    let eq = fdm(&net).expect("fdm");
    assert!(eq.max_free_residual_norm(&net.is_support) < 1e-8);
}
