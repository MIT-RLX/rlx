// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Arch chain equilibrium (jax_fdm `examples/arch/arch.py` setup).

use rlx_fdm::{Network, fdm, goals};

#[test]
fn arch_chain_sags_and_equilibrates() {
    let net = Network::arch_chain(5.0, 10, -1.0, -0.2);
    let eq = fdm(&net).expect("fdm");

    // Interior free nodes move down under gravity-like load.
    assert!(
        eq.xyz[3 * 5 + 2] < -0.01,
        "mid-span should sag, z={}",
        eq.xyz[3 * 5 + 2]
    );

    let r = eq.max_free_residual_norm(&net.is_support);
    assert!(r < 1e-8, "free-node residual {r}");

    // Supports stay at input height.
    assert!((eq.xyz[2] - 0.0).abs() < 1e-10);
    assert!((eq.xyz[3 * 10 + 2] - 0.0).abs() < 1e-10);

    assert!(goals::mean_edge_length(&eq) > 0.4);
    assert!(goals::total_loadpath_proxy(&eq) > 0.0);
}

#[test]
fn validation_rejects_no_supports() {
    let mut net = Network::arch_chain(1.0, 2, -1.0, 0.0);
    net.is_support = vec![false; 3];
    assert!(fdm(&net).is_err());
}
