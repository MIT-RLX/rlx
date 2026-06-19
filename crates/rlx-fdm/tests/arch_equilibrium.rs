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
