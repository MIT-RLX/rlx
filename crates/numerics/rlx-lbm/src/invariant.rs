// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The moment round-trip oracle.
//!
//! A moment-encoded solver *is* its reconstruction: the populations are never
//! stored, so `reconstruct` defines the scheme. That leaves nothing to compare
//! it against — a CPU/GPU parity test compares two copies of the same formula,
//! and a regression test compares it to whatever it produced yesterday.
//!
//! The one thing that constrains it from outside is the definition of a moment:
//!
//! ```text
//! Σ f_i = ρ        Σ f_i c_α = ρ u_α        Σ f_i c_α c_β = ρ (S_αβ + cs² δ_αβ)
//! ```
//!
//! If `moments(reconstruct(m)) ≠ m`, the scheme is inconsistent no matter how
//! plausible its output looks. This matters more than it sounds: a mis-scaled
//! reconstruction does not blow up, it silently rescales the off-equilibrium
//! stress — which is the viscosity. The flow still looks like a flow. It is the
//! forward-path analogue of a finite-difference gradient check.
//!
//! # What it does not catch
//!
//! The Hermite **truncation order**. The third-order tensors are orthogonal to
//! the first three moments by construction — they live in the lattice's ghost
//! moment space — so a second-order reconstruction round-trips *exactly* while
//! changing the populations, and therefore the dynamics. Verified in
//! `round_trip_is_blind_to_truncation_order`.
//!
//! So the two oracles cover different things and neither subsumes the other:
//!
//! | oracle | constrains |
//! |---|---|
//! | this round trip | how the reconstruction is **assembled** — coefficients, scales, index layout |
//! | Taylor–Green decay vs `exp(−2νk²t)` | what the scheme **does** — effective viscosity, truncation order |
//!
//! The bug this was written against is squarely in the first column: applying
//! `1/(2cs⁴)` as 1.5 instead of 4.5 while double-counting the equilibrium terms
//! yields `S_recovered = u ⊗ u + S/3`, caught here at ~1e-3 against a 1e-12
//! threshold.

use crate::moment::{
    Moments2d, Moments3d, moments_d2q9, moments_d3q27, reconstruct_d2q9, reconstruct_d3q27,
};

/// How far a state drifts through one encode/decode cycle.
#[derive(Debug, Clone, Copy, Default)]
pub struct Residual {
    /// `|ρ' − ρ|`.
    pub rho: f64,
    /// `max |u'_α − u_α|`.
    pub u: f64,
    /// `max |S'_αβ − S_αβ|`.
    pub s: f64,
}

impl Residual {
    /// The largest component — one number to threshold on.
    pub fn max(&self) -> f64 {
        self.rho.max(self.u).max(self.s)
    }
    /// True when every component is within `tol`.
    pub fn within(&self, tol: f64) -> bool {
        self.max() <= tol
    }
}

/// Round-trip residual for a D2Q9 state.
pub fn round_trip_d2q9(m: &Moments2d) -> Residual {
    let back = moments_d2q9(&reconstruct_d2q9(m));
    Residual {
        rho: (back.rho - m.rho).abs(),
        u: (0..2)
            .map(|i| (back.u[i] - m.u[i]).abs())
            .fold(0.0, f64::max),
        s: (0..3)
            .map(|i| (back.s[i] - m.s[i]).abs())
            .fold(0.0, f64::max),
    }
}

/// Round-trip residual for a D3Q27 state.
pub fn round_trip_d3q27(m: &Moments3d) -> Residual {
    let back = moments_d3q27(&reconstruct_d3q27(m));
    Residual {
        rho: (back.rho - m.rho).abs(),
        u: (0..3)
            .map(|i| (back.u[i] - m.u[i]).abs())
            .fold(0.0, f64::max),
        s: (0..6)
            .map(|i| (back.s[i] - m.s[i]).abs())
            .fold(0.0, f64::max),
    }
}

/// Deterministic spread of physically reasonable D2Q9 states: equilibrium plus a
/// small off-equilibrium stress, at low Mach number.
pub fn sample_states_2d() -> Vec<Moments2d> {
    let mut v = Vec::new();
    for &rho in &[1.0f64, 0.85, 1.2] {
        for &(ux, uy) in &[(0.0, 0.0), (0.05, -0.03), (-0.08, 0.06), (0.1, 0.1)] {
            for &k in &[0.0f64, 1e-4, 1e-3, 5e-3] {
                v.push(Moments2d {
                    rho,
                    u: [ux, uy],
                    s: [ux * ux + k, uy * uy - 0.6 * k, ux * uy + 0.5 * k],
                });
            }
        }
    }
    v
}

/// Deterministic spread of D3Q27 states.
pub fn sample_states_3d() -> Vec<Moments3d> {
    let mut v = Vec::new();
    for &rho in &[1.0f64, 0.9, 1.15] {
        for &(ux, uy, uz) in &[
            (0.0, 0.0, 0.0),
            (0.05, -0.03, 0.02),
            (-0.07, 0.04, -0.06),
            (0.09, 0.09, -0.09),
        ] {
            for &k in &[0.0f64, 1e-4, 1e-3, 5e-3] {
                v.push(Moments3d {
                    rho,
                    u: [ux, uy, uz],
                    s: [
                        ux * ux + k,
                        uy * uy - 0.6 * k,
                        uz * uz - 0.4 * k,
                        ux * uy + 0.5 * k,
                        ux * uz - 0.2 * k,
                        uy * uz + 0.3 * k,
                    ],
                });
            }
        }
    }
    v
}
