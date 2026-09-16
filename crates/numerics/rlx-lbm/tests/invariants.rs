// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Properties of the moment-encoded scheme, none of which reference a second
//! implementation.

use rlx_lbm::invariant::{round_trip_d2q9, round_trip_d3q27, sample_states_2d, sample_states_3d};
use rlx_lbm::moment::{
    Moments2d, Moments3d, collide_d2q9, moments_d2q9, omega, reconstruct_d2q9, reconstruct_d3q27,
};
use rlx_lbm::sim::{taylor_green, taylor_green_decay};

/// The load-bearing invariant: taking the moments of the reconstructed
/// populations must return the moments you started from.
///
/// This is what a reconstruction that is wrong-but-self-consistent fails. It is
/// checkable in double precision to machine epsilon, so the threshold needs no
/// tuning.
#[test]
fn d2q9_reconstruction_round_trips() {
    for m in sample_states_2d() {
        let r = round_trip_d2q9(&m);
        assert!(
            r.within(1e-12),
            "D2Q9 round-trip residual {r:?} for {m:?} — the reconstruction and the \
             moment definition disagree"
        );
    }
}

#[test]
fn d3q27_reconstruction_round_trips() {
    for m in sample_states_3d() {
        let r = round_trip_d3q27(&m);
        assert!(r.within(1e-12), "D3Q27 round-trip residual {r:?} for {m:?}");
    }
}

/// A test state with nonzero velocity and a genuine off-equilibrium stress.
fn probe_state() -> Moments2d {
    Moments2d {
        rho: 1.0,
        u: [0.08, -0.05],
        s: [0.08 * 0.08 + 1e-3, 0.05 * 0.05 - 6e-4, -0.08 * 0.05 + 5e-4],
    }
}

/// **What the round-trip catches: a mis-scaled reconstruction.**
///
/// The concrete failure this is modelled on: re-adding the equilibrium terms
/// `4.5(c·u)² − 1.5|u|²` (which `H²:S` already contains, since `S` includes
/// `u ⊗ u`) *and* applying `1/(2cs⁴)` as 1.5 instead of 4.5. The result is
/// self-consistent, stable, and produces plausible flow — but the recovered
/// second moment is `u ⊗ u + S/3`, so the off-equilibrium stress that carries
/// viscosity comes back at a third of its size.
///
/// Nothing differential sees this: every backend implementing the same formula
/// agrees. The round trip sees it immediately, and pins the exact error law.
#[test]
fn round_trip_catches_mis_scaled_reconstruction() {
    use rlx_lbm::lattice::{CS2, D2Q9_C, D2Q9_W, h2};
    let m = probe_state();
    let mut f = [0.0f64; 9];
    for i in 0..9 {
        let c = &D2Q9_C[i];
        let cu = (c[0] as f64) * m.u[0] + (c[1] as f64) * m.u[1];
        let u2 = m.u[0] * m.u[0] + m.u[1] * m.u[1];
        let hs = h2(c, 0, 0) * m.s[0] + h2(c, 1, 1) * m.s[1] + 2.0 * h2(c, 0, 1) * m.s[2];
        f[i] = m.rho * D2Q9_W[i] * (1.0 + cu / CS2 + 4.5 * cu * cu - 1.5 * u2 + 1.5 * hs);
    }
    let back = moments_d2q9(&f);

    // Mass and momentum survive — which is why it looks fine.
    assert!(
        (back.rho - m.rho).abs() < 1e-12,
        "mass should be unaffected"
    );
    for a in 0..2 {
        assert!((back.u[a] - m.u[a]).abs() < 1e-12, "momentum unaffected");
    }
    // The second moment does not.
    let s_err = (0..3)
        .map(|i| (back.s[i] - m.s[i]).abs())
        .fold(0.0, f64::max);
    assert!(
        s_err > 1e-4,
        "mis-scaled reconstruction round-tripped ({s_err:.2e}) — the oracle would \
         then be useless"
    );
    // And the error follows exactly `S_recovered = u ⊗ u + S/3`.
    let uu = [m.u[0] * m.u[0], m.u[1] * m.u[1], m.u[0] * m.u[1]];
    for i in 0..3 {
        assert!(
            (back.s[i] - (uu[i] + m.s[i] / 3.0)).abs() < 1e-15,
            "error law mismatch at S[{i}]"
        );
    }
    // The correct reconstruction round-trips on the same state.
    assert!(round_trip_d2q9(&m).within(1e-12));
}

/// **What the round-trip does *not* catch: Hermite truncation order.**
///
/// Dropping the third-order terms leaves `ρ`, `ρu` and `Π` untouched, because
/// the third-order Hermite tensors are orthogonal to the first three moments by
/// construction — they live in the lattice's ghost-moment space. So a
/// second-order reconstruction passes the round trip exactly while changing the
/// populations by ~1e-4, and therefore changing the dynamics.
///
/// This is a real limitation of the oracle and is recorded rather than papered
/// over: the round trip constrains *assembly*, the Taylor–Green decay test
/// constrains *dynamics*, and neither substitutes for the other.
#[test]
fn round_trip_is_blind_to_truncation_order() {
    use rlx_lbm::lattice::{CS2, D2Q9_C, D2Q9_W, h2};
    let m = probe_state();
    let mut f2 = [0.0f64; 9];
    for i in 0..9 {
        let c = &D2Q9_C[i];
        let cu = (c[0] as f64) * m.u[0] + (c[1] as f64) * m.u[1];
        let hs = h2(c, 0, 0) * m.s[0] + h2(c, 1, 1) * m.s[1] + 2.0 * h2(c, 0, 1) * m.s[2];
        f2[i] = m.rho * D2Q9_W[i] * (1.0 + cu / CS2 + hs / (2.0 * CS2 * CS2));
    }
    let back = moments_d2q9(&f2);
    let s_err = (0..3)
        .map(|i| (back.s[i] - m.s[i]).abs())
        .fold(0.0, f64::max);
    assert!(
        s_err < 1e-14,
        "second-order truncation is expected to round-trip exactly, got {s_err:.2e}"
    );

    // But the populations genuinely differ, so the terms are not no-ops.
    let f3 = reconstruct_d2q9(&m);
    let df = (0..9).map(|i| (f3[i] - f2[i]).abs()).fold(0.0, f64::max);
    assert!(
        df > 1e-6,
        "third-order terms changed nothing ({df:.2e}) — then they should be removed"
    );
}

/// Collision conserves mass and momentum exactly (with no force).
#[test]
fn collision_conserves_mass_and_momentum() {
    for m in sample_states_2d() {
        for &nu in &[0.001f64, 0.01, 0.1] {
            let post = collide_d2q9(&m, nu, [0.0, 0.0]);
            assert!((post.rho - m.rho).abs() < 1e-14, "mass changed");
            for a in 0..2 {
                assert!(
                    (post.u[a] - m.u[a]).abs() < 1e-14,
                    "momentum changed on axis {a}: {} -> {}",
                    m.u[a],
                    post.u[a]
                );
            }
        }
    }
}

/// An equilibrium state is a fixed point of collision — nothing to relax.
#[test]
fn equilibrium_is_a_collision_fixed_point() {
    for &(ux, uy) in &[(0.0, 0.0), (0.05, -0.03), (-0.1, 0.07)] {
        let m = Moments2d::equilibrium(1.0, [ux, uy]);
        let post = collide_d2q9(&m, 0.01, [0.0, 0.0]);
        for i in 0..3 {
            assert!(
                (post.s[i] - m.s[i]).abs() < 1e-14,
                "equilibrium moved under collision: S[{i}] {} -> {}",
                m.s[i],
                post.s[i]
            );
        }
    }
}

/// Relaxation must be monotone toward equilibrium and never overshoot for
/// `Ω ∈ (0, 2)`.
#[test]
fn collision_relaxes_toward_equilibrium() {
    let u = [0.06f64, -0.04];
    let eq = Moments2d::equilibrium(1.0, u);
    let m = Moments2d {
        rho: 1.0,
        u,
        s: [eq.s[0] + 2e-3, eq.s[1] - 1e-3, eq.s[2] + 1.5e-3],
    };
    let before: f64 = (0..3).map(|i| (m.s[i] - eq.s[i]).abs()).sum();
    for &nu in &[0.005f64, 0.05, 0.5] {
        let post = collide_d2q9(&m, nu, [0.0, 0.0]);
        let after: f64 = (0..3).map(|i| (post.s[i] - eq.s[i]).abs()).sum();
        assert!(
            after < before,
            "ν={nu} (Ω={:.3}): deviation grew {before:.3e} -> {after:.3e}",
            omega(nu)
        );
    }
}

/// `Σ f_i` from a reconstructed rest state must be exactly `ρ`, and every
/// population must equal its weight — the simplest possible pin on the
/// zeroth-order term.
#[test]
fn rest_state_reproduces_the_weights() {
    use rlx_lbm::lattice::{D2Q9_W, D3Q27_W};
    let f = reconstruct_d2q9(&Moments2d::equilibrium(1.0, [0.0, 0.0]));
    for i in 0..9 {
        assert!((f[i] - D2Q9_W[i]).abs() < 1e-15, "D2Q9 rest f[{i}]");
    }
    let f3 = reconstruct_d3q27(&Moments3d::equilibrium(1.0, [0.0, 0.0, 0.0]));
    for i in 0..27 {
        assert!((f3[i] - D3Q27_W[i]).abs() < 1e-15, "D3Q27 rest f[{i}]");
    }
}

/// Mass is conserved exactly over a full simulation, not just per-op.
#[test]
fn simulation_conserves_mass() {
    let mut fld = taylor_green(16, 0.02, 0.01);
    let m0 = fld.mass();
    for _ in 0..40 {
        fld.step();
    }
    let m1 = fld.mass();
    assert!((m1 - m0).abs() / m0 < 1e-12, "mass drifted {m0} -> {m1}");
}

/// **Taylor–Green decay vs the analytic solution.**
///
/// The external oracle: a decaying vortex loses energy as `exp(−2νk²t)`, so the
/// measured decay *is* a measurement of the solver's effective viscosity. A
/// reconstruction with a wrong coefficient reproduces the vortex shape and the
/// mass exactly while getting this number wrong — which is why shape-based eyeball
/// validation does not catch it.
#[test]
fn taylor_green_decays_at_the_analytic_rate() {
    let n = 32;
    let u0 = 0.02;
    for &nu in &[0.01f64, 0.03] {
        let mut fld = taylor_green(n, u0, nu);
        let steps = 200;
        let u_start = fld.max_speed();
        for _ in 0..steps {
            fld.step();
        }
        let measured = fld.max_speed() / u_start;
        let analytic = taylor_green_decay(n, nu, steps);
        let rel = (measured - analytic).abs() / analytic;
        assert!(
            rel < 0.05,
            "ν={nu}: measured decay {measured:.6} vs analytic {analytic:.6} \
             (relative error {rel:.3}) — effective viscosity is off"
        );
        eprintln!("TGV ν={nu}: measured {measured:.6}, analytic {analytic:.6}, rel {rel:.4}");
    }
}

/// **Lattice-mirror equivariance.** The D2Q9 lattice is symmetric under
/// `y → −y`, so the solver must commute with that reflection:
/// `step(mirror(F)) == mirror(step(F))`, to machine precision.
///
/// This is the property that a swapped direction index, a wrong opposite-pair
/// table, or a sign dropped from an odd moment breaks — the same class as the
/// RoPE table-stride bug, where three backends agreed and all three were wrong.
/// Unlike "the vortex still looks symmetric", it holds exactly, so there is no
/// tolerance to tune away.
#[test]
fn scheme_is_equivariant_under_lattice_mirror() {
    let n = 12;
    let nu = 0.01;

    // Mirror in y: reindex y → (n − y) mod n and flip every odd-in-y moment.
    let mirror = |f: &rlx_lbm::sim::Field2d| -> rlx_lbm::sim::Field2d {
        let mut out = f.clone();
        for y in 0..n {
            for x in 0..n {
                let m = f.at(x, (n - y) % n);
                out.cells[y * n + x] = Moments2d {
                    rho: m.rho,
                    u: [m.u[0], -m.u[1]],
                    s: [m.s[0], m.s[1], -m.s[2]],
                };
            }
        }
        out
    };

    let base = taylor_green(n, 0.03, nu);

    let mut a = mirror(&base); // step(mirror(F))
    a.step();
    let mut b = base.clone(); // mirror(step(F))
    b.step();
    let b = mirror(&b);

    for i in 0..n * n {
        let (p, q) = (a.cells[i], b.cells[i]);
        assert!((p.rho - q.rho).abs() < 1e-13, "rho at {i}");
        for k in 0..2 {
            assert!((p.u[k] - q.u[k]).abs() < 1e-13, "u[{k}] at {i}");
        }
        for k in 0..3 {
            assert!(
                (p.s[k] - q.s[k]).abs() < 1e-13,
                "S[{k}] at {i}: {} vs {}",
                p.s[k],
                q.s[k]
            );
        }
    }
}
