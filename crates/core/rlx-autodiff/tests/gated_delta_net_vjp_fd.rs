// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::GatedDeltaNet` VJP against central finite differences, across positions.
//!
//! GatedDeltaNet is a *recurrent* linear-attention scan: the running state `S`
//! is damped by `exp(g[t])`, updated with `outer(k[t], (v[t] − k[t]·S)·β[t])`,
//! and read out as `q[t] · S`. So `out[t]` depends on `k[s]`, `v[s]`, `g[s]`,
//! `β[s]` for **every** `s ≤ t`, not just `s == t`.
//!
//! `unfuse_fused_for_autodiff` unrolls that loop, and a decomposition that
//! threads the state correctly in the forward direction can still fail to carry
//! gradient backwards along it. That failure is invisible to a same-position
//! check — the diagonal `∂out[t]/∂·[t]` stays exactly right — and invisible to
//! any forward parity test. It shows up only as a cross-position gradient that
//! is *identically zero* where the model is genuinely sensitive.
//!
//! So the assertions here are split: the diagonal must be numerically right,
//! and the strictly-lower triangle must be non-zero wherever finite differences
//! say the dependence is real.

use rlx_autodiff::{GradWithLossOptions, Wrt, grad_with_loss_wrt};
use rlx_ir::{DType, Graph, Shape};

const B: usize = 1;
const S: usize = 4;
const H: usize = 2;
const N: usize = 4;
/// q/k/v are `[B, S, H, N]`.
const QKV: usize = B * S * H * N;
/// g/beta are `[B, S, H]`.
const GB: usize = B * S * H;

fn gdn_graph() -> Graph {
    let f = DType::F32;
    let bshn = Shape::new(&[B, S, H, N], f);
    let bsh = Shape::new(&[B, S, H], f);
    let mut g = Graph::new("gdn");
    let q = g.input("q", bshn.clone());
    let k = g.input("k", bshn.clone());
    let v = g.input("v", bshn.clone());
    let gate = g.input("g", bsh.clone());
    let beta = g.input("beta", bsh);
    let y = g.gated_delta_net(q, k, v, gate, beta, N, bshn);
    g.set_outputs(vec![y]);
    g
}

fn hashed(seed: u64, i: usize) -> f32 {
    let mut x = seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    ((x >> 40) as f32) / 8_388_608.0 - 1.0
}

struct Inputs {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    g: Vec<f32>,
    beta: Vec<f32>,
}

fn inputs() -> Inputs {
    Inputs {
        q: (0..QKV).map(|i| 0.4 * hashed(1, i)).collect(),
        k: (0..QKV).map(|i| 0.4 * hashed(2, i)).collect(),
        v: (0..QKV).map(|i| 0.4 * hashed(3, i)).collect(),
        // Decay must be negative so exp(g) < 1 and the scan stays stable.
        g: (0..GB).map(|i| -0.3 + 0.1 * hashed(4, i)).collect(),
        beta: (0..GB).map(|i| 0.5 + 0.2 * hashed(5, i)).collect(),
    }
}

/// `∂(Σ cotangent·out)/∂k`, by autodiff and by finite differences, with the
/// cotangent supported on target position `t` only.
fn grads_at_target(t: usize) -> (Vec<f32>, Vec<f64>) {
    let x = inputs();
    let mut cot = vec![0.0f32; QKV];
    for h in 0..H {
        for j in 0..N {
            cot[t * H * N + h * N + j] = 0.5 + 0.25 * ((j % 3) as f32);
        }
    }

    let bwd = grad_with_loss_wrt(
        &gdn_graph(),
        &[Wrt::Leaf("k".into())],
        GradWithLossOptions::STRICT.with_aux(false),
    );
    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    let ad = compiled.run(&[
        ("q", &x.q[..]),
        ("k", &x.k[..]),
        ("v", &x.v[..]),
        ("g", &x.g[..]),
        ("beta", &x.beta[..]),
        ("d_output", &cot[..]),
    ])[1]
        .clone();

    let mut fwd = rlx::Session::new(rlx::Device::Cpu).compile(gdn_graph());
    let mut probe = |k: &[f32]| -> f64 {
        fwd.run(&[
            ("q", &x.q[..]),
            ("k", k),
            ("v", &x.v[..]),
            ("g", &x.g[..]),
            ("beta", &x.beta[..]),
        ])[0]
            .iter()
            .zip(&cot)
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum()
    };

    let eps = 5e-3f32;
    let fd: Vec<f64> = (0..QKV)
        .map(|i| {
            let mut kp = x.k.clone();
            let mut km = x.k.clone();
            kp[i] += eps;
            km[i] -= eps;
            (probe(&kp) - probe(&km)) / (2.0 * eps as f64)
        })
        .collect();
    (ad, fd)
}

/// Largest magnitude over the `[H, N]` block belonging to position `s`.
fn at_position(values: &[f64], s: usize) -> f64 {
    (0..H * N)
        .map(|j| values[s * H * N + j].abs())
        .fold(0.0, f64::max)
}

fn at_position_f32(values: &[f32], s: usize) -> f64 {
    (0..H * N)
        .map(|j| values[s * H * N + j].abs() as f64)
        .fold(0.0, f64::max)
}

#[test]
fn gated_delta_net_vjp_matches_finite_differences() {
    for t in 0..S {
        let (ad, fd) = grads_at_target(t);
        for i in 0..QKV {
            assert!(
                (fd[i] - ad[i] as f64).abs() < 5e-3,
                "target t={t}, k[{i}]: autodiff {} vs finite-difference {}",
                ad[i],
                fd[i]
            );
        }
    }
}

/// The recurrence itself: `out[t]` must be sensitive to `k[s]` for `s < t`, and
/// the VJP must report that sensitivity rather than zero.
#[test]
fn gated_delta_net_vjp_carries_gradient_across_positions() {
    let mut checked = 0usize;
    for t in 1..S {
        let (ad, fd) = grads_at_target(t);
        for s in 0..t {
            let fd_mag = at_position(&fd, s);
            let ad_mag = at_position_f32(&ad, s);
            // Only assert where the forward genuinely depends on position s.
            if fd_mag < 1e-2 {
                continue;
            }
            checked += 1;
            assert!(
                ad_mag > 0.5 * fd_mag,
                "out[{t}] is sensitive to k[{s}] (finite difference {fd_mag:.5}) but the \
                 VJP reports {ad_mag:.5} — gradient is not flowing through the recurrent state"
            );
        }
    }
    assert!(
        checked > 0,
        "no cross-position dependence was strong enough to test — the scan decayed to nothing"
    );
    eprintln!("checked {checked} cross-position (target, source) pairs");
}

/// The upper triangle must stay exactly zero: the scan is causal.
#[test]
fn gated_delta_net_vjp_is_causal() {
    for t in 0..S - 1 {
        let (ad, _fd) = grads_at_target(t);
        for s in t + 1..S {
            assert_eq!(
                at_position_f32(&ad, s),
                0.0,
                "out[{t}] received gradient from a later position k[{s}]"
            );
        }
    }
}
