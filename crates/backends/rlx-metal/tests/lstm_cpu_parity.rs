// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Op::Lstm` on Metal must match the CPU thunk, at every width and length.
//!
//! Regression for a kernel that returned **NaN** past a single SIMD group: the
//! native MSL LSTM was gated on `hidden <= 1024` (the threadgroup limit), but
//! measured against `execute_lstm_f32` it only agrees while the threadgroup is
//! one SIMD group. It needed *both* a wide hidden and a long sequence to show,
//! which is why a small smoke test never caught it:
//!
//! ```text
//!   h=32,s=256 exact | h=33,s=256 exact | h=40,s=256 55% NaN
//!   h=64,s=32  exact | h=64,s=40  exact | h=64,s=48   2 NaN | h=64,s=256 81% NaN
//! ```
//!
//! Found downstream: an EEG port with a 128-unit BiLSTM over 256 steps came back
//! entirely NaN on Metal while CPU and MLX were finite.
//!
//!     cargo test -p rlx-metal --test lstm_cpu_parity -- --nocapture

// The tolerance check is written `!(maxd < tol)` rather than `maxd >= tol` on
// purpose: comparisons against NaN are all false, so the `>=` form reports a
// NaN run as passing. Given this file exists for a kernel that returned NaN,
// that is the one failure mode it must not be able to miss. Clippy's rewrite is
// right in general and wrong here.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use rlx_ir::op::Op;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

const F32: DType = DType::F32;

/// Deterministic pseudo-random fill — CPU and Metal must see identical bytes.
fn fill(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut z = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ seed;
            z ^= z >> 30;
            z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z ^= z >> 27;
            ((z >> 40) as f32 / 8_388_608.0 - 0.5) * scale
        })
        .collect()
}

fn run(device: Device, b: usize, s: usize, d: usize, h: usize, bidir: bool) -> Vec<f32> {
    let dirs = if bidir { 2 } else { 1 };
    let mut g = Graph::new("lstm_parity");
    let x = g.input("x", Shape::new(&[b, s, d], F32));
    let wih = g.param("w_ih", Shape::new(&[4 * h * d * dirs], F32));
    let whh = g.param("w_hh", Shape::new(&[4 * h * h * dirs], F32));
    let bias = g.param("bias", Shape::new(&[4 * h * dirs], F32));
    let y = g.add_node(
        Op::Lstm {
            hidden_size: h,
            num_layers: 1,
            bidirectional: bidir,
            carry: false,
        },
        vec![x, wih, whh, bias],
        Shape::new(&[b, s, dirs * h], F32),
    );
    g.set_outputs(vec![y]);

    let mut c = Session::new(device).compile(g);
    c.set_param("w_ih", &fill(4 * h * d * dirs, 1, 0.1));
    c.set_param("w_hh", &fill(4 * h * h * dirs, 2, 0.1));
    c.set_param("bias", &fill(4 * h * dirs, 3, 0.1));
    c.finalize_params();
    c.run(&[("x", &fill(b * s * d, 4, 1.0))]).remove(0)
}

#[test]
fn metal_matches_cpu_at_every_width_and_length() {
    if rlx_ir::env::skip_unless_device("metal", true, rlx_runtime::is_available(Device::Metal)) {
        eprintln!("metal not available in this build — skipped");
        return;
    }
    // 8/32 exercise the native single-SIMD-group kernel; 40/64/128 the host
    // fallback the gate now routes to. Sequence lengths straddle the point where
    // the old kernel started diverging.
    let mut failures = Vec::new();
    for &h in &[8usize, 32, 40, 64, 128] {
        for &s in &[32usize, 64, 256] {
            for &bidir in &[false, true] {
                let cpu = run(Device::Cpu, 1, s, h, h, bidir);
                let met = run(Device::Metal, 1, s, h, h, bidir);
                let nan = met.iter().filter(|v| v.is_nan()).count();
                let maxd = cpu
                    .iter()
                    .zip(&met)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                eprintln!("  h={h:3} s={s:3} bidir={bidir:5}: nan={nan:6} max|Δ|={maxd:.3e}");
                // 1e-5 absolute on outputs bounded by tanh ∈ [-1, 1].
                if nan > 0 || !(maxd < 1e-5) {
                    failures.push(format!(
                        "h={h} s={s} bidir={bidir}: nan={nan} max|Δ|={maxd:.3e}"
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "metal/cpu LSTM disagreement:\n  {}",
        failures.join("\n  ")
    );
}
