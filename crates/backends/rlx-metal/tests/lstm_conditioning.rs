// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Is the LSTM parity test measuring a kernel, or measuring chaos?**
//!
//! A previous change relaxed the native-MSL gate from `hidden <= 32` to the
//! threadgroup limit, arguing that the wide-hidden Metal/CPU divergence was an
//! ill-conditioned test rather than a kernel defect: an LSTM is a nonlinear
//! recurrence, and if it is expansive then ANY reassociation difference — which
//! two correct implementations are entitled to have — grows without bound over
//! the sequence.
//!
//! That claim is testable **without a GPU**: perturb the input by ~1 ulp and run
//! the CPU forward against itself. If a 1e-6 input change moves the CPU output
//! by O(1), the configuration is chaotic and a Metal-vs-CPU tolerance of 1e-5 is
//! not a meaningful test at that width. If the CPU tracks itself, the recurrence
//! is contractive and any large disagreement is somebody's bug.
//!
//! This exists because I concluded "kernel defect" from Metal-vs-CPU numbers
//! alone and should not have: the reference disagreeing with a perturbed copy of
//! itself is the control that separates the two explanations.
//!
//!     cargo test -p rlx-metal --test lstm_conditioning -- --nocapture

use rlx_ir::op::Op;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

const F32: DType = DType::F32;

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

/// CPU forward with `x` supplied by the caller, so two nearby inputs can be
/// compared through the identical code path.
fn cpu_run_w(s: usize, h: usize, x: &[f32], whh_rel: f32) -> Vec<f32> {
    let mut g = Graph::new("lstm_cond_w");
    let xi = g.input("x", Shape::new(&[1, s, h], F32));
    let wih = g.param("w_ih", Shape::new(&[4 * h * h], F32));
    let whh = g.param("w_hh", Shape::new(&[4 * h * h], F32));
    let bias = g.param("bias", Shape::new(&[4 * h], F32));
    let y = g.add_node(
        Op::Lstm {
            hidden_size: h,
            num_layers: 1,
            bidirectional: false,
            carry: false,
        },
        vec![xi, wih, whh, bias],
        Shape::new(&[1, s, h], F32),
    );
    g.set_outputs(vec![y]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let mut w = fill(4 * h * h, 2, 0.1);
    // RANDOM per-element sign, not a uniform scale. Scaling every weight by the
    // same factor is a highly structured perturbation that a recurrence can be
    // insensitive to while still amplifying the UNSTRUCTURED ~1 ulp differences
    // two real implementations actually have. Getting this wrong is how a
    // sensitivity test reports "contractive" for a configuration that is not.
    for (i, v) in w.iter_mut().enumerate() {
        let sign = if (i * 2654435761) % 2 == 0 { 1.0 } else { -1.0 };
        *v *= 1.0 + sign * whh_rel;
    }
    c.set_param("w_ih", &fill(4 * h * h, 1, 0.1));
    c.set_param("w_hh", &w);
    c.set_param("bias", &fill(4 * h, 3, 0.1));
    c.finalize_params();
    c.run(&[("x", x)]).remove(0)
}

/// The control that matters: two correct implementations differ by ~1 ulp in the
/// RECURRENT matrix product at every step, not by a one-off input nudge. This
/// injects that as a 1e-7 relative change to `w_hh` and asks how far the CPU
/// reference then diverges from itself.
#[test]
fn report_sensitivity_to_a_one_ulp_recurrent_weight_change() {
    println!("  CPU-vs-CPU, w_hh perturbed by 1e-7 relative (≈1 ulp)\n");
    println!("    h   seq   max|Δ| self   first |Δ|>1e-5 at t");
    for &h in &[32usize, 44, 48, 64, 96] {
        for &s in &[32usize, 64, 128] {
            let x = fill(s * h, 4, 1.0);
            let a = cpu_run_w(s, h, &x, 0.0);
            let b = cpu_run_w(s, h, &x, 1e-7);
            let d = a
                .iter()
                .zip(&b)
                .map(|(p, q)| (p - q).abs())
                .fold(0f32, f32::max);
            let first = a
                .iter()
                .zip(&b)
                .position(|(p, q)| (p - q).abs() > 1e-5)
                .map(|i| i / h);
            println!("  {h:3}  {s:4}   {d:.3e}   {first:?}");
        }
    }
}

fn cpu_run(s: usize, h: usize, x: &[f32]) -> Vec<f32> {
    let mut g = Graph::new("lstm_cond");
    let xi = g.input("x", Shape::new(&[1, s, h], F32));
    let wih = g.param("w_ih", Shape::new(&[4 * h * h], F32));
    let whh = g.param("w_hh", Shape::new(&[4 * h * h], F32));
    let bias = g.param("bias", Shape::new(&[4 * h], F32));
    let y = g.add_node(
        Op::Lstm {
            hidden_size: h,
            num_layers: 1,
            bidirectional: false,
            carry: false,
        },
        vec![xi, wih, whh, bias],
        Shape::new(&[1, s, h], F32),
    );
    g.set_outputs(vec![y]);
    let mut c = Session::new(Device::Cpu).compile(g);
    c.set_param("w_ih", &fill(4 * h * h, 1, 0.1));
    c.set_param("w_hh", &fill(4 * h * h, 2, 0.1));
    c.set_param("bias", &fill(4 * h, 3, 0.1));
    c.finalize_params();
    c.run(&[("x", x)]).remove(0)
}

#[test]
fn report_how_far_the_cpu_reference_diverges_from_itself() {
    println!("  CPU-vs-CPU sensitivity to a 1e-6 relative input perturbation");
    println!("  (Metal/CPU parity is asserted at 1e-5 absolute)\n");
    println!("    h   seq   max|Δ| self   verdict");
    for &h in &[8usize, 32, 40, 48, 64] {
        for &s in &[64usize, 256] {
            let x = fill(s * h, 4, 1.0);
            let mut x2 = x.clone();
            for v in x2.iter_mut() {
                *v *= 1.0 + 1e-6;
            }
            let a = cpu_run(s, h, &x);
            let b = cpu_run(s, h, &x2);
            let d = a
                .iter()
                .zip(&b)
                .map(|(p, q)| (p - q).abs())
                .fold(0f32, f32::max);
            // If a 1e-6 input nudge moves the output by more than the parity
            // tolerance, that tolerance cannot distinguish a bug from arithmetic.
            let verdict = if d < 1e-5 {
                "contractive — parity is meaningful"
            } else {
                "EXPANSIVE — 1e-5 parity is not meaningful here"
            };
            println!("  {h:3}  {s:4}   {d:.3e}   {verdict}");
        }
    }
}
