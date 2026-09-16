// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Is the native Metal LSTM's drift from CPU at long sequences a *race*, or just a
//! well-behaved kernel amplifying a benign floating-point difference?
//!
//! An LSTM recurrence with unconstrained random weights is chaotic: a 1e-7 seed
//! difference (which fp reassociation alone guarantees, since the GPU sums the
//! hidden dot product in a different order) grows exponentially over hundreds of
//! steps. So "Metal != CPU at s=256" is *not* on its own evidence of a bug.
//!
//! Two discriminators:
//!   * **determinism** — a race makes the kernel disagree with *itself* across runs;
//!     fp reassociation is deterministic.
//!   * **conditioning** — shrink `w_hh` so the recurrence is contractive; a correct
//!     kernel then tracks CPU to fp precision at any sequence length, while a race
//!     stays broken.
#![cfg(all(feature = "cpu", feature = "metal"))]
use rlx_ir::op::Op;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

mod common;

fn mk(n: usize, seed: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            ((((i.wrapping_mul(2654435761).wrapping_add(seed)) % 1000) as f32) / 500.0 - 1.0)
                * scale
        })
        .collect()
}

fn build(b: usize, s: usize, inp: usize, h: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("lstm_diagnose");
    let x = g.input("x", Shape::new(&[b, s, inp], f));
    let wih = g.input("w_ih", Shape::new(&[4 * h * inp], f));
    let whh = g.input("w_hh", Shape::new(&[4 * h * h], f));
    let bias = g.input("bias", Shape::new(&[4 * h], f));
    let out = g.add_node(
        Op::Lstm {
            hidden_size: h,
            num_layers: 1,
            bidirectional: false,
            carry: false,
        },
        vec![x, wih, whh, bias],
        Shape::new(&[b, s, h], f),
    );
    g.set_outputs(vec![out]);
    g
}

fn run(dev: Device, b: usize, s: usize, inp: usize, h: usize, whh_scale: f32) -> Vec<f32> {
    let xd = mk(b * s * inp, 1, 1.0);
    let wihd = mk(4 * h * inp, 2, 1.0);
    let whhd = mk(4 * h * h, 3, whh_scale);
    let bd = mk(4 * h, 4, 1.0);
    let slots: [(&str, &[f32]); 4] = [("x", &xd), ("w_ih", &wihd), ("w_hh", &whhd), ("bias", &bd)];
    let mut c = Session::new(dev).compile(build(b, s, inp, h));
    c.run(&slots).pop().unwrap()
}

fn max_delta(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// A race would make Metal disagree with itself; fp reassociation would not.
#[test]
fn metal_lstm_is_deterministic() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Metal, "metal") {
        eprintln!("skip: no Metal device");
        return;
    }
    for &(h, s) in &[(128usize, 256usize), (256, 256), (256, 48)] {
        let a = run(Device::Metal, 1, s, 12, h, 1.0);
        let b = run(Device::Metal, 1, s, 12, h, 1.0);
        let c = run(Device::Metal, 1, s, 12, h, 1.0);
        let d = max_delta(&a, &b).max(max_delta(&b, &c));
        println!("h={h} s={s}: metal-vs-metal max_abs={d:.3e}");
        assert_eq!(
            d, 0.0,
            "h={h} s={s}: Metal LSTM is non-deterministic — that is a race"
        );
    }
}

/// With a contractive recurrence the dynamics are no longer chaotic, so a correct
/// kernel must track CPU to fp precision even at s=256.
#[test]
fn metal_lstm_matches_cpu_when_well_conditioned() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Metal, "metal") {
        eprintln!("skip: no Metal device");
        return;
    }
    for &(h, s) in &[(128usize, 256usize), (256, 256)] {
        for &scale in &[0.05f32, 0.2] {
            let cpu = run(Device::Cpu, 1, s, 12, h, scale);
            let gpu = run(Device::Metal, 1, s, 12, h, scale);
            let d = max_delta(&cpu, &gpu);
            println!("h={h} s={s} w_hh*{scale}: cpu-vs-metal max_abs={d:.3e}");
            assert!(d < 1e-4, "h={h} s={s} scale={scale}: max_abs={d:.3e}");
        }
    }
}

/// How fast does a *deliberate* 1-ULP-scale perturbation grow on CPU alone? This is
/// the conditioning of the problem, independent of any backend.
#[test]
fn cpu_lstm_amplifies_tiny_perturbations() {
    let _gpu = common::serialize_gpu();
    let (h, s, inp) = (128usize, 256usize, 12usize);
    let base = run(Device::Cpu, 1, s, inp, h, 1.0);

    // Perturb one input element by ~1e-7 relative and re-run on CPU only.
    let mut xd = mk(s * inp, 1, 1.0);
    xd[0] += 1e-6;
    let wihd = mk(4 * h * inp, 2, 1.0);
    let whhd = mk(4 * h * h, 3, 1.0);
    let bd = mk(4 * h, 4, 1.0);
    let slots: [(&str, &[f32]); 4] = [("x", &xd), ("w_ih", &wihd), ("w_hh", &whhd), ("bias", &bd)];
    let mut c = Session::new(Device::Cpu).compile(build(1, s, inp, h));
    let perturbed = c.run(&slots).pop().unwrap();

    let d = max_delta(&base, &perturbed);
    println!("cpu-only: 1e-6 input perturbation -> max_abs={d:.3e} after {s} steps");
    println!(
        "(if this is O(0.1-1), the s=256 config is chaotic and cpu-vs-metal drift is expected)"
    );
}
