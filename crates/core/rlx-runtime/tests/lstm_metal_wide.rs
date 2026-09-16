// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Native Metal `Op::Lstm` vs the CPU reference across the (hidden, seq) space.
//!
//! The MSL kernel is threadgroup-cooperative — one threadgroup per batch item,
//! one thread per hidden unit, `h_prev` shared in threadgroup memory — so it is
//! sensitive to how wide the threadgroup gets. This sweep is the guard on the
//! wide-hidden path (`hidden > 32`, i.e. more than one SIMD group).
//!
//! **The recurrence is deliberately contractive here.** With unit-scale random
//! weights an LSTM is chaotic: a 1e-6 perturbation grows to ~1e-1 over 256 steps
//! *on CPU alone*, so CPU-vs-GPU reassociation differences necessarily blow up and
//! the comparison measures conditioning rather than correctness. Scaling `w_hh`
//! down makes the dynamics stable, which is what lets this assert real tolerances.
//! `lstm_metal_diagnose.rs` holds the evidence for that claim.
#![cfg(all(feature = "cpu", feature = "metal"))]
use rlx_ir::op::Op;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

mod common;

/// Keeps the recurrence contractive — see the module note.
const WHH_SCALE: f32 = 0.1;

fn mk(n: usize, seed: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            ((((i.wrapping_mul(2654435761).wrapping_add(seed)) % 1000) as f32) / 500.0 - 1.0)
                * scale
        })
        .collect()
}

fn build(b: usize, s: usize, inp: usize, h: usize, bidirectional: bool) -> Graph {
    let f = DType::F32;
    let dirs = if bidirectional { 2 } else { 1 };
    let mut g = Graph::new("lstm_metal_wide");
    let x = g.input("x", Shape::new(&[b, s, inp], f));
    let wih = g.input("w_ih", Shape::new(&[dirs * 4 * h * inp], f));
    let whh = g.input("w_hh", Shape::new(&[dirs * 4 * h * h], f));
    let bias = g.input("bias", Shape::new(&[dirs * 4 * h], f));
    let out = g.add_node(
        Op::Lstm {
            hidden_size: h,
            num_layers: 1,
            bidirectional,
            carry: false,
        },
        vec![x, wih, whh, bias],
        Shape::new(&[b, s, dirs * h], f),
    );
    g.set_outputs(vec![out]);
    g
}

/// `(max_abs_delta, nan_count)` for Metal vs CPU on one shape.
fn compare(b: usize, s: usize, inp: usize, h: usize, bidir: bool) -> (f32, usize) {
    let dirs = if bidir { 2 } else { 1 };
    let xd = mk(b * s * inp, 1, 1.0);
    let wihd = mk(dirs * 4 * h * inp, 2, 1.0);
    let whhd = mk(dirs * 4 * h * h, 3, WHH_SCALE);
    let bd = mk(dirs * 4 * h, 4, 1.0);
    let slots: [(&str, &[f32]); 4] = [("x", &xd), ("w_ih", &wihd), ("w_hh", &whhd), ("bias", &bd)];
    let run = |dev| {
        let mut c = Session::new(dev).compile(build(b, s, inp, h, bidir));
        c.run(&slots).pop().unwrap()
    };
    let cpu = run(Device::Cpu);
    let gpu = run(Device::Metal);
    assert_eq!(cpu.len(), gpu.len());
    let nans = gpu.iter().filter(|v| !v.is_finite()).count();
    let maxd = cpu
        .iter()
        .zip(&gpu)
        .filter(|(_, g)| g.is_finite())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, f32::max);
    (maxd, nans)
}

/// Hidden sizes spanning one SIMD group (32) up to the kernel's ceiling, at short
/// and long sequences. NaN here would mean threads that never ran left their
/// `h_sh` slots uninitialized — the dispatch outrunning
/// `maxTotalThreadsPerThreadgroup`.
#[test]
fn lstm_metal_matches_cpu_across_hidden_and_seq() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Metal, "metal") {
        eprintln!("skip: no Metal device");
        return;
    }
    let mut failures = Vec::new();
    for &h in &[16usize, 32, 33, 40, 64, 128, 256, 512] {
        for &s in &[8usize, 48, 256] {
            for &bidir in &[false, true] {
                let (maxd, nans) = compare(1, s, 12, h, bidir);
                let tag = if bidir { "bidir" } else { "uni  " };
                println!("h={h:<4} s={s:<4} {tag} max_abs={maxd:.3e} nans={nans}");
                if nans > 0 || maxd > 1e-4 {
                    failures.push(format!("h={h} s={s} {tag}: max_abs={maxd:.3e} nans={nans}"));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "Metal LSTM diverged:\n{}",
        failures.join("\n")
    );
}

/// The rlx-ocr2 recognizer's shape: bidirectional h=128 over a text line.
#[test]
fn lstm_metal_recognizer_shape() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Metal, "metal") {
        eprintln!("skip: no Metal device");
        return;
    }
    for &(s, inp) in &[(40usize, 192usize), (160, 192), (160, 128)] {
        let (maxd, nans) = compare(1, s, inp, 128, true);
        println!("recognizer seq={s} in={inp}: max_abs={maxd:.3e} nans={nans}");
        assert_eq!(nans, 0, "NaN at seq={s} in={inp}");
        assert!(maxd < 1e-4, "delta {maxd:.3e} at seq={s} in={inp}");
    }
}

/// Batch > 1 puts several threadgroups in flight at once.
#[test]
fn lstm_metal_batched() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Metal, "metal") {
        eprintln!("skip: no Metal device");
        return;
    }
    for &b in &[2usize, 8] {
        let (maxd, nans) = compare(b, 64, 24, 128, true);
        println!("batch={b}: max_abs={maxd:.3e} nans={nans}");
        assert_eq!(nans, 0, "NaN at batch={b}");
        assert!(maxd < 1e-4, "delta {maxd:.3e} at batch={b}");
    }
}
