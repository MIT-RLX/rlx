// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native Metal decode kernels — `Op::ArgMax`/`Op::ArgMin` last-axis reduction
//! and `Op::Sample` (temperature / top-k / top-p / Philox) vs the CPU
//! reference. These used to read the full logits row (~vocab) back to host
//! every decode token; they now run entirely on-GPU:
//!   * `argreduce_lastaxis` — one threadgroup folds a `[outer, N]` row.
//!   * `sample_logits`      — one threadgroup per batch row; Philox stream and
//!     filtering match `rlx-cpu sample_row` bit-for-bit.
//!
//! All cases run in ONE `#[test]` so the independent `Session`s execute
//! serially (matches the prefill-parity harness convention).

#![cfg(target_os = "macos")]

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

/// ArgMax/ArgMin over the LAST axis of `[outer, n]` (inner == 1, the decode
/// logits shape that routes to the cooperative `argreduce_lastaxis` kernel).
fn argreduce_lastaxis(outer: usize, n: usize, is_max: bool) -> (Vec<f32>, Vec<f32>) {
    let mut g = Graph::new("argmax_lastaxis");
    let x = g.input("x", Shape::new(&[outer, n], DType::F32));
    let op = if is_max {
        Op::ArgMax {
            axis: 1,
            keep_dim: false,
        }
    } else {
        Op::ArgMin {
            axis: 1,
            keep_dim: false,
        }
    };
    let y = g.add_node(op, vec![x], Shape::new(&[outer], DType::F32));
    g.set_outputs(vec![y]);

    // Deterministic pseudo-random with unique per-row extrema.
    let x_data: Vec<f32> = (0..outer * n)
        .map(|i| (((i as u64).wrapping_mul(2654435761) % 100003) as f32) * 0.001 - 50.0)
        .collect();

    let mut m = Session::new(Device::Metal).compile(g.clone());
    let metal = m.run(&[("x", x_data.as_slice())]).remove(0);
    let mut c = Session::new(Device::Cpu).compile(g);
    let cpu = c.run(&[("x", x_data.as_slice())]).remove(0);
    (metal, cpu)
}

fn build_sample(b: usize, v: usize, top_k: usize, top_p: f32, temp: f32, seed: u64) -> Graph {
    let mut g = Graph::new("sample");
    let logits = g.input("logits", Shape::new(&[b, v], DType::F32));
    let y = g.sample(
        logits,
        top_k,
        top_p,
        temp,
        seed,
        Shape::new(&[b], DType::F32),
    );
    g.set_outputs(vec![y]);
    g
}

fn sample_logits(b: usize, v: usize) -> Vec<f32> {
    // Non-uniform, distinct-per-row logits so top-k/top-p select a real subset.
    (0..b * v)
        .map(|i| {
            let r = (i % v) as f32;
            (r * 0.37).sin() * 3.0 + (r * 0.013).cos() * 1.5 + ((i / v) as f32) * 0.1
        })
        .collect()
}

fn run_sample(device: Device, b: usize, v: usize, k: usize, p: f32, t: f32, seed: u64) -> Vec<f32> {
    let data = sample_logits(b, v);
    let mut exe = Session::new(device).compile(build_sample(b, v, k, p, t, seed));
    exe.run(&[("logits", data.as_slice())]).remove(0)
}

#[test]
fn metal_argmax_and_sample_match_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }

    // ── 1. ArgMax/ArgMin last-axis EXACT vs CPU (decode-shaped rows) ──
    for &(outer, n) in &[(1usize, 4096usize), (3, 131072), (8, 257), (5, 1)] {
        for is_max in [true, false] {
            let (metal, cpu) = argreduce_lastaxis(outer, n, is_max);
            assert_eq!(metal.len(), outer);
            for (j, (a, b)) in metal.iter().zip(&cpu).enumerate() {
                assert_eq!(
                    *a as i64, *b as i64,
                    "argreduce[{j}] is_max={is_max} outer={outer} n={n}: metal {a} vs cpu {b}"
                );
            }
        }
    }
    eprintln!("argmax/argmin last-axis: EXACT vs CPU");

    // ── 2. Sample greedy (top_k=1) == ArgMax over the same logits ──
    for &(b, v) in &[(1usize, 512usize), (4, 4096)] {
        let greedy = run_sample(Device::Metal, b, v, 1, 1.0, 1.0, 1);
        // ArgMax of identical logits.
        let mut g = Graph::new("am");
        let logits = g.input("logits", Shape::new(&[b, v], DType::F32));
        let y = g.add_node(
            Op::ArgMax {
                axis: 1,
                keep_dim: false,
            },
            vec![logits],
            Shape::new(&[b], DType::F32),
        );
        g.set_outputs(vec![y]);
        let mut m = Session::new(Device::Metal).compile(g);
        let am = m
            .run(&[("logits", sample_logits(b, v).as_slice())])
            .remove(0);
        assert_eq!(
            greedy, am,
            "greedy(top_k=1) must equal argmax (b={b} v={v})"
        );
    }
    eprintln!("sample greedy(top_k=1) == argmax");

    // ── 3. temperature-0 greedy == argmax (other code path) ──
    {
        let (b, v) = (4usize, 1024usize);
        let g0 = run_sample(Device::Metal, b, v, 0, 1.0, 0.0, 5);
        let c0 = run_sample(Device::Cpu, b, v, 0, 1.0, 0.0, 5);
        assert_eq!(g0, c0, "temp0 greedy metal vs cpu");
    }

    // ── 4. top-k / top-p / combined: EXACT token vs CPU (fixed seed) ──
    // Exact match implies the filtered probability distribution agrees to
    // within float error (the inverse-CDF would otherwise diverge); this is
    // strictly stronger than "probs match within 1e-4".
    let cases: &[(&str, usize, usize, usize, f32, f32, u64)] = &[
        ("topk8", 4, 512, 8, 1.0, 0.8, 42),
        ("topk40", 2, 32000, 40, 1.0, 0.8, 7),
        ("topp", 3, 4096, 0, 0.9, 1.0, 123),
        ("topk-topp", 5, 8192, 50, 0.95, 0.7, 999),
        ("temp-only", 4, 2048, 0, 1.0, 1.2, 314),
    ];
    for &(name, b, v, k, p, t, seed) in cases {
        let metal = run_sample(Device::Metal, b, v, k, p, t, seed);
        let cpu = run_sample(Device::Cpu, b, v, k, p, t, seed);
        assert_eq!(metal, cpu, "sample {name}: metal {metal:?} vs cpu {cpu:?}");
        eprintln!("sample {name}: {metal:?} == cpu");
    }
}
