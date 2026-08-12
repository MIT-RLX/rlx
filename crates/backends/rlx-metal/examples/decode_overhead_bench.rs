// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Steady-state cost of one decode-shaped forward pass, split encode/commit/wait.
//!
//! `sgemm_check` shows the decode regime is not compute-bound: at `m = 6` a pass
//! takes ~0.37 ms while `m = 60` — ten times the work — takes ~0.28 ms. A
//! non-monotonic curve like that means a fixed per-pass overhead dominates, and
//! tuning the kernel cannot help until you know which part of the overhead it is.
//!
//! The existing `RLX_METAL_TRACE=1` split reports that, but every test in the
//! suite runs its graph once or twice, so the only samples available are the
//! first — which carry one-time MSL compilation and pipeline creation (~790 ms)
//! and say nothing about steady state. This runs the *same* compiled executable
//! many times so the warm samples are the ones you read.
//!
//! Run with the split enabled and aggregate the warm iterations:
//!
//! ```text
//! RLX_METAL_TRACE=1 cargo run --release -p rlx-metal --example decode_overhead_bench 2>&1 \
//!   | grep metal-trace | tail -n +11 \
//!   | sed -E 's/.*encode=([0-9.]+).*commit=([0-9.]+).*wait=([0-9.]+).*/\1 \2 \3/' \
//!   | awk '{e+=$1;c+=$2;w+=$3;n++} END {printf "encode %.1fus commit %.1fus wait %.1fus (n=%d)\n", e/n,c/n,w/n,n}'
//! ```
#![cfg(target_os = "macos")]

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    // Decode shape: one token through a couple of projections. Small enough
    // that compute cannot plausibly dominate, which is the point.
    let (m, k, n) = (1usize, 768usize, 768usize);

    let mut g = Graph::new("decode_overhead");
    let x = g.input("x", Shape::new(&[m, k], DType::F32));
    let w1 = g.param("w1", Shape::new(&[k, n], DType::F32));
    let w2 = g.param("w2", Shape::new(&[n, n], DType::F32));
    let h = g.add_node(Op::MatMul, vec![x, w1], Shape::new(&[m, n], DType::F32));
    let h = g.add_node(
        Op::Activation(rlx_ir::op::Activation::Silu),
        vec![h],
        Shape::new(&[m, n], DType::F32),
    );
    let out = g.add_node(Op::MatMul, vec![h, w2], Shape::new(&[m, n], DType::F32));
    g.set_outputs(vec![out]);

    // Marginal cost of one more dispatch: same shapes, N chained matmuls. If
    // the pass scales with op count rather than with FLOPs, the cost is
    // per-dispatch and the fix is fusion, not faster math.
    let chain_p50 = |ops: usize| -> f64 {
        let mut g = Graph::new("chain");
        let x = g.input("x", Shape::new(&[m, k], DType::F32));
        let w = g.param("w", Shape::new(&[k, n], DType::F32));
        let mut cur = x;
        for _ in 0..ops {
            cur = g.add_node(Op::MatMul, vec![cur, w], Shape::new(&[m, n], DType::F32));
        }
        g.set_outputs(vec![cur]);
        let mut c = Session::new(Device::Metal).compile(g);
        c.set_param("w", &vec![0.01f32; k * n]);
        let xd = vec![0.5f32; m * k];
        for _ in 0..10 {
            let _ = c.run(&[("x", xd.as_slice())]);
        }
        let mut v: Vec<f64> = (0..iters)
            .map(|_| {
                let t = std::time::Instant::now();
                let _ = c.run(&[("x", xd.as_slice())]);
                t.elapsed().as_secs_f64() * 1e6
            })
            .collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };

    let mut compiled = Session::new(Device::Metal).compile(g);
    compiled.set_param("w1", &vec![0.01f32; k * n]);
    compiled.set_param("w2", &vec![0.01f32; n * n]);
    let x_data = vec![0.5f32; m * k];

    // Warm-up: the first passes pay MSL compilation and pipeline creation, and
    // are exactly the samples that made the existing numbers unreadable.
    for _ in 0..10 {
        let _ = compiled.run(&[("x", x_data.as_slice())]);
    }

    let t0 = std::time::Instant::now();
    let mut per_iter = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = std::time::Instant::now();
        let _ = compiled.run(&[("x", x_data.as_slice())]);
        per_iter.push(t.elapsed().as_secs_f64() * 1e6);
    }
    let total = t0.elapsed().as_secs_f64() * 1e6;

    per_iter.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| per_iter[((per_iter.len() - 1) as f64 * p) as usize];
    println!("decode-shaped pass [m={m} k={k} n={n}], {iters} warm iterations");
    println!(
        "  total/iter  mean {:.1}µs  p50 {:.1}µs  p90 {:.1}µs  min {:.1}µs",
        total / iters as f64,
        pct(0.50),
        pct(0.90),
        per_iter[0]
    );
    println!("  (encode/commit/wait split: re-run with RLX_METAL_TRACE=1)");

    // The floor: submit an empty command buffer and wait. Whatever this costs
    // is Metal's submit→complete round trip and is not reclaimable by anything
    // we do on the encode side. The gap between it and the figure above is the
    // part that is actually ours.
    let dev = rlx_metal::device::metal_device().expect("Metal device");
    let mut empty = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = std::time::Instant::now();
        rlx_metal::mtl::autoreleasepool(|| {
            let cb = dev.queue.new_command_buffer();
            let enc = cb.compute_command_encoder_with_dispatch_type(
                rlx_metal::mtl::MTLDispatchType::Serial,
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });
        empty.push(t.elapsed().as_secs_f64() * 1e6);
    }
    empty.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let e50 = empty[empty.len() / 2];
    println!("empty command buffer (submit→complete floor), {iters} iterations");
    println!(
        "  p50 {:.1}µs  min {:.1}µs   → floor is {:.0}% of the decode pass p50",
        e50,
        empty[0],
        100.0 * e50 / pct(0.50)
    );

    let (c1, c2, c4) = (chain_p50(1), chain_p50(2), chain_p50(4));
    println!("matmul chain p50 (same shapes, more dispatches)");
    println!("  1 op {c1:.1}µs   2 ops {c2:.1}µs   4 ops {c4:.1}µs");
    println!(
        "  marginal cost per extra dispatch ≈ {:.1}µs  (FLOPs scale with op count too, \
         but at m=1 the math is ~0.6µs total)",
        (c4 - c1) / 3.0
    );
}
