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

//! Demonstrates how Metal's per-commit GPU sync latency dominates small-op
//! latency, and how pipelining N commits with a single sync amortises it.
//!
//! Background: rlx's per-op trace data shows that on Apple Silicon a single
//! command-buffer submit costs ~150 µs of `wait_until_completed` regardless
//! of how much work it contains. ICB / persistent kernels / JIT cache all
//! attack `encode` (3-25 µs); they don't change `wait`. The actual lever for
//! small-op throughput is to commit many forward passes back-to-back and
//! only wait once.
//!
//! Uses the generic `CompiledGraph` API — same code works on CPU (where
//! the pipelining methods fall back to serial `run`).
//!
//! cargo run --release -p rlx-runtime --example pipeline_demo --features metal

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    use rlx_ir::op::Activation;
    use rlx_ir::*;
    use rlx_runtime::{Device, Session};
    use std::time::Instant;

    let n = 4096;
    let build = || {
        let mut g = Graph::new("silu_chain");
        let f = DType::F32;
        let x = g.input("x", Shape::new(&[n], f));
        let y = g.activation(Activation::Silu, x, Shape::new(&[n], f));
        g.set_outputs(vec![y]);
        g
    };
    let x_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001 - 0.5).collect();

    let mut metal = Session::new(Device::Metal).compile(build());

    // Warmup
    for _ in 0..5 {
        let _ = metal.run(&[("x", &x_data)]);
    }

    let n_iters = 200usize;

    // ── Mode A: serial run() ──
    let t0 = Instant::now();
    for _ in 0..n_iters {
        let _ = metal.run(&[("x", &x_data)]);
    }
    let serial_total = t0.elapsed();

    // ── Mode B: commit_no_wait + sync_pending ──
    let t0 = Instant::now();
    for _ in 0..n_iters {
        metal.commit_no_wait(&[("x", &x_data)]);
    }
    metal.sync_pending();
    let pipelined_total = t0.elapsed();

    println!("\nN = {n_iters} iterations of silu over a {n}-element vector:");
    println!(
        "  serial             : {:>8} µs total, {:>6.2} µs/iter",
        serial_total.as_micros(),
        serial_total.as_secs_f64() * 1e6 / n_iters as f64
    );
    println!(
        "  commit_no_wait     : {:>8} µs total, {:>6.2} µs/iter  (outputs stomped)",
        pipelined_total.as_micros(),
        pipelined_total.as_secs_f64() * 1e6 / n_iters as f64
    );
    let speedup_b = serial_total.as_secs_f64() / pipelined_total.as_secs_f64();
    println!("  speedup            : {speedup_b:.2}x");

    // ── Mode C: run_pipelined with per-commit output snapshots ──
    // Production-ready API. Each commit gets its own output buffer; outputs
    // survive subsequent commits stomping the shared arena.
    let n_pipe = 64usize;
    let inputs_per_run: Vec<Vec<(&str, &[f32])>> = (0..n_pipe)
        .map(|_| vec![("x", x_data.as_slice())])
        .collect();

    let _ = metal.run_pipelined(&inputs_per_run); // warmup

    let t0 = Instant::now();
    let outs = metal.run_pipelined(&inputs_per_run);
    let pipe_real_total = t0.elapsed();

    println!("\nN = {n_pipe} iterations of run_pipelined (per-run output snapshots):");
    println!(
        "  total              : {:>8} µs total, {:>6.2} µs/iter",
        pipe_real_total.as_micros(),
        pipe_real_total.as_secs_f64() * 1e6 / n_pipe as f64
    );

    // Correctness: every run produces identical outputs (same input each
    // time) and matches what `serial run()` produces.
    let reference = metal.run(&[("x", &x_data)]);
    let mut max_err = 0f32;
    for run_out in &outs {
        for (a, b) in run_out[0].iter().zip(reference[0].iter()) {
            max_err = max_err.max((a - b).abs());
        }
    }
    println!(
        "  per-run output max_err vs serial reference: {max_err:.2e} {}",
        if max_err < 1e-6 { "✓" } else { "✗ FAIL" }
    );

    println!();
    println!("(serial pays ~150µs of wait_until_completed per call. Pipelined");
    println!(" amortizes it across N commits — the per-iter cost approaches");
    println!(" pure encode + GPU compute, ~10-30 µs.)");
    println!();
    println!("Same API works on CPU — there pipelining is a no-op (BLAS is");
    println!(" synchronous), so the trait defaults preserve correctness without");
    println!(" perf overhead.");
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("pipeline_demo requires --features metal on macOS");
}
