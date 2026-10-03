// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-element cost of one optimizer step.
//!
//! The host optimizer is 40–55% of a training step at MNIST-MLP scale and does
//! not vary with the device, so this is the number to move. Run with
//! `--release`; a debug build measures the debug build.
//!
//!     cargo run --release -p rlx-optim --example step_bench
//!     cargo run --release -p rlx-optim --example step_bench --features parallel

use std::time::Instant;

use rlx_optim::{Adam, AdamW, Lion, Optimizer, Sgd};

/// A deterministic spread of magnitudes, so the timing is not measuring
/// denormal or all-equal fast paths.
fn fill(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / 8_388_608.0 - 1.0) * 0.1
        })
        .collect()
}

fn bench(label: &str, n: usize, mut opt: Box<dyn Optimizer>) -> f64 {
    let shape = [n];
    let mut param = fill(n, 7);
    let grad = fill(n, 11);
    // Warm: the first call allocates the moment buffers.
    for _ in 0..5 {
        opt.step("w", &shape, &mut param, &grad);
        opt.end_iteration();
    }
    let iters = (200_000_000 / n).clamp(20, 2000);
    let start = Instant::now();
    for _ in 0..iters {
        opt.step("w", &shape, &mut param, &grad);
        opt.end_iteration();
    }
    let per_step = start.elapsed().as_secs_f64() / iters as f64;
    let per_elem_ns = per_step * 1e9 / n as f64;
    println!(
        "  {label:22} {:>9.1} µs/step  {per_elem_ns:>6.2} ns/element  {:>6.2} GB/s",
        per_step * 1e6,
        // Adam touches p, m, v (read+write) and g (read): 7 × 4 bytes per element.
        (n as f64 * 7.0 * 4.0) / per_step / 1e9
    );
    per_elem_ns
}

fn main() {
    for n in [64 * 1024usize, 1024 * 1024, 8 * 1024 * 1024] {
        println!(
            "\n{} elements ({:.1} MiB per buffer)",
            n,
            (n * 4) as f64 / 1048576.0
        );
        bench("adamw (f64 default)", n, Box::new(AdamW::new(1e-3)));
        bench(
            "adamw f32Math",
            n,
            Box::new(AdamW::new(1e-3).with_f32_math(true)),
        );
        bench("adam (f64 default)", n, Box::new(Adam::new(1e-3)));
        bench(
            "adam f32Math",
            n,
            Box::new(Adam::new(1e-3).with_f32_math(true)),
        );
        bench("sgd+momentum", n, {
            let mut o = Sgd::new(1e-3);
            o.momentum = 0.9;
            Box::new(o)
        });
        bench("lion", n, Box::new(Lion::new(1e-3)));
    }
}
