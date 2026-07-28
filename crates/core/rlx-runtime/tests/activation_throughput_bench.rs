// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Throughput benchmark for the warm-tier activation kernels — the perf side of
//! the codegen north-star check.
//!
//! The standalone (un-fused) activation kernels are **memory-bandwidth bound**:
//! each element is one load + one store around a handful of ALU ops. So the
//! *arithmetic* — whether trivial `relu` or the generated A&S `gelu`/`erf`
//! polynomial — is hidden under the memory traffic, and generating it from the
//! `rlxsl` manifest costs no throughput vs a hand-written kernel. This bench
//! makes that measurable: on a GPU, `gelu`/`erf` track `relu` (ratio ≈ 1×). CPU
//! is shown as a reference (its activations stay hand-written Rust, and `exp`
//! is genuinely compute-heavy there — that cost is inherent to the math, not
//! the codegen).
//!
//! Run it (not part of CI):
//!   cargo test -p rlx-runtime --features gpu,metal \
//!       --test activation_throughput_bench -- --ignored --nocapture
//! On the CUDA/Vulkan rig: `--features cuda,vulkan,gpu`.

#![cfg(feature = "cpu")]

use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session, is_available, supports_graph};
use std::time::Instant;

#[allow(clippy::vec_init_then_push)]
fn available_backends() -> Vec<Device> {
    let mut v: Vec<Device> = Vec::new();
    #[cfg(all(feature = "metal", target_os = "macos"))]
    v.push(Device::Metal);
    #[cfg(all(feature = "mlx", target_os = "macos"))]
    v.push(Device::Mlx);
    #[cfg(feature = "gpu")]
    v.push(Device::Gpu);
    #[cfg(feature = "cuda")]
    v.push(Device::Cuda);
    #[cfg(feature = "rocm")]
    v.push(Device::Rocm);
    #[cfg(feature = "vulkan")]
    v.push(Device::Vulkan);
    v.retain(|&d| is_available(d));
    v
}

fn act_graph(act: Activation, n: usize) -> Graph {
    let mut g = Graph::new("act_bench");
    let x = g.input("x", Shape::new(&[n], DType::F32));
    let y = g.activation(act, x, Shape::new(&[n], DType::F32));
    g.set_outputs(vec![y]);
    g
}

#[test]
#[ignore = "perf benchmark — run with --ignored --nocapture (see file header)"]
fn activation_kernel_throughput() {
    const N: usize = 1 << 23; // 8.4M f32 = 32 MB per buffer
    const ITERS: usize = 30;
    let bytes_per_iter = (N * 4 * 2) as f64; // one load + one store, f32

    let x: Vec<f32> = (0..N).map(|i| ((i % 2000) as f32) * 0.005 - 5.0).collect();

    // Trivial → complex: relu (1 op), silu (exp+div), gelu (~erf poly), erf (A&S).
    let acts = [
        Activation::Relu,
        Activation::Silu,
        Activation::Gelu,
        Activation::Erf,
    ];

    let mut devices = vec![Device::Cpu];
    devices.extend(available_backends());

    println!(
        "\nActivation-kernel throughput — N={N} ({} MB/buffer), {ITERS} iters, end-to-end (incl. H2D/D2H).",
        N * 4 / (1 << 20)
    );
    println!(
        "Warm-tier kernels are bandwidth-bound: on GPU, gelu/erf track relu (~1x) → the\n\
         generated arithmetic is effectively free. CPU is a hand-written reference.\n"
    );
    println!(
        "{:<9} {:<5} {:>11} {:>10} {:>10}",
        "backend", "act", "Gelem/s", "GB/s", "vs relu"
    );

    for &dev in &devices {
        let mut relu_rate = 0.0f64;
        for (i, &act) in acts.iter().enumerate() {
            let g = act_graph(act, N);
            if !supports_graph(dev, &g) {
                continue;
            }
            let mut c = Session::new(dev).compile(g);
            for _ in 0..3 {
                let _ = c.run(&[("x", &x)]); // warmup
            }
            let t = Instant::now();
            for _ in 0..ITERS {
                let _ = c.run(&[("x", &x)]);
            }
            let secs = t.elapsed().as_secs_f64();
            let rate = (N * ITERS) as f64 / secs; // elem/s
            let gbps = bytes_per_iter * ITERS as f64 / secs / 1e9;
            if i == 0 {
                relu_rate = rate;
            }
            let ratio = if relu_rate > 0.0 {
                rate / relu_rate
            } else {
                1.0
            };
            println!(
                "{:<9} {:<5} {:>11.2} {:>10.1} {:>9.2}x",
                format!("{dev:?}"),
                format!("{act:?}"),
                rate / 1e9,
                gbps,
                ratio
            );
        }
    }
    println!();
}
