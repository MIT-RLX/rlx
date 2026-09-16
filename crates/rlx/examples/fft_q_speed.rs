// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What `Op::FftQ` actually costs, per backend and against the float `Op::Fft`.
//!
//! See also `rlx-cpu --example fft_q_kernel` for the two kernels compared with
//! no graph around them.
//!
//! Every backend but CPU runs this op as a *host fallback*: the same CPU kernel,
//! reached through a device→host copy and a host→device copy. So the useful
//! questions are (a) how much the round trip costs, and (b) whether fixed-point
//! beats float on the CPU at all — which is what decides whether it is worth
//! using outside the embedded target it was built for.
//!
//! Run: cargo run --release --example fft_q_speed --features metal,gpu,cuda,rocm

use rlx_ir::fft::{FftNorm, FftQScale};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};
use std::time::Instant;

fn build_q(outer: usize, n: usize, scale: FftQScale) -> Graph {
    let mut g = Graph::new("fft_q");
    let x = g.input("x", Shape::new(&[outer, 2 * n], DType::F32));
    let xi = g.add_node(
        Op::Cast { to: DType::I32 },
        vec![x],
        Shape::new(&[outer, 2 * n], DType::I32),
    );
    let y = g.fft_q(xi, false, FftNorm::Backward, scale);
    let yf = g.add_node(
        Op::Cast { to: DType::F32 },
        vec![y],
        Shape::new(&[outer, 2 * n], DType::F32),
    );
    g.set_outputs(vec![yf]);
    g
}

/// The two `Cast` nodes alone — the tax `build_q` pays that `build_f` does not.
fn build_cast_only(outer: usize, n: usize) -> Graph {
    let mut g = Graph::new("cast_only");
    let x = g.input("x", Shape::new(&[outer, 2 * n], DType::F32));
    let xi = g.add_node(
        Op::Cast { to: DType::I32 },
        vec![x],
        Shape::new(&[outer, 2 * n], DType::I32),
    );
    let yf = g.add_node(
        Op::Cast { to: DType::F32 },
        vec![xi],
        Shape::new(&[outer, 2 * n], DType::F32),
    );
    g.set_outputs(vec![yf]);
    g
}

fn build_f(outer: usize, n: usize) -> Graph {
    let mut g = Graph::new("fft_f");
    let x = g.input("x", Shape::new(&[outer, 2 * n], DType::F32));
    let y = g.fft(x, false);
    g.set_outputs(vec![y]);
    g
}

/// Minimum of `reps` timed runs after 5 warmups, in microseconds.
///
/// Minimum, not median: these are latency measurements on shared machines, and
/// the floor is the number that reflects the work rather than the contention.
/// A median on a loaded rig produced a negative "net" cost, which is how this
/// benchmark announced it was measuring noise.
fn time_graph(dev: Device, g: Graph, xf: &[f32], reps: usize) -> Option<f64> {
    if !rlx_runtime::is_available(dev) {
        return None;
    }
    let mut sess = Session::new(dev).compile(g);
    for _ in 0..5 {
        let _ = sess.run(&[("x", xf)]);
    }
    let mut us: Vec<f64> = (0..reps)
        .map(|_| {
            let t = Instant::now();
            let _ = sess.run(&[("x", xf)]);
            t.elapsed().as_secs_f64() * 1e6
        })
        .collect();
    us.sort_by(f64::total_cmp);
    Some(us[0])
}

fn main() {
    let devices = [
        ("cpu", Device::Cpu),
        ("metal", Device::Metal),
        ("wgpu", Device::Gpu),
        ("cuda", Device::Cuda),
        ("rocm", Device::Rocm),
    ];

    for &(outer, n) in &[(1usize, 256usize), (1, 1024), (64, 1024), (256, 1024)] {
        let elems = outer * 2 * n;
        let xf: Vec<f32> = (0..elems)
            .map(|i| ((i as f64 * 0.13).sin() * 15000.0).round() as f32)
            .collect();
        let reps = if elems > 200_000 { 51 } else { 201 };

        println!("\n=== outer={outer} n={n}  ({elems} f32) ===");
        println!(
            "{:<7} {:>10} {:>10} {:>10} {:>10} {:>8}",
            "backend", "FftQ us", "casts us", "FftQ-net", "Fft(f32)", "net/f"
        );
        for (name, dev) in devices {
            // PerStage keeps the result inside f32's exact-integer range at n=1024.
            let q = time_graph(dev, build_q(outer, n, FftQScale::PerStage), &xf, reps);
            let c = time_graph(dev, build_cast_only(outer, n), &xf, reps);
            let f = time_graph(dev, build_f(outer, n), &xf, reps);
            match (q, c, f) {
                (Some(q), Some(c), Some(f)) => {
                    let net = q - c;
                    println!(
                        "{name:<7} {q:>10.1} {c:>10.1} {net:>10.1} {f:>10.1} {:>7.2}x",
                        net / f
                    );
                }
                _ => println!(
                    "{name:<7} {:>10} {:>10} {:>10} {:>10} {:>8}",
                    "n/a", "n/a", "n/a", "n/a", ""
                ),
            }
        }
    }
}
