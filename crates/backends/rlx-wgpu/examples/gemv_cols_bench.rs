// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Decode GEMV (`m = 1`) time per output shape for one `matmul_gemv` column
//! width, checked against the CPU backend.
//!
//! The column width is process-wide (`RLX_WGPU_GEMV_COLS` = 32 / 16 / 8, read
//! once), so compare widths by **alternating processes**:
//!
//! ```sh
//! for i in 1 2 3; do for c in 32 16 8; do
//!   RLX_WGPU_GEMV_COLS=$c cargo run --release -p rlx-wgpu --example gemv_cols_bench
//! done; done
//! ```
//! Each line: `cols m k n min_ms median_ms max_err`. `min` is the least
//! contended sample — the number to rank by on a shared GPU.

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session, is_available};
use std::time::Instant;

const SHAPES: &[(usize, usize, usize)] = &[
    (1, 3072, 1024),
    (1, 2048, 1024),
    (1, 1024, 1024),
    (1, 1024, 3072),
    (1, 1024, 4096),
    (1, 1024, 6144),
    (1, 4096, 4096),
];
/// Distinct weight matrices per graph: keeps the working set past the SLC and
/// amortizes submit + readback.
const REPS: usize = 12;
const ITERS: usize = 9;

fn fill(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.37 + seed).sin() * 0.5)
        .collect()
}

fn graph(m: usize, k: usize, n: usize) -> Graph {
    let mut g = Graph::new("gemv_bench");
    let x = g.input("x", Shape::new(&[m, k], DType::F32));
    let outs = (0..REPS)
        .map(|i| {
            let w = g.param(format!("w{i}"), Shape::new(&[k, n], DType::F32));
            g.matmul(x, w, Shape::new(&[m, n], DType::F32))
        })
        .collect();
    g.set_outputs(outs);
    g
}

fn main() {
    if !is_available(Device::Gpu) {
        eprintln!("no wgpu adapter");
        return;
    }
    let cols = rlx_ir::env::var("RLX_WGPU_GEMV_COLS").unwrap_or_else(|| "32".into());
    for &(m, k, n) in SHAPES {
        let x = fill(m * k, 0.7);
        let ws: Vec<Vec<f32>> = (0..REPS).map(|i| fill(k * n, 0.3 + i as f32)).collect();
        let run = |dev: Device| {
            let mut c = Session::new(dev).compile(graph(m, k, n));
            for (i, w) in ws.iter().enumerate() {
                c.set_param(&format!("w{i}"), w);
            }
            c
        };
        let mut cpu = run(Device::Cpu);
        let want = cpu.run(&[("x", &x)]);
        let mut gpu = run(Device::Gpu);
        let _ = gpu.run(&[("x", &x)]);
        let mut samples = Vec::with_capacity(ITERS);
        let mut got = Vec::new();
        for _ in 0..ITERS {
            let t = Instant::now();
            got = gpu.run(&[("x", &x)]);
            samples.push(t.elapsed().as_secs_f64() * 1e3 / REPS as f64);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let max_err = got
            .iter()
            .zip(&want)
            .flat_map(|(g, w)| g.iter().zip(w).map(|(a, b)| (a - b).abs()))
            .fold(0f32, f32::max);
        println!(
            "cols={cols} {m} {k} {n} min={:.4} median={:.4} max_err={max_err:.2e}",
            samples[0],
            samples[ITERS / 2]
        );
    }
}
