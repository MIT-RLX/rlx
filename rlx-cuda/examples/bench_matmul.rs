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

//! Minimal CUDA-only matmul micro-bench. No rlx-cpu / rlx-runtime —
//! avoids the CPU-BLAS link tax on hosts without Accelerate / OpenBLAS.
//!
//! cargo run --release -p rlx-cuda --example bench_matmul

use std::time::Instant;

use rlx_cuda::backend::CudaExecutable;
use rlx_ir::{DType, Graph, Shape};

fn bench(m: usize, k: usize, n: usize, warmup: usize, iters: usize) {
    let mut g = Graph::new("mm");
    let x = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[k, n], DType::F32));
    let y = g.matmul(x, w, Shape::new(&[m, n], DType::F32));
    g.set_outputs(vec![y]);

    let mut exe = CudaExecutable::compile(g);
    let wv: Vec<f32> = (0..k * n).map(|i| (i as f32) * 1e-3).collect();
    exe.set_param("w", &wv);
    let xv: Vec<f32> = (0..m * k).map(|i| (i as f32) * 1e-3).collect();

    for _ in 0..warmup {
        let _ = exe.run(&[("x", &xv)]);
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = exe.run(&[("x", &xv)]);
    }
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    let flops = 2.0 * (m * k * n) as f64;
    let gflops = flops / dt / 1e9;
    println!(
        "  M={:>5} K={:>5} N={:>5}   {:>8.3} ms   {:>8.1} GFLOP/s",
        m,
        k,
        n,
        dt * 1e3,
        gflops
    );
}

fn main() {
    if !rlx_cuda::is_available() {
        println!("CUDA not available on this host — exiting.");
        return;
    }
    println!("rlx-cuda matmul bench");
    println!("---------------------");
    let cases: &[(usize, usize, usize)] = &[
        (128, 128, 128),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8, 4096, 4096),
        (1024, 4096, 4096),
    ];
    for &(m, k, n) in cases {
        bench(m, k, n, /*warmup*/ 3, /*iters*/ 20);
    }
}
