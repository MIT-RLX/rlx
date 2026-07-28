// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU batched FFT: serial row loop vs rayon-parallel batch, toggled in-process
//! via RLX_FFT_CPU_PARALLEL. Rows are independent, so batch parallelism should
//! scale with core count for the batched-block workloads (welch/learned FFT,
//! GPU-resident-alternative STFT) that run native CPU FFT.
//!
//! ```sh
//! cargo run -p rlx-bench --release --example bench_fft_cpu_parallel --features cpu
//! ```

use std::io::Write;

use rlx_driver::Device;
use rlx_ir::{DType, Graph, Op, Shape, Tick};
use rlx_runtime::{CompiledGraph, Session};

fn fft_graph(batch: usize, n: usize) -> Graph {
    let mut g = Graph::new("cpu_par");
    let len = batch * n * 2;
    let mut bytes = Vec::with_capacity(len * 4);
    for i in 0..len {
        bytes.extend_from_slice(&((i as f32 * 0.013).sin()).to_le_bytes());
    }
    let x = g.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(&[batch, n * 2], DType::F32),
    );
    let y = g.fft(x, false);
    g.set_outputs(vec![y]);
    g
}

fn set_par(on: bool) {
    // SAFETY: single-threaded benchmark driver toggling a process-local gate.
    unsafe {
        std::env::set_var("RLX_FFT_CPU_PARALLEL", if on { "1" } else { "0" });
    }
}

fn main() {
    let empty: &[(&str, &[f32])] = &[];
    let median = |c: &mut CompiledGraph| -> u64 {
        for _ in 0..3 {
            let _ = c.run(empty);
        }
        let mut s = Vec::with_capacity(15);
        for _ in 0..15 {
            let t0 = Tick::now();
            let _ = c.run(empty);
            s.push(Tick::now().elapsed_ns(t0));
        }
        s.sort_unstable();
        s[s.len() / 2]
    };

    println!("CPU FFT: serial vs rayon-parallel batch (single forward FFT)");
    println!(
        "  {:>5} {:>6}  {:>12} {:>12}  {:>8}",
        "n", "batch", "serial µs", "parallel µs", "speedup"
    );
    let _ = std::io::stdout().flush();

    for &n in &[256usize, 512, 1024, 2048, 4096] {
        for &batch in &[16usize, 64, 256, 1024] {
            let mut c = Session::new(Device::Cpu).compile(fft_graph(batch, n));
            set_par(false);
            let ser = median(&mut c);
            set_par(true);
            let par = median(&mut c);
            let per = |t: u64| t as f64 / 1000.0;
            println!(
                "  {n:>5} {batch:>6}  {:>10.1}µs {:>10.1}µs  {:>7.2}×",
                per(ser),
                per(par),
                ser as f64 / par.max(1) as f64,
            );
            let _ = std::io::stdout().flush();
        }
        println!();
    }
}
