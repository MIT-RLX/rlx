// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Licensed under the GNU General Public License, version 3.

//! CPU FFT: radix-2 vs radix-4 for pure powers of four, toggled in-process via
//! RLX_FFT_RADIX4. Serial (RLX_FFT_CPU_PARALLEL=0) to isolate the per-row kernel
//! — radix-4 halves the stage count (log4 vs log2), so fewer array sweeps.
//!
//! ```sh
//! cargo run -p rlx-bench --release --example bench_fft_cpu_radix4
//! ```

use std::io::Write;

use rlx_driver::Device;
use rlx_ir::{DType, Graph, Op, Shape, Tick};
use rlx_runtime::{CompiledGraph, Session};

fn fft_graph(batch: usize, n: usize) -> Graph {
    let mut g = Graph::new("cpu_r4");
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

fn setv(k: &str, v: &str) {
    // SAFETY: single-threaded benchmark driver toggling process-local gates.
    unsafe {
        std::env::set_var(k, v);
    }
}

fn main() {
    setv("RLX_FFT_CPU_PARALLEL", "0"); // isolate the per-row kernel
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

    println!("CPU FFT (serial): radix-2 vs radix-4, pure powers of four");
    println!(
        "  {:>5} {:>6}  {:>11} {:>11}  {:>8}",
        "n", "batch", "radix2 µs", "radix4 µs", "speedup"
    );
    let _ = std::io::stdout().flush();

    for &n in &[64usize, 256, 1024, 4096, 16384] {
        for &batch in &[64usize, 256] {
            let mut c = Session::new(Device::Cpu).compile(fft_graph(batch, n));
            setv("RLX_FFT_RADIX4", "0");
            let r2 = median(&mut c);
            setv("RLX_FFT_RADIX4", "1");
            let r4 = median(&mut c);
            println!(
                "  {n:>5} {batch:>6}  {:>9.1}µs {:>9.1}µs  {:>7.2}×",
                r2 as f64 / 1000.0,
                r4 as f64 / 1000.0,
                r2 as f64 / r4.max(1) as f64,
            );
            let _ = std::io::stdout().flush();
        }
        println!();
    }
}
