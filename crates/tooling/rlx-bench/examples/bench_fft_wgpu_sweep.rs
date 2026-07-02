// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Licensed under the GNU General Public License, version 3.

//! wgpu-only FFT batch sweep: multi-kernel (old) vs on-chip radix-4/8 (new),
//! single forward FFT, toggled in-process via RLX_FFT_FAST. The on-chip path
//! uses a 32 KB workgroup; whether it beats the multi-kernel depends entirely
//! on having enough FFT rows (batch) to hide the resulting low occupancy.
//!
//! ```sh
//! cargo run -p rlx-bench --release --example bench_fft_wgpu_sweep \
//!     --features gpu,native-gpu-fft
//! ```

use std::io::Write;

use rlx_driver::Device;
use rlx_ir::{DType, Graph, Op, Shape, Tick};
use rlx_runtime::{CompiledGraph, Session};

fn fft_graph(batch: usize, n: usize) -> Graph {
    let mut g = Graph::new("wgpu_sweep");
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

fn set_fast(on: bool) {
    // SAFETY: single-threaded benchmark driver toggling a process-local gate.
    unsafe {
        std::env::set_var("RLX_FFT_FAST", if on { "1" } else { "0" });
    }
}

fn main() {
    let empty: &[(&str, &[f32])] = &[];
    let median = |c: &mut CompiledGraph| -> u64 {
        for _ in 0..5 {
            let _ = c.run(empty);
        }
        let mut s = Vec::with_capacity(25);
        for _ in 0..25 {
            let t0 = Tick::now();
            let _ = c.run(empty);
            s.push(Tick::now().elapsed_ns(t0));
        }
        s.sort_unstable();
        s[s.len() / 2]
    };

    println!("wgpu FFT: multi-kernel (old) vs on-chip radix-4/8 (new), single forward FFT");
    println!(
        "  {:>5} {:>6}  {:>11} {:>11}  {:>8}",
        "n", "batch", "old µs", "new µs", "new/old"
    );
    let _ = std::io::stdout().flush();

    for &n in &[2048usize, 4096] {
        for &batch in &[16usize, 64, 256, 512] {
            let mut c = Session::new(Device::Gpu).compile(fft_graph(batch, n));
            set_fast(false);
            let old = median(&mut c);
            set_fast(true);
            let new = median(&mut c);
            let per = |t: u64| t as f64 / 1000.0;
            println!(
                "  {n:>5} {batch:>6}  {:>9.1}µs {:>9.1}µs  {:>7.2}×",
                per(old),
                per(new),
                old as f64 / new.max(1) as f64,
            );
            let _ = std::io::stdout().flush();
        }
        println!();
    }
}
