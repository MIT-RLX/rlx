// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Licensed under the GNU General Public License, version 3.

//! native-gpu-fft real→complex fusion benefit (Metal): the real-FFT pipeline
//! `fft_real(signal)` = `Concat([signal, zeros])` → `Fft` → split. The fusion
//! drops the memory-bound 2N Concat and reads the signal directly (im=0).
//!
//! The fusion is decided at COMPILE time (thunk build), so the A/B compiles the
//! graph twice — once with `RLX_FFT_FUSE_REAL=0` (un-fused), once fused (default).
//!
//! ```sh
//! cargo run -p rlx-bench --release --example bench_fft_real_fusion \
//!     --features metal,native-gpu-fft
//! ```

use rlx_driver::Device;
use rlx_ir::fft::FftNorm;
use rlx_ir::{DType, Graph, Shape, Tick};
use rlx_runtime::Session;

fn build(batch: usize, n: usize) -> Graph {
    let mut g = Graph::new("fft_real_bench");
    let sig = g.input("sig", Shape::new(&[batch, n], DType::F32));
    let (re, im) = g.fft_real(sig, FftNorm::Backward);
    g.set_outputs(vec![re, im]);
    g
}

fn set_fuse(on: bool) {
    // SAFETY: single-threaded driver toggling a process-local compile-time gate.
    unsafe {
        std::env::set_var("RLX_FFT_FUSE_REAL", if on { "1" } else { "0" });
    }
}

fn main() {
    let dev = Device::Metal;
    println!("Metal real-FFT pipeline: fusion off vs on (drops the 2N Concat)");
    println!(
        "  {:>5} {:>6}  {:>11} {:>11}  {:>8}",
        "n", "batch", "unfused µs", "fused µs", "speedup"
    );

    let warmup = 10usize;
    let iters = 40usize;

    for &n in &[2048usize, 4096] {
        for &batch in &[16usize, 64, 256] {
            let sig: Vec<f32> = (0..batch * n).map(|i| (i as f32 * 0.013).sin()).collect();
            let sig_bytes: Vec<u8> = sig.iter().flat_map(|v| v.to_le_bytes()).collect();
            let inputs: [(&str, &[u8], DType); 1] = [("sig", &sig_bytes, DType::F32)];

            let median = |fuse: bool| -> u64 {
                set_fuse(fuse);
                let mut c = Session::new(dev).compile(build(batch, n));
                for _ in 0..warmup {
                    let _ = c.run_typed(&inputs);
                }
                let mut s = Vec::with_capacity(iters);
                for _ in 0..iters {
                    let t0 = Tick::now();
                    let _ = c.run_typed(&inputs);
                    s.push(Tick::now().elapsed_ns(t0));
                }
                s.sort_unstable();
                s[s.len() / 2]
            };

            let off = median(false);
            let on = median(true);
            println!(
                "  {n:>5} {batch:>6}  {:>9.1}µs {:>9.1}µs  {:>7.2}×",
                off as f64 / 1000.0,
                on as f64 / 1000.0,
                off as f64 / on.max(1) as f64,
            );
        }
        println!();
    }
}
