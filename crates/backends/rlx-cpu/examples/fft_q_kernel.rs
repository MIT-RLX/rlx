// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fixed-point vs float FFT, kernels only — no graph, no casts, no copies.
//!
//! This is the cleanest read on whether `Op::FftQ` is worth reaching for on a
//! host with an FPU. It is not the question the op was built to answer (that
//! one is about MCUs with no FPU at all), but it is the one people will ask.
//!
//! Run: cargo run --release -p rlx-cpu --example fft_q_kernel

use rlx_ir::fft::{FftNorm, FftQScale};
use std::time::Instant;

fn main() {
    println!(
        "{:<12} {:>14} {:>14} {:>8}",
        "shape", "fft1d_q32 us", "fft1d_f32 us", "q/f"
    );
    for &(outer, n) in &[(1usize, 256usize), (1, 1024), (64, 1024), (256, 1024)] {
        let reps = 51;
        let xi: Vec<i32> = (0..outer * 2 * n)
            .map(|i| ((i as f64 * 0.13).sin() * 15000.0) as i32)
            .collect();
        let xf: Vec<f32> = xi.iter().map(|&v| v as f32).collect();

        let (mut qs, mut fs) = (Vec::new(), Vec::new());
        for _ in 0..reps {
            let mut w = xi.clone();
            let t = Instant::now();
            rlx_cpu::thunk::fft1d_q32_block_parallel(
                &mut w,
                outer,
                n,
                false,
                FftNorm::Backward,
                FftQScale::PerStage,
            )
            .unwrap();
            qs.push(t.elapsed().as_secs_f64() * 1e6);

            let mut w = xf.clone();
            let t = Instant::now();
            // Same in-place block layout, so src == dst == byte offset 0.
            unsafe {
                rlx_cpu::thunk::execute_fft1d_f32(
                    0,
                    0,
                    outer,
                    n,
                    false,
                    FftNorm::Backward.tag(),
                    w.as_mut_ptr().cast::<u8>(),
                );
            }
            fs.push(t.elapsed().as_secs_f64() * 1e6);
        }
        qs.sort_by(f64::total_cmp);
        fs.sort_by(f64::total_cmp);
        let (q, f) = (qs[reps / 2], fs[reps / 2]);
        println!(
            "{:<12} {q:>14.1} {f:>14.1} {:>7.2}x",
            format!("{outer}x{n}"),
            q / f
        );
    }
}
