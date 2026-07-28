// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A/B benchmark for the `native-gpu-fft` on-chip FFT kernels on Metal/wgpu:
//!   * old — the multi-kernel DRAM-round-trip path (RLX_FFT_FAST=0)
//!   * r2  — single-kernel radix-2 in threadgroup memory (RADIX=2)
//!   * r4  — single-kernel radix-4 in-place (RADIX=4); n=2048 via leading radix-2
//!   * r8  — single-kernel radix-8 in-place (RADIX=8, default), pow-8 sizes only
//!
//! The dispatch reads `RLX_FFT_FAST` / `RLX_FFT_RADIX` live, so all three run on
//! the *same* compiled graph — only kernel routing changes. Each graph chains
//! `CHAIN` alternating fft/ifft (Ortho norm keeps values ~O(1)) so the single
//! output readback is amortized over many transforms.
//!
//! ```sh
//! just throttle
//! cargo run -p rlx-bench --release --example bench_fft_fast_ab \
//!     --features metal,gpu,native-gpu-fft
//! ```
//!
//! Radix-4 engages only for pure powers of four (n=4096 in range) on Metal;
//! n=2048 and wgpu fall back to r2, so r4≈r2 there (a built-in sanity check).

use rlx_driver::Device;
use rlx_ir::fft::FftNorm;
use rlx_ir::{DType, Graph, Op, Shape, Tick};
use rlx_runtime::{CompiledGraph, Session};

const CHAIN: usize = 32;

fn devices() -> Vec<(&'static str, Device)> {
    #[allow(unused_mut)]
    let mut out: Vec<(&'static str, Device)> = Vec::new();
    #[cfg(feature = "metal")]
    out.push(("metal", Device::Metal));
    #[cfg(feature = "gpu")]
    out.push(("wgpu", Device::Gpu));
    out
}

fn build_chain(batch: usize, n: usize, reps: usize) -> Graph {
    let mut g = Graph::new("fft_fast_ab");
    let len = batch * n * 2;
    let mut bytes = Vec::with_capacity(len * 4);
    for i in 0..len {
        bytes.extend_from_slice(&((i as f32 * 0.013).sin()).to_le_bytes());
    }
    let mut x = g.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(&[batch, n * 2], DType::F32),
    );
    for r in 0..reps {
        x = g.fft_norm(x, r % 2 == 1, FftNorm::Ortho);
    }
    g.set_outputs(vec![x]);
    g
}

/// Select the kernel path: "old" (multi-kernel), "r2", or "r4".
fn set_variant(v: &str) {
    // SAFETY: single-threaded benchmark driver toggling process-local gates.
    unsafe {
        match v {
            "old" => std::env::set_var("RLX_FFT_FAST", "0"),
            "r2" => {
                std::env::set_var("RLX_FFT_FAST", "1");
                std::env::set_var("RLX_FFT_RADIX", "2");
            }
            "r4" => {
                std::env::set_var("RLX_FFT_FAST", "1");
                std::env::set_var("RLX_FFT_RADIX", "4");
            }
            "r8" => {
                std::env::set_var("RLX_FFT_FAST", "1");
                std::env::set_var("RLX_FFT_RADIX", "8");
            }
            "r16" => {
                std::env::set_var("RLX_FFT_FAST", "1");
                std::env::set_var("RLX_FFT_RADIX", "16");
            }
            _ => unreachable!(),
        }
    }
}

fn output_f32(out: &[(Vec<u8>, DType)]) -> Vec<f32> {
    out[0]
        .0
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn main() {
    let devs = devices();
    if devs.is_empty() {
        eprintln!("no GPU backend compiled — build with --features metal,gpu,native-gpu-fft");
        return;
    }
    println!(
        "rlx-bench fft A/B/C — chain={CHAIN}, devices: {:?}",
        devs.iter().map(|(l, _)| *l).collect::<Vec<_>>()
    );
    println!("  old=multi-kernel DRAM | r2/r4/r8/r16 = on-chip radix-2/4/8/16 (pow-r sizes)\n");
    println!(
        "  {:5} {:>5} {:>4}  {:>7} {:>7} {:>7} {:>7} {:>7}  {:>6} {:>6} {:>6} {:>6}",
        "dev", "n", "bat", "old µs", "r2", "r4", "r8", "r16", "r2/o", "r4/o", "r8/o", "r16/o"
    );

    let warmup = 12usize;
    let iters = 80usize;
    let empty: &[(&str, &[f32])] = &[];
    let empty_typed: &[(&str, &[u8], DType)] = &[];

    let ns = [1024usize, 2048, 4096];
    let batches = [16usize, 64, 256];

    let median = |compiled: &mut CompiledGraph| -> u64 {
        for _ in 0..warmup {
            let _ = compiled.run(empty);
        }
        let mut s = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Tick::now();
            let _ = compiled.run(empty);
            s.push(Tick::now().elapsed_ns(t0));
        }
        s.sort_unstable();
        s[s.len() / 2]
    };

    for &(label, dev) in &devs {
        for &n in &ns {
            for &batch in &batches {
                let mut compiled = Session::new(dev).compile(build_chain(batch, n, CHAIN));

                // Parity: highest-radix path (r16) vs old (the reference) — the
                // largest algorithmic gap among the variants — on the same graph.
                set_variant("old");
                let old_ref = output_f32(&compiled.run_typed(empty_typed));
                set_variant("r16");
                let r16_ref = output_f32(&compiled.run_typed(empty_typed));
                let max_diff = old_ref
                    .iter()
                    .zip(&r16_ref)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);

                set_variant("old");
                let old_ns = median(&mut compiled);
                set_variant("r2");
                let r2_ns = median(&mut compiled);
                set_variant("r4");
                let r4_ns = median(&mut compiled);
                set_variant("r8");
                let r8_ns = median(&mut compiled);
                set_variant("r16");
                let r16_ns = median(&mut compiled);

                let per = |t: u64| t as f64 / CHAIN as f64 / 1000.0;
                let sp = |t: u64| old_ns as f64 / t.max(1) as f64;
                println!(
                    "  {label:5} {n:>5} {batch:>4}  {:>7.2} {:>7.2} {:>7.2} {:>7.2} {:>7.2}  {:>5.2}× {:>5.2}× {:>5.2}× {:>5.2}×  Δ{:.0e}",
                    per(old_ns),
                    per(r2_ns),
                    per(r4_ns),
                    per(r8_ns),
                    per(r16_ns),
                    sp(r2_ns),
                    sp(r4_ns),
                    sp(r8_ns),
                    sp(r16_ns),
                    max_diff,
                );
            }
            println!();
        }
    }
}
