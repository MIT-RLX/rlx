// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Licensed under the GNU General Public License, version 3.

//! FFT micro-benchmark: sweep batch × N across every enabled backend.
//!
//! ```sh
//! just throttle
//! cargo run -p rlx-bench --release --example bench_fft --features metal,gpu,mlx
//! cargo run -p rlx-bench --release --example bench_fft --features cuda
//! RLX_BENCH_DISPATCH_ONLY=1 cargo run -p rlx-bench --release --example bench_fft --features gpu
//! scripts/bench_fft_rig.sh
//! ```

use rlx_bench::{BenchResult, patterns::FftPattern, run_benchmark, run_benchmark_dispatch_only};
use rlx_driver::Device;

fn devices() -> Vec<(&'static str, Device)> {
    let out = vec![("cpu", Device::Cpu)];
    #[cfg(feature = "metal")]
    out.push(("metal", Device::Metal));
    #[cfg(feature = "mlx")]
    out.push(("mlx", Device::Mlx));
    #[cfg(feature = "gpu")]
    out.push(("wgpu", Device::Gpu));
    #[cfg(feature = "cuda")]
    out.push(("cuda", Device::Cuda));
    #[cfg(feature = "rocm")]
    out.push(("rocm", Device::Rocm));
    out
}

fn gbps(bytes: u64, ns: u64) -> f64 {
    if ns == 0 {
        return 0.0;
    }
    bytes as f64 / (ns as f64 / 1e9) / 1e9
}

fn print_row(label: &str, pattern: &FftPattern, r: &BenchResult) {
    let bytes = pattern.traffic_bytes();
    let med = r.median_ns();
    let us_per_row = med as f64 / pattern.batch as f64 / 1000.0;
    println!(
        "  {label:5} {name:18} median={med_us:8.2}µs  {gbps:6.2} GB/s  {us_row:7.3}µs/row",
        name = pattern.label(),
        med_us = med as f64 / 1000.0,
        gbps = gbps(bytes, med),
        us_row = us_per_row,
    );
}

fn run_sweep(devs: &[(&'static str, Device)], batches: &[usize], ns: &[usize], tag: &str) {
    println!("=== {tag} ===");
    for &n in ns {
        for &batch in batches {
            let pattern = FftPattern {
                batch,
                n,
                inverse: false,
            };
            for &(label, dev) in devs {
                let r = if std::env::var("RLX_BENCH_DISPATCH_ONLY").ok().as_deref() == Some("1") {
                    run_benchmark_dispatch_only(&pattern, dev, 5, 30)
                } else {
                    run_benchmark(&pattern, dev, 5, 30)
                };
                print_row(label, &pattern, &r);
            }
            println!();
        }
    }
}

fn main() {
    let devs = devices();
    println!(
        "rlx-bench fft — devices: {:?}",
        devs.iter().map(|(l, _)| *l).collect::<Vec<_>>()
    );
    println!(
        "  {:5} {:18} {:>14}  {:>9}  {:>10}",
        "dev", "pattern", "median", "GB/s", "µs/row"
    );

    let batches = [1usize, 4, 16, 64, 256];
    let pow2_ns = [64usize, 256, 1024, 4096];
    run_sweep(&devs, &batches, &pow2_ns, "pow2 f32 GPU path");

    let non_pow2_batches = [1usize, 16, 64];
    let non_pow2_ns = [12usize, 15, 20];
    run_sweep(
        &devs,
        &non_pow2_batches,
        &non_pow2_ns,
        "non-pow2 host fallback",
    );
}
