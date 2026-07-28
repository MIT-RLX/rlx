// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Example: drive every canonical pattern on every available device.
//!
//! Run:
//! ```sh
//! cargo run -p rlx-bench --release --example bench_all                 # CPU only
//! cargo run -p rlx-bench --release --example bench_all --features metal
//! cargo run -p rlx-bench --release --example bench_all --features mlx
//! cargo run -p rlx-bench --release --example bench_all --features gpu  # wgpu
//! cargo run -p rlx-bench --release --example bench_all --features cuda
//! cargo run -p rlx-bench --release --example bench_all --features rocm
//! ```
//!
//! Throttle gating: prefix with `scripts/check-throttle.sh` before
//! publishing numbers.

use rlx_bench::{
    BenchmarkPattern,
    patterns::{LayerNormPattern, MatmulBiasReluPattern, MatmulPattern},
    run_benchmark,
};
use rlx_driver::Device;
#[allow(unused_imports)]
use rlx_runtime::is_available;

fn devices() -> Vec<(&'static str, Device)> {
    #[allow(unused_mut)]
    let mut out = vec![("cpu", Device::Cpu)];
    #[cfg(feature = "metal")]
    if is_available(Device::Metal) {
        out.push(("metal", Device::Metal));
    }
    #[cfg(feature = "mlx")]
    if is_available(Device::Mlx) {
        out.push(("mlx", Device::Mlx));
    }
    #[cfg(feature = "gpu")]
    if is_available(Device::Gpu) {
        out.push(("wgpu", Device::Gpu));
    }
    #[cfg(feature = "cuda")]
    if is_available(Device::Cuda) {
        out.push(("cuda", Device::Cuda));
    }
    #[cfg(feature = "rocm")]
    if is_available(Device::Rocm) {
        out.push(("rocm", Device::Rocm));
    }
    out
}

fn run_pattern<P: BenchmarkPattern>(pattern: &P, devs: &[(&str, Device)]) {
    println!("\n# {}", pattern.name());
    for &(label, dev) in devs {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_benchmark(pattern, dev, /*warmup*/ 3, /*runs*/ 20)
        }));
        match result {
            Ok(r) => println!("  {label:5} {r}"),
            Err(_) => eprintln!("  {label:5} FAILED (backend panic — see stderr)"),
        }
    }
}

fn main() {
    let devs = devices();
    println!(
        "rlx-bench / PLAN L5 — devices: {:?}",
        devs.iter().map(|(l, _)| *l).collect::<Vec<_>>()
    );

    run_pattern(&MatmulPattern { m: 8, k: 64, n: 64 }, &devs);
    run_pattern(
        &MatmulPattern {
            m: 512,
            k: 512,
            n: 512,
        },
        &devs,
    );
    run_pattern(
        &LayerNormPattern {
            rows: 32,
            hidden: 128,
        },
        &devs,
    );
    run_pattern(&MatmulBiasReluPattern { m: 8, k: 64, n: 64 }, &devs);
}
