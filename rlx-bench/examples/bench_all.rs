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

fn run_pattern<P: BenchmarkPattern>(pattern: &P, devs: &[(&str, Device)]) {
    println!("\n# {}", pattern.name());
    for &(label, dev) in devs {
        let r = run_benchmark(pattern, dev, /*warmup*/ 3, /*runs*/ 20);
        println!("  {label:5} {r}");
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
        &LayerNormPattern {
            rows: 32,
            hidden: 128,
        },
        &devs,
    );
    run_pattern(&MatmulBiasReluPattern { m: 8, k: 64, n: 64 }, &devs);
}
