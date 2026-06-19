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

//
// Matmul-only sweep: MLX (CPU) vs rlx-wgpu on the same graphs.
//
// ```sh
// cargo run -p rlx-bench --release --example bench_mlx_wgpu --features mlx,gpu
// ./rig.sh --windows cargo run --release -p rlx-bench --example bench_mlx_wgpu --features mlx,gpu
// ```

use rlx_bench::{patterns::MatmulPattern, run_benchmark, run_benchmark_dispatch_only};
use rlx_driver::Device;
#[allow(unused_imports)]
use rlx_runtime::is_available;

fn devices() -> Vec<(&'static str, Device)> {
    #[allow(unused_mut)]
    let mut out = vec![("cpu", Device::Cpu)];
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
    out
}

fn gflops(m: usize, k: usize, n: usize, median_ns: u64) -> f64 {
    let flops = 2.0 * m as f64 * k as f64 * n as f64;
    flops / (median_ns as f64 / 1e9)
}

fn main() {
    let devs = devices();
    println!(
        "rlx-bench mlx vs wgpu — matmul — devices: {:?}  (*wgpu = dispatch-only, no readback)",
        devs.iter().map(|(l, _)| *l).collect::<Vec<_>>()
    );

    let shapes = [
        (8, 64, 64),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ];

    for &(m, k, n) in &shapes {
        println!("\n# matmul {m}x{k}x{n}");
        let pattern = MatmulPattern { m, k, n };
        for &(label, dev) in &devs {
            let r = if label == "wgpu" {
                run_benchmark_dispatch_only(&pattern, dev, 3, 20)
            } else {
                run_benchmark(&pattern, dev, 3, 20)
            };
            let med = r.median_ns();
            let tag = if label == "wgpu" { "wgpu*" } else { label };
            println!(
                "  {tag:5} median={:.2}µs mean={:.2}µs  {:.2} GFLOP/s",
                med as f64 / 1000.0,
                r.mean_ns() as f64 / 1000.0,
                gflops(m, k, n, med)
            );
        }
    }
}
