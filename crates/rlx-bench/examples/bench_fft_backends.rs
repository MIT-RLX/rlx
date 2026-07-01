// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Licensed under the GNU General Public License, version 3.

//! Cross-backend FFT correctness + timing matrix for the Apple RLX backends.
//!
//! Runs `Op::Fft` on every available `Device`, compares element-wise to the CPU
//! reference, and reports each backend's actual FFT execution path. The eight
//! "backends" the user asks about collapse to five real code paths:
//!
//!   metal / mpsgraph → Device::Metal  — custom MSL (native-gpu-fft radix-2/4/8/16)
//!   wgpu             → Device::Gpu     — custom WGSL (native-gpu-fft radix-2, n≤2048)
//!   mlx              → Device::Mlx     — MLX's own vector FFT (vendor)
//!   coreml / ane     → Device::Ane     — host CPU (rlx-coreml host_exec; ANE has no FFT)
//!   blas / accelerate→ Device::Cpu     — host CPU FFT (BLAS only affects matmul)
//!
//! ```sh
//! cargo run -p rlx-bench --release --example bench_fft_backends \
//!     --features metal,gpu,mlx,coreml,accelerate,native-gpu-fft
//! ```

use rlx_driver::Device;
use rlx_ir::{DType, Graph, Op, Shape, Tick};
use rlx_runtime::Session;

/// (label shown, the 8 user-facing names it covers, device, fft path note).
fn backends() -> Vec<(&'static str, &'static str, Device, &'static str)> {
    #[allow(unused_mut)]
    let mut out: Vec<(&'static str, &'static str, Device, &'static str)> =
        vec![("cpu", "blas/accelerate", Device::Cpu, "host FFT")];
    #[cfg(feature = "metal")]
    out.push((
        "metal",
        "metal/mpsgraph",
        Device::Metal,
        "MSL radix-2/4/8/16",
    ));
    #[cfg(feature = "gpu")]
    out.push(("wgpu", "wgpu", Device::Gpu, "WGSL radix-2 (n<=2048)"));
    #[cfg(feature = "mlx")]
    out.push(("mlx", "mlx", Device::Mlx, "MLX vendor FFT"));
    #[cfg(feature = "coreml")]
    out.push(("ane", "coreml/ane", Device::Ane, "host FFT (no ANE FFT)"));
    out
}

fn fft_graph(batch: usize, n: usize, inverse: bool) -> Graph {
    let mut g = Graph::new("fft_backends");
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
    let y = g.fft(x, inverse);
    g.set_outputs(vec![y]);
    g
}

fn f32s(out: &[(Vec<u8>, DType)]) -> Vec<f32> {
    out[0]
        .0
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Run an FFT graph on `dev`; returns (output, median ns) or None on panic.
fn run_dev(dev: Device, batch: usize, n: usize) -> Option<(Vec<f32>, u64)> {
    std::panic::catch_unwind(|| {
        let empty: &[(&str, &[u8], DType)] = &[];
        let mut c = Session::new(dev).compile(fft_graph(batch, n, false));
        let out = f32s(&c.run_typed(empty));
        for _ in 0..3 {
            let _ = c.run_typed(empty);
        }
        let mut s = Vec::with_capacity(20);
        for _ in 0..20 {
            let t0 = Tick::now();
            let _ = c.run_typed(empty);
            s.push(Tick::now().elapsed_ns(t0));
        }
        s.sort_unstable();
        (out, s[s.len() / 2])
    })
    .ok()
}

fn main() {
    // Quiet the per-run panic spew; we report failures ourselves.
    std::panic::set_hook(Box::new(|_| {}));

    let devs = backends();
    println!("rlx-bench FFT cross-backend matrix (forward FFT, vs CPU reference)");
    println!("  native-gpu-fft = {}", cfg!(feature = "native-gpu-fft"));
    println!(
        "  RLX_FFT_FAST={} RLX_FFT_RADIX={}\n",
        std::env::var("RLX_FFT_FAST").unwrap_or_else(|_| "(default on)".into()),
        std::env::var("RLX_FFT_RADIX").unwrap_or_else(|_| "(default 8)".into()),
    );
    println!(
        "  {:6} {:18} {:24} {:>5} {:>5}  {:>11} {:>10} {:>7}",
        "dev", "covers", "fft path", "n", "batch", "med µs", "max|Δ|", "status"
    );

    let batch = 64usize;
    for &n in &[2048usize, 4096] {
        // CPU reference for this size.
        let reference = run_dev(Device::Cpu, batch, n).map(|(o, _)| o);
        for &(label, covers, dev, path) in &devs {
            let (status, med, diff) = match run_dev(dev, batch, n) {
                None => ("PANIC".to_string(), 0u64, f32::NAN),
                Some((out, med)) => match &reference {
                    Some(r) if r.len() == out.len() => {
                        let d = r
                            .iter()
                            .zip(&out)
                            .map(|(a, b)| (a - b).abs())
                            .fold(0.0f32, f32::max);
                        let ok = d < 1e-2;
                        ((if ok { "PASS" } else { "DIFF" }).to_string(), med, d)
                    }
                    _ => ("noref".to_string(), med, f32::NAN),
                },
            };
            println!(
                "  {label:6} {covers:18} {path:24} {n:>5} {batch:>5}  {:>9.2}µs {diff:>10.2e} {status:>7}",
                med as f64 / 1000.0,
            );
        }
        println!();
    }
}
