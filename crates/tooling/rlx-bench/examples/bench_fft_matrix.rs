// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Licensed under the GNU General Public License, version 3.

//! Comprehensive FFT/IFFT benchmark matrix: every variant × precision × size ×
//! batch, across every available backend, with a CPU-parity check per cell.
//!
//! Variants:  fft (forward complex), ifft (inverse complex), rfft (forward real).
//!            irfft is defined but excluded from the sweep (its Hermitian mirror
//!            hard-crashes at large batch — a separate gather-batch bug).
//! Precision: f32 / f64 (2N real-block) and c64 (interleaved complex f32) for the
//!            complex variants; f32 for the real variants.
//! Backends:  cpu always; metal / gpu(wgpu) / mlx / coreml(ane) / cuda gated by
//!            cargo features. GPU native FFT is f32-pow2 only; f64/c64 run on the
//!            host CPU path (Metal) or are rejected (wgpu, f32-only arena → `—`).
//!
//! ```sh
//! # Apple:
//! cargo run -p rlx-bench --release --example bench_fft_matrix \
//!     --features metal,gpu,mlx,coreml,native-gpu-fft
//! # Linux CUDA + Vulkan-wgpu rig:
//! cargo run -p rlx-bench --release --example bench_fft_matrix \
//!     --features cuda,gpu,native-gpu-fft
//! ```

use std::io::Write;

use rlx_driver::Device;
use rlx_ir::{DType, FftNorm, Graph, GraphExt, NodeId, Op, Shape, Tick};
use rlx_runtime::Session;

#[derive(Clone, Copy, PartialEq)]
enum Variant {
    Fft,  // forward complex
    Ifft, // inverse complex
    Rfft, // forward real
    // Defined (build/precisions handle it) but not swept — its Hermitian mirror
    // crashes at large batch. Re-add to the loop in main() once that's fixed.
    #[allow(dead_code)]
    Irfft, // inverse real
}

impl Variant {
    fn label(self) -> &'static str {
        match self {
            Variant::Fft => "fft",
            Variant::Ifft => "ifft",
            Variant::Rfft => "rfft",
            Variant::Irfft => "irfft",
        }
    }
    fn precisions(self) -> &'static [DType] {
        match self {
            // Complex variants exercise all three storage layouts.
            Variant::Fft | Variant::Ifft => &[DType::F32, DType::F64, DType::C64],
            // Real variants: real→complex path is f32 (f64 rfft/irfft unsupported).
            Variant::Rfft | Variant::Irfft => &[DType::F32],
        }
    }
}

fn backends() -> Vec<(&'static str, Device)> {
    #[allow(unused_mut)]
    let mut out: Vec<(&'static str, Device)> = vec![("cpu", Device::Cpu)];
    #[cfg(feature = "metal")]
    out.push(("metal", Device::Metal));
    #[cfg(feature = "gpu")]
    out.push(("wgpu", Device::Gpu));
    #[cfg(feature = "mlx")]
    out.push(("mlx", Device::Mlx));
    #[cfg(feature = "coreml")]
    out.push(("ane", Device::Ane));
    #[cfg(feature = "cuda")]
    out.push(("cuda", Device::Cuda));
    out
}

/// Byte payload for a constant of `count` scalar elements in `dt`.
fn payload(dt: DType, count: usize) -> Vec<u8> {
    let mut b = Vec::new();
    for i in 0..count {
        let v = (i as f32 * 0.013).sin();
        match dt {
            DType::F64 => b.extend_from_slice(&(v as f64).to_le_bytes()),
            _ => b.extend_from_slice(&v.to_le_bytes()),
        }
    }
    b
}

/// Build a graph for one (variant, precision, n, batch) and set its output(s).
fn build(variant: Variant, dt: DType, n: usize, batch: usize) -> Graph {
    let mut g = Graph::new("fft_matrix");
    match variant {
        Variant::Fft | Variant::Ifft => {
            let inverse = variant == Variant::Ifft;
            let (shape, count) = match dt {
                // 2N real-block (first N real, next N imag) per row.
                DType::F32 | DType::F64 => (Shape::new(&[batch, 2 * n], dt), batch * 2 * n),
                // Interleaved complex64: n complex per row (axis extent = n).
                DType::C64 => (Shape::new(&[batch, n], DType::C64), batch * 2 * n),
                _ => unreachable!(),
            };
            let data = payload(if dt == DType::C64 { DType::F32 } else { dt }, count);
            let x = g.add_node(Op::Constant { data }, vec![], shape);
            let y = g.fft(x, inverse);
            g.set_outputs(vec![y]);
        }
        Variant::Rfft => {
            let x = g.add_node(
                Op::Constant {
                    data: payload(dt, batch * n),
                },
                vec![],
                Shape::new(&[batch, n], dt),
            );
            let (re, im) = g.rfft(x, FftNorm::Forward);
            let last = g.shape(re).rank() - 1;
            let block = g.concat_(vec![re, im], last);
            g.set_outputs(vec![block]);
        }
        Variant::Irfft => {
            let half = n / 2 + 1;
            let re: NodeId = g.add_node(
                Op::Constant {
                    data: payload(dt, batch * half),
                },
                vec![],
                Shape::new(&[batch, half], dt),
            );
            let im: NodeId = g.add_node(
                Op::Constant {
                    data: payload(dt, batch * half),
                },
                vec![],
                Shape::new(&[batch, half], dt),
            );
            let y = g.irfft(re, im, n, FftNorm::Forward);
            g.set_outputs(vec![y]);
        }
    }
    g
}

/// Flatten all outputs to f32 for a uniform parity comparison.
fn flat_f32(out: &[(Vec<u8>, DType)]) -> Vec<f32> {
    let mut v = Vec::new();
    for (bytes, dt) in out {
        match dt {
            DType::F64 => v.extend(
                bytes
                    .chunks_exact(8)
                    .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32),
            ),
            _ => v.extend(
                bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap())),
            ),
        }
    }
    v
}

/// Run a graph on `dev`; returns (flattened output, median ns) or None on panic.
fn run_dev(
    variant: Variant,
    dt: DType,
    n: usize,
    batch: usize,
    dev: Device,
) -> Option<(Vec<f32>, u64)> {
    std::panic::catch_unwind(|| {
        let empty: &[(&str, &[u8], DType)] = &[];
        let mut c = Session::new(dev).compile(build(variant, dt, n, batch));
        let out = flat_f32(&c.run_typed(empty));
        for _ in 0..3 {
            let _ = c.run_typed(empty);
        }
        let mut s = Vec::with_capacity(15);
        for _ in 0..15 {
            let t0 = Tick::now();
            let _ = c.run_typed(empty);
            s.push(Tick::now().elapsed_ns(t0));
        }
        s.sort_unstable();
        (out, s[s.len() / 2])
    })
    .ok()
}

/// Relative parity vs the CPU reference: max|Δ| / max|cpu|. Correct for FFT,
/// whose output magnitude scales with n. Returns (rel_error, len_ok).
fn rel_diff(cpu: &[f32], dev: &[f32]) -> (f32, bool) {
    if cpu.len() != dev.len() {
        return (f32::INFINITY, false);
    }
    let denom = cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
    let maxd = cpu
        .iter()
        .zip(dev)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    (maxd / denom, true)
}

fn dt_label(dt: DType) -> &'static str {
    match dt {
        DType::F32 => "f32",
        DType::F64 => "f64",
        DType::C64 => "c64",
        _ => "?",
    }
}

fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    let devs = backends();
    let sizes = [256usize, 1024, 4096, 16384, 65536];
    let batch = 64usize;

    println!("rlx FFT/IFFT matrix — median µs, batch={batch}, PASS = max|Δ| vs CPU < tol");
    print!("  {:<7} {:<5} {:>7}", "variant", "prec", "n");
    for (name, _) in &devs {
        print!("  {:>11}", name);
    }
    println!("   parity");

    // Irfft is excluded from the default sweep: correct at batch=1 (see
    // cpu_fft::rfft_irfft_roundtrip_mirror_all_backends) but its Hermitian mirror
    // (a gather) hard-crashes at large batch — a separate gather-batch bug.
    for &variant in &[Variant::Fft, Variant::Ifft, Variant::Rfft] {
        for &dt in variant.precisions() {
            for &n in &sizes {
                // CPU reference first.
                let cpu = run_dev(variant, dt, n, batch, Device::Cpu);
                let cpu_out = cpu.map(|v| v.0).unwrap_or_default();
                // Relative tolerance: f32 GPUs recompute twiddles in f32 (vs the
                // CPU's f64 recurrence), so allow a looser bound than f64.
                let tol = if dt == DType::F64 { 1e-4 } else { 5e-3 };
                print!("  {:<7} {:<5} {:>7}", variant.label(), dt_label(dt), n);
                // Per-backend timing (with a parity flag) + a summary of any that
                // fail parity vs CPU. Flag: ' ' ok, '!' rel>tol, '#' len mismatch.
                let mut fails: Vec<String> = Vec::new();
                for (name, dev) in &devs {
                    match run_dev(variant, dt, n, batch, *dev) {
                        Some((out, ns)) => {
                            let flag = if *dev == Device::Cpu {
                                ' '
                            } else {
                                let (rel, ok) = rel_diff(&cpu_out, &out);
                                if !ok {
                                    fails.push(format!("{name}#"));
                                    '#'
                                } else if rel >= tol {
                                    fails.push(format!("{name}({rel:.0e})"));
                                    '!'
                                } else {
                                    ' '
                                }
                            };
                            print!("  {:>9.1}µs{}", ns as f64 / 1000.0, flag);
                        }
                        None => print!("  {:>11} ", "—"),
                    }
                }
                if fails.is_empty() {
                    println!("   ok");
                } else {
                    println!("   BAD: {}", fails.join(" "));
                }
                let _ = std::io::stdout().flush();
            }
            println!();
        }
    }
}
