// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Cross-backend bench for GGUF `Op::DequantMatMul` (legacy + IQ schemes).
//!
//! ```sh
//! cargo run -p rlx-bench --release --example bench_gguf_dequant
//! cargo run -p rlx-bench --release --example bench_gguf_dequant --features metal
//! cargo run -p rlx-bench --release --example bench_gguf_dequant --features metal,mlx,gpu
//! ```

use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const WARMUP: usize = 3;
const ITERS: usize = 20;

struct Case {
    label: &'static str,
    scheme: QuantScheme,
    ggml: rlx_gguf::GgmlType,
    k: usize,
    n: usize,
    m: usize,
    tol: f32,
}

fn devices() -> Vec<(Device, &'static str)> {
    #[allow(unused_mut)]
    let mut v = vec![(Device::Cpu, "cpu")];
    #[cfg(feature = "metal")]
    v.push((Device::Metal, "metal"));
    #[cfg(feature = "mlx")]
    v.push((Device::Mlx, "mlx"));
    #[cfg(feature = "gpu")]
    v.push((Device::Gpu, "wgpu"));
    v
}

fn build_graph(case: &Case, packed_len: usize) -> Graph {
    let mut g = Graph::new("bench_gguf_dq");
    let x = g.input("x", Shape::new(&[case.m, case.k], DType::F32));
    let w = g.param("w", Shape::new(&[packed_len], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul {
            scheme: case.scheme,
        },
        vec![x, w],
        Shape::new(&[case.m, case.n], DType::F32),
    );
    g.set_outputs(vec![y]);
    g
}

fn reference(case: &Case, packed: &[u8], x: &[f32]) -> Vec<f32> {
    let w_ref = rlx_cpu::dequant_cache::gguf_weight_f32(0, packed, case.k, case.n, case.scheme);
    let mut out = vec![0f32; case.m * case.n];
    for r in 0..case.m {
        for c in 0..case.n {
            let mut acc = 0f32;
            for i in 0..case.k {
                acc += x[r * case.k + i] * w_ref[c * case.k + i];
            }
            out[r * case.n + c] = acc;
        }
    }
    out
}

fn median(mut xs: Vec<u128>) -> u128 {
    xs.sort_unstable();
    xs[xs.len() / 2]
}

fn bench_case(case: &Case) {
    let w_row: Vec<f32> = (0..case.k * case.n)
        .map(|i| ((i as f32) * 0.011).sin() * 0.5)
        .collect();
    let packed = rlx_gguf::quantize(&w_row, case.ggml).expect("quantize");
    let x: Vec<f32> = (0..case.m * case.k)
        .map(|i| 0.015 * (i as f32 + 1.0))
        .collect();
    let cpu_ref = reference(case, &packed, &x);
    let io_bytes = (case.m * case.k + case.m * case.n) * 4 + packed.len();

    println!(
        "\n=== {} ({:?}, m={} k={} n={}) ===",
        case.label, case.scheme, case.m, case.k, case.n
    );

    for (device, label) in devices() {
        let graph = build_graph(case, packed.len());
        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut exe = Session::new(device).compile(graph);
            exe.set_param_typed("w", &packed, DType::U8);
            for _ in 0..WARMUP {
                let _ = exe.run(&[("x", x.as_slice())]);
            }
            let mut samples = Vec::with_capacity(ITERS);
            let mut last = Vec::new();
            for _ in 0..ITERS {
                let t = Instant::now();
                last = exe.run(&[("x", x.as_slice())]).pop().unwrap_or_default();
                samples.push(t.elapsed().as_nanos());
            }
            (last, median(samples))
        }));
        let Some((out, lat_ns)) = result.ok() else {
            println!("  {:<6} unsupported / failed", label);
            continue;
        };
        let max_abs = if out.len() == cpu_ref.len() {
            out.iter()
                .zip(&cpu_ref)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max)
        } else {
            f32::INFINITY
        };
        let ok = max_abs <= case.tol;
        let gbps = io_bytes as f64 / (lat_ns as f64 / 1e9) / 1e9;
        println!(
            "  {:<6} lat {:.2} ms  {:.2} GB/s  {} (max|Δ|={:.2e})",
            label,
            lat_ns as f64 / 1e6,
            gbps,
            if ok { "PASS" } else { "FAIL" },
            max_abs
        );
    }
}

fn main() {
    let cases = [
        Case {
            label: "Q4_0",
            scheme: QuantScheme::GgufQ4_0,
            ggml: rlx_gguf::GgmlType::Q4_0,
            k: 256,
            n: 64,
            m: 4,
            tol: 1e-3,
        },
        Case {
            label: "Q4_K",
            scheme: QuantScheme::GgufQ4K,
            ggml: rlx_gguf::GgmlType::Q4K,
            k: 256,
            n: 64,
            m: 4,
            tol: 1e-3,
        },
        Case {
            label: "IQ2_XXS",
            scheme: QuantScheme::GgufIQ2XXS,
            ggml: rlx_gguf::GgmlType::IQ2XXS,
            k: 256,
            n: 64,
            m: 4,
            tol: 0.12,
        },
        Case {
            label: "IQ4_NL",
            scheme: QuantScheme::GgufIQ4NL,
            ggml: rlx_gguf::GgmlType::IQ4NL,
            k: 256,
            n: 64,
            m: 4,
            tol: 1e-3,
        },
        Case {
            label: "TQ2_0",
            scheme: QuantScheme::GgufTQ2_0,
            ggml: rlx_gguf::GgmlType::TQ2_0,
            k: 256,
            n: 64,
            m: 4,
            tol: 0.08,
        },
        Case {
            label: "IQ1_S",
            scheme: QuantScheme::GgufIQ1S,
            ggml: rlx_gguf::GgmlType::IQ1S,
            k: 256,
            n: 64,
            m: 1,
            tol: 0.10,
        },
    ];
    for case in &cases {
        bench_case(case);
    }
}
