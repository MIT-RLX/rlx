// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native low-precision `Op::ScaledMatMul` on CUDA with a *parameterized*
//! `ScaledFormat::Custom` minifloat (the `fNeXmY` family, e.g. `f4e3m0`).
//!
//! These research formats have no FP8 tensor core, so they run the on-device
//! decode-and-accumulate kernel (`scaled_lowp_general.cu`). Its generic path
//! unpacks `(exp, mant, bias)` from the packed `kernel_id()` descriptor (top-bit
//! sentinel) and decodes without a per-format `switch` case — so a new format
//! needs no kernel edit. The seven named ids (`0..=6`) keep the existing switch,
//! which the last test exercises to prove the hardware path is unaffected.
//!
//! Skips (no-op) on CUDA-less hosts via `rlx_cuda::is_available()`.

use rlx_cuda::backend::CudaExecutable;
use rlx_ir::{DType, Graph, Op, ScaleLayout, ScaledFormat, Shape};

/// Build the quantize → scaled-GEMM graph (TN: lhs [m,k], rhs [n,k], out [m,n]).
fn build_scaled_mm_graph(
    fmt: ScaledFormat,
    layout: ScaleLayout,
    m: usize,
    k: usize,
    n: usize,
) -> Graph {
    let (ls_shape, rs_shape) = match layout {
        ScaleLayout::PerTensor => (Shape::new(&[1], DType::F32), Shape::new(&[1], DType::F32)),
        _ => {
            let nb = k.div_ceil(layout.block() as usize);
            (
                Shape::new(&[m, nb], DType::U8),
                Shape::new(&[n, nb], DType::U8),
            )
        }
    };
    let mut g = Graph::new("cuda_scaled_mm");
    let lhs_in = g.input("lhs", Shape::new(&[m, k], DType::F32));
    let rhs_in = g.input("rhs", Shape::new(&[n, k], DType::F32));
    let ls = g.add_node(
        Op::ScaledQuantScale {
            format: fmt,
            scale_layout: layout,
        },
        vec![lhs_in],
        ls_shape,
    );
    let lq = g.add_node(
        Op::ScaledQuantize {
            format: fmt,
            scale_layout: layout,
        },
        vec![lhs_in, ls],
        Shape::new(&[m, k], DType::U8),
    );
    let rs = g.add_node(
        Op::ScaledQuantScale {
            format: fmt,
            scale_layout: layout,
        },
        vec![rhs_in],
        rs_shape,
    );
    let rq = g.add_node(
        Op::ScaledQuantize {
            format: fmt,
            scale_layout: layout,
        },
        vec![rhs_in, rs],
        Shape::new(&[n, k], DType::U8),
    );
    let y = g.add_node(
        Op::ScaledMatMul {
            lhs_format: fmt,
            rhs_format: fmt,
            scale_layout: layout,
            has_bias: false,
        },
        vec![lq, rq, ls, rs],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    g
}

fn f32_matmul_tn(lhs: &[f32], rhs: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                acc += lhs[i * k + p] * rhs[j * k + p];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

/// f4e3m0 with grid-aligned inputs (amax 2 → per-tensor scale 0.125, an exact
/// power of two) reconstructs bit-exactly, so the on-device scaled GEMM must
/// equal a plain f32 matmul of the inputs. A broken descriptor unpack (wrong
/// exp/mant/bias) would corrupt the decode and fail here.
#[test]
fn cuda_scaled_matmul_f4e3m0_grid_is_exact() {
    if !rlx_cuda::is_available() {
        eprintln!("skip: CUDA unavailable");
        return;
    }
    let fmt = ScaledFormat::custom(3, 0); // f4e3m0
    assert_eq!(fmt.to_string(), "f4e3m0");
    let (m, k, n) = (4usize, 32usize, 6usize);
    let grid = [2.0f32, -1.0, 0.5, -0.25, 1.0, -2.0, 0.25, -0.5]; // all |·| <= 2
    let lhs: Vec<f32> = (0..m * k).map(|i| grid[i % grid.len()]).collect();
    let rhs: Vec<f32> = (0..n * k).map(|i| grid[(i * 3 + 1) % grid.len()]).collect();

    let mut exe =
        CudaExecutable::compile(build_scaled_mm_graph(fmt, ScaleLayout::PerTensor, m, k, n));
    let out = exe
        .run(&[("lhs", lhs.as_slice()), ("rhs", rhs.as_slice())])
        .remove(0);
    let reference = f32_matmul_tn(&lhs, &rhs, m, k, n);
    assert_eq!(out.len(), m * n);
    let max_abs = out
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("cuda f4e3m0 grid: max_abs_vs_f32={max_abs:.3e}");
    assert!(
        max_abs <= 5e-3,
        "f4e3m0 grid GEMM max_abs {max_abs} (expected near-exact)"
    );
}

/// f4e3m0 on smooth data with block-MX (E8M0) scaling still tracks the f32
/// matmul despite 0 mantissa bits — proves the generic quantize + decode kernels
/// run end-to-end on-device and produce a sane result.
#[test]
fn cuda_scaled_matmul_f4e3m0_tracks_f32() {
    if !rlx_cuda::is_available() {
        eprintln!("skip: CUDA unavailable");
        return;
    }
    let fmt = ScaledFormat::custom(3, 0);
    let (m, k, n) = (4usize, 64usize, 8usize);
    let lhs: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.13).sin() * 1.5).collect();
    let rhs: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.07).cos() * 1.2).collect();

    let mut exe = CudaExecutable::compile(build_scaled_mm_graph(fmt, ScaleLayout::mx(), m, k, n));
    let out = exe
        .run(&[("lhs", lhs.as_slice()), ("rhs", rhs.as_slice())])
        .remove(0);
    assert!(
        out.iter().all(|v| v.is_finite()),
        "f4e3m0 produced non-finite"
    );
    let reference = f32_matmul_tn(&lhs, &rhs, m, k, n);
    let cos = cosine(&out, &reference);
    eprintln!("cuda f4e3m0 mx-block: cosine_vs_f32={cos:.4}");
    assert!(cos >= 0.7, "f4e3m0 mx-block cosine {cos} < 0.7");
}

/// The named E4M3 format through the SAME general decode kernel (block layout
/// forces the decode path, switch id 0, top bit clear) must be unaffected by
/// the generic-descriptor addition.
#[test]
fn cuda_scaled_matmul_named_e4m3_unaffected() {
    if !rlx_cuda::is_available() {
        eprintln!("skip: CUDA unavailable");
        return;
    }
    let (m, k, n) = (4usize, 64usize, 8usize);
    let lhs: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.13).sin() * 1.5).collect();
    let rhs: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.07).cos() * 1.2).collect();

    let mut exe = CudaExecutable::compile(build_scaled_mm_graph(
        ScaledFormat::F8E4M3,
        ScaleLayout::mx(),
        m,
        k,
        n,
    ));
    let out = exe
        .run(&[("lhs", lhs.as_slice()), ("rhs", rhs.as_slice())])
        .remove(0);
    let reference = f32_matmul_tn(&lhs, &rhs, m, k, n);
    let cos = cosine(&out, &reference);
    eprintln!("cuda e4m3 mx-block: cosine_vs_f32={cos:.5}");
    assert!(cos >= 0.99, "e4m3 mx-block cosine {cos} < 0.99");
}

/// Larger GEMM spanning multiple 16×16 tiles in m, n, AND k, with none a
/// multiple of 16 — exercises the shared-memory tiling boundaries and the
/// block-scale indexing across k-tiles. (The tiled kernel's job is to track the
/// f32 matmul; exact-vs-CPU no longer holds since accumulation is tile-blocked.)
#[test]
fn cuda_scaled_matmul_f4e3m0_multitile() {
    if !rlx_cuda::is_available() {
        eprintln!("skip: CUDA unavailable");
        return;
    }
    let fmt = ScaledFormat::custom(3, 0);
    let (m, k, n) = (37usize, 80usize, 45usize);
    let lhs: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.017).sin() * 1.3).collect();
    let rhs: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.023).cos() * 1.1).collect();
    let mut exe = CudaExecutable::compile(build_scaled_mm_graph(fmt, ScaleLayout::mx(), m, k, n));
    let out = exe
        .run(&[("lhs", lhs.as_slice()), ("rhs", rhs.as_slice())])
        .remove(0);
    assert_eq!(out.len(), m * n);
    assert!(
        out.iter().all(|v| v.is_finite()),
        "multitile produced non-finite"
    );
    let cos = cosine(&out, &f32_matmul_tn(&lhs, &rhs, m, k, n));
    eprintln!("cuda f4e3m0 multitile {m}x{k}x{n}: cosine_vs_f32={cos:.4}");
    assert!(cos >= 0.7, "multitile cosine {cos} < 0.7");
}

/// Throughput of the tiled decode GEMM at a GEMM-heavy size. `#[ignore]`d — run
/// explicitly: `... --test cuda_scaled_custom decode_bench -- --ignored --nocapture`.
#[test]
#[ignore]
fn cuda_scaled_matmul_decode_bench() {
    if !rlx_cuda::is_available() {
        return;
    }
    let fmt = ScaledFormat::custom(3, 0);
    let (m, k, n) = (1024usize, 1024usize, 1024usize);
    let lhs: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001).sin()).collect();
    let rhs: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.001).cos()).collect();
    let mut exe = CudaExecutable::compile(build_scaled_mm_graph(fmt, ScaleLayout::mx(), m, k, n));
    let _ = exe.run(&[("lhs", lhs.as_slice()), ("rhs", rhs.as_slice())]); // warmup
    let iters = 20u32;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let _ = exe.run(&[("lhs", lhs.as_slice()), ("rhs", rhs.as_slice())]);
    }
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    let gflops = 2.0 * (m * k * n) as f64 / dt / 1e9;
    eprintln!(
        "cuda decode GEMM {m}x{k}x{n} (tiled): {:.3} ms/iter, {:.1} GFLOP/s (incl. quantize + upload)",
        dt * 1e3,
        gflops
    );
}

/// Per-tensor quantize→dequantize reference matching the kernels exactly:
/// scale = amax / max_finite, then decode(encode(x/scale)) * scale.
fn cpu_quant_dequant_per_tensor(x: &[f32], fmt: ScaledFormat) -> Vec<f32> {
    use rlx_ir::lowp_codec::{decode, encode};
    let maxf = fmt.max_finite();
    let amax = x.iter().fold(0f32, |a, &v| a.max(v.abs()));
    let scale = if amax > 0.0 { amax / maxf } else { 1.0 };
    x.iter()
        .map(|&v| {
            let q = if scale != 0.0 { v / scale } else { 0.0 };
            decode(fmt, encode(fmt, q)) * scale
        })
        .collect()
}

fn build_quant_dequant_graph(fmt: ScaledFormat, rows: usize, cols: usize) -> Graph {
    let layout = ScaleLayout::PerTensor;
    let mut g = Graph::new("cuda_qdq");
    let x = g.input("x", Shape::new(&[rows, cols], DType::F32));
    let scale = g.add_node(
        Op::ScaledQuantScale {
            format: fmt,
            scale_layout: layout,
        },
        vec![x],
        Shape::new(&[1], DType::F32),
    );
    let codes = g.add_node(
        Op::ScaledQuantize {
            format: fmt,
            scale_layout: layout,
        },
        vec![x, scale],
        Shape::new(&[rows, cols], DType::U8),
    );
    let recon = g.add_node(
        Op::ScaledDequantize {
            format: fmt,
            scale_layout: layout,
        },
        vec![codes, scale],
        Shape::new(&[rows, cols], DType::F32),
    );
    g.set_outputs(vec![recon]);
    g
}

/// Sweep many `(exp, mant, bias)` splits — both parameterized `Custom` and named
/// — and assert the on-device quantize→dequantize matches the CPU oracle
/// (`rlx_ir::lowp_codec`) bit-for-bit. This validates the generic descriptor
/// decode/encode path on real hardware across formats the fixed `switch` never
/// covered, and confirms the modified NVRTC kernel (incl. the inf-saturation
/// branch, which is compile-validated here) still decodes the named ids exactly.
#[test]
fn cuda_scaled_quant_dequant_matches_cpu_oracle_sweep() {
    if !rlx_cuda::is_available() {
        eprintln!("skip: CUDA unavailable");
        return;
    }
    let (rows, cols) = (3usize, 32usize);
    let x: Vec<f32> = (0..rows * cols)
        .map(|i| (i as f32 * 0.21).sin() * 1.7)
        .collect();
    // Custom + non-native named formats take the general *decode* kernel;
    // F8E4M3/F8E5M2 per-tensor take the *native* fp8 quantize kernels
    // (`scaled_lowp.cu`) — both must reproduce the CPU oracle exactly. (The
    // native fp8 quantize now encodes in closed form, so it NVRTC-compiles
    // without <cuda_fp8.h>; on this Ampere card there are no fp8 tensor cores,
    // so we exercise it via quantize→dequantize, not the cublasLt GEMM.)
    let formats = [
        ScaledFormat::custom(3, 0), // f4e3m0
        ScaledFormat::custom(2, 1), // f4e2m1-shaped
        ScaledFormat::custom(2, 2), // f5e2m2
        ScaledFormat::custom(3, 2), // f6e3m2-shaped
        ScaledFormat::custom(4, 3), // f8e4m3-shaped (finite)
        ScaledFormat::custom(3, 4), // f8e3m4
        ScaledFormat::F8E4M3,       // native fp8 per-tensor quantize kernel
        ScaledFormat::F8E5M2,       // native fp8 per-tensor quantize kernel
        ScaledFormat::F8E4M3Fnuz,   // named fnuz → decode path
        ScaledFormat::F8E5M2Fnuz,   // named fnuz → decode path
        ScaledFormat::F6E3M2,       // named FP6 → decode path
        ScaledFormat::F4E2M1,       // named FP4 → decode path
    ];
    for fmt in formats {
        let mut exe = CudaExecutable::compile(build_quant_dequant_graph(fmt, rows, cols));
        let gpu = exe.run(&[("x", x.as_slice())]).remove(0);
        let cpu = cpu_quant_dequant_per_tensor(&x, fmt);
        assert_eq!(gpu.len(), cpu.len());
        let max_abs = gpu
            .iter()
            .zip(&cpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("cuda {fmt}: gpu-vs-cpu-oracle max_abs={max_abs:.3e}");
        assert!(
            max_abs <= 1e-6,
            "{fmt}: CUDA decode != CPU oracle (max_abs {max_abs})"
        );
    }
}
