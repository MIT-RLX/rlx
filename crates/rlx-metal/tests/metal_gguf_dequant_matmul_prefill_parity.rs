// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Packed GGUF `Op::DequantMatMul` PREFILL parity (m > 1) — Metal vs CPU.
//!
//! Decode (m == 1) uses the fused `q4k_mv_f32*` GEMV kernels. Prefill
//! (seq > 1 → m > 1) takes the thunk `else` branch: the `dequant_gguf` MSL
//! kernel expands the packed Q4_K / Q6_K weight `[n, k]` to an f32 scratch,
//! then `encode_mps_sgemm_bt` runs a real `MPSMatrixMultiplication` sgemm
//! (B^T). This is NOT the MPSGraph path (which never sees packed GGUF — see
//! `can_lower_dequant_in_mps`). These cases prove the m > 1 GEMM matches the
//! CPU `DequantMatMul` reference over the same packed bytes.

#![cfg(target_os = "macos")]

use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

/// Build `[m, n] = x[m, k] @ dequant(w)[n, k]^T` for a packed GGUF weight and
/// compare Metal against the CPU reference over identical packed bytes.
/// Returns the max abs elementwise difference.
fn run_case(
    scheme: QuantScheme,
    ggml: rlx_gguf::GgmlType,
    m: usize,
    k: usize,
    n: usize,
) -> Option<f32> {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return None;
    }

    // Weight is row-major [n, k]: output column c owns k contiguous values.
    let w_row: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.011).sin() * 0.5)
        .collect();
    let packed = rlx_gguf::quantize(&w_row, ggml).expect("quantize");
    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.017).sin()).collect();

    let mut g = Graph::new("gguf_dq_prefill");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul { scheme },
        vec![x_in, w],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);

    let run = |device: Device| -> Vec<f32> {
        let mut c = Session::new(device).compile(g.clone());
        c.set_param_typed("w", &packed, DType::U8);
        c.run(&[("x", x.as_slice())]).remove(0)
    };

    let metal = run(Device::Metal);
    let cpu = run(Device::Cpu);
    assert_eq!(metal.len(), m * n, "metal output len");
    assert_eq!(cpu.len(), m * n, "cpu output len");
    let max_abs = cpu
        .iter()
        .zip(&metal)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("gguf_dequant_matmul {scheme:?} m={m} k={k} n={n}: max_abs={max_abs:.6e}");
    Some(max_abs)
}

/// All cases run inside ONE `#[test]` so they execute serially on a single
/// thread. The m > 1 prefill path uses `MPSMatrixMultiplication`, whose global
/// matrix/kernel cache is not safe to build concurrently from the independent
/// `Session`s that parallel test threads create — splitting into separate
/// `#[test]` fns SIGSEGVs under the default multi-thread harness.
#[test]
fn gguf_dequant_matmul_prefill_matches_cpu() {
    // (scheme, ggml, m, k, n, tol, label)
    let cases: &[(
        QuantScheme,
        rlx_gguf::GgmlType,
        usize,
        usize,
        usize,
        f32,
        &str,
    )] = &[
        // Task-specified prefill shape: m=4, one Q4_K superblock (k=256), n=8.
        (
            QuantScheme::GgufQ4K,
            rlx_gguf::GgmlType::Q4K,
            4,
            256,
            8,
            1e-3,
            "Q4_K prefill",
        ),
        (
            QuantScheme::GgufQ6K,
            rlx_gguf::GgmlType::Q6K,
            4,
            256,
            8,
            1e-3,
            "Q6_K prefill",
        ),
        // k=512 (two superblocks), wider n, larger m → real GEMM tiling.
        (
            QuantScheme::GgufQ4K,
            rlx_gguf::GgmlType::Q4K,
            7,
            512,
            16,
            1e-3,
            "Q4_K multi-superblock",
        ),
        (
            QuantScheme::GgufQ6K,
            rlx_gguf::GgmlType::Q6K,
            7,
            512,
            16,
            1e-3,
            "Q6_K multi-superblock",
        ),
        // Decode (m=1) regression guard: fused GEMV path must stay correct.
        (
            QuantScheme::GgufQ4K,
            rlx_gguf::GgmlType::Q4K,
            1,
            256,
            8,
            1e-3,
            "Q4_K decode",
        ),
    ];
    for &(scheme, ggml, m, k, n, tol, label) in cases {
        if let Some(max_abs) = run_case(scheme, ggml, m, k, n) {
            assert!(
                max_abs < tol,
                "{label} Metal vs CPU max_abs={max_abs} (tol {tol})"
            );
        }
    }
}
