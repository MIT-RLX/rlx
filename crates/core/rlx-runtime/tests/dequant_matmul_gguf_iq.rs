// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// End-to-end `Op::DequantMatMul` for IQ / legacy GGUF schemes encoded via
// `rlx_gguf::quantize` — compares graph output to dequant + matmul reference.

mod common;

use common::GpuTestGuard;
use rlx_ir::quant::QuantScheme;
use rlx_ir::*;
use rlx_runtime::{Device, Session};

fn run_dequant_matmul_case(
    device: Device,
    scheme: QuantScheme,
    ggml: rlx_gguf::GgmlType,
    k: usize,
    n: usize,
    m: usize,
    rel_tol: f32,
) {
    let _gpu_guard = GpuTestGuard::acquire(device);
    let w_row: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.013).sin() * 0.45)
        .collect();
    let packed = rlx_gguf::quantize(&w_row, ggml).expect("quantize");
    let w_ref = rlx_cpu::dequant_cache::gguf_weight_f32(0, &packed, k, n, scheme);
    let x: Vec<f32> = (0..m * k).map(|i| 0.02 * (i as f32 + 1.0)).collect();

    let mut expected = vec![0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0f32;
            for i in 0..k {
                acc += x[r * k + i] * w_ref[c * k + i];
            }
            expected[r * n + c] = acc;
        }
    }

    let mut g = Graph::new("dq_matmul_iq");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w_packed = g.param("w_packed", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul { scheme },
        vec![x_in, w_packed],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);

    let session = Session::new(device);
    let mut compiled = session.compile(g);
    compiled.set_param_typed("w_packed", &packed, DType::U8);
    let actual = compiled.run(&[("x", x.as_slice())]).pop().unwrap();

    for i in 0..actual.len() {
        let rel = (actual[i] - expected[i]).abs() / expected[i].abs().max(1.0);
        assert!(
            rel < rel_tol,
            "{device:?} {scheme:?} mismatch at {i}: got {} expected {} (rel {:.2e})",
            actual[i],
            expected[i],
            rel
        );
    }
}

#[test]
fn dequant_matmul_iq2_xxs_matches_reference() {
    run_dequant_matmul_case(
        Device::Cpu,
        QuantScheme::GgufIQ2XXS,
        rlx_gguf::GgmlType::IQ2XXS,
        256,
        4,
        2,
        0.08,
    );
}

#[test]
fn dequant_matmul_q4_1_matches_reference() {
    run_dequant_matmul_case(
        Device::Cpu,
        QuantScheme::GgufQ4_1,
        rlx_gguf::GgmlType::Q4_1,
        32,
        8,
        3,
        1e-2,
    );
}

#[test]
fn dequant_matmul_q5_0_matches_reference() {
    run_dequant_matmul_case(
        Device::Cpu,
        QuantScheme::GgufQ5_0,
        rlx_gguf::GgmlType::Q5_0,
        32,
        8,
        3,
        1e-2,
    );
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn dequant_matmul_iq2_xxs_metal_matches_cpu() {
    run_dequant_matmul_case(
        Device::Metal,
        QuantScheme::GgufIQ2XXS,
        rlx_gguf::GgmlType::IQ2XXS,
        256,
        4,
        2,
        0.08,
    );
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn dequant_matmul_q4_1_metal_matches_cpu() {
    run_dequant_matmul_case(
        Device::Metal,
        QuantScheme::GgufQ4_1,
        rlx_gguf::GgmlType::Q4_1,
        32,
        8,
        3,
        1e-2,
    );
}

#[test]
#[cfg(feature = "gpu")]
fn dequant_matmul_iq2_xxs_wgpu_matches_cpu() {
    if !rlx_runtime::is_available(Device::Gpu) {
        eprintln!("wgpu adapter unavailable, skipping");
        return;
    }
    run_dequant_matmul_case(
        Device::Gpu,
        QuantScheme::GgufIQ2XXS,
        rlx_gguf::GgmlType::IQ2XXS,
        256,
        4,
        2,
        0.08,
    );
}

#[test]
#[cfg(all(target_os = "macos", feature = "mlx"))]
fn dequant_matmul_q4_1_mlx_matches_cpu() {
    if !rlx_runtime::is_available(Device::Mlx) {
        eprintln!("MLX unavailable, skipping");
        return;
    }
    run_dequant_matmul_case(
        Device::Mlx,
        QuantScheme::GgufQ4_1,
        rlx_gguf::GgmlType::Q4_1,
        32,
        8,
        3,
        1e-2,
    );
}

#[test]
#[cfg(all(target_os = "macos", feature = "mlx"))]
fn dequant_matmul_iq2_xxs_mlx_matches_cpu() {
    if !rlx_runtime::is_available(Device::Mlx) {
        eprintln!("MLX unavailable, skipping");
        return;
    }
    run_dequant_matmul_case(
        Device::Mlx,
        QuantScheme::GgufIQ2XXS,
        rlx_gguf::GgmlType::IQ2XXS,
        256,
        4,
        2,
        0.08,
    );
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn dequant_matmul_iq2_s_metal_fused_mv() {
    run_dequant_matmul_case(
        Device::Metal,
        QuantScheme::GgufIQ2S,
        rlx_gguf::GgmlType::IQ2S,
        256,
        4,
        1,
        0.08,
    );
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn dequant_matmul_iq3_s_metal_fused_mv() {
    run_dequant_matmul_case(
        Device::Metal,
        QuantScheme::GgufIQ3S,
        rlx_gguf::GgmlType::IQ3S,
        256,
        4,
        1,
        0.08,
    );
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn dequant_matmul_iq1_s_metal_fused_mv() {
    run_dequant_matmul_case(
        Device::Metal,
        QuantScheme::GgufIQ1S,
        rlx_gguf::GgmlType::IQ1S,
        256,
        4,
        1,
        0.10,
    );
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn dequant_matmul_iq1_m_metal_fused_mv() {
    run_dequant_matmul_case(
        Device::Metal,
        QuantScheme::GgufIQ1M,
        rlx_gguf::GgmlType::IQ1M,
        256,
        4,
        1,
        0.10,
    );
}

#[test]
#[cfg(feature = "cuda")]
fn dequant_matmul_iq2_xxs_cuda_matches_cpu() {
    if !rlx_runtime::is_available(Device::Cuda) {
        eprintln!("CUDA unavailable, skipping");
        return;
    }
    run_dequant_matmul_case(
        Device::Cuda,
        QuantScheme::GgufIQ2XXS,
        rlx_gguf::GgmlType::IQ2XXS,
        256,
        4,
        2,
        0.08,
    );
}
