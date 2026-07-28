// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! End-to-end test: `Op::DequantGroupedMatMul` on synthetic MoE expert stacks.
//!
//! GPU backend tests acquire [`common::GpuTestGuard`] so they stay safe under
//! the default parallel `cargo test` harness (Metal + wgpu init races otherwise
//! SIGSEGV on Apple Silicon).

mod common;

use common::GpuTestGuard;
use rlx_ir::quant::QuantScheme;
use rlx_ir::*;
use rlx_runtime::{Device, Session};

const QK_K: usize = 256;

fn build_one_q8_k_block(scale: f32, qs: &[i8; QK_K]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(276);
    bytes.extend_from_slice(&scale.to_le_bytes());
    for &q in qs {
        bytes.push(q as u8);
    }
    for _ in 0..(QK_K / 16) {
        bytes.extend_from_slice(&0i16.to_le_bytes());
    }
    bytes
}

/// Packed expert stack: `num_experts` slabs of shape `[n, k]` in GGUF Q8_K layout.
fn build_q8k_expert_stack(
    num_experts: usize,
    k: usize,
    n: usize,
    expert_scales: &[f32],
) -> Vec<u8> {
    assert_eq!(expert_scales.len(), num_experts);
    assert_eq!((k * n) % QK_K, 0);
    let qs: [i8; QK_K] = std::array::from_fn(|i| (i as i32 - 128) as i8);
    let mut packed = Vec::with_capacity(num_experts * n * 292);
    for &scale in expert_scales {
        for _ in 0..n {
            packed.extend_from_slice(&build_one_q8_k_block(scale, &qs));
        }
    }
    packed
}

fn reference_grouped_q8k(
    x: &[f32],
    packed: &[u8],
    expert_idx: &[f32],
    m: usize,
    k: usize,
    n: usize,
    num_experts: usize,
) -> Vec<f32> {
    let slab = (k * n) / QK_K * QuantScheme::GgufQ8K.gguf_block_bytes() as usize;
    let mut out = vec![0f32; m * n];
    for row in 0..m {
        let e = expert_idx[row] as usize;
        assert!(e < num_experts);
        let w_ref = rlx_gguf::dequant_q8_k(&packed[e * slab..(e + 1) * slab], k * n).unwrap();
        for c in 0..n {
            let mut acc = 0f32;
            for i in 0..k {
                acc += x[row * k + i] * w_ref[c * k + i];
            }
            out[row * n + c] = acc;
        }
    }
    out
}

fn run_grouped_q8k_case(device: Device) {
    let _gpu_guard = GpuTestGuard::acquire(device);
    let k = 256;
    let n = 4;
    let m = 5;
    let num_experts = 3;
    // Distinct scales so routing to the wrong expert is obvious.
    let expert_scales = [0.25f32, 0.5, 1.0];
    let packed = build_q8k_expert_stack(num_experts, k, n, &expert_scales);
    let x: Vec<f32> = (0..m * k).map(|i| 0.01 * (i as f32 + 1.0)).collect();
    // Non-contiguous expert ids exercise sort + unpermute in the kernel.
    let expert_idx = vec![1.0, 0.0, 2.0, 1.0, 0.0];
    let expected = reference_grouped_q8k(&x, &packed, &expert_idx, m, k, n, num_experts);

    let mut g = Graph::new("dq_grouped_matmul_q8k");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w_packed = g.param("w_packed", Shape::new(&[packed.len()], DType::U8));
    let idx_in = g.input("expert_idx", Shape::new(&[m], DType::F32));
    let y = g.add_node(
        Op::DequantGroupedMatMul {
            scheme: QuantScheme::GgufQ8K,
        },
        vec![x_in, w_packed, idx_in],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);

    let session = Session::new(device);
    let mut compiled = session.compile(g);
    compiled.set_param_typed("w_packed", &packed, DType::U8);
    let actual = compiled
        .run(&[("x", x.as_slice()), ("expert_idx", expert_idx.as_slice())])
        .pop()
        .unwrap();

    assert_eq!(actual.len(), expected.len());
    for i in 0..actual.len() {
        let diff = (actual[i] - expected[i]).abs();
        let rel = diff / expected[i].abs().max(1.0);
        assert!(
            rel < 1e-4,
            "{device:?} grouped Q8K mismatch at {i}: got {} expected {} (rel {:.2e})",
            actual[i],
            expected[i],
            rel
        );
    }
}

#[test]
fn dequant_grouped_matmul_q8k_matches_per_expert_reference() {
    run_grouped_q8k_case(Device::Cpu);
}

/// Same synthetic stack through F32 `GroupedMatMul` must match packed path.
#[test]
fn dequant_grouped_matmul_q8k_matches_f32_grouped_matmul() {
    let k = 256;
    let n = 4;
    let m = 5;
    let num_experts = 3;
    let expert_scales = [0.25f32, 0.5, 1.0];
    let packed = build_q8k_expert_stack(num_experts, k, n, &expert_scales);
    let x: Vec<f32> = (0..m * k).map(|i| 0.01 * (i as f32 + 1.0)).collect();
    let expert_idx = vec![1.0, 0.0, 2.0, 1.0, 0.0];

    let slab = (k * n) / QK_K * QuantScheme::GgufQ8K.gguf_block_bytes() as usize;
    let mut w_f32 = vec![0f32; num_experts * k * n];
    for e in 0..num_experts {
        let deq = rlx_gguf::dequant_q8_k(&packed[e * slab..(e + 1) * slab], k * n).unwrap();
        // GroupedMatMul uses sgemm with B stored row-major [k, n].
        for i in 0..k {
            for j in 0..n {
                w_f32[e * k * n + i * n + j] = deq[j * k + i];
            }
        }
    }

    let mut g_packed = Graph::new("dq_gmm_packed");
    let x_in = g_packed.input("x", Shape::new(&[m, k], DType::F32));
    let w_p = g_packed.param("w_packed", Shape::new(&[packed.len()], DType::U8));
    let idx_in = g_packed.input("expert_idx", Shape::new(&[m], DType::F32));
    let y_packed = g_packed.add_node(
        Op::DequantGroupedMatMul {
            scheme: QuantScheme::GgufQ8K,
        },
        vec![x_in, w_p, idx_in],
        Shape::new(&[m, n], DType::F32),
    );
    g_packed.set_outputs(vec![y_packed]);

    let mut g_f32 = Graph::new("dq_gmm_f32");
    let x2 = g_f32.input("x", Shape::new(&[m, k], DType::F32));
    let w2 = g_f32.param("w", Shape::new(&[num_experts, k, n], DType::F32));
    let idx2 = g_f32.input("expert_idx", Shape::new(&[m], DType::F32));
    let y_f32 = g_f32.add_node(
        Op::GroupedMatMul,
        vec![x2, w2, idx2],
        Shape::new(&[m, n], DType::F32),
    );
    g_f32.set_outputs(vec![y_f32]);

    let session = Session::new(Device::Cpu);
    let mut exe_packed = session.compile(g_packed);
    exe_packed.set_param_typed("w_packed", &packed, DType::U8);
    let packed_out = exe_packed
        .run(&[("x", x.as_slice()), ("expert_idx", expert_idx.as_slice())])
        .pop()
        .unwrap();

    let mut exe_f32 = session.compile(g_f32);
    exe_f32.set_param("w", &w_f32);
    let f32_out = exe_f32
        .run(&[("x", x.as_slice()), ("expert_idx", expert_idx.as_slice())])
        .pop()
        .unwrap();

    for i in 0..packed_out.len() {
        let diff = (packed_out[i] - f32_out[i]).abs();
        let rel = diff / f32_out[i].abs().max(1.0);
        assert!(
            rel < 1e-4,
            "packed vs F32 GroupedMatMul at {i}: {} vs {} (rel {:.2e})",
            packed_out[i],
            f32_out[i],
            rel
        );
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn dequant_grouped_matmul_q8k_metal_matches_cpu() {
    run_grouped_q8k_case(Device::Metal);
}

const QK4: usize = 32;

/// Packed expert stack: `num_experts` slabs of shape `[n, k]` in GGUF Q4_0 layout.
fn build_q4_0_expert_stack(
    num_experts: usize,
    k: usize,
    n: usize,
    expert_scales: &[f32],
) -> Vec<u8> {
    assert_eq!(expert_scales.len(), num_experts);
    assert_eq!((k * n) % QK4, 0);
    let mut packed = Vec::new();
    for (e, &scale) in expert_scales.iter().enumerate() {
        let slab: Vec<f32> = (0..n * k)
            .map(|i| scale * ((i as f32 + e as f32 * 0.1).sin() * 0.5))
            .collect();
        packed.extend(rlx_gguf::quantize::quantize_q4_0(&slab).expect("quantize_q4_0"));
    }
    packed
}

fn reference_grouped_q4_0(
    x: &[f32],
    packed: &[u8],
    expert_idx: &[f32],
    m: usize,
    k: usize,
    n: usize,
    num_experts: usize,
) -> Vec<f32> {
    let block_bytes = QuantScheme::GgufQ4_0.gguf_block_bytes() as usize;
    let slab = (k * n) / QK4 * block_bytes;
    let mut out = vec![0f32; m * n];
    for row in 0..m {
        let e = expert_idx[row] as usize;
        assert!(e < num_experts);
        let w_ref = rlx_gguf::dequant_q4_0(&packed[e * slab..(e + 1) * slab], k * n).unwrap();
        for c in 0..n {
            let mut acc = 0f32;
            for i in 0..k {
                acc += x[row * k + i] * w_ref[c * k + i];
            }
            out[row * n + c] = acc;
        }
    }
    out
}

fn run_grouped_q4_0_case(device: Device) {
    let _gpu_guard = GpuTestGuard::acquire(device);
    let k = 32;
    let n = 4;
    let m = 5;
    let num_experts = 3;
    let expert_scales = [0.12f32, 0.24, 0.36];
    let packed = build_q4_0_expert_stack(num_experts, k, n, &expert_scales);
    let x: Vec<f32> = (0..m * k).map(|i| 0.02 * (i as f32 + 1.0)).collect();
    let expert_idx = vec![2.0, 0.0, 1.0, 2.0, 0.0];
    let expected = reference_grouped_q4_0(&x, &packed, &expert_idx, m, k, n, num_experts);

    let mut g = Graph::new("dq_grouped_matmul_q4_0");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w_packed = g.param("w_packed", Shape::new(&[packed.len()], DType::U8));
    let idx_in = g.input("expert_idx", Shape::new(&[m], DType::F32));
    let y = g.add_node(
        Op::DequantGroupedMatMul {
            scheme: QuantScheme::GgufQ4_0,
        },
        vec![x_in, w_packed, idx_in],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);

    let session = Session::new(device);
    let mut compiled = session.compile(g);
    compiled.set_param_typed("w_packed", &packed, DType::U8);
    let actual = compiled
        .run(&[("x", x.as_slice()), ("expert_idx", expert_idx.as_slice())])
        .pop()
        .unwrap();

    assert_eq!(actual.len(), expected.len());
    for i in 0..actual.len() {
        let diff = (actual[i] - expected[i]).abs();
        let rel = diff / expected[i].abs().max(1.0);
        assert!(
            rel < 1e-3,
            "{device:?} grouped Q4_0 mismatch at {i}: got {} expected {} (rel {:.2e})",
            actual[i],
            expected[i],
            rel
        );
    }
}

#[test]
fn dequant_grouped_matmul_q4_0_matches_per_expert_reference() {
    run_grouped_q4_0_case(Device::Cpu);
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn dequant_grouped_matmul_q4_0_metal_matches_cpu() {
    run_grouped_q4_0_case(Device::Metal);
}

#[test]
#[cfg(feature = "gpu")]
fn dequant_grouped_matmul_q4_0_wgpu_matches_cpu() {
    if !rlx_runtime::is_available(Device::Gpu) {
        eprintln!("wgpu adapter unavailable, skipping");
        return;
    }
    run_grouped_q4_0_case(Device::Gpu);
}

#[test]
#[cfg(feature = "cuda")]
fn dequant_grouped_matmul_q4_0_cuda_matches_cpu() {
    if !rlx_runtime::is_available(Device::Cuda) {
        eprintln!("CUDA unavailable, skipping");
        return;
    }
    run_grouped_q4_0_case(Device::Cuda);
}

const QK256: usize = 256;

fn build_iq_expert_stack(
    num_experts: usize,
    k: usize,
    n: usize,
    ggml: rlx_gguf::GgmlType,
    expert_scales: &[f32],
) -> Vec<u8> {
    assert_eq!(expert_scales.len(), num_experts);
    assert_eq!((k * n) % QK256, 0);
    let mut packed = Vec::new();
    for (e, &scale) in expert_scales.iter().enumerate() {
        let slab: Vec<f32> = (0..n * k)
            .map(|i| scale * ((i as f32 + e as f32 * 0.1).sin() * 0.5))
            .collect();
        packed.extend(rlx_gguf::quantize(&slab, ggml).expect("quantize"));
    }
    packed
}

fn reference_grouped_iq(
    x: &[f32],
    packed: &[u8],
    expert_idx: &[f32],
    m: usize,
    k: usize,
    n: usize,
    num_experts: usize,
    scheme: QuantScheme,
) -> Vec<f32> {
    let block_bytes = scheme.gguf_block_bytes() as usize;
    let block_elems = scheme.gguf_block_size() as usize;
    let slab = (k * n) / block_elems * block_bytes;
    let mut out = vec![0f32; m * n];
    for row in 0..m {
        let e = expert_idx[row] as usize;
        assert!(e < num_experts);
        let w_ref = rlx_cpu::dequant_cache::gguf_weight_f32(
            0,
            &packed[e * slab..(e + 1) * slab],
            k,
            n,
            scheme,
        );
        for c in 0..n {
            let mut acc = 0f32;
            for i in 0..k {
                acc += x[row * k + i] * w_ref[c * k + i];
            }
            out[row * n + c] = acc;
        }
    }
    out
}

fn run_grouped_iq_case(device: Device, scheme: QuantScheme, ggml: rlx_gguf::GgmlType) {
    let _gpu_guard = GpuTestGuard::acquire(device);
    let k = 256;
    let n = 4;
    let m = 5;
    let num_experts = 3;
    let expert_scales = [0.15f32, 0.30, 0.45];
    let packed = build_iq_expert_stack(num_experts, k, n, ggml, &expert_scales);
    let x: Vec<f32> = (0..m * k).map(|i| 0.02 * (i as f32 + 1.0)).collect();
    let expert_idx = vec![2.0, 0.0, 1.0, 2.0, 0.0];
    let expected = reference_grouped_iq(&x, &packed, &expert_idx, m, k, n, num_experts, scheme);

    let mut g = Graph::new("dq_grouped_matmul_iq");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w_packed = g.param("w_packed", Shape::new(&[packed.len()], DType::U8));
    let idx_in = g.input("expert_idx", Shape::new(&[m], DType::F32));
    let y = g.add_node(
        Op::DequantGroupedMatMul { scheme },
        vec![x_in, w_packed, idx_in],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);

    let session = Session::new(device);
    let mut compiled = session.compile(g);
    compiled.set_param_typed("w_packed", &packed, DType::U8);
    let actual = compiled
        .run(&[("x", x.as_slice()), ("expert_idx", expert_idx.as_slice())])
        .pop()
        .unwrap();

    assert_eq!(actual.len(), expected.len());
    for i in 0..actual.len() {
        let diff = (actual[i] - expected[i]).abs();
        let rel = diff / expected[i].abs().max(1.0);
        assert!(
            rel < 5e-2,
            "{device:?} grouped {scheme:?} mismatch at {i}: got {} expected {} (rel {:.2e})",
            actual[i],
            expected[i],
            rel
        );
    }
}

#[test]
fn dequant_grouped_matmul_iq2_xxs_matches_per_expert_reference() {
    run_grouped_iq_case(
        Device::Cpu,
        QuantScheme::GgufIQ2XXS,
        rlx_gguf::GgmlType::IQ2XXS,
    );
}

#[test]
fn dequant_grouped_matmul_iq3_xxs_matches_per_expert_reference() {
    run_grouped_iq_case(
        Device::Cpu,
        QuantScheme::GgufIQ3XXS,
        rlx_gguf::GgmlType::IQ3XXS,
    );
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn dequant_grouped_matmul_iq2_xxs_metal_matches_cpu() {
    run_grouped_iq_case(
        Device::Metal,
        QuantScheme::GgufIQ2XXS,
        rlx_gguf::GgmlType::IQ2XXS,
    );
}

#[test]
#[cfg(feature = "gpu")]
fn dequant_grouped_matmul_iq2_xxs_wgpu_matches_cpu() {
    if !rlx_runtime::is_available(Device::Gpu) {
        eprintln!("wgpu adapter unavailable, skipping");
        return;
    }
    run_grouped_iq_case(
        Device::Gpu,
        QuantScheme::GgufIQ2XXS,
        rlx_gguf::GgmlType::IQ2XXS,
    );
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn dequant_grouped_matmul_iq3_xxs_metal_matches_cpu() {
    run_grouped_iq_case(
        Device::Metal,
        QuantScheme::GgufIQ3XXS,
        rlx_gguf::GgmlType::IQ3XXS,
    );
}

#[test]
#[cfg(feature = "gpu")]
fn dequant_grouped_matmul_iq3_xxs_wgpu_matches_cpu() {
    if !rlx_runtime::is_available(Device::Gpu) {
        eprintln!("wgpu adapter unavailable, skipping");
        return;
    }
    run_grouped_iq_case(
        Device::Gpu,
        QuantScheme::GgufIQ3XXS,
        rlx_gguf::GgmlType::IQ3XXS,
    );
}

#[test]
#[cfg(feature = "cuda")]
fn dequant_grouped_matmul_iq2_xxs_cuda_matches_cpu() {
    if !rlx_runtime::is_available(Device::Cuda) {
        eprintln!("CUDA unavailable, skipping");
        return;
    }
    run_grouped_iq_case(
        Device::Cuda,
        QuantScheme::GgufIQ2XXS,
        rlx_gguf::GgmlType::IQ2XXS,
    );
}

#[test]
#[cfg(feature = "cuda")]
fn dequant_grouped_matmul_iq3_xxs_cuda_matches_cpu() {
    if !rlx_runtime::is_available(Device::Cuda) {
        eprintln!("CUDA unavailable, skipping");
        return;
    }
    run_grouped_iq_case(
        Device::Cuda,
        QuantScheme::GgufIQ3XXS,
        rlx_gguf::GgmlType::IQ3XXS,
    );
}

#[test]
#[cfg(feature = "rocm")]
fn dequant_grouped_matmul_iq2_xxs_rocm_matches_cpu() {
    if !rlx_runtime::is_available(Device::Rocm) {
        eprintln!("ROCm unavailable, skipping");
        return;
    }
    run_grouped_iq_case(
        Device::Rocm,
        QuantScheme::GgufIQ2XXS,
        rlx_gguf::GgmlType::IQ2XXS,
    );
}

#[test]
#[cfg(feature = "rocm")]
fn dequant_grouped_matmul_iq3_xxs_rocm_matches_cpu() {
    if !rlx_runtime::is_available(Device::Rocm) {
        eprintln!("ROCm unavailable, skipping");
        return;
    }
    run_grouped_iq_case(
        Device::Rocm,
        QuantScheme::GgufIQ3XXS,
        rlx_gguf::GgmlType::IQ3XXS,
    );
}

#[test]
fn dequant_grouped_matmul_iq2_s_matches_per_expert_reference() {
    run_grouped_iq_case(Device::Cpu, QuantScheme::GgufIQ2S, rlx_gguf::GgmlType::IQ2S);
}

#[test]
fn dequant_grouped_matmul_iq3_s_matches_per_expert_reference() {
    run_grouped_iq_case(Device::Cpu, QuantScheme::GgufIQ3S, rlx_gguf::GgmlType::IQ3S);
}

#[test]
fn dequant_grouped_matmul_tq2_0_matches_per_expert_reference() {
    run_grouped_iq_case(
        Device::Cpu,
        QuantScheme::GgufTQ2_0,
        rlx_gguf::GgmlType::TQ2_0,
    );
}

#[test]
fn dequant_grouped_matmul_iq1_s_matches_per_expert_reference() {
    run_grouped_iq_case(Device::Cpu, QuantScheme::GgufIQ1S, rlx_gguf::GgmlType::IQ1S);
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn dequant_grouped_matmul_iq2_s_metal_matches_cpu() {
    run_grouped_iq_case(
        Device::Metal,
        QuantScheme::GgufIQ2S,
        rlx_gguf::GgmlType::IQ2S,
    );
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn dequant_grouped_matmul_iq3_s_metal_matches_cpu() {
    run_grouped_iq_case(
        Device::Metal,
        QuantScheme::GgufIQ3S,
        rlx_gguf::GgmlType::IQ3S,
    );
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn dequant_grouped_matmul_tq2_0_metal_matches_cpu() {
    run_grouped_iq_case(
        Device::Metal,
        QuantScheme::GgufTQ2_0,
        rlx_gguf::GgmlType::TQ2_0,
    );
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn dequant_grouped_matmul_iq1_s_metal_matches_cpu() {
    run_grouped_iq_case(
        Device::Metal,
        QuantScheme::GgufIQ1S,
        rlx_gguf::GgmlType::IQ1S,
    );
}

#[test]
#[cfg(feature = "gpu")]
fn dequant_grouped_matmul_iq2_s_wgpu_matches_cpu() {
    if !rlx_runtime::is_available(Device::Gpu) {
        eprintln!("wgpu adapter unavailable, skipping");
        return;
    }
    run_grouped_iq_case(Device::Gpu, QuantScheme::GgufIQ2S, rlx_gguf::GgmlType::IQ2S);
}

#[test]
#[cfg(feature = "gpu")]
fn dequant_grouped_matmul_iq3_s_wgpu_matches_cpu() {
    if !rlx_runtime::is_available(Device::Gpu) {
        eprintln!("wgpu adapter unavailable, skipping");
        return;
    }
    run_grouped_iq_case(Device::Gpu, QuantScheme::GgufIQ3S, rlx_gguf::GgmlType::IQ3S);
}

#[test]
#[cfg(feature = "gpu")]
fn dequant_grouped_matmul_tq2_0_wgpu_matches_cpu() {
    if !rlx_runtime::is_available(Device::Gpu) {
        eprintln!("wgpu adapter unavailable, skipping");
        return;
    }
    run_grouped_iq_case(
        Device::Gpu,
        QuantScheme::GgufTQ2_0,
        rlx_gguf::GgmlType::TQ2_0,
    );
}

#[test]
#[cfg(feature = "gpu")]
fn dequant_grouped_matmul_iq1_s_wgpu_matches_cpu() {
    if !rlx_runtime::is_available(Device::Gpu) {
        eprintln!("wgpu adapter unavailable, skipping");
        return;
    }
    run_grouped_iq_case(Device::Gpu, QuantScheme::GgufIQ1S, rlx_gguf::GgmlType::IQ1S);
}

#[test]
#[cfg(feature = "cuda")]
fn dequant_grouped_matmul_iq2_s_cuda_matches_cpu() {
    if !rlx_runtime::is_available(Device::Cuda) {
        eprintln!("CUDA unavailable, skipping");
        return;
    }
    run_grouped_iq_case(
        Device::Cuda,
        QuantScheme::GgufIQ2S,
        rlx_gguf::GgmlType::IQ2S,
    );
}

#[test]
#[cfg(feature = "cuda")]
fn dequant_grouped_matmul_iq3_s_cuda_matches_cpu() {
    if !rlx_runtime::is_available(Device::Cuda) {
        eprintln!("CUDA unavailable, skipping");
        return;
    }
    run_grouped_iq_case(
        Device::Cuda,
        QuantScheme::GgufIQ3S,
        rlx_gguf::GgmlType::IQ3S,
    );
}

#[test]
#[cfg(feature = "cuda")]
fn dequant_grouped_matmul_tq2_0_cuda_matches_cpu() {
    if !rlx_runtime::is_available(Device::Cuda) {
        eprintln!("CUDA unavailable, skipping");
        return;
    }
    run_grouped_iq_case(
        Device::Cuda,
        QuantScheme::GgufTQ2_0,
        rlx_gguf::GgmlType::TQ2_0,
    );
}
