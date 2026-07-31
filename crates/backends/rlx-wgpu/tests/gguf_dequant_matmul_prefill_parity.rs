// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Packed GGUF `Op::DequantMatMul` prefill parity (m > 1) — wgpu vs CPU.

use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

fn run_case(
    scheme: QuantScheme,
    ggml: rlx_gguf::GgmlType,
    m: usize,
    k: usize,
    n: usize,
) -> Option<f32> {
    if !rlx_runtime::is_available(Device::Gpu) {
        eprintln!("skip: wgpu unavailable");
        return None;
    }

    let w_row: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.011).sin() * 0.5)
        .collect();
    let packed = rlx_gguf::quantize(&w_row, ggml).expect("quantize");
    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.017).sin()).collect();

    let mut g = Graph::new("wgpu_gguf_dq_prefill");
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

    let gpu = run(Device::Gpu);
    let cpu = run(Device::Cpu);
    assert_eq!(gpu.len(), m * n);
    assert_eq!(cpu.len(), m * n);
    let max_abs = cpu
        .iter()
        .zip(&gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("wguf_dequant_matmul {scheme:?} m={m} k={k} n={n}: max_abs={max_abs:.6e}");
    Some(max_abs)
}

#[test]
fn wgpu_gguf_dequant_matmul_prefill_matches_cpu() {
    let cases: &[(QuantScheme, rlx_gguf::GgmlType, usize, usize, usize, f32)] = &[
        (
            QuantScheme::GgufQ4K,
            rlx_gguf::GgmlType::Q4K,
            4,
            256,
            8,
            1e-2,
        ),
        (
            QuantScheme::GgufQ6K,
            rlx_gguf::GgmlType::Q6K,
            4,
            256,
            8,
            1e-2,
        ),
        (
            QuantScheme::GgufQ4K,
            rlx_gguf::GgmlType::Q4K,
            21,
            256,
            640,
            1e-2,
        ),
        (
            QuantScheme::GgufQ4K,
            rlx_gguf::GgmlType::Q4K,
            21,
            640,
            2048,
            1e-2,
        ),
        (
            QuantScheme::GgufQ6K,
            rlx_gguf::GgmlType::Q6K,
            21,
            2048,
            640,
            1e-2,
        ),
        // Gemma 4 E2B double-wide MLP shapes (intermediate=12288) — the wgpu
        // KV-shared block error concentrates in these. gate/up: k=1536→n=12288;
        // down: k=12288→n=1536.
        (
            QuantScheme::GgufQ4K,
            rlx_gguf::GgmlType::Q4K,
            5,
            1536,
            12288,
            1e-2,
        ),
        (
            QuantScheme::GgufQ4K,
            rlx_gguf::GgmlType::Q4K,
            5,
            12288,
            1536,
            1e-2,
        ),
        // Gemma 3 270M unsloth GGUF: q/k Q5_0, v Q8_0 (hidden=640).
        (
            QuantScheme::GgufQ5_0,
            rlx_gguf::GgmlType::Q5_0,
            21,
            640,
            1024,
            1e-2,
        ),
        (
            QuantScheme::GgufQ5_0,
            rlx_gguf::GgmlType::Q5_0,
            21,
            640,
            256,
            1e-2,
        ),
        (
            QuantScheme::GgufQ8_0,
            rlx_gguf::GgmlType::Q8_0,
            21,
            640,
            256,
            1e-2,
        ),
        // Custom 1-bit Q1_0 (prism-ml Bonsai-27B): 128-elem blocks. k a
        // multiple of 128; wgpu on-device dequant_gguf path vs CPU.
        (
            QuantScheme::GgufQ1_0,
            rlx_gguf::GgmlType::Q1_0,
            5,
            512,
            256,
            1e-2,
        ),
        // Bonsai-ish prefill tiles (hidden=5120).
        (
            QuantScheme::GgufQ1_0,
            rlx_gguf::GgmlType::Q1_0,
            33,
            512,
            512,
            1e-2,
        ),
        (
            QuantScheme::GgufQ1_0,
            rlx_gguf::GgmlType::Q1_0,
            8,
            5120,
            5120,
            2e-2,
        ),
        (
            QuantScheme::GgufQ1_0,
            rlx_gguf::GgmlType::Q1_0,
            33,
            5120,
            5120,
            2e-2,
        ),
    ];
    for (scheme, ggml, m, k, n, tol) in cases {
        let Some(max_abs) = run_case(*scheme, *ggml, *m, *k, *n) else {
            return;
        };
        assert!(
            max_abs <= *tol,
            "wgpu prefill {scheme:?} m={m} max_abs {max_abs} > {tol}"
        );
    }
}

/// Pack one FV5 block (104 bytes) from 256 five-value codes in {-2,-1,0,1,2}.
fn pack_fv5_block(codes: &[i8], s_lo: f32, s_hi: f32) -> Vec<u8> {
    let mut b = vec![0u8; 104];
    b[0..4].copy_from_slice(&s_lo.to_le_bytes());
    b[4..8].copy_from_slice(&s_hi.to_le_bytes());
    for (j, &c) in codes.iter().enumerate() {
        let (byte, bit) = (j / 8, 1u8 << (j % 8));
        let (p, ng, hi) = match c {
            1 => (true, false, false),
            2 => (true, false, true),
            -1 => (false, true, false),
            -2 => (false, true, true),
            _ => (false, false, false),
        };
        if p {
            b[8 + byte] |= bit;
        }
        if ng {
            b[40 + byte] |= bit;
        }
        if hi {
            b[72 + byte] |= bit;
        }
    }
    b
}

// End-to-end FV5 (Neutrino ternary) DequantMatMul: exercises the full
// lower → dispatch → dequant-to-scratch → matmul path on wgpu vs CPU.
// FV5 has no float quantizer (packs are made offline), so we pack directly.
#[test]
fn wgpu_fv5_dequant_matmul_prefill_matches_cpu() {
    if !rlx_runtime::is_available(Device::Gpu) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let (m, k, n) = (4usize, 256usize, 8usize); // k a multiple of 256 → 1 block/row
    let mut packed = Vec::new();
    for row in 0..n {
        let codes: [i8; 256] = std::array::from_fn(|j| match (j + row) % 5 {
            0 => 0,
            1 => 1,
            2 => 2,
            3 => -1,
            _ => -2,
        });
        packed.extend_from_slice(&pack_fv5_block(&codes, 0.05, 0.2));
    }
    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.017).sin()).collect();

    let mut g = Graph::new("wgpu_fv5_dq_prefill");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufFV5,
        },
        vec![x_in, w],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);

    let run = |device: Device| -> Vec<f32> {
        let mut c = Session::new(device).compile(g.clone());
        c.set_param_typed("w", &packed, DType::U8);
        c.run(&[("x", x.as_slice())]).remove(0)
    };
    let gpu = run(Device::Gpu);
    let cpu = run(Device::Cpu);
    let max_abs = cpu
        .iter()
        .zip(&gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("wgpu FV5 matmul m={m} k={k} n={n}: max_abs={max_abs:.6e}");
    assert!(max_abs <= 1e-4, "wgpu FV5 prefill max_abs {max_abs} > 1e-4");
}
