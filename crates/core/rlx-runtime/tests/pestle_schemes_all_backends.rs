// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::DequantMatMul` for the two schemes `Doses-AI/Pestle-27B-Ternary-GGUF`
//! needs — **`Q2_0`** (the factor pairs, 978 tensors) and **`G8_0`** (the
//! `token_embd` / `output` tables) — must agree with the CPU reference on
//! every backend.
//!
//! This is not a formality. `dequant_gguf.{msl,cu,wgsl}` are
//! `if (scheme_id == N) { … return; }` chains with **no default branch**, and
//! a scheme reaches them via the shared `define_gguf_gpu_dequant_ids!` table.
//! An id in that table with no matching branch does not raise — the kernel
//! writes nothing and the weights read as **zeros**. That was the state of
//! `Q2_0` on wgpu and CUDA before this test existed: registered id, no
//! branch, silent garbage. `tests/…_scheme_kernel_coverage.rs` in `rlx-ir`
//! guards the table statically; this one checks the arithmetic dynamically.
//!
//! Runs on whatever is available and reports the rest, so the same file
//! covers CPU/Metal/MLX/wgpu on a Mac and CUDA/ROCm/Vulkan on a rig.

use rlx_ir::quant::QuantScheme;
use rlx_ir::*;
use rlx_runtime::{Device, Session};

mod common;

const DEVICES: &[Device] = &[
    Device::Cpu,
    Device::Metal,
    Device::Mlx,
    Device::Cuda,
    Device::Rocm,
    Device::Gpu, // wgpu
    Device::Vulkan,
];

/// Host reference: dequantize `[n, k]` packed bytes, then `x @ Wᵀ`.
fn reference(
    packed: &[u8],
    m: usize,
    k: usize,
    n: usize,
    x: &[f32],
    scheme: QuantScheme,
) -> Vec<f32> {
    let w = match scheme {
        QuantScheme::GgufQ2_0 => rlx_gguf::q2_dequant::dequant_q2_0(packed, n * k).unwrap(),
        QuantScheme::GgufG8_0 => rlx_gguf::g8_dequant::dequant_g8_0(packed, n * k).unwrap(),
        other => panic!("no host reference for {other:?}"),
    };
    let mut out = vec![0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            out[r * n + c] = (0..k).map(|j| x[r * k + j] * w[c * k + j]).sum();
        }
    }
    out
}

fn run_on(
    device: Device,
    packed: &[u8],
    m: usize,
    k: usize,
    n: usize,
    x: &[f32],
    scheme: QuantScheme,
) -> Vec<f32> {
    let mut g = Graph::new("pestle_scheme");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul { scheme },
        vec![x_in, w],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut c = Session::new(device).compile(g);
    c.set_param_typed("w", packed, DType::U8);
    c.run(&[("x", x)]).remove(0)
}

/// Q2_0: f16 group scale + 128 two-bit codes → `(q−1)·d`, 34 bytes / 128.
/// Codes span all four values so a kernel that mapped `q` instead of `q−1`
/// (or dropped the `+2d` code) diverges rather than merely drifting.
fn q2_0_fixture(k: usize, n: usize) -> Vec<u8> {
    let w: Vec<f32> = (0..n * k)
        .map(|i| [-0.5f32, 0.0, 0.5, 1.0][i % 4] * (1.0 + ((i / 128) % 3) as f32))
        .collect();
    rlx_gguf::q2_dequant::quantize_q2_0(&w).expect("quantize Q2_0")
}

/// G8_0: four bf16 scales — one per group of 8 — + 32 two-bit codes,
/// 16 bytes / 32. The four scales are spread 64× apart, so a kernel using a
/// single block-wide scale, or reading the scales as f16 instead of bf16,
/// fails outright.
fn g8_0_fixture(k: usize, n: usize) -> Vec<u8> {
    let w: Vec<f32> = (0..n * k)
        .map(|i| {
            let d = 0.125 * 4f32.powi(((i / 8) % 4) as i32);
            [-1.0f32, 0.0, 1.0][i % 3] * d
        })
        .collect();
    rlx_gguf::g8_dequant::quantize_g8_0(&w).expect("quantize G8_0")
}

fn check(scheme: QuantScheme, fixture: fn(usize, usize) -> Vec<u8>, k: usize) {
    let n = 8usize;
    let packed = fixture(k, n);
    let mut ran = 0usize;
    let mut skipped = Vec::new();

    // m == 1 is the decode GEMV path, m > 1 the prefill path; several
    // backends route those through different kernels.
    for &m in &[1usize, 4] {
        let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.013).sin()).collect();
        let want = reference(&packed, m, k, n, &x, scheme);
        for &dev in DEVICES {
            if common::skip_unless(dev) {
                if m == 1 {
                    skipped.push(dev);
                }
                continue;
            }
            let got = run_on(dev, &packed, m, k, n, &x, scheme);
            assert_eq!(got.len(), want.len(), "{scheme} on {dev:?}: output length");

            // A missing kernel branch yields all-zeros — the specific failure
            // this test exists for, and worth its own message.
            assert!(
                got.iter().any(|v| *v != 0.0),
                "{scheme} on {dev:?} (m={m}) returned all zeros — \
                 the kernel almost certainly has no branch for scheme id {:?}",
                scheme.gpu_dequant_scheme_id()
            );

            // Relative, not absolute: rlx-cpu's `m == 1` Q2_0 GEMV quantizes
            // activations to int8 and dots in integer (`q2_0_dot_q8_block`,
            // llama.cpp-style), so it is deliberately approximate — ~1e-3
            // relative — while the dequant-then-f32-matmul backends land at
            // ~1e-7. One bound that accepts the int8 path still rejects any
            // real decode error, all of which are O(1) relative: a wrong
            // code→value map, a block-wide instead of per-group-of-8 scale,
            // f16-vs-bf16 scale bits, or a shifted bit offset.
            let scale = want
                .iter()
                .map(|v| v.abs())
                .fold(0.0f32, f32::max)
                .max(1e-6);
            let worst = want
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let dot: f32 = want.iter().zip(&got).map(|(a, b)| a * b).sum();
            let na = want.iter().map(|v| v * v).sum::<f32>().sqrt();
            let nb = got.iter().map(|v| v * v).sum::<f32>().sqrt();
            let cos = dot / (na * nb).max(1e-12);
            assert!(
                worst / scale < 5e-3 && cos > 0.9999,
                "{scheme} on {dev:?} (m={m}): rel |Δ| = {} (worst {worst} / scale {scale}), cos = {cos}",
                worst / scale
            );
            ran += 1;
        }
    }
    assert!(ran > 0, "{scheme}: no backend was available to test");
    eprintln!("{scheme}: verified on {ran} device/shape combos; unavailable here: {skipped:?}");
}

#[test]
fn q2_0_dequant_matmul_matches_cpu_on_every_backend() {
    let _gpu = common::serialize_gpu();
    // 256 = two 128-element Q2_0 blocks per row, so block iteration is
    // exercised rather than a single-block special case.
    check(QuantScheme::GgufQ2_0, q2_0_fixture, 256);
}

#[test]
fn g8_0_dequant_matmul_matches_cpu_on_every_backend() {
    let _gpu = common::serialize_gpu();
    check(QuantScheme::GgufG8_0, g8_0_fixture, 128);
}
