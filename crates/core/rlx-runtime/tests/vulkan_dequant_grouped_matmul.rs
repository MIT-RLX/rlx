// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native fused GGUF dequant + grouped (MoE) GEMM on Vulkan vs the rlx-cpu
//! oracle.
//!
//! `Op::DequantGroupedMatMul` took the CPU host fallback on Vulkan.
//! `dequant_grouped_matmul.comp` handles the three schemes Vulkan's existing
//! `dequant_matmul.comp` decodes — Q4_K, Q6_K, Q1_0 — by decoding the routed
//! expert's row inside the accumulation loop.
//!
//! That is deliberately *not* CUDA's design. CUDA dequantises a whole `k*n`
//! slab into a scratch arena region and then runs a dense GEMM, which needs
//! `k*n*4` bytes of scratch per call plus a host round trip to sort tokens by
//! expert. Decoding in the loop needs neither, which is the point for the large
//! expert stacks this op exists for.
//!
//! The gate is a tolerance rather than equality because the two implementations
//! accumulate the same products in a different association — rlx-cpu dequantises
//! a row then dots it, this decodes block by block. The decode itself is exact
//! integer/f16 arithmetic, so the tolerance is tight.
//!
//! Coverage that matters here: **more than one expert, and tokens that actually
//! route to different ones**. A single-expert case passes with the expert index
//! ignored entirely, which is the most likely way to get this wrong.

#![cfg(all(feature = "vulkan", feature = "cpu"))]

use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

mod common;

const QK_K: usize = 256;

fn wave(n: usize, phase: f32, amp: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * phase).sin() * amp).collect()
}

/// Packed expert stack: `experts` slabs of `[n, k]` in GGUF Q4_K layout.
///
/// Each expert gets a distinct amplitude so a kernel that ignores the expert
/// index produces visibly wrong numbers rather than a near-miss.
fn q4k_stack(experts: usize, k: usize, n: usize) -> Vec<u8> {
    assert_eq!(k % QK_K, 0);
    let blocks_per_row = k / QK_K;
    let mut packed = vec![0u8; experts * n * blocks_per_row * 144];
    let mut at = 0usize;
    for e in 0..experts {
        let amp = 1.0 + e as f32;
        for row in 0..n {
            for b in 0..blocks_per_row {
                let src: Vec<f32> = (0..QK_K)
                    .map(|i| {
                        let t = (e * 977 + row * 131 + b * 17 + i) as f32;
                        (t * 0.037).sin() * amp
                    })
                    .collect();
                rlx_gguf::quantize::quantize_q4_k_block(&src, &mut packed[at..at + 144]);
                at += 144;
            }
        }
    }
    packed
}

fn build(m: usize, k: usize, n: usize, experts: usize) -> Graph {
    let mut g = Graph::new("dq_grouped");
    let x = g.input("x", Shape::new(&[m, k], DType::F32));
    let blocks = (k * n) / QK_K * experts;
    let w = g.input("w", Shape::new(&[blocks * 144], DType::U8));
    let idx = g.input("idx", Shape::new(&[m], DType::F32));
    let y = g.dequant_grouped_matmul_packed(
        x,
        w,
        idx,
        QuantScheme::GgufQ4K,
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    g
}

fn run(device: Device, m: usize, k: usize, n: usize, experts: usize, idx: &[f32]) -> Vec<f32> {
    let x = wave(m * k, 0.11, 1.0);
    let packed = q4k_stack(experts, k, n);
    let xb: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let ib: Vec<u8> = idx.iter().flat_map(|v| v.to_le_bytes()).collect();
    let bytes = Session::new(device)
        .compile(build(m, k, n, experts))
        .run_typed(&[
            ("x", &xb, DType::F32),
            ("w", &packed, DType::U8),
            ("idx", &ib, DType::F32),
        ])
        .remove(0)
        .0;
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn close(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);
    let mut worst = 0f32;
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        let rel = (a - b).abs() / scale;
        worst = worst.max(rel);
        // 5e-5, not 1e-5. Both sides now decode the same exact integer/f16
        // codes, but they associate the sum differently (rlx-cpu dequantises a
        // row then dots it; the shader decodes block by block), and these
        // dot products cancel hard — `sum|x_i*w_i|` is ~315 for a result of
        // ~2.6, so f32 rounding shows up ~120x amplified against `scale`.
        // The old 1e-5 was never exercised on real values: until the U8
        // `run_typed` byte path was fixed on both sides, both backends decoded
        // the *same* f32-lane garbage and matched bit-exactly, so any
        // tolerance passed. Observed worst case here is 1.0044e-5.
        assert!(
            rel <= 5e-5,
            "{what}: element {i} — vulkan {a} vs cpu {b} (rel {rel:e})"
        );
    }
    eprintln!("{what}: max rel diff {worst:.2e}");
}

#[test]
fn q4k_grouped_matmul_matches_cpu() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Vulkan, "vulkan") {
        return;
    }
    let (m, k, n, experts) = (5usize, 256usize, 4usize, 3usize);
    // Tokens spread across all three experts, and not in order — a kernel that
    // read the index off by a row would still look plausible on a sorted one.
    let idx = vec![2.0f32, 0.0, 1.0, 2.0, 0.0];
    let cpu = run(Device::Cpu, m, k, n, experts, &idx);
    let vk = run(Device::Vulkan, m, k, n, experts, &idx);
    close("q4k grouped", &vk, &cpu);
}

#[test]
fn every_token_on_one_expert_still_matches() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Vulkan, "vulkan") {
        return;
    }
    // The degenerate routing, run for each expert in turn: this is what catches
    // an expert-slab stride that is wrong by a constant factor.
    let (m, k, n, experts) = (3usize, 256usize, 5usize, 3usize);
    for e in 0..experts {
        let idx = vec![e as f32; m];
        let cpu = run(Device::Cpu, m, k, n, experts, &idx);
        let vk = run(Device::Vulkan, m, k, n, experts, &idx);
        close(&format!("q4k all-on-expert-{e}"), &vk, &cpu);
    }
}

#[test]
fn multi_block_k_matches_cpu() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Vulkan, "vulkan") {
        return;
    }
    // k = 512 is two Q4_K super-blocks per row, so the per-row block stride is
    // exercised rather than assumed.
    let (m, k, n, experts) = (4usize, 512usize, 3usize, 2usize);
    let idx = vec![1.0f32, 0.0, 1.0, 0.0];
    let cpu = run(Device::Cpu, m, k, n, experts, &idx);
    let vk = run(Device::Vulkan, m, k, n, experts, &idx);
    close("q4k k=512", &vk, &cpu);
}

#[test]
fn out_of_range_expert_index_is_clamped_like_cpu() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Vulkan, "vulkan") {
        return;
    }
    // Not a supported input so much as a guard that a bad index cannot read
    // outside the stack: the shader clamps, and the result has to stay finite.
    let (m, k, n, experts) = (3usize, 256usize, 3usize, 2usize);
    let idx = vec![0.0f32, 1.0, 5.0];
    let vk = run(Device::Vulkan, m, k, n, experts, &idx);
    assert!(
        vk.iter().all(|v| v.is_finite()),
        "clamped expert index produced non-finite output: {vk:?}"
    );
}
