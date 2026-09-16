// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native `Op::DequantMoEWeights` on Vulkan vs the rlx-cpu oracle.
//!
//! Dequantises a packed GGUF expert stack — `[E]` slabs of `[N, K]` blocks — to
//! dense `[E, K, N]` f32, i.e. with a per-expert transpose:
//!
//!   `out[e, i, j] = dequant(slab_e)[j * k + i]`
//!
//! **Bit-exact.** The whole op is block decode plus a permutation; there is no
//! accumulation, so nothing legitimately differs and any tolerance would be
//! hiding something.
//!
//! What makes this worth a careful gate is the addressing. Because of the
//! transpose, adjacent output elements come from *different* blocks, so the
//! shader decodes one element at a time rather than expanding a super-block —
//! and that single-element form has to invert, index for index, the layout the
//! block-at-a-time decoders build. Q4_K walks sub-blocks in pairs (32 low
//! nibbles then 32 high, with different scales); Q6_K emits two halves of four
//! quarters sharing a `qh` byte. Invert either wrong and every weight is still
//! a plausible weight.

#![cfg(all(feature = "vulkan", feature = "cpu"))]

use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Dim, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

mod common;

const QK_K: usize = 256;

/// `experts` slabs of `[n, k]` in the given GGUF layout, each expert with a
/// distinct amplitude so a wrong slab stride is visible rather than subtle.
fn stack(scheme: QuantScheme, experts: usize, k: usize, n: usize) -> Vec<u8> {
    let block_bytes = scheme.gguf_block_bytes() as usize;
    let blocks_per_row = k / QK_K;
    let mut packed = vec![0u8; experts * n * blocks_per_row * block_bytes];
    let mut at = 0usize;
    for e in 0..experts {
        let amp = 1.0 + e as f32;
        for row in 0..n {
            for b in 0..blocks_per_row {
                let src: Vec<f32> = (0..QK_K)
                    .map(|i| {
                        let t = (e * 811 + row * 149 + b * 23 + i) as f32;
                        (t * 0.041).sin() * amp
                    })
                    .collect();
                let dst = &mut packed[at..at + block_bytes];
                match scheme {
                    QuantScheme::GgufQ4K => rlx_gguf::quantize::quantize_q4_k_block(&src, dst),
                    QuantScheme::GgufQ6K => rlx_gguf::quantize::quantize_q6_k_block(&src, dst),
                    other => panic!("unsupported in this fixture: {other:?}"),
                }
                at += block_bytes;
            }
        }
    }
    packed
}

fn build(scheme: QuantScheme, experts: usize, k: usize, n: usize, packed_len: usize) -> Graph {
    let mut g = Graph::new("moe_w");
    let w = g.input("w", Shape::new(&[packed_len], DType::U8));
    let y = g.add_node(
        Op::DequantMoEWeights { scheme },
        vec![w],
        Shape::from_dims(
            &[Dim::Static(experts), Dim::Static(k), Dim::Static(n)],
            DType::F32,
        ),
    );
    g.set_outputs(vec![y]);
    g
}

fn run(device: Device, scheme: QuantScheme, experts: usize, k: usize, n: usize) -> Vec<f32> {
    let packed = stack(scheme, experts, k, n);
    let bytes = Session::new(device)
        .compile(build(scheme, experts, k, n, packed.len()))
        .run_typed(&[("w", &packed, DType::U8)])
        .remove(0)
        .0;
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn exact(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{what}: element {i} — vulkan {a} vs cpu {b}"
        );
    }
}

#[test]
fn q4k_matches_cpu_bit_exactly() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Vulkan, "vulkan") {
        return;
    }
    for (experts, k, n) in [(3usize, 256usize, 4usize), (2, 512, 3)] {
        let cpu = run(Device::Cpu, QuantScheme::GgufQ4K, experts, k, n);
        let vk = run(Device::Vulkan, QuantScheme::GgufQ4K, experts, k, n);
        exact(&format!("q4k E={experts} K={k} N={n}"), &vk, &cpu);
    }
}

#[test]
fn q6k_matches_cpu_bit_exactly() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Vulkan, "vulkan") {
        return;
    }
    // Q6_K's inversion is the fiddlier of the two — two halves of four quarters
    // sharing a `qh` byte, with the quarter selecting the 2-bit field, the
    // nibble, the `ql` byte and the scale.
    for (experts, k, n) in [(3usize, 256usize, 4usize), (2, 512, 3)] {
        let cpu = run(Device::Cpu, QuantScheme::GgufQ6K, experts, k, n);
        let vk = run(Device::Vulkan, QuantScheme::GgufQ6K, experts, k, n);
        exact(&format!("q6k E={experts} K={k} N={n}"), &vk, &cpu);
    }
}

#[test]
fn single_expert_still_transposes() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Vulkan, "vulkan") {
        return;
    }
    // One expert removes the slab stride from the picture, so a failure here is
    // the transpose or the element inversion and nothing else.
    let cpu = run(Device::Cpu, QuantScheme::GgufQ4K, 1, 256, 5);
    let vk = run(Device::Vulkan, QuantScheme::GgufQ4K, 1, 256, 5);
    exact("q4k single expert", &vk, &cpu);
}
