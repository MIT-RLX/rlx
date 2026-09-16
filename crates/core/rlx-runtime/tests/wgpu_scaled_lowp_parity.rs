// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native `scaled_lowp.wgsl` (quant-scale / quantize / dequantize / decode-GEMM)
//! vs the rlx-cpu oracle, across every `ScaledFormat` and scale layout.
//!
//! These four ops used to take the generic CPU host route on wgpu — a readback,
//! a CPU pass and an upload per op — even though the harder grouped (MoE)
//! variant already had a native decode kernel. That kernel is MXFP4-only, so
//! this is a full codec port rather than a rewire, and a codec port is exactly
//! the thing that goes subtly wrong: a mis-decoded exponent bias or a
//! mis-rounded code produces plausible numbers, not an error.
//!
//! Two gates, split where the two implementations genuinely stop being the same
//! computation:
//!
//! * **Round-trip is bit-exact.** quantize → dequantize is elementwise, so
//!   there is no accumulation order to differ. Every format, every layout, every
//!   element. This is what pins the codec.
//! * **The GEMM is 1e-5 relative.** `scaled_matmul_decode` accumulates over a
//!   16-wide tile while rlx-cpu accumulates in a straight loop, so f32 sums
//!   land in a different order. That is a reordering difference, not a decode
//!   difference — and the round-trip gate above is what rules the latter out.

#![cfg(all(feature = "gpu", feature = "cpu"))]

use rlx_ir::{DType, Graph, ScaleLayout, ScaledFormat, Shape};
use rlx_runtime::{Device, Session};

mod common;

/// Every named format, plus one parameterized `fNeXmY` so the generic decode
/// path (top-bit-set `kernel_id`) is covered too.
fn formats() -> Vec<(&'static str, ScaledFormat)> {
    vec![
        ("e4m3", ScaledFormat::F8E4M3),
        ("e5m2", ScaledFormat::F8E5M2),
        ("e4m3fnuz", ScaledFormat::F8E4M3Fnuz),
        ("e5m2fnuz", ScaledFormat::F8E5M2Fnuz),
        ("e2m3", ScaledFormat::F6E2M3),
        ("e3m2", ScaledFormat::F6E3M2),
        ("e2m1", ScaledFormat::F4E2M1),
        ("custom f4e3m0", ScaledFormat::custom(3, 0)),
    ]
}

fn layouts() -> Vec<(&'static str, ScaleLayout)> {
    vec![
        ("per_tensor", ScaleLayout::PerTensor),
        ("mx_e8m0", ScaleLayout::mx()),
        ("nvfp4", ScaleLayout::nvfp4()),
    ]
}

fn wave(n: usize, phase: f32, amp: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * phase).sin() * amp).collect()
}

fn run(device: Device, g: Graph, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    Session::new(device).compile(g).run(inputs).remove(0)
}

// ---------------------------------------------------------------------------
// Codec: quantize -> dequantize, bit-exact
// ---------------------------------------------------------------------------

fn roundtrip_graph(rows: usize, cols: usize, fmt: ScaledFormat, layout: ScaleLayout) -> Graph {
    let mut g = Graph::new("lowp_rt");
    let x = g.input("x", Shape::new(&[rows, cols], DType::F32));
    let (codes, scale) = g.scaled_quantize(x, fmt, layout);
    let y = g.scaled_dequantize(codes, scale, fmt, layout);
    g.set_outputs(vec![y]);
    g
}

#[test]
fn quantize_dequantize_roundtrip_is_bit_exact_on_wgpu() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }
    // `cols` is not a multiple of the MX block (32) or the NVFP4 group (16), so
    // the ragged last block is exercised rather than assumed away.
    let (rows, cols) = (5usize, 40usize);
    let x = wave(rows * cols, 0.21, 3.0);

    for (fname, fmt) in formats() {
        for (lname, layout) in layouts() {
            let cpu = run(
                Device::Cpu,
                roundtrip_graph(rows, cols, fmt, layout),
                &[("x", &x)],
            );
            let gpu = run(
                Device::Gpu,
                roundtrip_graph(rows, cols, fmt, layout),
                &[("x", &x)],
            );
            assert_eq!(
                gpu.len(),
                cpu.len(),
                "{fname}/{lname}: length {} vs {}",
                gpu.len(),
                cpu.len()
            );
            for (i, (a, b)) in gpu.iter().zip(&cpu).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{fname}/{lname}: element {i} — wgpu {a} vs cpu {b} (input {})",
                    x[i]
                );
            }
        }
    }
}

#[test]
fn roundtrip_survives_the_hard_inputs() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }
    // Zeros, denormals, values far past every format's max (so the encoder's
    // saturate path runs), and exact negatives.
    let mut x = vec![
        0.0f32,
        -0.0,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        1e-30,
        -1e-30,
        1e30,
        -1e30,
        6.0,
        -6.0,
        448.0,
        -448.0,
        57344.0,
        -57344.0,
        0.5,
        -0.5,
    ];
    x.extend(wave(16, 0.9, 900.0));
    let (rows, cols) = (2usize, 16usize);

    for (fname, fmt) in formats() {
        for (lname, layout) in layouts() {
            let cpu = run(
                Device::Cpu,
                roundtrip_graph(rows, cols, fmt, layout),
                &[("x", &x)],
            );
            let gpu = run(
                Device::Gpu,
                roundtrip_graph(rows, cols, fmt, layout),
                &[("x", &x)],
            );
            for (i, (a, b)) in gpu.iter().zip(&cpu).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{fname}/{lname}: element {i} — wgpu {a} vs cpu {b} (input {})",
                    x[i]
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// decode-GEMM
// ---------------------------------------------------------------------------

fn matmul_graph(
    m: usize,
    k: usize,
    n: usize,
    fmt: ScaledFormat,
    layout: ScaleLayout,
    bias: bool,
) -> Graph {
    let mut g = Graph::new("lowp_mm");
    // TN: both operands are K-last, so `w` is [n, k]. `scaled_matmul_bias`
    // takes f32 and inserts the ScaledQuantScale + ScaledQuantize chain itself,
    // so this one graph exercises all four kernels.
    let x = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.input("w", Shape::new(&[n, k], DType::F32));
    let b = if bias {
        Some(g.input("b", Shape::new(&[n], DType::F32)))
    } else {
        None
    };
    let y = g.scaled_matmul_bias(x, w, b, fmt, layout);
    g.set_outputs(vec![y]);
    g
}

fn close(what: &str, gpu: &[f32], cpu: &[f32]) {
    assert_eq!(gpu.len(), cpu.len(), "{what}: length");
    let scale = cpu.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    let mut worst = 0f32;
    for (i, (a, b)) in gpu.iter().zip(cpu).enumerate() {
        let rel = (a - b).abs() / scale;
        worst = worst.max(rel);
        assert!(
            rel <= 1e-5,
            "{what}: element {i} — wgpu {a} vs cpu {b} (rel {rel:e})"
        );
    }
    eprintln!("{what}: max rel diff {worst:.2e}");
}

#[test]
fn scaled_matmul_decode_matches_cpu_on_wgpu() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }
    // m and n straddle the kernel's 16x16 tile so the edge guards run; k is not
    // a multiple of the tile or of the MX block.
    let (m, k, n) = (18usize, 40usize, 20usize);
    let x = wave(m * k, 0.13, 1.5);
    let w = wave(n * k, 0.07, 1.2);
    let b: Vec<f32> = wave(n, 0.31, 0.4);

    for (fname, fmt) in formats() {
        for (lname, layout) in layouts() {
            for bias in [false, true] {
                let g = || matmul_graph(m, k, n, fmt, layout, bias);
                let mut inputs: Vec<(&str, &[f32])> = vec![("x", &x), ("w", &w)];
                if bias {
                    inputs.push(("b", &b));
                }
                let cpu = run(Device::Cpu, g(), &inputs);
                let gpu = run(Device::Gpu, g(), &inputs);
                close(&format!("{fname}/{lname}/bias={bias}"), &gpu, &cpu);
            }
        }
    }
}

#[test]
fn scaled_matmul_decode_handles_a_single_tile() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }
    // Everything smaller than one 16x16 tile — the all-edges case.
    let (m, k, n) = (3usize, 7usize, 5usize);
    let x = wave(m * k, 0.41, 2.0);
    let w = wave(n * k, 0.17, 1.7);
    for (fname, fmt) in formats() {
        let layout = ScaleLayout::mx();
        let g = || matmul_graph(m, k, n, fmt, layout, false);
        let cpu = run(Device::Cpu, g(), &[("x", &x), ("w", &w)]);
        let gpu = run(Device::Gpu, g(), &[("x", &x), ("w", &w)]);
        close(&format!("{fname}/single-tile"), &gpu, &cpu);
    }
}
