// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Native low-precision `Op::ScaledDequantize` on Metal. Apple GPUs have no FP8
//! units, so Metal runs the decode host fallback over unified memory — the SAME
//! `rlx-cpu` oracle the CPU backend uses — so Metal must match CPU bit-for-bit.
//! `ScaledDequantize` is the inverse of `ScaledQuantize` and is what the
//! `ScaledMatMul` backward (straight-through QAT) emits to rebuild operands; this
//! test runs a quantize→dequantize round-trip and checks it reconstructs f32.

#![cfg(target_os = "macos")]

use rlx_ir::{DType, Graph, Op, ScaleLayout, ScaledFormat, Shape};
use rlx_runtime::{Device, Session};

fn run_case(fmt: ScaledFormat, rows: usize, cols: usize, cos_thresh: f32) {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let layout = ScaleLayout::PerTensor;
    let x: Vec<f32> = (0..rows * cols)
        .map(|i| (i as f32 * 0.21).sin() * 1.7)
        .collect();

    let mut g = Graph::new("scaled_dequant_rt");
    let x_in = g.input("x", Shape::new(&[rows, cols], DType::F32));
    let scale = g.add_node(
        Op::ScaledQuantScale {
            format: fmt,
            scale_layout: layout,
        },
        vec![x_in],
        Shape::new(&[1], DType::F32),
    );
    let codes = g.add_node(
        Op::ScaledQuantize {
            format: fmt,
            scale_layout: layout,
        },
        vec![x_in, scale],
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

    let run = |device: Device| -> Vec<f32> {
        Session::new(device)
            .compile(g.clone())
            .run(&[("x", x.as_slice())])
            .remove(0)
    };

    let metal = run(Device::Metal);
    let cpu = run(Device::Cpu);
    assert_eq!(metal.len(), rows * cols);

    // Metal host fallback uses the same oracle as CPU → bit-identical.
    let max_abs = cpu
        .iter()
        .zip(&metal)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs < 1e-6, "{fmt} Metal vs CPU max_abs={max_abs}");

    // The round-trip reconstructs the original f32 (cosine vs input).
    let dot: f32 = metal.iter().zip(&x).map(|(a, b)| a * b).sum();
    let na = metal.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    let cos = dot / (na * nb);
    eprintln!("scaled_dequant {fmt}: metal-vs-cpu max_abs={max_abs:.2e} cosine-vs-input={cos:.5}");
    assert!(
        cos >= cos_thresh,
        "{fmt} round-trip cosine {cos} < {cos_thresh}"
    );
}

#[test]
fn metal_scaled_dequant_e4m3_matches_cpu() {
    run_case(ScaledFormat::F8E4M3, 4, 16, 0.999);
}

#[test]
fn metal_scaled_dequant_e5m2_matches_cpu() {
    run_case(ScaledFormat::F8E5M2, 4, 16, 0.99);
}

#[test]
fn metal_scaled_dequant_fp4_matches_cpu() {
    run_case(ScaledFormat::F4E2M1, 4, 16, 0.93);
}
