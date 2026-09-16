// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native INT8 `Op::Quantize` / `Op::Dequantize` / `Op::QMatMul` / `Op::QConv2d`
//! on wgpu vs the rlx-cpu oracle.
//!
//! All four took the generic CPU host route on wgpu. The kernels are ported
//! from Vulkan's, with both of that port's bugs already fixed: the affine table
//! is `vec4`-strided (a scalar array in a WGSL uniform has a 16-byte stride, and
//! the Vulkan twin read `scale`/`zero_point` out of the padding and emitted
//! zeros for every tensor) and the packed-i8 writers own a whole word (WGSL has
//! no byte store, so a per-element launch loses three of every four results).
//!
//! Everything here is integer or exactly-rounded, so **every gate is exact**.
//! There is no accumulation-order argument to make: `QMatMul` accumulates in
//! i32, which is associative.
//!
//! Two details the gates are specifically shaped to catch:
//!
//! * **Rounding.** Rust's `f32::round` breaks ties away from zero; WGSL's
//!   `round` breaks them to even. `tie_breaking_inputs_round_the_same_way` feeds
//!   exact halves, where the two disagree.
//! * **The bias dtype.** An I32 tensor holds raw i32 on the CPU arena and a
//!   widened f32 *value* on the GPU arena. Feeding it as f32 lanes through
//!   `run` would therefore mean different numbers on each side — so the bias
//!   goes in through `run_typed` as real I32 bytes, which each backend converts
//!   for itself.

#![cfg(all(feature = "gpu", feature = "cpu"))]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

mod common;

fn wave(n: usize, phase: f32, amp: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * phase).sin() * amp).collect()
}

fn f32_bytes(xs: &[f32]) -> Vec<u8> {
    xs.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn i32_bytes(xs: &[i32]) -> Vec<u8> {
    xs.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Run typed on both devices and require bit-equality of the f32 output.
fn both(what: &str, build: impl Fn() -> Graph, inputs: &[(&str, &[u8], DType)]) {
    let run = |device: Device| -> Vec<f32> {
        let bytes = Session::new(device)
            .compile(build())
            .run_typed(inputs)
            .remove(0)
            .0;
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let cpu = run(Device::Cpu);
    let gpu = run(Device::Gpu);
    assert_eq!(
        gpu, cpu,
        "{what}: wgpu != cpu\n  gpu={gpu:?}\n  cpu={cpu:?}"
    );
}

// ---------------------------------------------------------------------------
// Quantize / Dequantize
// ---------------------------------------------------------------------------

fn roundtrip(rows: usize, cols: usize, scale: f32, zp: i32) -> Graph {
    let mut g = Graph::new("q_rt");
    let x = g.input("x", Shape::new(&[rows, cols], DType::F32));
    let q = g.quantize(x, scale, zp);
    let y = g.dequantize(q, scale, zp);
    g.set_outputs(vec![y]);
    g
}

#[test]
fn quantize_dequantize_roundtrip_matches_cpu() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }
    // `cols` is not a multiple of 4, so the ragged tail word — the one a
    // word-owner launch has to leave alone past the end — is exercised.
    let (rows, cols) = (5usize, 7usize);
    let x = wave(rows * cols, 0.31, 2.0);
    let xb = f32_bytes(&x);
    for (scale, zp) in [
        (0.05f32, 0i32),
        (0.05, 3),
        (0.25, -7),
        (1.0, 0),
        (0.01, 40),
        (2.0, -128),
    ] {
        both(
            &format!("roundtrip scale={scale} zp={zp}"),
            || roundtrip(rows, cols, scale, zp),
            &[("x", &xb, DType::F32)],
        );
    }
}

#[test]
fn saturating_inputs_clamp_the_same_way() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }
    // Well past ±128 codes in both directions, plus zero and denormals, so the
    // clamp arms run on both sides.
    let x = vec![
        0.0f32,
        -0.0,
        1e30,
        -1e30,
        1000.0,
        -1000.0,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        6.4,
        -6.4,
        6.35,
        -6.35,
    ];
    let xb = f32_bytes(&x);
    for (scale, zp) in [(0.05f32, 0i32), (0.05, 3), (0.25, -7)] {
        both(
            &format!("saturate scale={scale} zp={zp}"),
            || roundtrip(3, 4, scale, zp),
            &[("x", &xb, DType::F32)],
        );
    }
}

#[test]
fn tie_breaking_inputs_round_the_same_way() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }
    // Exact halves: `x / scale` lands on `n + 0.5` for every entry. Rust rounds
    // away from zero, WGSL's `round` rounds to even — so a kernel that used the
    // builtin gets half of these wrong and nothing else in the suite notices.
    let scale = 0.25f32;
    let x: Vec<f32> = (-8..8).map(|n| (n as f32 + 0.5) * scale).collect();
    let xb = f32_bytes(&x);
    for zp in [0i32, 3, -5] {
        both(
            &format!("ties zp={zp}"),
            || roundtrip(4, 4, scale, zp),
            &[("x", &xb, DType::F32)],
        );
    }
}

#[test]
fn per_channel_affine_matches_cpu() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }
    // Per-channel is where the affine table's stride matters: with one channel
    // a wrongly strided table can still accidentally read index 0.
    let (rows, cols) = (5usize, 6usize);
    let x = wave(rows * cols, 0.23, 3.0);
    let xb = f32_bytes(&x);
    let scales: Vec<f32> = (0..cols).map(|c| 0.02 * (c as f32 + 1.0)).collect();
    let zps: Vec<i32> = (0..cols).map(|c| c as i32 * 3 - 7).collect();
    both(
        "per-channel axis=1",
        || {
            let mut g = Graph::new("q_pc");
            let xi = g.input("x", Shape::new(&[rows, cols], DType::F32));
            let q = g.quantize_per_channel(xi, 1, scales.clone(), zps.clone());
            let y = g.dequantize_per_channel(q, 1, scales.clone(), zps.clone());
            g.set_outputs(vec![y]);
            g
        },
        &[("x", &xb, DType::F32)],
    );
}

// ---------------------------------------------------------------------------
// QMatMul
// ---------------------------------------------------------------------------

#[test]
fn q_matmul_matches_cpu() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }
    // N is not a multiple of 4 so the ragged output word is shared, and M > 1
    // so a lost write would show as a whole stale row rather than a tail.
    let (m, k, n) = (6usize, 8usize, 7usize);
    let x = wave(m * k, 0.31, 2.0);
    let w = wave(k * n, 0.17, 1.5);
    let xb = f32_bytes(&x);
    let wb = f32_bytes(&w);
    // Real I32 bias bytes — see the module note on the bias dtype.
    let bias: Vec<i32> = (0..n).map(|i| i as i32 * 5 - 12).collect();
    let bb = i32_bytes(&bias);

    for (mult, out_zp) in [(0.1f32, 0i32), (0.02, 7), (0.5, -20)] {
        both(
            &format!("q_matmul mult={mult} out_zp={out_zp}"),
            || {
                let mut g = Graph::new("qmm");
                let xf = g.input("x", Shape::new(&[m, k], DType::F32));
                let wf = g.input("w", Shape::new(&[k, n], DType::F32));
                let bi = g.input("b", Shape::new(&[n], DType::I32));
                let xq = g.quantize(xf, 0.25, 0);
                let wq = g.quantize(wf, 0.25, 0);
                let y = g.q_matmul(
                    xq,
                    wq,
                    bi,
                    0,
                    0,
                    out_zp,
                    mult,
                    Shape::new(&[m, n], DType::I8),
                );
                // scale 1 / zp 0: the f32 output *is* the i8 code.
                let out = g.dequantize(y, 1.0, 0);
                g.set_outputs(vec![out]);
                g
            },
            &[
                ("x", &xb, DType::F32),
                ("w", &wb, DType::F32),
                ("b", &bb, DType::I32),
            ],
        );
    }
}

#[test]
fn q_matmul_honours_operand_zero_points() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }
    let (m, k, n) = (4usize, 5usize, 6usize);
    let x = wave(m * k, 0.41, 2.0);
    let w = wave(k * n, 0.23, 1.5);
    let xb = f32_bytes(&x);
    let wb = f32_bytes(&w);
    let bb = i32_bytes(&vec![0i32; n]);
    both(
        "q_matmul nonzero operand zps",
        || {
            let mut g = Graph::new("qmm_zp");
            let xf = g.input("x", Shape::new(&[m, k], DType::F32));
            let wf = g.input("w", Shape::new(&[k, n], DType::F32));
            let bi = g.input("b", Shape::new(&[n], DType::I32));
            let xq = g.quantize(xf, 0.25, 3);
            let wq = g.quantize(wf, 0.25, -2);
            let y = g.q_matmul(xq, wq, bi, 3, -2, 0, 0.1, Shape::new(&[m, n], DType::I8));
            let out = g.dequantize(y, 1.0, 0);
            g.set_outputs(vec![out]);
            g
        },
        &[
            ("x", &xb, DType::F32),
            ("w", &wb, DType::F32),
            ("b", &bb, DType::I32),
        ],
    );
}

// ---------------------------------------------------------------------------
// QConv2d
// ---------------------------------------------------------------------------

#[test]
fn q_conv2d_matches_cpu() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }
    // Padding, stride and dilation all non-trivial so the bounds guard runs, and
    // an output count that is not a multiple of 4 so the ragged word is shared.
    let (n, c_in, h, w, c_out, kh, kw) = (2usize, 3usize, 6usize, 5usize, 4usize, 3usize, 3usize);
    let x = wave(n * c_in * h * w, 0.29, 2.0);
    let wt = wave(c_out * c_in * kh * kw, 0.19, 1.4);
    let xb = f32_bytes(&x);
    let wb = f32_bytes(&wt);
    let bias: Vec<i32> = (0..c_out).map(|i| i as i32 * 4 - 6).collect();
    let bb = i32_bytes(&bias);

    for (stride, padding, dilation) in [
        (vec![1usize, 1], vec![0usize, 0], vec![1usize, 1]),
        (vec![2, 1], vec![1, 1], vec![1, 1]),
        (vec![1, 1], vec![1, 0], vec![2, 1]),
    ] {
        let s = stride.clone();
        let p = padding.clone();
        let d = dilation.clone();
        both(
            &format!("q_conv2d stride={s:?} pad={p:?} dil={d:?}"),
            move || {
                let mut g = Graph::new("qconv");
                let xf = g.input("x", Shape::new(&[n, c_in, h, w], DType::F32));
                let wf = g.input("w", Shape::new(&[c_out, c_in, kh, kw], DType::F32));
                let bi = g.input("b", Shape::new(&[c_out], DType::I32));
                let xq = g.quantize(xf, 0.25, 0);
                let wq = g.quantize(wf, 0.25, 0);
                let ho = (h + 2 * p[0] - d[0] * (kh - 1) - 1) / s[0] + 1;
                let wo = (w + 2 * p[1] - d[1] * (kw - 1) - 1) / s[1] + 1;
                let y = g.q_conv2d(
                    xq,
                    wq,
                    bi,
                    vec![kh, kw],
                    s.clone(),
                    p.clone(),
                    d.clone(),
                    1,
                    0,
                    0,
                    0,
                    0.05,
                    Shape::new(&[n, c_out, ho, wo], DType::I8),
                );
                let out = g.dequantize(y, 1.0, 0);
                g.set_outputs(vec![out]);
                g
            },
            &[
                ("x", &xb, DType::F32),
                ("w", &wb, DType::F32),
                ("b", &bb, DType::I32),
            ],
        );
    }
}
