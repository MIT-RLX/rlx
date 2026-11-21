// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The GGUF schemes laid out exactly as ggml's blocks — `NVFP4` (type 40, 64
//! values / 36 bytes), `MXFP4` (type 39, halves nibble order), `Q2_0` (type
//! 42, group 64) — and packed `BF16`, through `Op::DequantMatMul` and the
//! expert-indexed `Op::DequantGroupedMatMul`, against the host decode on every
//! available backend. `rlx-gguf`'s decoders are checked against ggml's own
//! (transcribed) in that crate, so agreement here means ggml-exact.

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
    Device::Gpu,
    Device::Vulkan,
    Device::Ane, // CoreML
];

fn ggml_type(scheme: QuantScheme) -> rlx_gguf::GgmlType {
    match scheme {
        QuantScheme::GgufNVFP4 => rlx_gguf::GgmlType::NVFP4,
        QuantScheme::GgufMXFP4 => rlx_gguf::GgmlType::MXFP4,
        QuantScheme::GgufQ2_0G64 => rlx_gguf::GgmlType::Q2_0,
        QuantScheme::GgufBF16 => rlx_gguf::GgmlType::BF16,
        other => panic!("no ggml type for {other:?}"),
    }
}

/// `[rows, k]` weights spanning several blocks with distinct scales.
fn weights(rows: usize, k: usize, scheme: QuantScheme) -> Vec<f32> {
    (0..rows * k)
        .map(|i| {
            let v = ((i as f32) * 0.37).sin() * (1.0 + ((i / 16) % 5) as f32);
            if scheme == QuantScheme::GgufQ2_0G64 {
                v.signum() * (v.abs() > 0.8) as i32 as f32 // ternary: exact
            } else {
                v
            }
        })
        .collect()
}

fn encode(w: &[f32], scheme: QuantScheme) -> Vec<u8> {
    rlx_gguf::quantize(w, ggml_type(scheme)).expect("quantize")
}

fn decode(bytes: &[u8], n: usize, scheme: QuantScheme) -> Vec<f32> {
    rlx_gguf::dequant_typed(ggml_type(scheme), bytes, n, "w").expect("decode")
}

fn close(want: &[f32], got: &[f32], what: &str) {
    assert_eq!(want.len(), got.len(), "{what}: length");
    assert!(
        got.iter().any(|v| *v != 0.0),
        "{what}: all zeros (no kernel branch?)"
    );
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    let worst = want
        .iter()
        .zip(got)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        worst / scale < 1e-4,
        "{what}: relative |Δ| {}",
        worst / scale
    );
}

fn dense(
    device: Device,
    packed: &[u8],
    m: usize,
    k: usize,
    n: usize,
    x: &[f32],
    s: QuantScheme,
) -> Vec<f32> {
    let mut g = Graph::new("ggml_scheme");
    let xi = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul { scheme: s },
        vec![xi, w],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut c = Session::new(device).compile(g);
    c.set_param_typed("w", packed, DType::U8);
    c.run(&[("x", x)]).remove(0)
}

#[allow(clippy::too_many_arguments)]
fn grouped(
    device: Device,
    packed: &[u8],
    m: usize,
    k: usize,
    n: usize,
    x: &[f32],
    idx: &[f32],
    s: QuantScheme,
) -> Vec<f32> {
    let mut g = Graph::new("ggml_scheme_grouped");
    let xi = g.input("x", Shape::new(&[m, k], DType::F32));
    let e = g.input("e", Shape::new(&[m], DType::F32));
    let w = g.param("w", Shape::new(&[packed.len()], DType::U8));
    let y = g.dequant_grouped_matmul_packed(xi, w, e, s, Shape::new(&[m, n], DType::F32));
    g.set_outputs(vec![y]);
    let mut c = Session::new(device).compile(g);
    c.set_param_typed("w", packed, DType::U8);
    c.run(&[("x", x), ("e", idx)]).remove(0)
}

fn check(scheme: QuantScheme) {
    let _gpu = common::serialize_gpu();
    let (k, n, experts) = (256usize, 8usize, 3usize);
    let mut ran = Vec::new();
    for &m in &[1usize, 4] {
        let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.013).sin()).collect();
        // Dense: one [n, k] weight.
        let packed = encode(&weights(n, k, scheme), scheme);
        let w = decode(&packed, n * k, scheme);
        let want: Vec<f32> = (0..m * n)
            .map(|o| (0..k).map(|j| x[o / n * k + j] * w[o % n * k + j]).sum())
            .collect();
        // Grouped: `experts` slabs of [n, k], one expert per row.
        let bank = encode(&weights(experts * n, k, scheme), scheme);
        let wb = decode(&bank, experts * n * k, scheme);
        let idx: Vec<f32> = (0..m).map(|r| ((r * 2 + 1) % experts) as f32).collect();
        let want_g: Vec<f32> = (0..m * n)
            .map(|o| {
                let (r, c) = (o / n, o % n);
                let e = idx[r] as usize;
                (0..k).map(|j| x[r * k + j] * wb[(e * n + c) * k + j]).sum()
            })
            .collect();
        for &dev in DEVICES {
            if common::skip_unless(dev) {
                continue;
            }
            close(
                &want,
                &dense(dev, &packed, m, k, n, &x, scheme),
                &format!("{scheme} {dev:?} m={m}"),
            );
            close(
                &want_g,
                &grouped(dev, &bank, m, k, n, &x, &idx, scheme),
                &format!("{scheme} grouped {dev:?} m={m}"),
            );
            ran.push(dev);
        }
    }
    assert!(!ran.is_empty(), "{scheme}: no backend available");
    eprintln!("{scheme}: verified on {ran:?}");
}

#[test]
fn nvfp4_matches_host_on_every_backend() {
    check(QuantScheme::GgufNVFP4);
}

#[test]
fn mxfp4_matches_host_on_every_backend() {
    check(QuantScheme::GgufMXFP4);
}

#[test]
fn ggml_q2_0_matches_host_on_every_backend() {
    check(QuantScheme::GgufQ2_0G64);
}

#[test]
fn bf16_matches_host_on_every_backend() {
    check(QuantScheme::GgufBF16);
}

/// MLX `mxfp8` experts (FP8 E4M3 codes × E8M0 group scales, decoded to f32
/// by the loader) through `Op::DequantGroupedMatMulMlx`.
#[test]
fn mlx_mxfp8_grouped_matches_host_on_every_backend() {
    let _gpu = common::serialize_gpu();
    let (k, n, experts, gs) = (128usize, 8usize, 3usize, 32usize);
    let codes: Vec<u8> = (0..experts * n * k)
        .map(|i| {
            let c = ((i * 37 + 11) % 256) as u8;
            if c & 0x7F == 0x7F { c ^ 1 } else { c } // no NaN codes
        })
        .collect();
    let scales: Vec<f32> = (0..experts * n * (k / gs))
        .map(|i| 2f32.powi((i % 7) as i32 - 6))
        .collect();
    let biases = vec![0f32; scales.len()];
    let w: Vec<f32> = (0..experts * n * k)
        .map(|i| {
            let (row, c) = (i / k, i % k);
            rlx_gguf::mx_dequant::e4m3_scale_to_f32(codes[i]) * scales[row * (k / gs) + c / gs]
        })
        .collect();
    let scheme = QuantScheme::MlxMxfp8 {
        group_size: gs as u32,
    };
    let mut ran = Vec::new();
    for &m in &[1usize, 4] {
        let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.013).sin()).collect();
        let idx: Vec<f32> = (0..m).map(|r| ((r * 2 + 1) % experts) as f32).collect();
        let want: Vec<f32> = (0..m * n)
            .map(|o| {
                let (r, c) = (o / n, o % n);
                let e = idx[r] as usize;
                (0..k).map(|j| x[r * k + j] * w[(e * n + c) * k + j]).sum()
            })
            .collect();
        for &dev in DEVICES {
            if common::skip_unless(dev) {
                continue;
            }
            let mut g = Graph::new("mxfp8_grouped");
            let xi = g.input("x", Shape::new(&[m, k], DType::F32));
            let e = g.input("e", Shape::new(&[m], DType::F32));
            let wq = g.param("w", Shape::new(&[codes.len()], DType::U8));
            let sc = g.param("s", Shape::new(&[experts, n, k / gs], DType::F32));
            let bi = g.param("b", Shape::new(&[experts, n, k / gs], DType::F32));
            let y = g.add_node(
                Op::DequantGroupedMatMulMlx { scheme },
                vec![xi, wq, sc, bi, e],
                Shape::new(&[m, n], DType::F32),
            );
            g.set_outputs(vec![y]);
            let mut c = Session::new(dev).compile(g);
            c.set_param_typed("w", &codes, DType::U8);
            c.set_param("s", &scales);
            c.set_param("b", &biases);
            let got = c.run(&[("x", &x), ("e", &idx)]).remove(0);
            close(&want, &got, &format!("mxfp8 grouped {dev:?} m={m}"));
            ran.push(dev);
        }
    }
    eprintln!("mlx mxfp8 grouped: verified on {ran:?}");
}
