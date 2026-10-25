// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! GGUF `Q8_0` on MLX runs as native affine 8-bit `quantized_matmul`
//! (`scale = d`, codes `q + 128`, `bias = -128·d`) instead of dequantizing
//! every weight to a resident f32 copy on the host.
//!
//! The mapping is exact, so the result must match the dequantized weights
//! to f32 reduction-order noise — including the extreme code -128 and a
//! one-row (decode-shaped) matmul. `RLX_MLX_Q8_0_DEQUANT=1` is the old path.
#![cfg(feature = "mlx")]

use rlx_ir::quant::QuantScheme;
use rlx_ir::*;
use rlx_runtime::{Device, Session};

mod common;

/// `n` rows of `k` weights as Q8_0 blocks, with varied scales and codes
/// spanning the full i8 range.
fn q8_0(n: usize, k: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * k / 32 * 34);
    for b in 0..n * k / 32 {
        let d = half::f16::from_f32(0.001 + (b % 17) as f32 * 0.0007);
        out.extend_from_slice(&d.to_le_bytes());
        for j in 0..32 {
            let q: i8 = if (b + j) % 29 == 0 {
                -128
            } else {
                (((b * 31 + j * 7) % 255) as i32 - 127) as i8
            };
            out.push(q as u8);
        }
    }
    out
}

fn reference(w: &[u8], x: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let wf = rlx_gguf::dequant_typed(rlx_gguf::GgmlType::Q8_0, w, n * k, "w").unwrap();
    (0..m * n)
        .map(|i| {
            let (r, c) = (i / n, i % n);
            (0..k)
                .map(|j| f64::from(x[r * k + j]) * f64::from(wf[c * k + j]))
                .sum::<f64>() as f32
        })
        .collect()
}

fn run(device: Device, w: &[u8], x: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut g = Graph::new("q8_0_affine");
    let xi = g.input("x", Shape::new(&[m, k], DType::F32));
    let wp = g.param("w", Shape::new(&[w.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ8_0,
        },
        vec![xi, wp],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut c = Session::new(device).compile(g);
    c.set_param_typed("w", w, DType::U8);
    c.run(&[("x", x)])[0].clone()
}

#[test]
fn q8_0_matches_dequantized_weights_on_mlx() {
    if common::skip_unless_available(Device::Mlx, "mlx") {
        return;
    }
    let _gpu = common::GpuTestGuard::acquire(Device::Mlx);
    let (k, n) = (96, 40);
    let w = q8_0(n, k);
    for m in [1usize, 37] {
        let x: Vec<f32> = (0..m * k)
            .map(|i| ((i * 13 % 97) as f32 - 48.0) / 50.0)
            .collect();
        let want = reference(&w, &x, m, k, n);
        let got = run(Device::Mlx, &w, &x, m, k, n);
        let scale = want.iter().fold(0f32, |a, v| a.max(v.abs()));
        let err = got
            .iter()
            .zip(&want)
            .fold(0f32, |a, (g, w)| a.max((g - w).abs()));
        assert!(
            err <= 1e-5 * scale.max(1.0),
            "m={m}: MLX Q8_0 off by {err} (scale {scale})"
        );
    }
}
