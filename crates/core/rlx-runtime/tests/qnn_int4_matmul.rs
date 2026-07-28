// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Session-level int4 static-weight MatMul on `Device::Hexagon`.
//! IR packs signed nibbles; QNN CPU stages them as BW_SCALE_OFFSET bitwidth=4.

#![cfg(feature = "qnn")]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn qnn_available() -> bool {
    std::env::var_os("RLX_QNN_BACKEND_LIB").is_some() || std::env::var_os("QNN_SDK_ROOT").is_some()
}

fn pack_i4(vals: &[i8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len().div_ceil(2));
    for chunk in vals.chunks(2) {
        let lo = (chunk[0] as u8) & 0x0f;
        let hi = chunk.get(1).map(|&b| (b as u8) & 0x0f).unwrap_or(0);
        out.push(lo | (hi << 4));
    }
    out
}

#[test]
fn int4_matmul_matches_oracle() {
    if !qnn_available() {
        eprintln!("skip: no QNN backend");
        return;
    }

    let (m, k, n) = (2usize, 4, 3);
    let scale = 0.25f32;
    let zp = 0i32;
    let w_i4: Vec<i8> = vec![1, -2, 3, 4, -5, 6, 7, -8, 0, -1, 2, -3];
    let w_packed = pack_i4(&w_i4);
    let x: Vec<f32> = (0..m * k).map(|i| 0.25 * (i as f32 - 3.0)).collect();

    let mut g = Graph::new("i4mm");
    let xi = g.input("x", Shape::new(&[m, k], DType::F32));
    let wi = g.param("w", Shape::new(&[k, n], DType::I8));
    let wd = g.dequantize(wi, scale, zp);
    let y = g.matmul(xi, wd, Shape::new(&[m, n], DType::F32));
    g.set_outputs(vec![y]);

    let mut hex = Session::new(Device::Hexagon).compile(g);
    hex.set_param_typed("w", &w_packed, DType::I8);
    let hex_out = hex.run(&[("x", x.as_slice())]);

    let w_f: Vec<f32> = w_i4
        .iter()
        .map(|&q| scale * (q as f32 - zp as f32))
        .collect();
    let mut want = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += x[i * k + kk] * w_f[kk * n + j];
            }
            want[i * n + j] = acc;
        }
    }

    let md = hex_out[0]
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(md < 1e-3, "Hexagon int4 MatMul vs oracle max_diff {md}");
    eprintln!("Session(Device::Hexagon) int4 MatMul OK (max_diff={md:.2e})");
}
