// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! End-to-end: a MatMul dispatched to `Device::Hexagon` (the QNN FFI runtime
//! backend) matches the CPU reference, through the normal `Session` API — the
//! same path `../rlx-models` uses.
//!
//! Needs a QNN backend library (`QNN_SDK_ROOT` or `RLX_QNN_BACKEND_LIB`); skips
//! otherwise so host/CI without the SDK stays green. Validated in Docker
//! against `libQnnCpu.so` (see `crates/rlx-qnn/docker/`).

#![cfg(feature = "qnn")]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn qnn_available() -> bool {
    std::env::var_os("RLX_QNN_BACKEND_LIB").is_some() || std::env::var_os("QNN_SDK_ROOT").is_some()
}

#[test]
fn matmul_runs_on_hexagon_and_matches_oracle() {
    if !qnn_available() {
        eprintln!("skip: no QNN backend (set QNN_SDK_ROOT or RLX_QNN_BACKEND_LIB)");
        return;
    }

    let (m, k, n) = (8usize, 16, 4);
    let build = || {
        let mut g = Graph::new("mm");
        let a = g.input("in0", Shape::new(&[m, k], DType::F32));
        let b = g.input("in1", Shape::new(&[k, n], DType::F32));
        let y = g.matmul(a, b, Shape::new(&[m, n], DType::F32));
        g.set_outputs(vec![y]);
        g
    };

    let in0: Vec<f32> = (0..m * k)
        .map(|i| ((i % 7) as i32 - 3) as f32 * 0.5)
        .collect();
    let in1: Vec<f32> = (0..k * n)
        .map(|i| ((i % 5) as i32 - 2) as f32 * 0.25)
        .collect();
    let inputs: [(&str, &[f32]); 2] = [("in0", &in0), ("in1", &in1)];

    let mut hex = Session::new(Device::Hexagon).compile(build());
    let hex_out = hex.run(&inputs);

    // Oracle: row-major matmul in plain Rust.
    let mut want = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += in0[i * k + kk] * in1[kk * n + j];
            }
            want[i * n + j] = acc;
        }
    }

    let max_diff = hex_out[0]
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-3, "Hexagon vs oracle max_diff {max_diff}");
    eprintln!("Session(Device::Hexagon) matmul matches oracle (max_diff={max_diff:.2e})");
}

#[test]
fn matmul_add_runs_on_hexagon_and_matches_oracle() {
    if !qnn_available() {
        eprintln!("skip: no QNN backend (set QNN_SDK_ROOT or RLX_QNN_BACKEND_LIB)");
        return;
    }

    let (m, k, n) = (4usize, 8, 3);
    let build = || {
        let mut g = Graph::new("mm_add");
        let x = g.input("x", Shape::new(&[m, k], DType::F32));
        let w = g.input("w", Shape::new(&[k, n], DType::F32));
        let c = g.input("c", Shape::new(&[m, n], DType::F32));
        let mm = g.matmul(x, w, Shape::new(&[m, n], DType::F32));
        let y = g.binary(
            rlx_ir::op::BinaryOp::Add,
            mm,
            c,
            Shape::new(&[m, n], DType::F32),
        );
        g.set_outputs(vec![y]);
        g
    };

    let xv: Vec<f32> = (0..m * k)
        .map(|i| ((i % 7) as i32 - 3) as f32 * 0.5)
        .collect();
    let wv: Vec<f32> = (0..k * n)
        .map(|i| ((i % 5) as i32 - 2) as f32 * 0.25)
        .collect();
    let cv: Vec<f32> = (0..m * n).map(|i| i as f32 * 0.1 - 0.5).collect();
    let inputs: [(&str, &[f32]); 3] = [("x", &xv), ("w", &wv), ("c", &cv)];

    let mut hex = Session::new(Device::Hexagon).compile(build());
    let out = hex.run(&inputs);

    // Oracle: x·w + c (row-major).
    let mut want = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = cv[i * n + j];
            for kk in 0..k {
                acc += xv[i * k + kk] * wv[kk * n + j];
            }
            want[i * n + j] = acc;
        }
    }

    let max_diff = out[0]
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-3,
        "matmul+add Hexagon vs oracle max_diff {max_diff}"
    );
    eprintln!("Session(Device::Hexagon) matmul+add matches oracle (max_diff={max_diff:.2e})");
}
