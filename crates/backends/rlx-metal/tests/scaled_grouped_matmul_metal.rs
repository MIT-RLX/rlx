// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU-vs-Metal parity for `Op::ScaledGroupedMatMul` (MXFP4 grouped MoE GEMM).
//! CPU runs the fused decode-and-segment oracle; Metal runs the portable
//! decompose (ScaledDequantize + Transpose + native MPS GroupedMatMul). Same
//! packed codes + E8M0 block scales pushed through both must agree.

// Compares against a live Metal device, so it can only run on macOS — same
// gate the rest of this directory uses.
#![cfg(target_os = "macos")]

use rlx_ir::{DType, Graph, ScaleLayout, ScaledFormat, Shape};
use rlx_runtime::{Device, Session};

fn build(m: usize, k: usize, n: usize, e_cnt: usize) -> Graph {
    let mut g = Graph::new("sgmm");
    let x = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.input("w", Shape::new(&[e_cnt, n, k], DType::F32));
    let idx = g.input("idx", Shape::new(&[m], DType::F32));
    let y = g.scaled_grouped_matmul(x, w, idx, ScaledFormat::F4E2M1, ScaleLayout::mx());
    g.set_outputs(vec![y]);
    g
}

fn parity(m: usize, k: usize, n: usize, e_cnt: usize, idx: Vec<f32>) {
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32) * 0.031).sin() * 1.2)
        .collect();
    let w: Vec<f32> = (0..e_cnt * n * k)
        .map(|i| ((i as f32) * 0.017).cos() * 1.1)
        .collect();

    let run = |dev: Device| -> Vec<f32> {
        let mut c = Session::new(dev).compile(build(m, k, n, e_cnt));
        c.run(&[
            ("x", x.as_slice()),
            ("w", w.as_slice()),
            ("idx", idx.as_slice()),
        ])[0]
            .clone()
    };

    let cpu = run(Device::Cpu);
    let met = run(Device::Metal);
    assert_eq!(cpu.len(), m * n);
    let err = cpu
        .iter()
        .zip(&met)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("scaled_grouped_matmul CPU-vs-Metal (m={m}): max_abs_err={err:e}");
    assert!(
        err < 1e-3,
        "MXFP4 grouped Metal must match CPU (m={m}): err {err:e}\ncpu={cpu:?}\nmetal={met:?}"
    );
}

#[test]
fn scaled_grouped_matmul_cpu_metal_gemv() {
    // Single-row decode per expert.
    parity(1, 64, 8, 3, vec![2.0]);
}

#[test]
fn scaled_grouped_matmul_cpu_metal_gemm() {
    // Prefill: mixed expert routing across rows.
    parity(6, 64, 8, 3, vec![0.0, 1.0, 2.0, 1.0, 0.0, 2.0]);
}
