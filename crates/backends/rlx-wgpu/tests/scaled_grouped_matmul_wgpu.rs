// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU-vs-wgpu parity for `Op::ScaledGroupedMatMul` (MXFP4 grouped MoE GEMM).
//! CPU runs the fused decode oracle; wgpu runs the portable decompose
//! (ScaledDequantize + Transpose + native WGSL GroupedMatMul). Runs only when a
//! wgpu device is present; otherwise a graceful no-op.

use rlx_ir::{DType, Graph, ScaleLayout, ScaledFormat, Shape};

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
    if !rlx_wgpu::is_available() {
        return;
    }
    use rlx_runtime::{Device, Session};

    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32) * 0.031).sin() * 1.2)
        .collect();
    let w: Vec<f32> = (0..e_cnt * n * k)
        .map(|i| ((i as f32) * 0.017).cos() * 1.1)
        .collect();

    let cpu = Session::new(Device::Cpu);
    let mut cc = cpu.compile(build(m, k, n, e_cnt));
    let want = cc
        .run(&[("x", &x), ("w", &w), ("idx", &idx)])
        .into_iter()
        .next()
        .unwrap();

    let mut gc = Session::new(Device::Gpu).compile(build(m, k, n, e_cnt));
    let got = gc
        .run(&[("x", &x), ("w", &w), ("idx", &idx)])
        .into_iter()
        .next()
        .unwrap();

    let err = want
        .iter()
        .zip(got.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("scaled_grouped_matmul CPU-vs-wgpu (m={m}): max_abs_err={err:e}");
    assert!(
        err < 1e-3,
        "MXFP4 grouped wgpu must match CPU (m={m}): err {err:e}"
    );
}

#[test]
fn scaled_grouped_matmul_cpu_wgpu_gemv() {
    parity(1, 64, 8, 3, vec![2.0]);
}

#[test]
fn scaled_grouped_matmul_cpu_wgpu_gemm() {
    parity(6, 64, 8, 3, vec![0.0, 1.0, 2.0, 1.0, 0.0, 2.0]);
}
