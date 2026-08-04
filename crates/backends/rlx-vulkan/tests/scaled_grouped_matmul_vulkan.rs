// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU-vs-Vulkan parity for `Op::ScaledGroupedMatMul` (MXFP4 grouped MoE GEMM).
//! CPU runs the fused decode oracle; Vulkan runs the native
//! `scaled_grouped_matmul_decode.comp` shader (per-routed-expert FP4 decode, no
//! f32 weight materialization). Runs only when a Vulkan device is present.

use rlx_ir::{DType, Graph, ScaleLayout, ScaledFormat, Shape};
use rlx_vulkan::backend::VulkanExecutable;

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
    if !rlx_vulkan::is_available() {
        return;
    }
    use rlx_runtime::{Device, Session};

    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32) * 0.031).sin() * 1.2)
        .collect();
    let w: Vec<f32> = (0..e_cnt * n * k)
        .map(|i| ((i as f32) * 0.017).cos() * 1.1)
        .collect();

    let mut cc = Session::new(Device::Cpu).compile(build(m, k, n, e_cnt));
    let want = cc
        .run(&[("x", &x), ("w", &w), ("idx", &idx)])
        .into_iter()
        .next()
        .unwrap();

    let mut exe = VulkanExecutable::compile(build(m, k, n, e_cnt));
    let got = exe
        .run(&[("x", &x), ("w", &w), ("idx", &idx)])
        .into_iter()
        .next()
        .unwrap();

    let err = want
        .iter()
        .zip(got.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("scaled_grouped_matmul CPU-vs-Vulkan (m={m}): max_abs_err={err:e}");
    assert!(
        err < 1e-3,
        "MXFP4 grouped Vulkan must match CPU (m={m}): err {err:e}"
    );
}

#[test]
fn scaled_grouped_matmul_cpu_vulkan_gemv() {
    parity(1, 64, 8, 3, vec![2.0]);
}

#[test]
fn scaled_grouped_matmul_cpu_vulkan_gemm() {
    parity(6, 64, 8, 3, vec![0.0, 1.0, 2.0, 1.0, 0.0, 2.0]);
}
