// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU-vs-CUDA parity for `Op::ScaledGroupedMatMul` (MXFP4 grouped MoE GEMM).
//! CPU runs the fused decode oracle; CUDA runs the native on-device
//! `scaled_grouped_matmul_decode` kernel (per-routed-expert FP4 decode, no f32
//! weight materialization). Runs only when a CUDA device is present.

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
    // `Session::new(Device::Cuda)` panics both without CUDA hardware AND when
    // rlx-runtime was built without its `cuda` feature. The runtime's own probe
    // covers both; `rlx_cuda::is_available()` would only catch the former.
    if !rlx_runtime::is_available(Device::Cuda) {
        eprintln!("skip: CUDA unavailable (no device, or runtime built without `cuda`)");
        return;
    }
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
    let cuda = run(Device::Cuda);
    assert_eq!(cpu.len(), m * n);
    let err = cpu
        .iter()
        .zip(&cuda)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("scaled_grouped_matmul CPU-vs-CUDA (m={m}): max_abs_err={err:e}");
    assert!(
        err < 1e-3,
        "MXFP4 grouped CUDA must match CPU (m={m}): err {err:e}\ncpu={cpu:?}\ncuda={cuda:?}"
    );
}

#[test]
fn scaled_grouped_matmul_cpu_cuda_gemv() {
    parity(1, 64, 8, 3, vec![2.0]);
}

#[test]
fn scaled_grouped_matmul_cpu_cuda_gemm() {
    parity(6, 64, 8, 3, vec![0.0, 1.0, 2.0, 1.0, 0.0, 2.0]);
}
