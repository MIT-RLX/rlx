// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU-vs-CUDA parity for native DenseSolve / BatchedDenseSolve (cuSOLVER /
//! cuBLAS). Skips cleanly when no CUDA device is present — exercise on GPU CI.

use rlx_cuda::CudaExecutable;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
}

fn cpu_run(g: Graph, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    Session::new(Device::Cpu)
        .compile(g)
        .run(inputs)
        .into_iter()
        .next()
        .unwrap()
}

#[test]
fn dense_solve_f32_matches_cpu() {
    if !rlx_cuda::is_available() {
        eprintln!("[rlx-cuda dense_solve] no CUDA device — skipping");
        return;
    }
    let n = 2usize;
    let mut g = Graph::new("dense_solve");
    let a = g.input("a", Shape::new(&[n, n], DType::F32));
    let b = g.input("b", Shape::new(&[n], DType::F32));
    let x = g.dense_solve(a, b, Shape::new(&[n], DType::F32));
    g.set_outputs(vec![x]);
    // [[2,1],[0,2]] * [1,2] = [4,4]
    let av = vec![2.0f32, 1.0, 0.0, 2.0];
    let bv = vec![4.0f32, 4.0];
    let want = cpu_run(g.clone(), &[("a", &av), ("b", &bv)]);
    let mut exe = CudaExecutable::compile(g);
    let got = exe
        .run(&[("a", &av), ("b", &bv)])
        .into_iter()
        .next()
        .unwrap();
    assert!(
        close(&got, &want, 1e-4),
        "DenseSolve CUDA vs CPU:\n got={got:?}\nwant={want:?}"
    );
    assert!(close(&got, &[1.0, 2.0], 1e-4));
}

#[test]
fn batched_dense_solve_f32_matches_cpu() {
    if !rlx_cuda::is_available() {
        eprintln!("[rlx-cuda dense_solve] no CUDA device — skipping batched");
        return;
    }
    let batch = 2usize;
    let n = 2usize;
    let mut g = Graph::new("batched_dense_solve");
    let a = g.input("a", Shape::new(&[batch, n, n], DType::F32));
    let b = g.input("b", Shape::new(&[batch, n], DType::F32));
    let x = g.batched_dense_solve(a, b, Shape::new(&[batch, n], DType::F32));
    g.set_outputs(vec![x]);
    let av = vec![
        2.0f32, 1.0, 0.0, 2.0, // batch 0
        2.0, 1.0, 0.0, 2.0, // batch 1
    ];
    let bv = vec![4.0f32, 4.0, 4.0, 4.0];
    let want = cpu_run(g.clone(), &[("a", &av), ("b", &bv)]);
    let mut exe = CudaExecutable::compile(g);
    let got = exe
        .run(&[("a", &av), ("b", &bv)])
        .into_iter()
        .next()
        .unwrap();
    assert!(
        close(&got, &want, 1e-4),
        "BatchedDenseSolve CUDA vs CPU:\n got={got:?}\nwant={want:?}"
    );
}
