// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU-vs-CUDA parity for native `Op::Conv3d` (cuDNN and `conv3d.cu`).

use std::sync::Mutex;

use rlx_cuda::CudaExecutable;
use rlx_cuda::device::{cuda_dnn_handle, last_conv3d_path};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

static PATH_LOCK: Mutex<()> = Mutex::new(());

/// Serialize every test that runs a conv3d. `RLX_CUDA_NO_CUDNN` is a
/// process-global env var and `last_conv3d_path()` a process-global tracker, so
/// a concurrent conv3d clobbers the path another test is about to assert on.
/// Every conv3d test must hold this — not only the ones reading the path back.
/// Poison-tolerant so one failure can't cascade into the rest.
fn path_lock() -> std::sync::MutexGuard<'static, ()> {
    PATH_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

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

fn make_conv3d_case() -> (Graph, Vec<f32>, Vec<f32>) {
    let mut g = Graph::new("conv3d_parity");
    let x = g.input("x", Shape::new(&[1, 1, 3, 3, 3], DType::F32));
    let w = g.input("w", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
    let y = g.conv3d(x, w, [1, 1, 1], [0, 0, 0], [1, 1, 1], 1);
    g.set_outputs(vec![y]);
    let xv: Vec<f32> = (1..=27).map(|v| v as f32).collect();
    let wv: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    (g, xv, wv)
}

#[test]
fn conv3d_matches_cpu() {
    let _guard = path_lock();
    if !rlx_cuda::is_available() {
        eprintln!("[rlx-cuda conv3d] no CUDA device — skipping");
        return;
    }
    let (g, xv, wv) = make_conv3d_case();
    let want = cpu_run(g.clone(), &[("x", &xv), ("w", &wv)]);
    let mut exe = CudaExecutable::compile(g);
    let got = exe
        .run(&[("x", &xv), ("w", &wv)])
        .into_iter()
        .next()
        .unwrap();
    assert!(
        close(&got, &want, 1e-4),
        "Conv3d CUDA vs CPU:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn conv3d_cudnn_matches_cpu() {
    let _guard = path_lock();
    if !rlx_cuda::is_available() {
        eprintln!("[rlx-cuda conv3d.cudnn] no CUDA device — skipping");
        return;
    }
    rlx_ir::env::unset("RLX_CUDA_NO_CUDNN");
    // Fail (don't silently skip) when CUDA is present but cuDNN is not —
    // that is the path this test exists to cover.
    assert!(
        cuda_dnn_handle().is_some(),
        "cuDNN unavailable: set RLX_CUDNN_DIR or put libcudnn.so on the loader path"
    );
    assert!(
        !rlx_ir::env::flag("RLX_CUDA_NO_CUDNN"),
        "RLX_CUDA_NO_CUDNN is set; unset it to exercise conv3d.cudnn"
    );

    let (g, xv, wv) = make_conv3d_case();
    let want = cpu_run(g.clone(), &[("x", &xv), ("w", &wv)]);
    let mut exe = CudaExecutable::compile(g);
    let got = exe
        .run(&[("x", &xv), ("w", &wv)])
        .into_iter()
        .next()
        .unwrap();
    assert_eq!(
        last_conv3d_path(),
        Some("cudnn"),
        "expected cuDNN Conv3d path, got {:?}",
        last_conv3d_path()
    );
    assert!(
        close(&got, &want, 1e-4),
        "Conv3d cuDNN vs CPU:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn conv3d_kernel_matches_cpu_when_no_cudnn() {
    let _guard = path_lock();
    if !rlx_cuda::is_available() {
        eprintln!("[rlx-cuda conv3d.kernel] no CUDA device — skipping");
        return;
    }
    rlx_ir::env::set("RLX_CUDA_NO_CUDNN", "1");

    let (g, xv, wv) = make_conv3d_case();
    let want = cpu_run(g.clone(), &[("x", &xv), ("w", &wv)]);
    let mut exe = CudaExecutable::compile(g);
    let got = exe
        .run(&[("x", &xv), ("w", &wv)])
        .into_iter()
        .next()
        .unwrap();

    rlx_ir::env::unset("RLX_CUDA_NO_CUDNN");

    assert_eq!(
        last_conv3d_path(),
        Some("kernel"),
        "expected KERNEL Conv3d path under RLX_CUDA_NO_CUDNN, got {:?}",
        last_conv3d_path()
    );
    assert!(
        close(&got, &want, 1e-4),
        "Conv3d kernel vs CPU:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn conv3d_identity_1x1x1_matches_input() {
    let _guard = path_lock();
    if !rlx_cuda::is_available() {
        eprintln!("[rlx-cuda conv3d] no CUDA device — skipping identity");
        return;
    }
    let mut g = Graph::new("conv3d_id");
    let x = g.input("x", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
    let w = g.input("w", Shape::new(&[1, 1, 1, 1, 1], DType::F32));
    let y = g.conv3d(x, w, [1, 1, 1], [0, 0, 0], [1, 1, 1], 1);
    g.set_outputs(vec![y]);
    let xv: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    let wv = vec![1.0f32];
    let mut exe = CudaExecutable::compile(g);
    let got = exe
        .run(&[("x", &xv), ("w", &wv)])
        .into_iter()
        .next()
        .unwrap();
    assert!(
        close(&got, &xv, 1e-5),
        "Conv3d 1x1x1 identity:\n got={got:?}\nwant={xv:?}"
    );
}
