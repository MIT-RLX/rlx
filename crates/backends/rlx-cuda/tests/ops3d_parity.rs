// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU-vs-CUDA parity for remaining 3-D kernels:
//! `ConvTranspose3d` (cuDNN BackwardData + `conv_transpose3d.cu`) and
//! `Pool3d` (`pool3d.cu`).

use std::sync::Mutex;

use rlx_cuda::CudaExecutable;
use rlx_cuda::device::{cuda_dnn_handle, last_conv_transpose3d_path};
use rlx_ir::op::ReduceOp;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

static PATH_LOCK: Mutex<()> = Mutex::new(());

/// Serialize every test that runs a ConvTranspose3d. `RLX_CUDA_NO_CUDNN` is a
/// process-global env var and `last_conv_transpose3d_path()` a process-global
/// tracker, so a concurrent CT3d run clobbers the path another test is about to
/// assert on. Every CT3d test must hold this — not only the ones that read the
/// path back. Poison-tolerant so one failure can't cascade `PoisonError` into
/// the rest and bury the real cause.
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

fn make_ct3d_case() -> (Graph, Vec<f32>, Vec<f32>) {
    let mut g = Graph::new("ct3d_parity");
    let x = g.input("x", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
    let w = g.input("w", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
    let y = g.conv_transpose3d(x, w, [2, 2, 2], [0, 0, 0], [1, 1, 1], [0, 0, 0], 1);
    g.set_outputs(vec![y]);
    let xv: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let wv: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    (g, xv, wv)
}

#[test]
fn conv_transpose3d_matches_cpu() {
    let _guard = path_lock();
    if !rlx_cuda::is_available() {
        eprintln!("[rlx-cuda ct3d] no CUDA device — skipping");
        return;
    }
    let (g, xv, wv) = make_ct3d_case();
    let want = cpu_run(g.clone(), &[("x", &xv), ("w", &wv)]);
    let mut exe = CudaExecutable::compile(g);
    let got = exe
        .run(&[("x", &xv), ("w", &wv)])
        .into_iter()
        .next()
        .unwrap();
    assert!(
        close(&got, &want, 1e-4),
        "ConvTranspose3d CUDA vs CPU:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn conv_transpose3d_cudnn_matches_cpu() {
    let _guard = path_lock();
    if !rlx_cuda::is_available() {
        eprintln!("[rlx-cuda ct3d.cudnn] no CUDA device — skipping");
        return;
    }
    rlx_ir::env::unset("RLX_CUDA_NO_CUDNN");
    rlx_ir::env::unset("RLX_CUDA_CONV_T_KERNEL");
    assert!(
        cuda_dnn_handle().is_some(),
        "cuDNN unavailable: set RLX_CUDNN_DIR or put libcudnn.so on the loader path"
    );
    let (g, xv, wv) = make_ct3d_case();
    let want = cpu_run(g.clone(), &[("x", &xv), ("w", &wv)]);
    let mut exe = CudaExecutable::compile(g);
    let got = exe
        .run(&[("x", &xv), ("w", &wv)])
        .into_iter()
        .next()
        .unwrap();
    assert_eq!(
        last_conv_transpose3d_path(),
        Some("cudnn"),
        "expected cuDNN CT3d path, got {:?}",
        last_conv_transpose3d_path()
    );
    assert!(
        close(&got, &want, 1e-4),
        "ConvTranspose3d cuDNN vs CPU:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn conv_transpose3d_kernel_matches_cpu_when_no_cudnn() {
    let _guard = path_lock();
    if !rlx_cuda::is_available() {
        eprintln!("[rlx-cuda ct3d.kernel] no CUDA device — skipping");
        return;
    }
    rlx_ir::env::set("RLX_CUDA_NO_CUDNN", "1");
    let (g, xv, wv) = make_ct3d_case();
    let want = cpu_run(g.clone(), &[("x", &xv), ("w", &wv)]);
    let mut exe = CudaExecutable::compile(g);
    let got = exe
        .run(&[("x", &xv), ("w", &wv)])
        .into_iter()
        .next()
        .unwrap();
    rlx_ir::env::unset("RLX_CUDA_NO_CUDNN");
    assert_eq!(
        last_conv_transpose3d_path(),
        Some("kernel"),
        "expected KERNEL CT3d path under RLX_CUDA_NO_CUDNN, got {:?}",
        last_conv_transpose3d_path()
    );
    assert!(
        close(&got, &want, 1e-4),
        "ConvTranspose3d kernel vs CPU:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn pool3d_max_matches_cpu() {
    if !rlx_cuda::is_available() {
        eprintln!("[rlx-cuda pool3d] no CUDA device — skipping max");
        return;
    }
    let mut g = Graph::new("pool3d_max");
    let x = g.input("x", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
    let y = g.add_node(
        Op::Pool {
            kind: ReduceOp::Max,
            kernel_size: vec![2, 2, 2],
            stride: vec![1, 1, 1],
            padding: vec![0, 0, 0],
        },
        vec![x],
        Shape::new(&[1, 1, 1, 1, 1], DType::F32),
    );
    g.set_outputs(vec![y]);
    let xv: Vec<f32> = (1..=8).map(|i| i as f32).collect();
    let want = cpu_run(g.clone(), &[("x", &xv)]);
    let mut exe = CudaExecutable::compile(g);
    let got = exe.run(&[("x", &xv)]).into_iter().next().unwrap();
    assert!(
        close(&got, &want, 1e-5),
        "Pool3d Max CUDA vs CPU:\n got={got:?}\nwant={want:?}"
    );
    assert!(close(&got, &[8.0], 1e-5));
}

#[test]
fn pool3d_avg_matches_cpu() {
    if !rlx_cuda::is_available() {
        eprintln!("[rlx-cuda pool3d] no CUDA device — skipping avg");
        return;
    }
    let mut g = Graph::new("pool3d_avg");
    let x = g.input("x", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
    let y = g.add_node(
        Op::Pool {
            kind: ReduceOp::Mean,
            kernel_size: vec![2, 2, 2],
            stride: vec![1, 1, 1],
            padding: vec![0, 0, 0],
        },
        vec![x],
        Shape::new(&[1, 1, 1, 1, 1], DType::F32),
    );
    g.set_outputs(vec![y]);
    let xv: Vec<f32> = (1..=8).map(|i| i as f32).collect();
    let want = cpu_run(g.clone(), &[("x", &xv)]);
    let mut exe = CudaExecutable::compile(g);
    let got = exe.run(&[("x", &xv)]).into_iter().next().unwrap();
    assert!(
        close(&got, &want, 1e-4),
        "Pool3d Mean CUDA vs CPU:\n got={got:?}\nwant={want:?}"
    );
    assert!(close(&got, &[4.5], 1e-4));
}

#[test]
fn pool3d_max_strided_matches_cpu() {
    if !rlx_cuda::is_available() {
        eprintln!("[rlx-cuda pool3d] no CUDA device — skipping strided");
        return;
    }
    let mut g = Graph::new("pool3d_strided");
    let x = g.input("x", Shape::new(&[1, 1, 4, 4, 4], DType::F32));
    let y = g.add_node(
        Op::Pool {
            kind: ReduceOp::Max,
            kernel_size: vec![2, 2, 2],
            stride: vec![2, 2, 2],
            padding: vec![0, 0, 0],
        },
        vec![x],
        Shape::new(&[1, 1, 2, 2, 2], DType::F32),
    );
    g.set_outputs(vec![y]);
    let xv: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let want = cpu_run(g.clone(), &[("x", &xv)]);
    let mut exe = CudaExecutable::compile(g);
    let got = exe.run(&[("x", &xv)]).into_iter().next().unwrap();
    assert!(
        close(&got, &want, 1e-5),
        "Pool3d Max strided CUDA vs CPU:\n got={got:?}\nwant={want:?}"
    );
}
