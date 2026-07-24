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

//! CPU-vs-CUDA parity for 3-D training backward ops:
//! `Conv3dBackwardInput` / `Conv3dBackwardWeight` (cuDNN + gather `.cu`) and
//! `MaxPool3dBackward` (kernel only).

use std::sync::Mutex;

use rlx_cuda::CudaExecutable;
use rlx_cuda::device::{cuda_dnn_handle, last_conv3d_bwd_path};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

static PATH_LOCK: Mutex<()> = Mutex::new(());

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

/// Forward geometry: x [1,1,3,3,3], w [1,1,2,2,2] → y [1,1,2,2,2].
fn make_conv3d_bwd_input() -> (Graph, Vec<f32>, Vec<f32>) {
    let mut g = Graph::new("c3d_bwd_in");
    let dy = g.input("dy", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
    let w = g.input("w", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
    let dx = g.conv3d_backward_input(
        dy,
        w,
        Shape::new(&[1, 1, 3, 3, 3], DType::F32),
        vec![2, 2, 2],
        vec![1, 1, 1],
        vec![0, 0, 0],
        vec![1, 1, 1],
        1,
    );
    g.set_outputs(vec![dx]);
    let dyv: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    let wv: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    (g, dyv, wv)
}

fn make_conv3d_bwd_weight() -> (Graph, Vec<f32>, Vec<f32>) {
    let mut g = Graph::new("c3d_bwd_w");
    let x = g.input("x", Shape::new(&[1, 1, 3, 3, 3], DType::F32));
    let dy = g.input("dy", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
    let dw = g.conv3d_backward_weight(
        x,
        dy,
        Shape::new(&[1, 1, 2, 2, 2], DType::F32),
        vec![2, 2, 2],
        vec![1, 1, 1],
        vec![0, 0, 0],
        vec![1, 1, 1],
        1,
    );
    g.set_outputs(vec![dw]);
    let xv: Vec<f32> = (1..=27).map(|v| v as f32).collect();
    let dyv: Vec<f32> = (1..=8).map(|v| v as f32 * 0.5).collect();
    (g, xv, dyv)
}

fn make_maxpool3d_bwd() -> (Graph, Vec<f32>, Vec<f32>) {
    let mut g = Graph::new("mp3d_bwd");
    let x = g.input("x", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
    let dy = g.input("dy", Shape::new(&[1, 1, 1, 1, 1], DType::F32));
    let dx = g.maxpool3d_backward(x, dy, vec![2, 2, 2], vec![1, 1, 1], vec![0, 0, 0]);
    g.set_outputs(vec![dx]);
    // Unique max at last index so the gradient has a single spike.
    let xv = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0];
    let dyv = vec![1.0];
    (g, xv, dyv)
}

#[test]
fn conv3d_backward_input_matches_cpu() {
    if !rlx_cuda::is_available() {
        eprintln!("[rlx-cuda c3d_bwd_in] no CUDA device — skipping");
        return;
    }
    let (g, dyv, wv) = make_conv3d_bwd_input();
    let want = cpu_run(g.clone(), &[("dy", &dyv), ("w", &wv)]);
    let mut exe = CudaExecutable::compile(g);
    let got = exe
        .run(&[("dy", &dyv), ("w", &wv)])
        .into_iter()
        .next()
        .unwrap();
    assert!(
        close(&got, &want, 1e-4),
        "Conv3dBackwardInput CUDA vs CPU:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn conv3d_backward_input_cudnn_matches_cpu() {
    let _guard = PATH_LOCK.lock().unwrap();
    if !rlx_cuda::is_available() {
        eprintln!("[rlx-cuda c3d_bwd_in.cudnn] no CUDA device — skipping");
        return;
    }
    rlx_ir::env::unset("RLX_CUDA_CONV_FORCE_GATHER");
    rlx_ir::env::unset("RLX_CUDA_NO_CUDNN");
    assert!(
        cuda_dnn_handle().is_some(),
        "cuDNN unavailable: set RLX_CUDNN_DIR or put libcudnn.so on the loader path"
    );
    let (g, dyv, wv) = make_conv3d_bwd_input();
    let want = cpu_run(g.clone(), &[("dy", &dyv), ("w", &wv)]);
    let mut exe = CudaExecutable::compile(g);
    let got = exe
        .run(&[("dy", &dyv), ("w", &wv)])
        .into_iter()
        .next()
        .unwrap();
    assert_eq!(
        last_conv3d_bwd_path(),
        Some("cudnn"),
        "expected cuDNN Conv3dBackwardInput, got {:?}",
        last_conv3d_bwd_path()
    );
    assert!(
        close(&got, &want, 1e-4),
        "Conv3dBackwardInput cuDNN vs CPU:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn conv3d_backward_input_kernel_matches_cpu() {
    let _guard = PATH_LOCK.lock().unwrap();
    if !rlx_cuda::is_available() {
        eprintln!("[rlx-cuda c3d_bwd_in.kernel] no CUDA device — skipping");
        return;
    }
    rlx_ir::env::set("RLX_CUDA_CONV_FORCE_GATHER", "1");
    let (g, dyv, wv) = make_conv3d_bwd_input();
    let want = cpu_run(g.clone(), &[("dy", &dyv), ("w", &wv)]);
    let mut exe = CudaExecutable::compile(g);
    let got = exe
        .run(&[("dy", &dyv), ("w", &wv)])
        .into_iter()
        .next()
        .unwrap();
    rlx_ir::env::unset("RLX_CUDA_CONV_FORCE_GATHER");
    assert_eq!(
        last_conv3d_bwd_path(),
        Some("kernel"),
        "expected gather kernel under RLX_CUDA_CONV_FORCE_GATHER, got {:?}",
        last_conv3d_bwd_path()
    );
    assert!(
        close(&got, &want, 1e-4),
        "Conv3dBackwardInput kernel vs CPU:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn conv3d_backward_weight_matches_cpu() {
    if !rlx_cuda::is_available() {
        eprintln!("[rlx-cuda c3d_bwd_w] no CUDA device — skipping");
        return;
    }
    let (g, xv, dyv) = make_conv3d_bwd_weight();
    let want = cpu_run(g.clone(), &[("x", &xv), ("dy", &dyv)]);
    let mut exe = CudaExecutable::compile(g);
    let got = exe
        .run(&[("x", &xv), ("dy", &dyv)])
        .into_iter()
        .next()
        .unwrap();
    assert!(
        close(&got, &want, 1e-4),
        "Conv3dBackwardWeight CUDA vs CPU:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn conv3d_backward_weight_cudnn_matches_cpu() {
    let _guard = PATH_LOCK.lock().unwrap();
    if !rlx_cuda::is_available() {
        eprintln!("[rlx-cuda c3d_bwd_w.cudnn] no CUDA device — skipping");
        return;
    }
    rlx_ir::env::unset("RLX_CUDA_CONV_FORCE_GATHER");
    rlx_ir::env::unset("RLX_CUDA_NO_CUDNN");
    assert!(
        cuda_dnn_handle().is_some(),
        "cuDNN unavailable: set RLX_CUDNN_DIR or put libcudnn.so on the loader path"
    );
    let (g, xv, dyv) = make_conv3d_bwd_weight();
    let want = cpu_run(g.clone(), &[("x", &xv), ("dy", &dyv)]);
    let mut exe = CudaExecutable::compile(g);
    let got = exe
        .run(&[("x", &xv), ("dy", &dyv)])
        .into_iter()
        .next()
        .unwrap();
    assert_eq!(
        last_conv3d_bwd_path(),
        Some("cudnn"),
        "expected cuDNN Conv3dBackwardWeight, got {:?}",
        last_conv3d_bwd_path()
    );
    assert!(
        close(&got, &want, 1e-4),
        "Conv3dBackwardWeight cuDNN vs CPU:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn maxpool3d_backward_matches_cpu() {
    if !rlx_cuda::is_available() {
        eprintln!("[rlx-cuda mp3d_bwd] no CUDA device — skipping");
        return;
    }
    let (g, xv, dyv) = make_maxpool3d_bwd();
    let want = cpu_run(g.clone(), &[("x", &xv), ("dy", &dyv)]);
    let mut exe = CudaExecutable::compile(g);
    let got = exe
        .run(&[("x", &xv), ("dy", &dyv)])
        .into_iter()
        .next()
        .unwrap();
    assert!(
        close(&got, &want, 1e-5),
        "MaxPool3dBackward CUDA vs CPU:\n got={got:?}\nwant={want:?}"
    );
}
