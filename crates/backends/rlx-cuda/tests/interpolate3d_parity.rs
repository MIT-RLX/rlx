// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU-vs-CUDA parity for `Op::Interpolate3d` (nearest NCDHW resample).

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
fn interpolate3d_2x_matches_cpu() {
    if !rlx_cuda::is_available() {
        eprintln!("[rlx-cuda interpolate3d] no CUDA device — skipping");
        return;
    }
    let mut g = Graph::new("interp3d_2x");
    let x = g.input("x", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
    let y = g.interpolate3d_nearest(x, [4, 4, 4]);
    g.set_outputs(vec![y]);
    let xv: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    let want = cpu_run(g.clone(), &[("x", &xv)]);
    let mut exe = CudaExecutable::compile(g);
    let got = exe.run(&[("x", &xv)]).into_iter().next().unwrap();
    assert!(
        close(&got, &want, 1e-6),
        "Interpolate3d 2x CUDA vs CPU:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn interpolate3d_asymmetric_matches_cpu() {
    if !rlx_cuda::is_available() {
        eprintln!("[rlx-cuda interpolate3d] no CUDA device — skipping asymmetric");
        return;
    }
    let mut g = Graph::new("interp3d_asym");
    let x = g.input("x", Shape::new(&[1, 2, 2, 3, 3], DType::F32));
    let y = g.interpolate3d_nearest(x, [3, 5, 4]);
    g.set_outputs(vec![y]);
    let xv: Vec<f32> = (0..36).map(|v| v as f32 * 0.1).collect();
    let want = cpu_run(g.clone(), &[("x", &xv)]);
    let mut exe = CudaExecutable::compile(g);
    let got = exe.run(&[("x", &xv)]).into_iter().next().unwrap();
    assert!(
        close(&got, &want, 1e-6),
        "Interpolate3d asymmetric CUDA vs CPU:\n got={got:?}\nwant={want:?}"
    );
}
