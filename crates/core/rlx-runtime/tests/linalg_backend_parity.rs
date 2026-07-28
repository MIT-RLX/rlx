// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-backend parity for `Op::Cholesky` and `Op::TriangularSolve` vs CPU.
//! GPU backends host-stage these to CPU LAPACK (the `DenseSolve` pattern).

#![allow(dead_code)]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session, is_available};

const N: usize = 5;

fn spd_and_b() -> (Vec<f32>, Vec<f32>) {
    let m: Vec<f32> = (0..N * N)
        .map(|i| ((i as f32) * 0.29 + 0.1).sin() * 0.5)
        .collect();
    let mut a = vec![0f32; N * N];
    for i in 0..N {
        for j in 0..N {
            let mut s = 0.0f32;
            for k in 0..N {
                s += m[i * N + k] * m[j * N + k];
            }
            a[i * N + j] = s + if i == j { N as f32 } else { 0.0 };
        }
    }
    let b: Vec<f32> = (0..N).map(|i| ((i as f32) * 0.5 + 0.3).cos()).collect();
    (a, b)
}

// Cholesky then solve `L·x = b` — exercises both ops in one graph.
fn run(device: Device, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut g = Graph::new("linalg");
    let ai = g.input("a", Shape::new(&[N, N], DType::F32));
    let bi = g.input("b", Shape::new(&[N, 1], DType::F32));
    let l = g.cholesky(ai, Shape::new(&[N, N], DType::F32));
    let x = g.triangular_solve(l, bi, true, false, Shape::new(&[N, 1], DType::F32));
    let det = g.det(ai, Shape::from_dims(&[], DType::F32));
    let ld = g.logdet(ai, Shape::from_dims(&[], DType::F32));
    g.set_outputs(vec![x, det, ld]);
    let outs = Session::new(device).compile(g).run(&[("a", a), ("b", b)]);
    // Concatenate [trisolve(chol) ; det ; logdet] → exercises all four ops.
    let mut r = outs[0].clone();
    r.extend_from_slice(&outs[1]);
    r.extend_from_slice(&outs[2]);
    r
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

#[test]
fn linalg_cpu_runs() {
    let (a, b) = spd_and_b();
    let out = run(Device::Cpu, &a, &b);
    assert_eq!(out.len(), N + 2); // trisolve(N) + det + logdet
}

macro_rules! backend_parity {
    ($name:ident, $feat:meta, $dev:expr) => {
        #[test]
        #[$feat]
        fn $name() {
            if !is_available($dev) {
                eprintln!("skip: {:?} unavailable", $dev);
                return;
            }
            let (a, b) = spd_and_b();
            let cpu = run(Device::Cpu, &a, &b);
            let dev = run($dev, &a, &b);
            let err = max_abs(&cpu, &dev);
            eprintln!("{:?}/CPU linalg max_abs={:.3e}", $dev, err);
            assert!(err < 1e-4, "linalg {:?} parity failed: {:.3e}", $dev, err);
        }
    };
}

backend_parity!(
    linalg_metal,
    cfg(all(feature = "metal", target_os = "macos")),
    Device::Metal
);
backend_parity!(
    linalg_mlx,
    cfg(all(feature = "mlx", target_os = "macos")),
    Device::Mlx
);
backend_parity!(linalg_wgpu, cfg(feature = "gpu"), Device::Gpu);
backend_parity!(linalg_cuda, cfg(feature = "cuda"), Device::Cuda);
backend_parity!(linalg_rocm, cfg(feature = "rocm"), Device::Rocm);
backend_parity!(linalg_vulkan, cfg(feature = "vulkan"), Device::Vulkan);
