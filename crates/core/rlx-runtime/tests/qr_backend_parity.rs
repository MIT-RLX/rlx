// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-backend parity for `Op::Qr` (Q/R) vs CPU (host-staged LAPACK).

#![allow(dead_code)]

use rlx_ir::op::QrPart;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

const M: usize = 4;
const N: usize = 3;
const K: usize = 3;

fn a_mat() -> Vec<f32> {
    (0..M * N)
        .map(|i| ((i as f32) * 0.53 + 0.3).sin() * 2.0 + 0.15 * i as f32)
        .collect()
}

fn run(device: Device, a: &[f32]) -> Vec<f32> {
    let mut g = Graph::new("qr");
    let ai = g.input("a", Shape::new(&[M, N], DType::F32));
    let q = g.qr(ai, QrPart::Q, Shape::new(&[M, K], DType::F32));
    let r = g.qr(ai, QrPart::R, Shape::new(&[K, N], DType::F32));
    g.set_outputs(vec![q, r]);
    let outs = Session::new(device).compile(g).run(&[("a", a)]);
    let mut v = outs[0].clone();
    v.extend_from_slice(&outs[1]);
    v
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

#[test]
fn qr_cpu_runs() {
    assert_eq!(run(Device::Cpu, &a_mat()).len(), M * K + K * N);
}

macro_rules! backend_parity {
    ($name:ident, $feat:meta, $dev:expr) => {
        #[test]
        #[$feat]
        fn $name() {
            if !rlx_runtime::is_available($dev) {
                eprintln!("skip: {:?} unavailable", $dev);
                return;
            }
            let a = a_mat();
            let cpu = run(Device::Cpu, &a);
            let dev = run($dev, &a);
            let err = max_abs(&cpu, &dev);
            eprintln!("{:?}/CPU qr max_abs={:.3e}", $dev, err);
            assert!(err < 1e-4, "qr {:?} parity failed: {:.3e}", $dev, err);
        }
    };
}

backend_parity!(
    qr_metal,
    cfg(all(feature = "metal", target_os = "macos")),
    Device::Metal
);
backend_parity!(
    qr_mlx,
    cfg(all(feature = "mlx", target_os = "macos")),
    Device::Mlx
);
backend_parity!(qr_wgpu, cfg(feature = "gpu"), Device::Gpu);
backend_parity!(qr_cuda, cfg(feature = "cuda"), Device::Cuda);
backend_parity!(qr_rocm, cfg(feature = "rocm"), Device::Rocm);
backend_parity!(qr_vulkan, cfg(feature = "vulkan"), Device::Vulkan);
