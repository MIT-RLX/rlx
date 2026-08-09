// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-backend parity for `Op::Sort` + `Op::ArgSort` vs CPU (host-staged).

#![allow(dead_code)]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

const R: usize = 4;
const C: usize = 8;

fn x() -> Vec<f32> {
    (0..R * C)
        .map(|i| ((i as f32) * 0.37).sin() * 3.0)
        .collect()
}

fn run(device: Device, xv: &[f32]) -> Vec<f32> {
    let mut g = Graph::new("sort");
    let xi = g.input("x", Shape::new(&[R, C], DType::F32));
    let s = g.sort(xi, 1, false, Shape::new(&[R, C], DType::F32));
    let a = g.argsort(xi, 1, true, Shape::new(&[R, C], DType::F32));
    g.set_outputs(vec![s, a]);
    let outs = Session::new(device).compile(g).run(&[("x", xv)]);
    let mut r = outs[0].clone();
    r.extend_from_slice(&outs[1]);
    r
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

#[test]
fn sort_cpu_runs() {
    assert_eq!(run(Device::Cpu, &x()).len(), 2 * R * C);
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
            let xv = x();
            let cpu = run(Device::Cpu, &xv);
            let dev = run($dev, &xv);
            let err = max_abs(&cpu, &dev);
            eprintln!("{:?}/CPU sort max_abs={:.3e}", $dev, err);
            assert!(err < 1e-5, "sort {:?} parity failed: {:.3e}", $dev, err);
        }
    };
}

backend_parity!(
    sort_metal,
    cfg(all(feature = "metal", target_os = "macos")),
    Device::Metal
);
backend_parity!(
    sort_mlx,
    cfg(all(feature = "mlx", target_os = "macos")),
    Device::Mlx
);
backend_parity!(sort_wgpu, cfg(feature = "gpu"), Device::Gpu);
backend_parity!(sort_cuda, cfg(feature = "cuda"), Device::Cuda);
backend_parity!(sort_rocm, cfg(feature = "rocm"), Device::Rocm);
backend_parity!(sort_vulkan, cfg(feature = "vulkan"), Device::Vulkan);
