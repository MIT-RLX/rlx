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

//! Cross-backend parity for `Op::Binary(BinaryOp::Atan2)` vs CPU (== `f32::atan2`).

#![allow(dead_code)]

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session, is_available};

const N: usize = 64;

fn inputs() -> (Vec<f32>, Vec<f32>) {
    // General position; avoid the origin and the b<0,a→0 branch cut.
    let a: Vec<f32> = (0..N)
        .map(|i| (i as f32 * 0.19).sin() * 2.0 + 0.05)
        .collect();
    let b: Vec<f32> = (0..N)
        .map(|i| (i as f32 * 0.11).cos() * 2.0 + 0.05)
        .collect();
    (a, b)
}

fn run(device: Device, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut g = Graph::new("atan2");
    let ai = g.input("a", Shape::new(&[N], DType::F32));
    let bi = g.input("b", Shape::new(&[N], DType::F32));
    let y = g.add_node(
        Op::Binary(BinaryOp::Atan2),
        vec![ai, bi],
        Shape::new(&[N], DType::F32),
    );
    g.set_outputs(vec![y]);
    Session::new(device)
        .compile(g)
        .run(&[("a", a), ("b", b)])
        .pop()
        .unwrap()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

#[test]
fn atan2_cpu_matches_f32() {
    let (a, b) = inputs();
    let out = run(Device::Cpu, &a, &b);
    for i in 0..N {
        assert!((out[i] - a[i].atan2(b[i])).abs() < 1e-6);
    }
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
            let (a, b) = inputs();
            let cpu = run(Device::Cpu, &a, &b);
            let dev = run($dev, &a, &b);
            let err = max_abs(&cpu, &dev);
            eprintln!("{:?}/CPU atan2 max_abs={:.3e}", $dev, err);
            assert!(err < 1e-5, "atan2 {:?} parity failed: {:.3e}", $dev, err);
        }
    };
}

backend_parity!(
    atan2_metal,
    cfg(all(feature = "metal", target_os = "macos")),
    Device::Metal
);
backend_parity!(
    atan2_mlx,
    cfg(all(feature = "mlx", target_os = "macos")),
    Device::Mlx
);
backend_parity!(atan2_wgpu, cfg(feature = "gpu"), Device::Gpu);
backend_parity!(atan2_cuda, cfg(feature = "cuda"), Device::Cuda);
backend_parity!(atan2_rocm, cfg(feature = "rocm"), Device::Rocm);
backend_parity!(atan2_vulkan, cfg(feature = "vulkan"), Device::Vulkan);
