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

//! Cross-backend parity for `Op::Svd` (U/S/Vt) vs CPU (host-staged LAPACK).

#![allow(dead_code)]

use rlx_ir::op::SvdPart;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session, is_available};

const M: usize = 4;
const N: usize = 3;
const K: usize = 3;

fn a_mat() -> Vec<f32> {
    (0..M * N)
        .map(|i| ((i as f32) * 0.41 + 0.2).sin() * 2.0 + 0.1 * i as f32)
        .collect()
}

// Concatenate [U ; S ; Vt] — host-staged to the same LAPACK, so bit-exact
// (no sign ambiguity across backends).
fn run(device: Device, a: &[f32]) -> Vec<f32> {
    let mut g = Graph::new("svd");
    let ai = g.input("a", Shape::new(&[M, N], DType::F32));
    let u = g.svd(ai, SvdPart::U, Shape::new(&[M, K], DType::F32));
    let s = g.svd(ai, SvdPart::S, Shape::new(&[K], DType::F32));
    let vt = g.svd(ai, SvdPart::Vt, Shape::new(&[K, N], DType::F32));
    g.set_outputs(vec![u, s, vt]);
    let outs = Session::new(device).compile(g).run(&[("a", a)]);
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
fn svd_cpu_runs() {
    assert_eq!(run(Device::Cpu, &a_mat()).len(), M * K + K + K * N);
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
            let a = a_mat();
            let cpu = run(Device::Cpu, &a);
            let dev = run($dev, &a);
            let err = max_abs(&cpu, &dev);
            eprintln!("{:?}/CPU svd max_abs={:.3e}", $dev, err);
            assert!(err < 1e-4, "svd {:?} parity failed: {:.3e}", $dev, err);
        }
    };
}

backend_parity!(
    svd_metal,
    cfg(all(feature = "metal", target_os = "macos")),
    Device::Metal
);
backend_parity!(
    svd_mlx,
    cfg(all(feature = "mlx", target_os = "macos")),
    Device::Mlx
);
backend_parity!(svd_wgpu, cfg(feature = "gpu"), Device::Gpu);
backend_parity!(svd_cuda, cfg(feature = "cuda"), Device::Cuda);
backend_parity!(svd_rocm, cfg(feature = "rocm"), Device::Rocm);
backend_parity!(svd_vulkan, cfg(feature = "vulkan"), Device::Vulkan);
