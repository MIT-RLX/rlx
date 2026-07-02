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

//! CPU vs GPU parity for rank-4 `[B, S, H, D]` attention (EEG-DINO layout).

#![allow(dead_code)]

use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

fn build_bshd_attn(b: usize, s: usize, nh: usize, dh: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("bshd_attn");
    let q = g.input("q", Shape::new(&[b, s, nh, dh], f));
    let k = g.input("k", Shape::new(&[b, s, nh, dh], f));
    let v = g.input("v", Shape::new(&[b, s, nh, dh], f));
    let out = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: dh,
            mask_kind: MaskKind::None,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v],
        Shape::new(&[b, s, nh, dh], f),
    );
    g.set_outputs(vec![out]);
    g
}

fn run(
    device: Device,
    b: usize,
    s: usize,
    nh: usize,
    dh: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
) -> Vec<f32> {
    let g = build_bshd_attn(b, s, nh, dh);
    let mut compiled = Session::new(device).compile(g);
    compiled
        .run(&[("q", q), ("k", k), ("v", v)])
        .into_iter()
        .next()
        .unwrap()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

fn deterministic_inputs(
    b: usize,
    s: usize,
    nh: usize,
    dh: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = b * s * nh * dh;
    let q: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).sin() * 0.5).collect();
    let k: Vec<f32> = (0..n).map(|i| (i as f32 * 0.11).cos() * 0.3).collect();
    let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.03) % 1.0 - 0.5).collect();
    (q, k, v)
}

#[test]
fn cpu_bshd_rank4_reference() {
    let (b, s, nh, dh) = (1, 191, 8, 25);
    let (q, k, v) = deterministic_inputs(b, s, nh, dh);
    let out = run(Device::Cpu, b, s, nh, dh, &q, &k, &v);
    assert_eq!(out.len(), b * s * nh * dh);
}

macro_rules! gpu_parity {
    ($name:ident, $feat:meta, $dev:expr) => {
        #[test]
        #[$feat]
        fn $name() {
            if !rlx_runtime::is_available($dev) {
                eprintln!("skip: {:?} unavailable", $dev);
                return;
            }
            // EEG-DINO small: B=1, S=191, H=8, D=25
            let (b, s, nh, dh) = (1, 191, 8, 25);
            let (q, k, v) = deterministic_inputs(b, s, nh, dh);
            let cpu = run(Device::Cpu, b, s, nh, dh, &q, &k, &v);
            let gpu = run($dev, b, s, nh, dh, &q, &k, &v);
            let err = max_abs(&cpu, &gpu);
            eprintln!("{}/CPU max_abs={err:.3e}", stringify!($dev));
            assert!(err < 1e-4, "BSHD rank-4 attention parity failed: {err:.3e}");
        }
    };
}

gpu_parity!(
    bshd_parity_metal,
    cfg(all(feature = "metal", target_os = "macos")),
    Device::Metal
);
gpu_parity!(
    bshd_parity_mlx,
    cfg(all(feature = "mlx", target_os = "macos")),
    Device::Mlx
);
gpu_parity!(bshd_parity_wgpu, cfg(feature = "gpu"), Device::Gpu);
gpu_parity!(bshd_parity_cuda, cfg(feature = "cuda"), Device::Cuda);
gpu_parity!(bshd_parity_rocm, cfg(feature = "rocm"), Device::Rocm);
