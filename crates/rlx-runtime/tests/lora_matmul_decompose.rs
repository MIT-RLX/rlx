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

//! `Op::LoraMatMul` standalone now runs on every backend. It hard-failed on
//! Metal/WGPU/CUDA/ROCm because its `unfuse` decomposition (x@W + scale·(x@A)@B)
//! only fired when *another* fused op was present — `LoraMatMul` was missing
//! from `FUSED_KINDS`, the set that triggers the unfuse pass. CPU is the
//! native reference.

#![cfg(feature = "cpu")]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

const M: usize = 3;
const K: usize = 8;
const N: usize = 6;
const R: usize = 2;

fn build() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("lora");
    let x = g.input("x", Shape::new(&[M, K], f));
    let w = g.input("w", Shape::new(&[K, N], f));
    let a = g.input("a", Shape::new(&[K, R], f));
    let b = g.input("b", Shape::new(&[R, N], f));
    let y = g.lora_matmul(x, w, a, b, 0.5, Shape::new(&[M, N], f));
    g.set_outputs(vec![y]);
    g
}

fn run(device: Device) -> Vec<f32> {
    let x: Vec<f32> = (0..M * K).map(|i| (i % 7) as f32 * 0.1 - 0.3).collect();
    let w: Vec<f32> = (0..K * N).map(|i| (i % 5) as f32 * 0.1).collect();
    let a: Vec<f32> = (0..K * R).map(|i| (i % 3) as f32 * 0.2).collect();
    let b: Vec<f32> = (0..R * N).map(|i| (i % 4) as f32 * 0.15).collect();
    Session::new(device)
        .compile(build())
        .run(&[("x", &x), ("w", &w), ("a", &a), ("b", &b)])
        .pop()
        .unwrap()
}

fn assert_close(what: &str, a: &[f32], b: &[f32]) {
    let m = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(m <= 1e-4, "{what}: max abs diff {m:e} > 1e-4");
    eprintln!("{what}: max abs diff {m:.2e}");
}

#[test]
fn lora_cpu_runs() {
    assert_eq!(run(Device::Cpu).len(), M * N);
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn lora_metal_matches_cpu() {
    assert_close("metal", &run(Device::Metal), &run(Device::Cpu));
}

#[test]
#[cfg(feature = "gpu")]
fn lora_wgpu_matches_cpu() {
    assert_close("wgpu", &run(Device::Gpu), &run(Device::Cpu));
}

#[test]
#[cfg(all(target_os = "macos", feature = "mlx"))]
fn lora_mlx_matches_cpu() {
    assert_close("mlx", &run(Device::Mlx), &run(Device::Cpu));
}
