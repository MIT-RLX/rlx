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

//! Cross-backend parity for the WGPU vision trio (`GroupNorm`, `LayerNorm2d`,
//! `ResizeNearest2x`) added via host-staging. CPU is the reference.

#![cfg(feature = "cpu")]
#![allow(dead_code)]

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

fn nchw(n: usize, c: usize, h: usize, w: usize) -> Shape {
    Shape::new(&[n, c, h, w], DType::F32)
}

fn ramp(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i + seed) % 17) as f32 - 8.0) * 0.1 + ((i % 5) as f32) * 0.03)
        .collect()
}

fn group_norm_graph(n: usize, c: usize, h: usize, w: usize, groups: usize) -> Graph {
    let mut g = Graph::new("gn");
    let x = g.input("x", nchw(n, c, h, w));
    let gamma = g.input("gamma", Shape::new(&[c], DType::F32));
    let beta = g.input("beta", Shape::new(&[c], DType::F32));
    let y = g.group_norm(x, gamma, beta, groups, 1e-5);
    g.set_outputs(vec![y]);
    g
}

fn layer_norm2d_graph(n: usize, c: usize, h: usize, w: usize) -> Graph {
    let mut g = Graph::new("ln2d");
    let x = g.input("x", nchw(n, c, h, w));
    let gamma = g.input("gamma", Shape::new(&[c], DType::F32));
    let beta = g.input("beta", Shape::new(&[c], DType::F32));
    let y = g.layer_norm2d(x, gamma, beta, 1e-6);
    g.set_outputs(vec![y]);
    g
}

fn resize_graph(n: usize, c: usize, h: usize, w: usize) -> Graph {
    let mut g = Graph::new("resize");
    let x = g.input("x", nchw(n, c, h, w));
    let y = g.add_node(Op::ResizeNearest2x, vec![x], nchw(n, c, h * 2, w * 2));
    g.set_outputs(vec![y]);
    g
}

fn assert_close(what: &str, actual: &[f32], reference: &[f32]) {
    assert_eq!(
        actual.len(),
        reference.len(),
        "{what}: len {} vs {}",
        actual.len(),
        reference.len()
    );
    let max = actual
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max <= 1e-4, "{what}: max abs diff {max:e} > 1e-4");
    eprintln!("{what}: max abs diff {max:.2e} (n={})", actual.len());
}

// ── GroupNorm ────────────────────────────────────────────────────────────
fn run_gn(device: Device, n: usize, c: usize, h: usize, w: usize, groups: usize) -> Vec<f32> {
    let x = ramp(n * c * h * w, 1);
    let gamma = ramp(c, 3);
    let beta = ramp(c, 7);
    let mut exe = Session::new(device).compile(group_norm_graph(n, c, h, w, groups));
    exe.run(&[
        ("x", x.as_slice()),
        ("gamma", gamma.as_slice()),
        ("beta", beta.as_slice()),
    ])
    .pop()
    .unwrap()
}

fn run_ln2d(device: Device, n: usize, c: usize, h: usize, w: usize) -> Vec<f32> {
    let x = ramp(n * c * h * w, 2);
    let gamma = ramp(c, 4);
    let beta = ramp(c, 8);
    let mut exe = Session::new(device).compile(layer_norm2d_graph(n, c, h, w));
    exe.run(&[
        ("x", x.as_slice()),
        ("gamma", gamma.as_slice()),
        ("beta", beta.as_slice()),
    ])
    .pop()
    .unwrap()
}

fn run_resize(device: Device, n: usize, c: usize, h: usize, w: usize) -> Vec<f32> {
    let x = ramp(n * c * h * w, 5);
    let mut exe = Session::new(device).compile(resize_graph(n, c, h, w));
    exe.run(&[("x", x.as_slice())]).pop().unwrap()
}

#[test]
#[cfg(feature = "gpu")]
fn group_norm_wgpu_matches_cpu() {
    for &(n, c, h, w, gr) in &[(1, 8, 4, 4, 2), (2, 32, 8, 8, 8), (1, 16, 16, 16, 4)] {
        assert_close(
            &format!("gn wgpu [{n},{c},{h},{w}] groups={gr}"),
            &run_gn(Device::Gpu, n, c, h, w, gr),
            &run_gn(Device::Cpu, n, c, h, w, gr),
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn layer_norm2d_wgpu_matches_cpu() {
    for &(n, c, h, w) in &[(1, 8, 4, 4), (2, 32, 8, 8), (1, 64, 16, 16)] {
        assert_close(
            &format!("ln2d wgpu [{n},{c},{h},{w}]"),
            &run_ln2d(Device::Gpu, n, c, h, w),
            &run_ln2d(Device::Cpu, n, c, h, w),
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn resize_nearest2x_wgpu_matches_cpu() {
    for &(n, c, h, w) in &[(1, 3, 8, 8), (2, 16, 12, 10), (1, 8, 32, 32)] {
        assert_close(
            &format!("resize wgpu [{n},{c},{h},{w}]"),
            &run_resize(Device::Gpu, n, c, h, w),
            &run_resize(Device::Cpu, n, c, h, w),
        );
    }
}

// ── MLX arms (GroupNorm newly added; LayerNorm2d/ResizeNearest2x cross-check) ──
#[test]
#[cfg(all(target_os = "macos", feature = "mlx"))]
fn group_norm_mlx_matches_cpu() {
    for &(n, c, h, w, gr) in &[(1, 8, 4, 4, 2), (2, 32, 8, 8, 8), (1, 16, 16, 16, 4)] {
        assert_close(
            &format!("gn mlx [{n},{c},{h},{w}] groups={gr}"),
            &run_gn(Device::Mlx, n, c, h, w, gr),
            &run_gn(Device::Cpu, n, c, h, w, gr),
        );
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn resize_nearest2x_metal_matches_cpu() {
    // Standalone resize is wrapped into a single-step TransformRegion; Metal
    // must unwrap it to the native resize thunk (was an unimplemented panic).
    for &(n, c, h, w) in &[(1, 4, 8, 8), (2, 16, 12, 10), (1, 8, 32, 32)] {
        assert_close(
            &format!("resize metal [{n},{c},{h},{w}]"),
            &run_resize(Device::Metal, n, c, h, w),
            &run_resize(Device::Cpu, n, c, h, w),
        );
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "mlx"))]
fn layer_norm2d_mlx_matches_cpu() {
    for &(n, c, h, w) in &[(1, 8, 4, 4), (2, 32, 8, 8)] {
        assert_close(
            &format!("ln2d mlx [{n},{c},{h},{w}]"),
            &run_ln2d(Device::Mlx, n, c, h, w),
            &run_ln2d(Device::Cpu, n, c, h, w),
        );
    }
}
