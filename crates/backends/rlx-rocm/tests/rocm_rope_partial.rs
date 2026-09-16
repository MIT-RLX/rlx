// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Forward `Op::Rope` with **partial** rotation and several heads per row.
//!
//! Qwen3.5 rotates 64 of each 256-wide head and packs 8 heads into a 2048-wide
//! row. A Jacobian-lens fit on ROCm disagreed with CPU by 0.24 relative, and a
//! node-level diff put the first divergence on exactly this op — worst absolute
//! error 8.5, so not a rounding question.
//!
//! `rope.cu` is shared with CUDA, so this pins whether the fault is the kernel's
//! arithmetic (it would fail on CUDA too) or something ROCm-specific about how
//! it is scheduled — an in-place arena slot, say, where each thread reads a
//! partner element another thread is concurrently overwriting.

#![cfg(target_os = "linux")]

use rlx_ir::op::RopeStyle;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn rope_graph(b: usize, s: usize, last: usize, head_dim: usize, n_rot: usize) -> Graph {
    let f = DType::F32;
    let half = head_dim / 2;
    let mut g = Graph::new("rope");
    let x = g.input("x", Shape::new(&[b, s, last], f));
    let cos = g.param("cos", Shape::new(&[s, half], f));
    let sin = g.param("sin", Shape::new(&[s, half], f));
    let y = g.add_node(
        rlx_ir::Op::Rope {
            head_dim,
            n_rot,
            style: RopeStyle::NeoX,
        },
        vec![x, cos, sin],
        Shape::new(&[b, s, last], f),
    );
    g.set_outputs(vec![y]);
    g
}

fn run(device: Device, b: usize, s: usize, last: usize, head_dim: usize, n_rot: usize) -> Vec<f32> {
    let g = rope_graph(b, s, last, head_dim, n_rot);
    let x: Vec<f32> = (0..b * s * last)
        .map(|i| ((i as f32) * 0.017).sin())
        .collect();
    let half = head_dim / 2;
    let cos: Vec<f32> = (0..s * half).map(|i| ((i as f32) * 0.011).cos()).collect();
    let sin: Vec<f32> = (0..s * half).map(|i| ((i as f32) * 0.011).sin()).collect();
    let mut c = Session::new(device).compile(g);
    c.set_param("cos", &cos);
    c.set_param("sin", &sin);
    c.run(&[("x", &x)]).remove(0)
}

fn check(b: usize, s: usize, last: usize, head_dim: usize, n_rot: usize) {
    let cpu = run(Device::Cpu, b, s, last, head_dim, n_rot);
    let gpu = run(Device::Rocm, b, s, last, head_dim, n_rot);
    assert_eq!(cpu.len(), gpu.len());
    let worst = cpu
        .iter()
        .zip(&gpu)
        .map(|(a, c)| (a - c).abs())
        .fold(0.0f32, f32::max);
    let heads = last / head_dim;
    eprintln!(
        "b{b} s{s} last{last} head_dim{head_dim} n_rot{n_rot} ({heads} head(s)): \
         worst |cpu - rocm| = {worst:.3e}"
    );
    assert!(
        worst < 1e-5,
        "ROCm Rope disagrees with CPU for last={last}, head_dim={head_dim}, n_rot={n_rot} \
         ({heads} heads packed): worst {worst:.3e}"
    );
}

#[test]
fn partial_rope_matches_cpu() {
    if rlx_ir::env::skip_unless_device("rocm", true, rlx_runtime::is_available(Device::Rocm)) {
        eprintln!("skip: ROCm unavailable");
        return;
    }
    // Qwen3.5's own shape first: 8 heads of 256, rotating 64.
    check(8, 24, 2048, 256, 64);
    // Two heads, and the single-head case for contrast.
    check(8, 24, 512, 256, 64);
    check(8, 24, 256, 256, 64);
    // Full rotation, several heads — no pass-through tail.
    check(4, 8, 512, 128, 128);
    // Small partial case.
    check(2, 6, 256, 64, 32);
}
