// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::RopeBackward` with **partial** RoPE and several heads per row.
//!
//! RoPE rotates the first `n_rot` channels of *every head* and passes the rest
//! through. When the last axis packs `heads · head_dim`, slicing that flat axis
//! at `n_rot` rotates head 0 and leaves every other head untouched — the shapes
//! all still line up, the forward is unaffected, and the gradient is quietly
//! wrong. Qwen3.5 is exactly this case (`head_dim` 256, `n_rot` 64), where it
//! cost ~1% relative error in an attention block's Jacobian.
//!
//! The single-head case (`last == head_dim`) never exercised it, which is why it
//! survived; both are covered here.

#![cfg(target_os = "macos")]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

/// `dy` → `RopeBackward` → out, with cos/sin as parameters.
fn rope_backward_graph(b: usize, s: usize, last: usize, head_dim: usize, n_rot: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("rope_bwd");
    let dy = g.input("dy", Shape::new(&[b, s, last], f));
    let cos = g.param("cos", Shape::new(&[s, n_rot / 2], f));
    let sin = g.param("sin", Shape::new(&[s, n_rot / 2], f));
    let y = g.add_node(
        rlx_ir::Op::RopeBackward {
            head_dim,
            n_rot,
            style: rlx_ir::op::RopeStyle::NeoX,
        },
        vec![dy, cos, sin],
        Shape::new(&[b, s, last], f),
    );
    g.set_outputs(vec![y]);
    g
}

fn run(device: Device, b: usize, s: usize, last: usize, head_dim: usize, n_rot: usize) -> Vec<f32> {
    let g = rope_backward_graph(b, s, last, head_dim, n_rot);
    let dy: Vec<f32> = (0..b * s * last)
        .map(|i| ((i as f32) * 0.017).sin())
        .collect();
    let half = n_rot / 2;
    let cos: Vec<f32> = (0..s * half).map(|i| ((i as f32) * 0.011).cos()).collect();
    let sin: Vec<f32> = (0..s * half).map(|i| ((i as f32) * 0.011).sin()).collect();
    let mut c = Session::new(device).compile(g);
    c.set_param("cos", &cos);
    c.set_param("sin", &sin);
    c.run(&[("dy", &dy)]).remove(0)
}

fn check(b: usize, s: usize, last: usize, head_dim: usize, n_rot: usize) {
    let cpu = run(Device::Cpu, b, s, last, head_dim, n_rot);
    let mlx = run(Device::Mlx, b, s, last, head_dim, n_rot);
    assert_eq!(cpu.len(), mlx.len());
    let worst = cpu
        .iter()
        .zip(&mlx)
        .map(|(a, c)| (a - c).abs())
        .fold(0.0f32, f32::max);
    let denom = cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
    eprintln!(
        "b{b} s{s} last{last} head_dim{head_dim} n_rot{n_rot}: worst |cpu - mlx| = {worst:.3e} \
         ({:.3e} relative)",
        worst / denom
    );
    assert!(
        worst / denom < 1e-5,
        "MLX RopeBackward disagrees with CPU for last={last}, head_dim={head_dim}, \
         n_rot={n_rot}: worst {worst:.3e} — check that the head axis is split out \
         before slicing at n_rot"
    );
}

#[test]
fn partial_rope_backward_matches_cpu() {
    if rlx_ir::env::skip_unless_device("mlx", true, rlx_runtime::is_available(Device::Mlx)) {
        eprintln!("skip: MLX unavailable");
        return;
    }
    // Several heads packed into the last axis, partial rotation — the case that
    // regressed. Qwen3.5's own shape first.
    check(8, 24, 512, 256, 64);
    check(2, 6, 256, 64, 32);
    check(1, 4, 128, 32, 16);
    // Full rotation with several heads: n_rot == head_dim, so there is no tail.
    check(2, 6, 256, 64, 64);
    // Single head, which is what the pre-fix code handled correctly.
    check(2, 6, 64, 64, 32);
    check(2, 6, 64, 64, 64);
}
