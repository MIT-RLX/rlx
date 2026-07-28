// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CUDA ↔ CPU gradient parity for native 3-D Conv / MaxPool.

use rlx_ir::op::ReduceOp;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::{Device, Session, is_available};

const F: DType = DType::F32;

fn target() -> Device {
    match std::env::var("RLX_PARITY_DEVICE") {
        Ok(s) => rlx_runtime::parse_device(&s).unwrap_or(Device::Cuda),
        Err(_) => Device::Cuda,
    }
}

fn seeded(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z ^= z >> 31;
            ((z >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn run(
    bwd: &Graph,
    dev: Device,
    params: &[(&str, Vec<f32>)],
    inputs: &[(&str, &[f32])],
) -> Vec<Vec<f32>> {
    let mut sess = Session::new(dev).compile(bwd.clone());
    for (n, v) in params {
        sess.set_param(n, v);
    }
    sess.run(inputs)
}

fn max_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn parity(
    name: &str,
    bwd: &Graph,
    params: &[(&str, Vec<f32>)],
    inputs: &[(&str, &[f32])],
    tol: f32,
) -> bool {
    let cpu = run(bwd, Device::Cpu, params, inputs);
    let gpu = run(bwd, target(), params, inputs);
    let mut bad = false;
    for (i, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
        let e = max_err(c, g);
        if e > tol {
            eprintln!("{name} out[{i}] max_err={e} (tol={tol})");
            bad = true;
        }
    }
    bad
}

fn sum_loss(g: &mut Graph, y: NodeId) -> NodeId {
    let rank = g.node(y).shape.rank();
    g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: (0..rank).collect(),
            keep_dim: false,
        },
        vec![y],
        Shape::from_dims(&[], F),
    )
}

#[test]
fn conv3d_grads_match_cpu() {
    if !is_available(target()) {
        eprintln!(
            "ops3d_backprop: {:?} unavailable — skipping conv3d",
            target()
        );
        return;
    }
    let mut g = Graph::new("c3d");
    let x = g.param("x", Shape::new(&[1, 1, 3, 3, 3], F));
    let w = g.param("w", Shape::new(&[1, 1, 2, 2, 2], F));
    let y = g.conv3d(x, w, [1, 1, 1], [0, 0, 0], [1, 1, 1], 1);
    let loss = sum_loss(&mut g, y);
    g.set_outputs(vec![loss]);
    let bwd = rlx_autodiff::grad_with_loss(&g, &[x, w]);
    let p = vec![("x", seeded(27, 3)), ("w", seeded(8, 5))];
    assert!(
        !parity("conv3d", &bwd, &p, &[("d_output", &[1.0])], 2e-3),
        "conv3d: gradients disagree CPU vs {:?}",
        target()
    );
}

#[test]
fn maxpool3d_grads_match_cpu() {
    if !is_available(target()) {
        eprintln!(
            "ops3d_backprop: {:?} unavailable — skipping maxpool3d",
            target()
        );
        return;
    }
    let mut g = Graph::new("mp3d");
    let x = g.param("x", Shape::new(&[1, 1, 2, 2, 2], F));
    let y = g.add_node(
        Op::Pool {
            kind: ReduceOp::Max,
            kernel_size: vec![2, 2, 2],
            stride: vec![1, 1, 1],
            padding: vec![0, 0, 0],
        },
        vec![x],
        Shape::new(&[1, 1, 1, 1, 1], F),
    );
    let loss = sum_loss(&mut g, y);
    g.set_outputs(vec![loss]);
    let bwd = rlx_autodiff::grad_with_loss(&g, &[x]);
    // Distinct values → unique argmax.
    let xv = vec![0.1, 0.3, -0.2, 0.5, 0.4, -0.1, 0.2, 0.8];
    assert!(
        !parity(
            "maxpool3d",
            &bwd,
            &[("x", xv)],
            &[("d_output", &[1.0])],
            1e-4
        ),
        "maxpool3d: gradients disagree CPU vs {:?}",
        target()
    );
}

#[test]
fn conv_transpose3d_grads_match_cpu() {
    if !is_available(target()) {
        eprintln!("ops3d_backprop: {:?} unavailable — skipping ct3d", target());
        return;
    }
    let mut g = Graph::new("ct3d");
    let x = g.param("x", Shape::new(&[1, 1, 2, 2, 2], F));
    let w = g.param("w", Shape::new(&[1, 1, 2, 2, 2], F));
    let y = g.conv_transpose3d(x, w, [2, 2, 2], [0, 0, 0], [1, 1, 1], [0, 0, 0], 1);
    let loss = sum_loss(&mut g, y);
    g.set_outputs(vec![loss]);
    let bwd = rlx_autodiff::grad_with_loss(&g, &[x, w]);
    let p = vec![("x", seeded(8, 11)), ("w", seeded(8, 13))];
    assert!(
        !parity("conv_transpose3d", &bwd, &p, &[("d_output", &[1.0])], 3e-3),
        "conv_transpose3d: gradients disagree CPU vs {:?}",
        target()
    );
}
