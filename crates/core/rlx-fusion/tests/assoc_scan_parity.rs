// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `ParallelizeAffineScan` (`assoc-scan`): affine scans rewritten to a
//! log-depth prefix scan must match the sequential `Op::Scan` on the CPU
//! executor, and non-affine bodies must be left alone.
#![cfg(feature = "assoc-scan")]

use rlx_fusion::assoc_scan::ParallelizeAffineScan;
use rlx_fusion::pass::Pass;
use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, NodeId, Op, OpKind, Shape};

const T: usize = 37; // not a power of two
const D: usize = 5;

fn sh(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

fn data(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            (((i as u32).wrapping_mul(2654435761) ^ seed.wrapping_mul(40503)) % 1000) as f32
                / 1000.0
                - 0.5
        })
        .collect()
}

fn run(g: &Graph, feeds: &[(&str, Vec<f32>)]) -> Vec<f32> {
    let plan = rlx_opt::memory::plan_memory(g);
    let mut arena = rlx_cpu::arena::Arena::from_plan(plan);
    let sched = rlx_cpu::thunk::compile_thunks(g, &arena);
    for node in g.nodes() {
        match &node.op {
            Op::Constant { data } if arena.has_buffer(node.id) && !data.is_empty() => {
                let buf = arena.slice_mut(node.id);
                for (dst, c) in buf.iter_mut().zip(data.chunks_exact(4)) {
                    *dst = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                }
            }
            Op::Input { name } => {
                let (_, v) = feeds.iter().find(|(n, _)| n == name).expect("feed");
                arena.slice_mut(node.id)[..v.len()].copy_from_slice(v);
            }
            _ => {}
        }
    }
    rlx_cpu::thunk::execute_thunks(&sched, arena.raw_buf_mut());
    let o = g.outputs[0];
    let n = g.node(o).shape.num_elements().unwrap();
    arena.slice_mut(o)[..n].to_vec()
}

fn has_scan(g: &Graph) -> bool {
    g.nodes().iter().any(|n| n.op.kind() == OpKind::Scan)
}

/// Builds `scan(h0, bcast=[c], xs=[x, y])` with a body from `body_fn`
/// (`carry, c, x_t, y_t → next`).
fn scan_graph(
    trajectory: bool,
    body_fn: impl Fn(&mut Graph, NodeId, NodeId, NodeId, NodeId) -> NodeId,
) -> Graph {
    let mut body = Graph::new("body");
    let h = body.input("h", sh(&[D]));
    let c = body.input("c", sh(&[D]));
    let x = body.input("x", sh(&[D]));
    let y = body.input("y", sh(&[D]));
    let next = body_fn(&mut body, h, c, x, y);
    body.set_outputs(vec![next]);

    let mut g = Graph::new("outer");
    let h0 = g.input("h0", sh(&[D]));
    let c = g.input("c", sh(&[D]));
    let xs = g.input("xs", sh(&[T, D]));
    let ys = g.input("ys", sh(&[T, D]));
    let out_shape = if trajectory { sh(&[T, D]) } else { sh(&[D]) };
    let scan = g.add_node(
        Op::Scan {
            body: Box::new(body),
            length: T as u32,
            save_trajectory: trajectory,
            num_bcast: 1,
            num_xs: 2,
            num_checkpoints: 0,
        },
        vec![h0, c, xs, ys],
        out_shape,
    );
    g.set_outputs(vec![scan]);
    g
}

fn check(name: &str, g: Graph, expect_rewrite: bool) {
    let feeds = [
        ("h0", data(D, 1)),
        ("c", data(D, 2)),
        ("xs", data(T * D, 3)),
        ("ys", data(T * D, 4)),
    ];
    let want = run(&g, &feeds);
    let rewritten = ParallelizeAffineScan::default().run(g);
    assert_eq!(
        !has_scan(&rewritten),
        expect_rewrite,
        "{name}: rewrite decision"
    );
    let got = run(&rewritten, &feeds);
    let err = want
        .iter()
        .zip(&got)
        .map(|(a, b)| (a - b).abs() / a.abs().max(1.0))
        .fold(0f32, f32::max);
    assert!(err < 1e-5, "{name}: max rel err {err}");
}

fn bin(g: &mut Graph, op: BinaryOp, a: NodeId, b: NodeId) -> NodeId {
    g.binary(op, a, b, sh(&[D]))
}

#[test]
fn ema_with_broadcast_decay() {
    // h = c*h + x
    for traj in [false, true] {
        let g = scan_graph(traj, |g, h, c, x, _| {
            let ch = bin(g, BinaryOp::Mul, c, h);
            bin(g, BinaryOp::Add, ch, x)
        });
        check("ema", g, true);
    }
}

#[test]
fn time_varying_iir_with_lifted_coefficients() {
    // h = (sigmoid(x) * h + tanh(y) * c) / (c + c): coefficients are
    // expressions of the per-step and broadcast inputs, lifted out of the body.
    for traj in [false, true] {
        let g = scan_graph(traj, |g, h, c, x, y| {
            let a = g.activation(Activation::Sigmoid, x, sh(&[D]));
            let ty = g.activation(Activation::Tanh, y, sh(&[D]));
            let ah = bin(g, BinaryOp::Mul, a, h);
            let b = bin(g, BinaryOp::Mul, ty, c);
            let s = bin(g, BinaryOp::Add, ah, b);
            let two = bin(g, BinaryOp::Add, c, c);
            bin(g, BinaryOp::Div, s, two)
        });
        check("iir", g, true);
    }
}

#[test]
fn cumulative_sum_and_product() {
    let sum = scan_graph(true, |g, h, _, x, _| bin(g, BinaryOp::Add, h, x));
    check("cumsum", sum, true);
    let prod = scan_graph(true, |g, h, _, x, _| {
        let a = g.activation(Activation::Exp, x, sh(&[D]));
        bin(g, BinaryOp::Mul, h, a)
    });
    check("cumprod", prod, true);
    // h - x: subtraction keeps a = 1.
    let diff = scan_graph(false, |g, h, _, x, _| bin(g, BinaryOp::Sub, h, x));
    check("cumdiff", diff, true);
}

#[test]
fn non_affine_bodies_are_left_alone() {
    // h*h is quadratic in the carry.
    let sq = scan_graph(false, |g, h, _, x, _| {
        let hh = bin(g, BinaryOp::Mul, h, h);
        bin(g, BinaryOp::Add, hh, x)
    });
    check("square", sq, false);
    // tanh(h) is not affine.
    let rnn = scan_graph(false, |g, h, _, x, _| {
        let s = bin(g, BinaryOp::Add, h, x);
        g.activation(Activation::Tanh, s, sh(&[D]))
    });
    check("tanh_rnn", rnn, false);
}
