// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CoreML's CPU engine (BNNS) mis-executes `transpose → reduce → transpose`: it
//! segfaults, or silently returns wrong sums (reproduced with coremltools and no
//! rlx code, macOS 26.6.2). rlx-autodiff emits that chain for bias gradients, so
//! CLEF's training graphs crashed. `hybrid::hoist_transpose_out_of_reduce`
//! rewrites every `reduce(transpose(x))`; each case here is checked against
//! rlx-cpu. Before the rewrite `bias_grad_chain` crashed the test process
//! (SIGSEGV) and its [3,32,32] variant was off by 98.

use rlx_coreml::CoremlExecutable;
use rlx_ir::op::ReduceOp;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::{Device, Session};

const F: DType = DType::F32;

fn input(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 37 % 101) as f32) * 0.03 - 1.0)
        .collect()
}

fn check(g: Graph, n: usize) {
    let x = input(n);
    let mut cpu = Session::new(Device::Cpu).compile(g.clone());
    let want = cpu.run(&[("x", &x)]).remove(0);
    let mut exe = CoremlExecutable::compile(g);
    let got = exe.run(&[("x", &x)]).expect("coreml run").remove(0);
    assert_eq!(got.len(), want.len());
    let err = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
    assert!(err <= 1e-4 * scale, "max_err {err} (scale {scale})");
}

fn transpose(g: &mut Graph, x: NodeId, perm: &[usize]) -> NodeId {
    let d: Vec<usize> = g
        .node(x)
        .shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect();
    let o: Vec<usize> = perm.iter().map(|&p| d[p]).collect();
    g.add_node(
        Op::Transpose {
            perm: perm.to_vec(),
        },
        vec![x],
        Shape::new(&o, F),
    )
}

fn reduce(g: &mut Graph, x: NodeId, op: ReduceOp, axes: &[usize], keep: bool) -> NodeId {
    let d: Vec<usize> = g
        .node(x)
        .shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect();
    let o: Vec<usize> = (0..d.len())
        .filter_map(|i| {
            if axes.contains(&i) {
                keep.then_some(1)
            } else {
                Some(d[i])
            }
        })
        .collect();
    g.add_node(
        Op::Reduce {
            op,
            axes: axes.to_vec(),
            keep_dim: keep,
        },
        vec![x],
        Shape::new(&o, F),
    )
}

/// The chain rlx-autodiff emits for a `[a,b,c] → [c]` bias gradient.
fn bias_grad_chain(a: usize, b: usize, c: usize) {
    let mut g = Graph::new("bias_grad");
    let x = g.input("x", Shape::new(&[a, b, c], F));
    let t1 = transpose(&mut g, x, &[1, 2, 0]);
    let r1 = reduce(&mut g, t1, ReduceOp::Sum, &[2], false);
    let t2 = transpose(&mut g, r1, &[1, 0]);
    let r2 = reduce(&mut g, t2, ReduceOp::Sum, &[1], false);
    g.set_outputs(vec![r2]);
    check(g, a * b * c);
}

#[test]
fn bias_grad_chain_crashing_shape() {
    bias_grad_chain(3, 64, 64);
}

#[test]
fn bias_grad_chain_wrong_value_shape() {
    bias_grad_chain(3, 32, 32);
}

/// Surviving axes come out of the permutation reordered, so the rewrite must
/// emit a transpose AFTER the reduce, with and without `keep_dim`.
#[test]
fn reordering_permutation_keep_and_drop() {
    for keep in [false, true] {
        for op in [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Max] {
            let mut g = Graph::new("reorder");
            let x = g.input("x", Shape::new(&[4, 5, 6, 7], F));
            let t = transpose(&mut g, x, &[3, 1, 0, 2]); // [7,5,4,6]
            let r = reduce(&mut g, t, op, &[1], keep); // surviving x-axes 3,0,2: not ascending
            g.set_outputs(vec![r]);
            check(g, 4 * 5 * 6 * 7);
        }
    }
}

/// A transpose with a reader besides the reduce must survive the rewrite.
#[test]
fn shared_transpose_is_kept() {
    let mut g = Graph::new("shared");
    let x = g.input("x", Shape::new(&[3, 8, 5], F));
    let t = transpose(&mut g, x, &[2, 0, 1]); // [5,3,8]
    let r = reduce(&mut g, t, ReduceOp::Sum, &[1, 2], false); // [5]
    let t2 = transpose(&mut g, t, &[1, 0, 2]); // [3,5,8], second reader of t
    let r2 = reduce(&mut g, t2, ReduceOp::Max, &[0], false); // [5,8]
    let flat = g.add_node(
        Op::Reshape {
            new_shape: vec![40],
        },
        vec![r2],
        Shape::new(&[40], F),
    );
    let cat = g.add_node(Op::Concat { axis: 0 }, vec![r, flat], Shape::new(&[45], F));
    g.set_outputs(vec![cat]);
    check(g, 3 * 8 * 5);
}
