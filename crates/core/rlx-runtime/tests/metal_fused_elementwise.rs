// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Pure elementwise programs run as one generated Metal kernel per region
//! (`rlx_metal::fused_ew`): against the CPU, on new data, with fan-out inside
//! a region, broadcast operands and a result read from the middle of one.
#![cfg(feature = "metal")]

mod common;

use rlx_ir::op::{Activation, BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::{Device, Session};

const N: usize = 37;
const S: usize = 53;

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

fn op(g: &mut Graph, o: BinaryOp, a: NodeId, b: NodeId) -> NodeId {
    g.binary(o, a, b, f32s(&[N, S]))
}

fn act(g: &mut Graph, a: Activation, x: NodeId) -> NodeId {
    g.activation(a, x, f32s(&[N, S]))
}

/// A DAG on `[N,1]` and `[1,S]`: each value read more than once. Returns the
/// graph's parts: a mid-way value and the final one.
fn dag(g: &mut Graph) -> (NodeId, NodeId) {
    use BinaryOp::*;
    let x = g.input("x", f32s(&[N, 1]));
    let y = g.input("y", f32s(&[1, S]));
    let data = 1.5f32.to_le_bytes().to_vec();
    let c = g.add_node(Op::Constant { data }, vec![], f32s(&[1]));
    let (a, b) = (op(g, Mul, x, y), op(g, Add, x, y));
    let (d, e) = (op(g, Div, a, b), op(g, Sub, b, a));
    let f = op(g, Mul, d, e);
    let h = op(g, Add, f, a);
    let i = op(g, Mul, h, c);
    let j = act(g, Activation::Abs, i);
    let k = act(g, Activation::Sqrt, j);
    let l = op(g, Max, k, d);
    let m = act(g, Activation::Recip, l);
    let n = act(g, Activation::Neg, m);
    let o = op(g, Add, n, e);
    (h, o)
}

fn data(n: usize, k: f32) -> Vec<f32> {
    (0..n)
        .map(|i| 0.75 + k * (i as f32 * 0.37).sin().abs())
        .collect()
}

fn close(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            x.is_finite() && (x - y).abs() <= 1e-5 * y.abs().max(1.0),
            "{what} [{i}]: {x} against {y}"
        );
    }
}

fn parity(build: impl Fn() -> Graph, what: &str) {
    if common::skip_unless(Device::Metal) {
        return;
    }
    let _gpu = common::GpuTestGuard::acquire(Device::Metal);
    let (mut cpu, mut gpu) = (
        Session::new(Device::Cpu).compile(build()),
        Session::new(Device::Metal).compile(build()),
    );
    for k in [1.0f32, 3.0, 0.25] {
        let (x, y) = (data(N, k), data(S, 2.0 * k));
        let ins: [(&str, &[f32]); 2] = [("x", &x), ("y", &y)];
        let (want, got) = (cpu.run(&ins), gpu.run(&ins));
        assert_eq!(want.len(), got.len());
        for (w, g) in want.iter().zip(&got) {
            close(g, w, what);
        }
    }
}

#[test]
fn a_dag_with_fan_out_and_broadcasts() {
    parity(
        || {
            let mut g = Graph::new("dag");
            let (_, o) = dag(&mut g);
            g.set_outputs(vec![o]);
            g
        },
        "one result",
    );
}

#[test]
fn a_result_from_the_middle_of_a_region() {
    parity(
        || {
            let mut g = Graph::new("two");
            let (h, o) = dag(&mut g);
            g.set_outputs(vec![o, h]);
            g
        },
        "two results",
    );
}

#[test]
fn a_graph_with_other_ops_is_left_alone() {
    parity(
        || {
            let mut g = Graph::new("mixed");
            let (_, o) = dag(&mut g);
            let r = g.reduce(o, ReduceOp::Sum, vec![1], false, f32s(&[N]));
            g.set_outputs(vec![r]);
            g
        },
        "with a reduction",
    );
}
