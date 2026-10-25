// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::While` runs as a loop at run time, not an unroll: with no
//! `max_iterations` (which an unroll cannot take), its count depending on
//! the data; with one, stopping there; and not at all when its condition
//! starts false. Each against the same loop on the host, on every device
//! this build has.

mod common;

use rlx_ir::op::{BinaryOp, CmpOp};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

fn scalar(g: &mut Graph, v: f32) -> rlx_ir::NodeId {
    g.add_node(
        Op::Constant {
            data: v.to_le_bytes().to_vec(),
        },
        vec![],
        f32s(&[1]),
    )
}

/// Collatz: carried `(steps, n)`; while `n > 1`: `n` halves when even, else
/// `3n + 1`; `steps + 1`. The value is the steps.
fn collatz(max_iterations: Option<usize>) -> Graph {
    let mut cond = Graph::new("cond");
    let _steps = cond.input("steps", f32s(&[1]));
    let n = cond.input("n", f32s(&[1]));
    let one = scalar(&mut cond, 1.0);
    let go = cond.add_node(
        Op::Compare(CmpOp::Gt),
        vec![n, one],
        Shape::new(&[1], DType::Bool),
    );
    cond.set_outputs(vec![go]);

    let mut body = Graph::new("body");
    let steps = body.input("steps", f32s(&[1]));
    let n = body.input("n", f32s(&[1]));
    let one = scalar(&mut body, 1.0);
    let two = scalar(&mut body, 2.0);
    let three = scalar(&mut body, 3.0);
    let half = body.binary(BinaryOp::Div, n, two, f32s(&[1]));
    let fl = body.add_node(
        Op::Activation(rlx_ir::op::Activation::Floor),
        vec![half],
        f32s(&[1]),
    );
    let even = body.add_node(
        Op::Compare(CmpOp::Eq),
        vec![half, fl],
        Shape::new(&[1], DType::Bool),
    );
    let even = body.add_node(Op::Cast { to: DType::F32 }, vec![even], f32s(&[1]));
    let tripled = body.binary(BinaryOp::Mul, n, three, f32s(&[1]));
    let odd = body.binary(BinaryOp::Add, tripled, one, f32s(&[1]));
    // even ? half : odd, as `odd + even · (half − odd)`.
    let gap = body.binary(BinaryOp::Sub, half, odd, f32s(&[1]));
    let pick = body.binary(BinaryOp::Mul, even, gap, f32s(&[1]));
    let next = body.binary(BinaryOp::Add, odd, pick, f32s(&[1]));
    let counted = body.binary(BinaryOp::Add, steps, one, f32s(&[1]));
    body.set_outputs(vec![counted, next]);

    let mut g = Graph::new("collatz");
    let n = g.input("n", f32s(&[1]));
    let zero = scalar(&mut g, 0.0);
    let w = g.add_node(
        Op::While {
            cond: Box::new(cond),
            body: Box::new(body),
            max_iterations,
        },
        vec![zero, n],
        f32s(&[1]),
    );
    g.set_outputs(vec![w]);
    g
}

fn on_host(mut n: u64, max: Option<usize>) -> f32 {
    let mut steps = 0usize;
    while n > 1 && max.is_none_or(|m| steps < m) {
        n = if n.is_multiple_of(2) {
            n / 2
        } else {
            3 * n + 1
        };
        steps += 1;
    }
    steps as f32
}

fn devices() -> Vec<Device> {
    let mut d = vec![Device::Cpu];
    for dev in [Device::Metal, Device::Cuda, Device::Rocm, Device::Gpu] {
        if rlx_runtime::is_available(dev) {
            d.push(dev);
        }
    }
    d
}

#[test]
fn a_while_loops_as_long_as_its_data_says() {
    for device in devices() {
        let _gpu = common::GpuTestGuard::acquire(device);
        for max in [None, Some(5), Some(1000)] {
            let mut c = Session::new(device).compile(collatz(max));
            for n in [1u64, 6, 7, 27, 97] {
                let got = c.run(&[("n", &[n as f32])]).remove(0);
                let want = on_host(n, max);
                assert_eq!(got, [want], "{device:?} n={n} max={max:?}");
            }
        }
    }
}
