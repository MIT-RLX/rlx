// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The CPU backend's binary-op fast paths give each element exactly the
//! arithmetic of a plain per-element loop: same-shape Add/Mul (NEON, split
//! across threads for a large tensor) and a single-value operand on either
//! side (a held scalar instead of the broadcast index walk), at sizes on both
//! sides of the parallel threshold and with a ragged tail.

use rlx_ir::op::{BinaryOp, Op};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn run(op: BinaryOp, a: &[f32], b: &[f32]) -> Vec<f32> {
    let f = DType::F32;
    let n = a.len().max(b.len());
    let mut g = Graph::new("binary");
    let x = g.input("a", Shape::new(&[a.len()], f));
    let y = g.input("b", Shape::new(&[b.len()], f));
    let out = g.add_node(Op::Binary(op), vec![x, y], Shape::new(&[n], f));
    g.set_outputs(vec![out]);
    Session::new(Device::Cpu)
        .compile(g)
        .run(&[("a", a), ("b", b)])
        .remove(0)
}

fn want(op: BinaryOp, a: f32, b: f32) -> f32 {
    match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div => a / b,
        BinaryOp::Max => a.max(b),
        BinaryOp::Min => a.min(b),
        _ => unreachable!(),
    }
}

#[test]
fn each_element_gets_the_plain_arithmetic() {
    let ops = [
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Mul,
        BinaryOp::Div,
        BinaryOp::Max,
        BinaryOp::Min,
    ];
    for n in [7usize, 1000, 2_375_683] {
        let a: Vec<f32> = (0..n)
            .map(|i| ((i * 2654435761) % 10007) as f32 * 1.37e-3 - 6.1)
            .collect();
        let b: Vec<f32> = (0..n)
            .map(|i| ((i * 40503) % 9973) as f32 * -2.9e-4 + 1.3)
            .collect();
        for op in ops {
            let same = run(op, &a, &b);
            let r_scalar = run(op, &a, &b[..1]);
            let l_scalar = run(op, &a[..1], &b);
            for i in 0..n {
                assert_eq!(
                    same[i].to_bits(),
                    want(op, a[i], b[i]).to_bits(),
                    "{op:?} n {n} same-shape at {i}"
                );
                assert_eq!(
                    r_scalar[i].to_bits(),
                    want(op, a[i], b[0]).to_bits(),
                    "{op:?} n {n} rhs scalar at {i}"
                );
                assert_eq!(
                    l_scalar[i].to_bits(),
                    want(op, a[0], b[i]).to_bits(),
                    "{op:?} n {n} lhs scalar at {i}"
                );
            }
        }
    }
}
