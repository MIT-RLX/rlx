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

//! Verify that the runtime's `LowerControlFlow` pre-pass lets the CPU
//! backend execute graphs containing `Op::If` and `Op::While` end-to-end
//! — neither op is in `CPU_SUPPORTED_OPS`, so without the rewrite the
//! legalize check would reject them.
//!
//! `Op::If` rewrites to `Where(predicate, then_inlined, else_inlined)`
//! and `Op::While` to a chain of body replicas up to `max_iterations`.
//! Both branches / all iterations always execute in the rewritten graph
//! (the trade-off documented in `rlx_opt::control_flow`).

#![cfg(feature = "cpu")]

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

#[test]
fn cpu_runs_op_if_via_lower_control_flow() {
    // then = relu(x), else = sigmoid(x). Predicate=1.0 selects then.
    let s = Shape::new(&[4], DType::F32);

    let mut then_g = Graph::new("then");
    let ti = then_g.input("c", s.clone());
    let to = then_g.activation(Activation::Relu, ti, s.clone());
    then_g.set_outputs(vec![to]);

    let mut else_g = Graph::new("else");
    let ei = else_g.input("c", s.clone());
    let eo = else_g.activation(Activation::Sigmoid, ei, s.clone());
    else_g.set_outputs(vec![eo]);

    let mut g = Graph::new("if_test");
    let x = g.input("x", s.clone());
    let pred = g.input("pred", Shape::new(&[1], DType::F32));
    let y = g.add_node(
        Op::If {
            then_branch: Box::new(then_g),
            else_branch: Box::new(else_g),
        },
        vec![pred, x],
        s.clone(),
    );
    g.set_outputs(vec![y]);

    let session = Session::new(Device::Cpu);
    let mut compiled = session.compile(g);
    let xs: Vec<f32> = vec![-1.0, 0.0, 1.0, 2.0];
    let pred_true = vec![1.0f32];
    let outs = compiled.run(&[("x", &xs), ("pred", &pred_true)]);
    // pred=1.0 → take Relu(x) = [0, 0, 1, 2].
    assert_eq!(
        outs[0],
        vec![0.0, 0.0, 1.0, 2.0],
        "Op::If should select then-branch when predicate is non-zero"
    );
}

#[test]
fn cpu_runs_op_while_via_lower_control_flow() {
    // body = c * c. Loop-carried single value, max_iterations=3.
    // Starting from x = [2, 3], after 3 iterations the value is
    // x ^ (2^3) = x^8.
    let s = Shape::new(&[2], DType::F32);

    let mut body_g = Graph::new("body");
    let bi = body_g.input("c", s.clone());
    let bo = body_g.binary(BinaryOp::Mul, bi, bi, s.clone());
    body_g.set_outputs(vec![bo]);

    let mut cond_g = Graph::new("cond");
    let ci = cond_g.input("c", s.clone());
    cond_g.set_outputs(vec![ci]);

    let mut g = Graph::new("while_test");
    let x = g.input("x", s.clone());
    let y = g.add_node(
        Op::While {
            cond: Box::new(cond_g),
            body: Box::new(body_g),
            max_iterations: Some(3),
        },
        vec![x],
        s.clone(),
    );
    g.set_outputs(vec![y]);

    let session = Session::new(Device::Cpu);
    let mut compiled = session.compile(g);
    let xs: Vec<f32> = vec![2.0, 3.0];
    let outs = compiled.run(&[("x", &xs)]);
    let want = vec![2.0_f32.powi(8), 3.0_f32.powi(8)]; // [256, 6561]
    assert!(
        outs[0].iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-3),
        "Op::While unroll should square 3 times: got {:?} want {want:?}",
        outs[0]
    );
}
