// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! A program built from constants reuses arena slots on the CPU: every run
//! must still recompute the same answer, with the constants intact.

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

#[test]
fn a_constant_program_gives_the_same_answer_on_every_run() {
    let s = Shape::new(&[4096], DType::F32);
    let v: Vec<f32> = (0..4096).map(|i| 1.0 + i as f32 / 4096.0).collect();
    let mut g = Graph::new("reruns");
    let data = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let c = g.add_node(Op::Constant { data }, vec![], s.clone());
    let (mut a, mut b) = (c, c);
    for _ in 0..12 {
        let p = g.binary(BinaryOp::Mul, a, c, s.clone());
        let q = g.binary(BinaryOp::Add, b, a, s.clone());
        (a, b) = (g.binary(BinaryOp::Div, p, q, s.clone()), q);
    }
    let o = g.binary(BinaryOp::Add, a, b, s.clone());
    g.set_outputs(vec![o]);

    let mut want = vec![0f32; 4096];
    for i in 0..4096 {
        let (c, mut a, mut b) = (v[i], v[i], v[i]);
        for _ in 0..12 {
            let (p, q) = (a * c, b + a);
            (a, b) = (p / q, q);
        }
        want[i] = a + b;
    }
    let mut m = Session::new(Device::Cpu).compile(g);
    for run in 0..4 {
        assert_eq!(m.run(&[]).remove(0), want, "run {run}");
    }
}
