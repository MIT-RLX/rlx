// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A `Scan` body gathers with an I32 index as any graph does. The CPU
//! compiled scan bodies without the integer-storage preparation the outer
//! graph gets, so an I32 index was read as f32 bits — about 3e-45 — and
//! every gather took row 0. TEN-VAD's exported pitch track (its Viterbi
//! backtrack gathers by state) found it.
#![cfg(feature = "cpu")]

use rlx_ir::{DType, Graph, GraphExt, Op, Shape};

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

#[test]
fn a_scan_body_gathers_at_its_integer_index() {
    // Each step: element `2 + t` of `x_t` (an index cast from the carry, and
    // a constant one), through a narrow view, as the carry.
    let mut body = Graph::new("pick");
    let carry = body.input("carry", f32s(&[2]));
    let xt = body.input("x", f32s(&[10]));
    let view = body.narrow_(xt, 0, 5, 5);
    let at = body.narrow_(carry, 0, 0, 1);
    let at = body.add_node(
        Op::Cast { to: DType::I32 },
        vec![at],
        Shape::new(&[1], DType::I32),
    );
    let by_cast = body.gather(view, at, 0, f32s(&[1]));
    let two = body.add_node(
        Op::Constant {
            data: 2i32.to_le_bytes().to_vec(),
        },
        vec![],
        Shape::new(&[1], DType::I32),
    );
    let by_const = body.gather(view, two, 0, f32s(&[1]));
    let next = body.concat_(vec![by_cast, by_const], 0);
    body.set_outputs(vec![next]);

    let mut g = Graph::new("outer");
    let init = g.input("init", f32s(&[2]));
    let xs = g.input("xs", f32s(&[3, 10]));
    let traj = g.add_node(
        Op::Scan {
            body: Box::new(body),
            length: 3,
            save_trajectory: true,
            num_bcast: 0,
            num_xs: 1,
            num_checkpoints: 0,
        },
        vec![init, xs],
        f32s(&[3, 2]),
    );
    g.set_outputs(vec![traj]);
    let mut c = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(g);
    let xs: Vec<f32> = (0..30).map(|v| v as f32 * 0.25).collect();
    // Step t picks element 5 + i of row t: i = the carry's first value
    // (1, then what it picked), and i = 2.
    let out = c.run(&[("init", &[1.0, 0.0]), ("xs", &xs)]).remove(0);
    let row = |t: usize, i: usize| xs[t * 10 + 5 + i];
    let first = row(0, 1);
    let second = row(1, first as usize);
    let third = row(2, second as usize);
    assert_eq!(
        out,
        [first, row(0, 2), second, row(1, 2), third, row(2, 2)],
        "{out:?}"
    );
}
