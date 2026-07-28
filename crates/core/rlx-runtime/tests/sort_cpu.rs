// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::Sort` / `Op::ArgSort` on CPU — values + indices, ascending/descending,
//! along either axis.

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn run(x: &[f32], dims: &[usize], axis: usize, descending: bool, arg: bool) -> Vec<f32> {
    let mut g = Graph::new("sort");
    let xi = g.input("x", Shape::new(dims, DType::F32));
    let out = if arg {
        g.argsort(xi, axis, descending, Shape::new(dims, DType::F32))
    } else {
        g.sort(xi, axis, descending, Shape::new(dims, DType::F32))
    };
    g.set_outputs(vec![out]);
    Session::new(Device::Cpu)
        .compile(g)
        .run(&[("x", x)])
        .pop()
        .unwrap()
}

#[test]
fn sort_argsort_cpu() {
    // [[3, 1, 2], [0.5, 4, -1]]
    let x = vec![3.0, 1.0, 2.0, 0.5, 4.0, -1.0];

    assert_eq!(
        run(&x, &[2, 3], 1, false, false),
        vec![1.0, 2.0, 3.0, -1.0, 0.5, 4.0]
    );
    assert_eq!(
        run(&x, &[2, 3], 1, false, true),
        vec![1.0, 2.0, 0.0, 2.0, 0.0, 1.0]
    );
    assert_eq!(
        run(&x, &[2, 3], 1, true, false),
        vec![3.0, 2.0, 1.0, 4.0, 0.5, -1.0]
    );
    // Axis 0: cols [3,0.5]→[0.5,3], [1,4]→[1,4], [2,-1]→[-1,2].
    assert_eq!(
        run(&x, &[2, 3], 0, false, false),
        vec![0.5, 1.0, -1.0, 3.0, 4.0, 2.0]
    );
}
