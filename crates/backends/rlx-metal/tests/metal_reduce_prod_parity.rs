// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::Reduce { Prod }` on Metal vs CPU. Product reduction previously fell off
//! the MPSGraph fast path (`_ => return None`); it now lowers to
//! `reductionProductWithTensor`. Result must match the CPU reference.

#![cfg(target_os = "macos")]

use rlx_ir::op::ReduceOp;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

fn build_prod(outer: usize, reduced: usize, inner: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("reduce_prod");
    let x = g.input("x", Shape::new(&[outer, reduced, inner], f));
    let y = g.add_node(
        Op::Reduce {
            op: ReduceOp::Prod,
            axes: vec![1],
            keep_dim: false,
        },
        vec![x],
        Shape::new(&[outer, inner], f),
    );
    g.set_outputs(vec![y]);
    g
}

#[test]
fn metal_reduce_prod_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let (outer, reduced, inner) = (3usize, 5usize, 4usize);
    // Values near 1 so the product over `reduced` stays well-conditioned.
    let x: Vec<f32> = (0..outer * reduced * inner)
        .map(|i| 0.7 + (((i * 37 + 3) % 11) as f32) * 0.06)
        .collect();

    let g = build_prod(outer, reduced, inner);
    let mut m = Session::new(Device::Metal).compile(g.clone());
    let metal = m.run(&[("x", &x)]).remove(0);
    let mut c = Session::new(Device::Cpu).compile(g);
    let cpu = c.run(&[("x", &x)]).remove(0);

    assert_eq!(metal.len(), outer * inner);
    for (j, (a, b)) in metal.iter().zip(&cpu).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "reduce_prod[{j}]: metal {a} vs cpu {b}"
        );
    }
}
