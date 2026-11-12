// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `ModelFlow::gather_rows` and the last-token gather: the rows picked per
//! batch item, and the input gradient reaching every batch item (CPU).
//!
//! For `loss = sum(gathered)` the gradient at `x[b, s, :]` is the number of
//! times position `s` was picked for item `b`, so repeats and batch > 1 are
//! both checked exactly. The last-token gather used to differentiate to zero
//! for every batch item but the first.

use rlx_flow::{MapWeights, ModelFlow};
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};

const B: usize = 2;
const S: usize = 5;
const C: usize = 3;

fn x0() -> Vec<f32> {
    (0..B * S * C).map(|i| i as f32 * 0.5 - 3.0).collect()
}

fn input_id(g: &Graph, name: &str) -> NodeId {
    g.nodes()
        .iter()
        .find(|n| matches!(&n.op, Op::Input { name: n2 } if n2 == name))
        .map(|n| n.id)
        .unwrap_or_else(|| panic!("no input {name}"))
}

/// Forward rows and `d loss / d x` for `loss = sum(output)`.
fn run(flow: ModelFlow, extra: (&str, Vec<f32>)) -> (Vec<f32>, Vec<f32>) {
    let built = flow.build(&mut MapWeights::default()).expect("flow build");
    let g = built.into_graph().expect("into_graph");
    let fwd = Session::new(Device::Cpu)
        .compile(g.clone())
        .run(&[("x", &x0()), (extra.0, &extra.1)])
        .swap_remove(0);
    let mut g = g;
    let out = g.outputs[0];
    let loss = g.sum(out, vec![0, 1, 2], false);
    g.set_outputs(vec![loss]);
    let x = input_id(&g, "x");
    let bwd = grad_with_loss(&g, &[x]);
    let r = Session::new(Device::Cpu).compile(bwd).run(&[
        ("x", &x0()),
        (extra.0, &extra.1),
        ("d_output", &[1.0]),
    ]);
    (fwd, r[1].clone())
}

fn expect_rows(fwd: &[f32], rows: &[[usize; 3]], k: usize) {
    let x = x0();
    for (b, r) in rows.iter().enumerate() {
        for (j, &s) in r.iter().take(k).enumerate() {
            let got = &fwd[(b * k + j) * C..(b * k + j + 1) * C];
            let want = &x[(b * S + s) * C..(b * S + s + 1) * C];
            assert_eq!(got, want, "item {b}, row {j} (position {s})");
        }
    }
}

fn expect_counts(dx: &[f32], rows: &[[usize; 3]], k: usize) {
    for (b, r) in rows.iter().enumerate() {
        for s in 0..S {
            let n = r.iter().take(k).filter(|&&p| p == s).count() as f32;
            for c in 0..C {
                assert_eq!(dx[(b * S + s) * C + c], n, "d x[{b}, {s}, {c}]");
            }
        }
    }
}

#[test]
fn gather_rows_picks_rows_and_routes_gradients_per_item() {
    let rows = [[0, 4, 4], [3, 1, 2]];
    let idx: Vec<f32> = rows.iter().flatten().map(|&p| p as f32).collect();
    let flow = ModelFlow::new("gather_rows")
        .input("x", Shape::new(&[B, S, C], DType::F32))
        .gather_rows("rows", B, 3);
    let (fwd, dx) = run(flow, ("rows", idx));
    assert_eq!(fwd.len(), B * 3 * C);
    expect_rows(&fwd, &rows, 3);
    expect_counts(&dx, &rows, 3);
}

#[test]
fn last_token_gather_routes_gradients_per_item() {
    let last = [[4, 0, 0], [1, 0, 0]];
    let flow = ModelFlow::new("last_token")
        .input("x", Shape::new(&[B, S, C], DType::F32))
        .input("last_token_idx", Shape::new(&[B], DType::F32))
        .gather_last_token_dynamic(B);
    let (fwd, dx) = run(flow, ("last_token_idx", vec![4.0, 1.0]));
    assert_eq!(fwd.len(), B * C);
    expect_rows(&fwd, &last, 1);
    expect_counts(&dx, &last, 1);
}
