// SPDX-License-Identifier: MIT OR Apache-2.0
//! Param-invariant subgraph hoisting: split a graph into a weight-only
//! `prepare` graph (run once) + a `main` graph that reads the prepared tensors
//! as bound inputs. Validates the split + manual prepare→bind→main orchestration
//! matches the unsplit run and is stable across forwards.

#![cfg(feature = "cpu")]

use rlx_compile::split_param_invariant;
use rlx_ir::op::BinaryOp;
use rlx_ir::*;
use rlx_runtime::{Device, Session};

/// `y = x @ (w · scale)` — `w·scale` is param-invariant (both are weights), so it
/// should hoist into `prepare`; the matmul stays in `main`.
fn build() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("hoist");
    let x = g.input("x", Shape::new(&[2, 4], f));
    let w = g.param("w", Shape::new(&[4, 3], f));
    let scale = g.param("scale", Shape::new(&[4, 3], f));
    let w_scaled = g.binary(BinaryOp::Mul, w, scale, Shape::new(&[4, 3], f));
    let y = g.matmul(x, w_scaled, Shape::new(&[2, 3], f));
    g.set_outputs(vec![y]);
    g
}

fn feeds() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1 - 0.3).collect();
    let w: Vec<f32> = (0..12).map(|i| (i as f32) * 0.05 - 0.2).collect();
    let scale: Vec<f32> = (0..12).map(|i| 0.5 + (i % 4) as f32 * 0.25).collect();
    (x, w, scale)
}

#[test]
fn split_structure() {
    let split = split_param_invariant(&build()).expect("Mul(w,scale) is hoistable");
    assert_eq!(split.boundary.len(), 1);
    // prepare: the Mul, weights, no matmul.
    assert!(
        split
            .prepare
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::Binary(BinaryOp::Mul)))
    );
    assert!(
        !split
            .prepare
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::MatMul))
    );
    // main: the matmul, the boundary Input, NO Mul (hoisted away).
    assert!(
        split
            .main
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::MatMul))
    );
    assert!(
        !split
            .main
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::Binary(BinaryOp::Mul)))
    );
    assert!(split.prepare_params.contains("w") && split.prepare_params.contains("scale"));
    assert!(split.main_params.is_empty(), "weights hoisted out of main");
}

#[test]
fn split_run_matches_unsplit_and_is_stable() {
    let (x, w, scale) = feeds();

    // Reference: the whole graph.
    let mut refc = Session::new(Device::Cpu).compile(build());
    refc.set_param("w", &w);
    refc.set_param("scale", &scale);
    let y_ref = refc.run(&[("x", x.as_slice())]).pop().unwrap();

    // Split + manual orchestration: run prepare once, bind, run main.
    let split = split_param_invariant(&build()).unwrap();
    let mut prep = Session::new(Device::Cpu).compile(split.prepare);
    prep.set_param("w", &w);
    prep.set_param("scale", &scale);
    let prepared = prep.run(&[]).pop().unwrap(); // = w·scale

    let mut main = Session::new(Device::Cpu).compile(split.main);
    assert!(
        main.bind_handle(&split.boundary[0], &prepared),
        "backend must support persistent bound handles"
    );
    let y_split = main.run(&[("x", x.as_slice())]).pop().unwrap();

    let maxd = y_ref
        .iter()
        .zip(&y_split)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(maxd <= 1e-5, "split vs unsplit diff {maxd}");

    // The bound boundary persists — a second forward (without re-binding) is
    // identical, i.e. the prepared tensor survives across runs.
    let y_split2 = main.run(&[("x", x.as_slice())]).pop().unwrap();
    assert_eq!(
        y_split, y_split2,
        "bound prepared tensor must persist across runs"
    );
}

/// End-to-end: `cache_param_invariant` makes `Session::compile_with` transparently
/// stage prepare→main. Same API as an ordinary compiled graph; identical results
/// and stable across many forwards. Weight re-set re-runs prepare.
#[test]
fn automatic_hoisting_matches_and_is_stable() {
    use rlx_runtime::CompileOptions;
    let (x, w, scale) = feeds();

    let mut refc = Session::new(Device::Cpu).compile(build());
    refc.set_param("w", &w);
    refc.set_param("scale", &scale);
    let y_ref = refc.run(&[("x", x.as_slice())]).pop().unwrap();

    let mut opts = CompileOptions::new();
    opts.cache_param_invariant = true;
    let mut c = Session::new(Device::Cpu).compile_with(build(), &opts);
    // Same set_param/run API — routing to prepare/main is transparent.
    c.set_param("w", &w);
    c.set_param("scale", &scale);
    let y = c.run(&[("x", x.as_slice())]).pop().unwrap();

    let maxd = y_ref
        .iter()
        .zip(&y)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(maxd <= 1e-5, "hoisted vs plain diff {maxd}");

    // Many forwards: prepare runs once, results stable.
    for _ in 0..5 {
        let yn = c.run(&[("x", x.as_slice())]).pop().unwrap();
        assert_eq!(y, yn);
    }

    // Re-set a weight → prepare re-runs → output reflects the new weight.
    let scale2: Vec<f32> = scale.iter().map(|v| v * 2.0).collect();
    c.set_param("scale", &scale2);
    let y_new = c.run(&[("x", x.as_slice())]).pop().unwrap();
    assert!(
        y.iter().zip(&y_new).any(|(a, b)| (a - b).abs() > 1e-4),
        "re-setting a weight must re-run prepare and change the output"
    );

    let mut refc2 = Session::new(Device::Cpu).compile(build());
    refc2.set_param("w", &w);
    refc2.set_param("scale", &scale2);
    let y_ref2 = refc2.run(&[("x", x.as_slice())]).pop().unwrap();
    let maxd2 = y_ref2
        .iter()
        .zip(&y_new)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        maxd2 <= 1e-5,
        "after weight update, hoisted vs plain diff {maxd2}"
    );
}
