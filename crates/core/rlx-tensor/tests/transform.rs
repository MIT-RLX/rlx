// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Composable function transforms (grad / vmap and their composition). Run:
//! `cargo test -p rlx-tensor --features transforms,eval`.
#![cfg(all(feature = "transforms", feature = "eval"))]

use rlx_tensor::{Func, shape};

fn approx(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "length mismatch: {a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-4, "{a:?} != {b:?}");
    }
}

/// f(x) = sum(x * x)
fn sum_sq() -> Func {
    Func::new("sum_sq", |s| {
        let x = s.input("x", shape![3]);
        (&x * &x).sum([0], false)
    })
}

#[test]
fn func_runs_forward() {
    let out = sum_sq().run(&[("x", &[1.0, 2.0, 3.0])]);
    approx(&out[0], &[14.0]); // 1 + 4 + 9
}

#[test]
fn grad_transform() {
    // d/dx sum(x*x) = 2x
    let df = sum_sq().grad(&["x"]);
    let out = df.run(&[("x", &[1.0, 2.0, 3.0])]);
    approx(&out[0], &[2.0, 4.0, 6.0]);
}

#[test]
fn vmap_transform() {
    // batch sum(x*x) over 2 rows
    let batched = sum_sq().vmap(&["x"], 2);
    let out = batched.run(&[("x", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]);
    approx(&out[0], &[14.0, 77.0]); // [1+4+9, 16+25+36]
}

#[test]
fn vmap_of_grad_composes() {
    // vmap(grad(f)) — batched gradient, one fused graph. d/dx = 2x per row.
    let batched_grad = sum_sq().grad(&["x"]).vmap(&["x"], 2);
    let out = batched_grad.run(&[("x", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]);
    approx(&out[0], &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn param_binding_forward() {
    // loss(x; w) = sum(x * w), weights bound once.
    let f = Func::new("lin", |s| {
        let x = s.input("x", shape![3]);
        let w = s.param("w", shape![3]);
        (&x * &w).sum([0], false)
    })
    .with_param("w", vec![2.0, 2.0, 2.0]);
    // sum([1,2,3] * [2,2,2]) = 12
    approx(&f.run(&[("x", &[1.0, 2.0, 3.0])])[0], &[12.0]);
}

#[test]
fn grad_wrt_param_uses_bound_value() {
    // loss(w) = sum(w * w)  ->  d/dw = 2w; gradient depends on the weight value.
    let f = Func::new("sqw", |s| {
        let w = s.param("w", shape![3]);
        (&w * &w).sum([0], false)
    })
    .with_param("w", vec![1.0, 2.0, 3.0]);
    let g = f.grad(&["w"]);
    // no inputs — the bound param flows through the transform.
    approx(&g.run(&[])[0], &[2.0, 4.0, 6.0]);
}

#[test]
fn jit_composes_and_runs() {
    // jit after the transform chain; the handle runs without recompiling.
    let f = sum_sq().grad(&["x"]).jit();
    approx(&f.run(&[("x", &[1.0, 2.0, 3.0])])[0], &[2.0, 4.0, 6.0]);
    approx(&f.run(&[("x", &[5.0, 6.0, 7.0])])[0], &[10.0, 12.0, 14.0]);
    // A clone shares the same compiled artifact.
    let g = f.clone();
    approx(&g.run(&[("x", &[0.0, 1.0, 2.0])])[0], &[0.0, 2.0, 4.0]);
}
