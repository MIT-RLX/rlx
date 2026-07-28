// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Capstone: fit a real linear model with gradient descent. Exercises the
//! whole stack — inputs + multiple params, matmul, broadcast bias, MSE loss,
//! grad w.r.t. several params, optimizer convergence. Run:
//! `cargo test -p rlx-tensor --features optim`.
#![cfg(feature = "optim")]

use rlx_tensor::{Func, Sgd, shape};

#[test]
fn linear_regression_learns_weights() {
    // Ground truth: y = X · w* + b*,  w* = [1.5, -2.0], b* = 0.5
    let xs: &[f32] = &[1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 3.0, 2.0]; // [4,2]
    let ys: &[f32] = &[0.0, 1.5, -2.0, 1.0]; // X·w* + b*  [4,1]

    // loss(x,y; w,b) = mean( (X @ w + b - y)^2 )
    let mut model = Func::new("linreg", |s| {
        let x = s.input("x", shape![4, 2]);
        let y = s.input("y", shape![4, 1]);
        let w = s.param("w", shape![2, 1]);
        let b = s.param("b", shape![1]);
        let diff = &(&x.matmul(&w) + &b) - &y;
        (&diff * &diff).mean([0, 1], false)
    })
    .with_param("w", vec![0.0, 0.0])
    .with_param("b", vec![0.0]);

    let mut opt = Sgd::new(0.1);
    let feed: &[(&str, &[f32])] = &[("x", xs), ("y", ys)];

    let mut first = f32::NAN;
    let mut last = f32::NAN;
    for i in 0..3000 {
        let (next, loss) = model.train_step(&mut opt, &["w", "b"], feed);
        model = next;
        if i == 0 {
            first = loss[0];
        }
        last = loss[0];
    }

    assert!(first > 1.0, "expected large initial loss, got {first}");
    assert!(last < 1e-4, "loss should converge to ~0, got {last}");

    // Recovered the ground-truth weights.
    let w = model.param_binding("w").unwrap();
    let b = model.param_binding("b").unwrap();
    assert!((w[0] - 1.5).abs() < 0.02, "w0 = {} (want 1.5)", w[0]);
    assert!((w[1] + 2.0).abs() < 0.02, "w1 = {} (want -2.0)", w[1]);
    assert!((b[0] - 0.5).abs() < 0.02, "b = {} (want 0.5)", b[0]);
}
