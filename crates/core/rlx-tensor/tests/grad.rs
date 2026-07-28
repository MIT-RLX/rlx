// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reverse-mode autodiff against closed-form gradients. Run with:
//! `cargo test -p rlx-tensor --features grad,eval`.
#![cfg(all(feature = "grad", feature = "eval"))]

use rlx_tensor::Tensor;

fn approx(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "length mismatch: {a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-4, "{a:?} != {b:?}");
    }
}

#[test]
fn grad_of_product_sum() {
    // loss = sum(a * b)  ->  d/da = b, d/db = a.
    // a and b come from independent graphs: exercises cross-graph identity.
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], [3]);
    let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], [3]);
    let loss = (&a * &b).sum([0], false);
    let g = loss.grad(&[&a, &b]);
    approx(&g[0].to_vec(), &[4.0, 5.0, 6.0]); // d/da = b
    approx(&g[1].to_vec(), &[1.0, 2.0, 3.0]); // d/db = a
}

#[test]
fn grad_accumulates_for_reused_input() {
    // loss = sum(x * x)  ->  d/dx = 2x (gradient flows through both operands).
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], [3]);
    let loss = (&x * &x).sum([0], false);
    let g = loss.grad(&[&x]);
    approx(&g[0].to_vec(), &[2.0, 4.0, 6.0]);
}

#[test]
fn grad_through_relu() {
    // loss = sum(relu(x))  ->  d/dx = 1 where x > 0 else 0.
    let x = Tensor::from_vec(vec![-1.0, 2.0, -3.0, 4.0], [4]);
    let loss = x.relu().sum([0], false);
    let g = loss.grad(&[&x]);
    approx(&g[0].to_vec(), &[0.0, 1.0, 0.0, 1.0]);
}

#[test]
fn grad_of_scaled_sum() {
    // loss = sum(x * 3)  ->  d/dx = 3.
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], [3]);
    let loss = (&x * 3.0f32).sum([0], false);
    let g = loss.grad(&[&x]);
    approx(&g[0].to_vec(), &[3.0, 3.0, 3.0]);
}

#[test]
fn grad_linear_layer() {
    // loss = sum(x @ w),  x:[1,2], w:[2,2]
    // d/dw[i,j] = sum_b x[b,i]  -> each row of grad is column-sum of x.
    let x = Tensor::from_vec(vec![1.0, 2.0], [1, 2]);
    let w = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], [2, 2]);
    let loss = x.matmul(&w).sum([0, 1], false);
    let g = loss.grad(&[&w]);
    // dL/dw = x^T @ ones[1,2] = [[1,1],[2,2]]
    approx(&g[0].to_vec(), &[1.0, 1.0, 2.0, 2.0]);
}
