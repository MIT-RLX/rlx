// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `pool2d` forward, validated against hand-computed references.
//! Run: `cargo test -p rlx-tensor --features eval`.
#![cfg(feature = "eval")]

use rlx_ir::op::ReduceOp;
use rlx_tensor::{Dim, Tensor};

fn approx(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "length mismatch: {a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-5, "{a:?} != {b:?}");
    }
}

fn dims(t: &Tensor) -> Vec<usize> {
    t.shape()
        .dims()
        .iter()
        .map(|d| match d {
            Dim::Static(n) => *n,
            Dim::Dynamic(_) => panic!("unexpected dynamic dim"),
        })
        .collect()
}

#[test]
fn max_pool_over_a_window_of_three() {
    // [1,1,1,7] with k=3 s=2 -> 3 outputs: max(1,3,2), max(2,5,4), max(4,0,6)
    let x = Tensor::from_vec(vec![1.0, 3.0, 2.0, 5.0, 4.0, 0.0, 6.0], [1, 1, 1, 7]);
    let out = x.max_pool2d([1, 3], [1, 2], [0, 0]);
    assert_eq!(dims(&out), vec![1, 1, 1, 3]);
    approx(&out.to_vec(), &[3.0, 5.0, 6.0]);
}

#[test]
fn max_pool_is_per_channel() {
    // Two channels must not mix: channel 1 is channel 0 negated.
    let x = Tensor::from_vec(
        vec![1.0, 4.0, 2.0, 3.0, -1.0, -4.0, -2.0, -3.0],
        [1, 2, 1, 4],
    );
    let out = x.max_pool2d([1, 2], [1, 2], [0, 0]);
    assert_eq!(dims(&out), vec![1, 2, 1, 2]);
    approx(&out.to_vec(), &[4.0, 3.0, -1.0, -2.0]);
}

#[test]
fn pooling_reduces_the_length_axis_in_nchw() {
    // The length-in-H convention 1-D convolution uses across rlx: [N,C,L,1].
    // windows are max(1,9,2) and max(2,8,3)
    let x = Tensor::from_vec(vec![1.0, 9.0, 2.0, 8.0, 3.0], [1, 1, 5, 1]);
    let out = x.max_pool2d([3, 1], [2, 1], [0, 0]);
    assert_eq!(dims(&out), vec![1, 1, 2, 1]);
    approx(&out.to_vec(), &[9.0, 8.0]);
}

#[test]
fn average_pooling_takes_the_mean() {
    let x = Tensor::from_vec(vec![1.0, 3.0, 2.0, 6.0], [1, 1, 1, 4]);
    let out = x.avg_pool2d([1, 2], [1, 2], [0, 0]);
    approx(&out.to_vec(), &[2.0, 4.0]);
}

#[test]
fn pool2d_selects_the_reduction() {
    let x = Tensor::from_vec(vec![1.0, 3.0, 2.0, 6.0], [1, 1, 1, 4]);
    approx(
        &x.pool2d(ReduceOp::Max, [1, 2], [1, 2], [0, 0]).to_vec(),
        &[3.0, 6.0],
    );
    approx(
        &x.pool2d(ReduceOp::Mean, [1, 2], [1, 2], [0, 0]).to_vec(),
        &[2.0, 4.0],
    );
}

#[test]
fn overlapping_windows_keep_every_maximum() {
    // k=3 s=1 sees every triple, so a lone peak appears in three outputs.
    let x = Tensor::from_vec(vec![0.0, 0.0, 7.0, 0.0, 0.0], [1, 1, 1, 5]);
    let out = x.max_pool2d([1, 3], [1, 1], [0, 0]);
    assert_eq!(dims(&out), vec![1, 1, 1, 3]);
    approx(&out.to_vec(), &[7.0, 7.0, 7.0]);
}

/// Backprop through pooling — the property QAT depends on, since a network with
/// a pool in it is otherwise untrainable.
#[cfg(feature = "grad")]
mod grad {
    use super::{Tensor, approx};

    #[test]
    fn max_pool_routes_gradient_to_the_winner_only() {
        // loss = sum(maxpool(x)). Each window contributes its maximum, so the
        // gradient is 1 at the argmax of each window and 0 everywhere else.
        let x = Tensor::from_vec(vec![1.0, 3.0, 2.0, 5.0, 4.0, 0.0], [1, 1, 1, 6]);
        let loss = x
            .max_pool2d([1, 2], [1, 2], [0, 0])
            .sum([0, 1, 2, 3], false);
        let g = loss.grad(&[&x]);
        // windows (1,3) (2,5) (4,0) -> winners at indices 1, 3, 4
        approx(&g[0].to_vec(), &[0.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn average_pool_splits_gradient_evenly() {
        let x = Tensor::from_vec(vec![1.0, 3.0, 2.0, 6.0], [1, 1, 1, 4]);
        let loss = x
            .avg_pool2d([1, 2], [1, 2], [0, 0])
            .sum([0, 1, 2, 3], false);
        let g = loss.grad(&[&x]);
        approx(&g[0].to_vec(), &[0.5, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn overlapping_windows_accumulate_gradient() {
        // k=3 s=1 over a lone peak: it wins all three windows, so its gradient
        // accumulates to 3.
        let x = Tensor::from_vec(vec![0.0, 0.0, 7.0, 0.0, 0.0], [1, 1, 1, 5]);
        let loss = x
            .max_pool2d([1, 3], [1, 1], [0, 0])
            .sum([0, 1, 2, 3], false);
        let g = loss.grad(&[&x]);
        approx(&g[0].to_vec(), &[0.0, 0.0, 3.0, 0.0, 0.0]);
    }
}
