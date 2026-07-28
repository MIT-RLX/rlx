// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tier-3 array manipulation: split/chunk/tile/flip/roll/pad.
//! Run: `cargo test -p rlx-tensor --features eval`.
#![cfg(feature = "eval")]

use rlx_tensor::Tensor;

fn approx(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "length mismatch: {a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-5, "{a:?} != {b:?}");
    }
}

#[test]
fn split_and_chunk() {
    let x = Tensor::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0], [5]);
    let parts = x.split(0, &[2, 3]);
    approx(&parts[0].to_vec(), &[0.0, 1.0]);
    approx(&parts[1].to_vec(), &[2.0, 3.0, 4.0]);
    // chunk: 5 into 2 -> [3, 2]
    let cs = x.chunk(0, 2);
    assert_eq!(cs.len(), 2);
    approx(&cs[0].to_vec(), &[0.0, 1.0, 2.0]);
    approx(&cs[1].to_vec(), &[3.0, 4.0]);
}

#[test]
fn tile_repeats() {
    let x = Tensor::from_vec(vec![1.0, 2.0], [2]);
    approx(&x.tile(0, 3).to_vec(), &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
}

#[test]
fn flip_reverses() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [4]);
    approx(&x.flip(0).to_vec(), &[4.0, 3.0, 2.0, 1.0]);
    // 2-D, flip rows (axis 0)
    let m = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
    approx(&m.flip(0).to_vec(), &[3.0, 4.0, 1.0, 2.0]);
}

#[test]
fn roll_shifts() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [4]);
    // shift 1 -> last element wraps to front
    approx(&x.roll(0, 1).to_vec(), &[4.0, 1.0, 2.0, 3.0]);
}

#[test]
fn pad_constant() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], [3]);
    approx(
        &x.pad(0, 2, 1, 0.0).to_vec(),
        &[0.0, 0.0, 1.0, 2.0, 3.0, 0.0],
    );
    // 2-D: pad columns (axis 1) on a [1,2] -> [1,4]
    let m = Tensor::from_vec(vec![5.0, 6.0], [1, 2]);
    approx(&m.pad(1, 1, 1, -1.0).to_vec(), &[-1.0, 5.0, 6.0, -1.0]);
}
