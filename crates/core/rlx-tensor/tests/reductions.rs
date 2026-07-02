// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Tier-2 numerics: cumsum, var/std, norm, logsumexp.
//! Run: `cargo test -p rlx-tensor --features eval`.
#![cfg(feature = "eval")]

use rlx_tensor::Tensor;

fn approx(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "length mismatch: {a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-4, "{a:?} != {b:?}");
    }
}

#[test]
fn cumsum_inclusive_and_exclusive() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [4]);
    approx(&x.cumsum(0, false).to_vec(), &[1.0, 3.0, 6.0, 10.0]);
    approx(&x.cumsum(0, true).to_vec(), &[0.0, 1.0, 3.0, 6.0]);
}

#[test]
fn var_and_std() {
    // [1,2,3,4,5]: mean 3, var = (4+1+0+1+4)/5 = 2, std = sqrt(2)
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], [5]);
    approx(&x.var([0], false).to_vec(), &[2.0]);
    approx(&x.std([0], false).to_vec(), &[2.0_f32.sqrt()]);
}

#[test]
fn norm_l2() {
    // sqrt(3^2 + 4^2) = 5
    let x = Tensor::from_vec(vec![3.0, 4.0], [2]);
    approx(&x.norm([0], false).to_vec(), &[5.0]);
}

#[test]
fn logsumexp_matches_naive() {
    let v = [1.0_f32, 2.0, 3.0];
    let x = Tensor::from_vec(v.to_vec(), [3]);
    let expect = v.iter().map(|x| x.exp()).sum::<f32>().ln();
    approx(&x.logsumexp(0, false).to_vec(), &[expect]);
}

#[test]
fn var_per_row() {
    // [[1,2,3],[2,2,2]] -> row vars [2/3, 0]
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 2.0, 2.0, 2.0], [2, 3]);
    approx(&x.var([1], false).to_vec(), &[2.0 / 3.0, 0.0]);
}
