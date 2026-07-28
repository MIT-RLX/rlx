// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! ndarray interop (feature `ndarray`). Run:
//! `cargo test -p rlx-tensor --features ndarray,eval`.
#![cfg(all(feature = "ndarray", feature = "eval"))]

use ndarray::{Array2, array};
use rlx_tensor::Tensor;

#[test]
fn from_ndarray_preserves_shape_and_data() {
    let a: Array2<f32> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let t = Tensor::from(a);
    assert_eq!(t.dims(), vec![2, 3]);
    assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn to_ndarray_roundtrip() {
    let a: Array2<f32> = array![[1.0, 2.0], [3.0, 4.0]];
    // Tensor op then back to ndarray.
    let t = Tensor::from(&a);
    let out = (&t + &t).to_ndarray(); // 2*a
    assert_eq!(out.shape(), &[2, 2]);
    let expected: Array2<f32> = array![[2.0, 4.0], [6.0, 8.0]];
    assert_eq!(out, expected.into_dyn());
}

#[test]
fn compute_through_rlx_matches_ndarray() {
    // matmul via rlx, compare to ndarray's dot.
    let a: Array2<f32> = array![[1.0, 2.0], [3.0, 4.0]];
    let b: Array2<f32> = array![[5.0, 6.0], [7.0, 8.0]];
    let got = Tensor::from(&a).matmul(&Tensor::from(&b)).to_ndarray();
    let want = a.dot(&b).into_dyn();
    assert_eq!(got, want);
}
