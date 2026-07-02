// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Matrix inverse (composite over DenseSolve → LAPACK). Needs a BLAS eval
//! build. Run: `cargo test -p rlx-tensor --features eval-blas`.
#![cfg(feature = "eval-blas")]

use rlx_tensor::Tensor;

fn approx(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "length mismatch: {a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-4, "{a:?} != {b:?}");
    }
}

#[test]
fn inv_matches_closed_form() {
    // A = [[4,7],[2,6]], det = 10, inv = [[0.6,-0.7],[-0.2,0.4]]
    let a = Tensor::from_vec(vec![4.0, 7.0, 2.0, 6.0], [2, 2]);
    approx(&a.inv().to_vec(), &[0.6, -0.7, -0.2, 0.4]);
}

#[test]
fn a_times_inv_is_identity() {
    let a = Tensor::from_vec(vec![4.0, 7.0, 2.0, 6.0], [2, 2]);
    let prod = a.matmul(&a.inv()).to_vec();
    approx(&prod, &[1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn solve_linear_system() {
    // A x = b, A=[[3,2],[1,2]], b=[5,5] -> x=[0,2.5]? check: 3*0+2*2.5=5, 1*0+2*2.5=5 ✓
    let a = Tensor::from_vec(vec![3.0, 2.0, 1.0, 2.0], [2, 2]);
    let b = Tensor::from_vec(vec![5.0, 5.0], [2, 1]);
    approx(&a.solve(&b).to_vec(), &[0.0, 2.5]);
}
