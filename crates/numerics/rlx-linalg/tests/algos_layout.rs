// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Output-layout contract for the LAPACK wrappers in `algos`.
//!
//! `eigh`'s packing is documented — eigenvalues ascending, eigenvector `i` as
//! *row* `i` — but nothing checked it, and the row-vs-column half of that
//! contract is easy to get backwards: LAPACK writes eigenvectors as columns in
//! column-major, and it is only the row-major *view* of the same bytes that
//! turns them into rows. A caller that assumes columns gets a plausible,
//! orthonormal, completely wrong basis, which downstream shows up as a small
//! numerical discrepancy rather than an obvious failure. That mistake has
//! already been made once in this workspace.
//!
//! Ground truth here is `numpy.linalg.eigh` / `numpy.linalg.solve` (numpy
//! 2.4.3), transposed into this crate's documented packing.

use rlx_linalg::algos;

const TOL: f64 = 1e-12;

/// Symmetric, distinct eigenvalues, and deliberately *not* structured so that
/// its eigenvector matrix is symmetric — otherwise rows and columns would agree
/// and the test would pass under either convention.
const A: [f64; 9] = [4.0, 1.0, -2.0, 1.0, 2.0, 0.5, -2.0, 0.5, 3.0];

#[test]
fn eigh_returns_eigenvalues_ascending() {
    let n = 3;
    let mut out = vec![0f64; n + n * n];
    algos::eigh(&A, n, &mut out).expect("eigh");
    let expect = [
        0.653_923_768_461_384_4,
        2.716_348_465_618_261_3,
        5.629_727_765_920_354,
    ];
    for (i, e) in expect.iter().enumerate() {
        assert!(
            (out[i] - e).abs() < TOL,
            "eigenvalue {i}: {} != {e}",
            out[i]
        );
    }
    assert!(
        out[0] < out[1] && out[1] < out[2],
        "eigenvalues are not ascending: {:?}",
        &out[..3]
    );
}

#[test]
fn eigh_packs_eigenvector_i_as_row_i() {
    // The defining property, applied row-wise: A·vᵢ = λᵢ·vᵢ. Sign-agnostic, so
    // it does not depend on LAPACK's arbitrary sign choice — but it *is*
    // layout-sensitive, which is the whole point. Reading the block as columns
    // instead fails this.
    let n = 3;
    let mut out = vec![0f64; n + n * n];
    algos::eigh(&A, n, &mut out).expect("eigh");
    let (vals, vecs) = out.split_at(n);

    for i in 0..n {
        let v = &vecs[i * n..(i + 1) * n];
        for r in 0..n {
            let av: f64 = (0..n).map(|c| A[r * n + c] * v[c]).sum();
            assert!(
                (av - vals[i] * v[r]).abs() < 1e-10,
                "row {i} is not an eigenvector: (A·v)[{r}]={av} != lambda*v[{r}]={}",
                vals[i] * v[r]
            );
        }
    }
}

#[test]
fn eigh_rows_are_orthonormal() {
    let n = 3;
    let mut out = vec![0f64; n + n * n];
    algos::eigh(&A, n, &mut out).expect("eigh");
    let vecs = &out[n..];
    for i in 0..n {
        for j in 0..n {
            let dot: f64 = (0..n).map(|k| vecs[i * n + k] * vecs[j * n + k]).sum();
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot - want).abs() < 1e-10,
                "<v{i}, v{j}> = {dot}, expected {want}"
            );
        }
    }
}

#[test]
fn eigh_matches_numpy_up_to_sign() {
    let n = 3;
    let mut out = vec![0f64; n + n * n];
    algos::eigh(&A, n, &mut out).expect("eigh");
    // numpy's columns, transposed into this crate's row packing.
    let expect: [[f64; 3]; 3] = [
        [
            -0.532_324_344_798_056_3,
            0.612_516_986_895_522,
            -0.584_340_425_351_310_5,
        ],
        [
            0.274_633_931_933_671_47,
            0.777_889_337_499_416_2,
            0.565_211_802_809_682,
        ],
        [
            -0.800_754_016_765_430_5,
            -0.140_396_294_000_767_56,
            0.582_307_380_397_053_3,
        ],
    ];
    for (i, e) in expect.iter().enumerate() {
        let v = &out[n + i * n..n + (i + 1) * n];
        // LAPACK fixes no sign convention, so accept either orientation.
        let same = v.iter().zip(e).all(|(a, b)| (a - b).abs() < 1e-10);
        let flipped = v.iter().zip(e).all(|(a, b)| (a + b).abs() < 1e-10);
        assert!(
            same || flipped,
            "eigenvector {i}: {v:?} matches neither {e:?} nor its negation"
        );
    }
}

#[test]
fn gesv_solves_row_major_multi_rhs() {
    // bcidecoders' Wiener/Kalman fits go through this; a transposed RHS would
    // still produce numbers, just the wrong ones.
    let n = 3;
    let m = 2;
    let a = [3.0, 1.0, 0.0, 1.0, 4.0, 2.0, 0.0, 2.0, 5.0];
    let b = [1.0, 2.0, 3.0, 0.0, -1.0, 4.0];
    let mut x = vec![0f64; n * m];
    algos::gesv(&a, &b, n, m, &mut x).expect("gesv");
    let expect = [
        -0.023_255_813_953_488_413,
        0.930_232_558_139_534_9,
        1.069_767_441_860_465_2,
        -0.790_697_674_418_604_7,
        -0.627_906_976_744_186,
        1.116_279_069_767_441_8,
    ];
    for (i, e) in expect.iter().enumerate() {
        assert!((x[i] - e).abs() < 1e-12, "X[{i}] = {} != {e}", x[i]);
    }
}
