// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Small dense linear algebra for [`Step::Linalg`](crate::Step::Linalg), in f64
//! (rlx-cpu promotes to f64 for LAPACK too).
//!
//! Factorizations are sequential — each pivot depends on the last — so they do
//! not map to one fragment per output element. Both executors run these
//! functions on the host: the GL executor reads the operands back, calls them,
//! and uploads the result, the way the native GPU backends host-stage linalg.

/// Which solve / factorization a [`Step::Linalg`](crate::Step::Linalg) runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinalgOp {
    /// `A = L·Lᵀ`; output `L` with the strict upper triangle zeroed.
    Cholesky,
    /// `op(A)·X = B`, `A` read as lower or upper triangular, `op(A) = Aᵀ` when
    /// `transpose`.
    TriSolve { lower: bool, transpose: bool },
    /// `A·X = B` by LU with partial pivoting.
    Solve,
    /// `det(A)`.
    Det,
    /// `log|det(A)|`.
    LogDet,
}

/// Run `op` on `batch` independent `[n, n]` matrices `a` (and right-hand sides
/// `b`, `[n, nrhs]` each, row-major). Returns the row-major result: `[n, n]`,
/// `[n, nrhs]` or one scalar per matrix.
pub(crate) fn run(
    op: LinalgOp,
    a: &[f32],
    b: Option<&[f32]>,
    n: usize,
    nrhs: usize,
    batch: usize,
) -> Vec<f32> {
    let mut out = Vec::new();
    for bi in 0..batch {
        let m: Vec<f64> = a[bi * n * n..(bi + 1) * n * n]
            .iter()
            .map(|&v| v as f64)
            .collect();
        let rhs = || -> Vec<f64> {
            b.map_or(&[][..], |b| &b[bi * n * nrhs..(bi + 1) * n * nrhs])
                .iter()
                .map(|&v| v as f64)
                .collect()
        };
        let r = match op {
            LinalgOp::Cholesky => cholesky(&m, n),
            LinalgOp::TriSolve { lower, transpose } => {
                tri_solve(&m, rhs(), n, nrhs, lower, transpose)
            }
            LinalgOp::Solve => {
                let (lu, perm, _) = lu(&m, n);
                lu_solve(&lu, &perm, rhs(), n, nrhs)
            }
            LinalgOp::Det => {
                let (lu, _, sign) = lu(&m, n);
                vec![(0..n).map(|i| lu[i * n + i]).product::<f64>() * sign]
            }
            LinalgOp::LogDet => {
                let (lu, _, _) = lu(&m, n);
                vec![(0..n).map(|i| lu[i * n + i].abs().ln()).sum()]
            }
        };
        out.extend(r.into_iter().map(|v| v as f32));
    }
    out
}

fn cholesky(a: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0f64; n * n];
    for j in 0..n {
        let d = a[j * n + j] - (0..j).map(|k| l[j * n + k] * l[j * n + k]).sum::<f64>();
        let ljj = d.sqrt();
        l[j * n + j] = ljj;
        for i in j + 1..n {
            let s = a[i * n + j] - (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum::<f64>();
            l[i * n + j] = s / ljj;
        }
    }
    l
}

/// Substitution with `op(A)`; the triangle `op(A)` occupies is lower exactly
/// when `lower != transpose`.
fn tri_solve(
    a: &[f64],
    mut x: Vec<f64>,
    n: usize,
    nrhs: usize,
    lower: bool,
    transpose: bool,
) -> Vec<f64> {
    let at = |i: usize, j: usize| {
        if transpose {
            a[j * n + i]
        } else {
            a[i * n + j]
        }
    };
    let forward = lower != transpose;
    for c in 0..nrhs {
        for s in 0..n {
            let i = if forward { s } else { n - 1 - s };
            let known = if forward { 0..i } else { i + 1..n };
            let acc: f64 = known.map(|j| at(i, j) * x[j * nrhs + c]).sum();
            x[i * nrhs + c] = (x[i * nrhs + c] - acc) / at(i, i);
        }
    }
    x
}

/// In-place LU with partial pivoting: `(LU, row permutation, det sign)`.
fn lu(a: &[f64], n: usize) -> (Vec<f64>, Vec<usize>, f64) {
    let mut m = a.to_vec();
    let mut perm: Vec<usize> = (0..n).collect();
    let mut sign = 1.0;
    for k in 0..n {
        let p = (k..n)
            .max_by(|&i, &j| m[i * n + k].abs().total_cmp(&m[j * n + k].abs()))
            .unwrap_or(k);
        if p != k {
            for j in 0..n {
                m.swap(k * n + j, p * n + j);
            }
            perm.swap(k, p);
            sign = -sign;
        }
        let piv = m[k * n + k];
        for i in k + 1..n {
            let f = m[i * n + k] / piv;
            m[i * n + k] = f;
            for j in k + 1..n {
                m[i * n + j] -= f * m[k * n + j];
            }
        }
    }
    (m, perm, sign)
}

fn lu_solve(lu: &[f64], perm: &[usize], b: Vec<f64>, n: usize, nrhs: usize) -> Vec<f64> {
    let mut x = vec![0f64; n * nrhs];
    for c in 0..nrhs {
        for i in 0..n {
            let s: f64 = (0..i).map(|j| lu[i * n + j] * x[j * nrhs + c]).sum();
            x[i * nrhs + c] = b[perm[i] * nrhs + c] - s;
        }
        for i in (0..n).rev() {
            let s: f64 = (i + 1..n).map(|j| lu[i * n + j] * x[j * nrhs + c]).sum();
            x[i * nrhs + c] = (x[i * nrhs + c] - s) / lu[i * n + i];
        }
    }
    x
}
