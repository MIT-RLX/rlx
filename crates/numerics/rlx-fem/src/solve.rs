// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compressed sparse row storage and a preconditioned conjugate gradient.
//!
//! Both the stiffness matrix and the Newton tangent of [`crate::assemble`] are
//! symmetric positive definite once constraints are eliminated, so conjugate
//! gradient is the matching solver and a factorisation would cost memory for no
//! gain. Jacobi preconditioning suffices because the conditioning of these
//! systems is driven by the coefficient contrast between regions — often four
//! orders of magnitude, and diagonal in character.
//!
//! This is deliberately dependency-free so the crate core builds under
//! `--no-default-features`. Where the graph runtime is already in play,
//! `rlx-sparse` offers the same solve plus BiCGSTAB, GMRES and incomplete
//! factorisations with backend kernels behind them.

/// Compressed sparse row matrix.
#[derive(Debug, Clone, Default)]
pub struct Csr {
    /// Row count; the matrix is square.
    pub n: usize,
    /// Row start offsets, length `n + 1`.
    pub indptr: Vec<usize>,
    /// Column index per stored entry.
    pub indices: Vec<usize>,
    /// Value per stored entry.
    pub values: Vec<f64>,
}

impl Csr {
    /// Assemble from `(row, col, value)` triplets, summing duplicates.
    pub fn from_triplets(n: usize, mut triplets: Vec<(usize, usize, f64)>) -> Csr {
        triplets.sort_unstable_by_key(|&(r, c, _)| (r, c));
        let mut indptr = vec![0usize; n + 1];
        let mut indices: Vec<usize> = Vec::with_capacity(triplets.len());
        let mut values: Vec<f64> = Vec::with_capacity(triplets.len());
        let mut row = 0usize;
        for (r, c, v) in triplets {
            while row < r {
                indptr[row + 1] = indices.len();
                row += 1;
            }
            if indptr[row] < indices.len() && indices.last() == Some(&c) {
                *values.last_mut().expect("non-empty") += v;
            } else {
                indices.push(c);
                values.push(v);
            }
        }
        for r in row..n {
            indptr[r + 1] = indices.len();
        }
        Csr {
            n,
            indptr,
            indices,
            values,
        }
    }

    /// `y = A * x`.
    pub fn mul_into(&self, x: &[f64], y: &mut [f64]) {
        for r in 0..self.n {
            let mut acc = 0.0;
            for k in self.indptr[r]..self.indptr[r + 1] {
                acc += self.values[k] * x[self.indices[k]];
            }
            y[r] = acc;
        }
    }

    /// Main diagonal, for Jacobi preconditioning.
    pub fn diagonal(&self) -> Vec<f64> {
        (0..self.n)
            .map(|r| {
                (self.indptr[r]..self.indptr[r + 1])
                    .find(|&k| self.indices[k] == r)
                    .map(|k| self.values[k])
                    .unwrap_or(0.0)
            })
            .collect()
    }
}

/// How a linear solve ended.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolveReport {
    /// Iterations taken.
    pub iterations: usize,
    /// Final relative residual.
    pub residual: f64,
    /// Whether the tolerance was met.
    pub converged: bool,
}
/// An incomplete Cholesky factorisation with no fill-in.
///
/// `L L^T ~= A`, with `L` confined to the sparsity of `A`'s lower triangle.
/// Applying it costs one forward and one back substitution per iteration —
/// about the price of a second mat-vec — and buys far more than it costs: on
/// the magnetostatic systems this solver was built for it cuts a
/// Jacobi-preconditioned solve from 520 iterations to 71 at 3168 unknowns, and
/// the wall clock by about two and a half times.
///
/// It is not unconditionally available. The factorisation needs the matrix to
/// stay positive definite under its own restricted pattern, and a Newton
/// tangent from a strongly saturating material need not be; where it breaks
/// down, [`Ic0::of`] returns `None` and the caller falls back to the diagonal.
pub struct Ic0 {
    n: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    values: Vec<f64>,
    diag: Vec<f64>,
}

impl Ic0 {
    /// Factorise, or decline to.
    pub fn of(a: &Csr) -> Option<Ic0> {
        let n = a.n;
        let (mut indptr, mut indices, mut values) = (vec![0usize; n + 1], Vec::new(), Vec::new());
        for r in 0..n {
            for k in a.indptr[r]..a.indptr[r + 1] {
                if a.indices[k] < r {
                    indices.push(a.indices[k]);
                    values.push(a.values[k]);
                }
            }
            indptr[r + 1] = indices.len();
        }
        let mut diag: Vec<f64> = a.diagonal();

        for r in 0..n {
            for k in indptr[r]..indptr[r + 1] {
                let c = indices[k];
                let mut sum = values[k];
                // The two rows dotted over the columns they share below `c`.
                // Both index lists are ascending, so this is a merge.
                let (mut i, mut j) = (indptr[r], indptr[c]);
                while i < indptr[r + 1] && j < indptr[c + 1] {
                    let (ci, cj) = (indices[i], indices[j]);
                    if ci >= c || cj >= c {
                        break;
                    }
                    match ci.cmp(&cj) {
                        core::cmp::Ordering::Less => i += 1,
                        core::cmp::Ordering::Greater => j += 1,
                        core::cmp::Ordering::Equal => {
                            sum -= values[i] * values[j];
                            i += 1;
                            j += 1;
                        }
                    }
                }
                // NaN counts as a breakdown too, which is why this is written
                // as a positive test rather than as `<= 0.0`.
                if !matches!(
                    diag[c].partial_cmp(&0.0),
                    Some(core::cmp::Ordering::Greater)
                ) {
                    return None;
                }
                values[k] = sum / diag[c];
            }
            let mut d = diag[r];
            for k in indptr[r]..indptr[r + 1] {
                d -= values[k] * values[k];
            }
            if !matches!(d.partial_cmp(&0.0), Some(core::cmp::Ordering::Greater)) || !d.is_finite()
            {
                return None;
            }
            diag[r] = d.sqrt();
        }
        Some(Ic0 {
            n,
            indptr,
            indices,
            values,
            diag,
        })
    }

    /// Solve `L L^T z = r` in place of the diagonal scaling.
    pub fn apply(&self, r: &[f64], z: &mut [f64]) {
        z.copy_from_slice(r);
        for i in 0..self.n {
            let mut s = z[i];
            for k in self.indptr[i]..self.indptr[i + 1] {
                s -= self.values[k] * z[self.indices[k]];
            }
            z[i] = s / self.diag[i];
        }
        for i in (0..self.n).rev() {
            z[i] /= self.diag[i];
            let zi = z[i];
            for k in self.indptr[i]..self.indptr[i + 1] {
                z[self.indices[k]] -= self.values[k] * zi;
            }
        }
    }
}

/// Jacobi-preconditioned conjugate gradient for a symmetric positive definite `a`.
///
/// Returns the solution together with a report. A system that did not converge
/// still returns its best iterate rather than an error, because a caller driving
/// an interactive solve is better served by a field with a warning attached than
/// by nothing at all.
pub fn pcg(a: &Csr, b: &[f64], tol: f64, max_iter: usize) -> (Vec<f64>, SolveReport) {
    let n = a.n;
    let mut x = vec![0.0; n];
    if n == 0 {
        return (
            x,
            SolveReport {
                iterations: 0,
                residual: 0.0,
                converged: true,
            },
        );
    }

    let inv_diag: Vec<f64> = a
        .diagonal()
        .into_iter()
        .map(|d| if d.abs() > 0.0 { 1.0 / d } else { 1.0 })
        .collect();

    let mut r = b.to_vec();
    let b_norm = dot(b, b).sqrt();
    if b_norm == 0.0 {
        return (
            x,
            SolveReport {
                iterations: 0,
                residual: 0.0,
                converged: true,
            },
        );
    }

    // Incomplete Cholesky where the matrix admits it, the diagonal where it
    // does not. Both are only preconditioners: neither changes the answer, and
    // the fallback is silent because there is nothing for a caller to do about
    // it — a tangent that will not factorise is still solved, just more slowly.
    let ic = Ic0::of(a);
    let precondition = |r: &[f64], z: &mut [f64]| match &ic {
        Some(f) => f.apply(r, z),
        None => {
            for i in 0..n {
                z[i] = r[i] * inv_diag[i];
            }
        }
    };

    let mut z = vec![0.0; n];
    precondition(&r, &mut z);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let mut ap = vec![0.0; n];

    for it in 1..=max_iter {
        a.mul_into(&p, &mut ap);
        let denom = dot(&p, &ap);
        if denom.abs() < f64::MIN_POSITIVE {
            break;
        }
        let alpha = rz / denom;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let res = dot(&r, &r).sqrt() / b_norm;
        if res < tol {
            return (
                x,
                SolveReport {
                    iterations: it,
                    residual: res,
                    converged: true,
                },
            );
        }
        precondition(&r, &mut z);
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;
        rz = rz_new;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
    }

    let res = dot(&r, &r).sqrt() / b_norm;
    (
        x,
        SolveReport {
            iterations: max_iter,
            residual: res,
            converged: false,
        },
    )
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triplets_sum_duplicates() {
        let m = Csr::from_triplets(2, vec![(0, 0, 1.0), (0, 0, 2.0), (1, 1, 4.0)]);
        let mut y = vec![0.0; 2];
        m.mul_into(&[1.0, 1.0], &mut y);
        assert_eq!(y, vec![3.0, 4.0]);
    }

    #[test]
    fn empty_rows_keep_the_offsets_consistent() {
        // Row 1 has no entries; indptr must still be monotone and addressable.
        let m = Csr::from_triplets(3, vec![(0, 0, 1.0), (2, 2, 3.0)]);
        assert_eq!(m.indptr, vec![0, 1, 1, 2]);
        let mut y = vec![0.0; 3];
        m.mul_into(&[1.0, 1.0, 1.0], &mut y);
        assert_eq!(y, vec![1.0, 0.0, 3.0]);
    }

    #[test]
    fn cg_solves_a_small_symmetric_system() {
        // [[4, 1], [1, 3]] x = [1, 2]  ->  x = [1/11, 7/11]
        let m = Csr::from_triplets(2, vec![(0, 0, 4.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)]);
        let (x, rep) = pcg(&m, &[1.0, 2.0], 1e-14, 100);
        assert!(rep.converged, "{rep:?}");
        assert!((x[0] - 1.0 / 11.0).abs() < 1e-12);
        assert!((x[1] - 7.0 / 11.0).abs() < 1e-12);
    }

    #[test]
    fn cg_solves_a_laplacian_chain() {
        // A 1D Laplacian with unit source has a parabolic discrete solution,
        // and conjugate gradient must find it to near machine precision.
        let n = 64;
        let mut t = Vec::new();
        for i in 0..n {
            t.push((i, i, 2.0));
            if i > 0 {
                t.push((i, i - 1, -1.0));
            }
            if i + 1 < n {
                t.push((i, i + 1, -1.0));
            }
        }
        let a = Csr::from_triplets(n, t);
        let (x, rep) = pcg(&a, &vec![1.0; n], 1e-12, 500);
        assert!(rep.converged, "{rep:?}");
        for i in 0..n {
            let xi = (i + 1) as f64;
            let expect = 0.5 * xi * (n as f64 + 1.0 - xi);
            assert!(
                (x[i] - expect).abs() / expect < 1e-8,
                "{} vs {expect}",
                x[i]
            );
        }
    }

    #[test]
    fn a_zero_right_hand_side_returns_immediately() {
        let m = Csr::from_triplets(3, vec![(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)]);
        let (x, rep) = pcg(&m, &[0.0; 3], 1e-12, 10);
        assert!(rep.converged && rep.iterations == 0);
        assert_eq!(x, vec![0.0; 3]);
    }
}
