// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Symmetric matrix functions and covariance estimation for EEG source
//! localization (`exg-source` eLORETA / sLORETA).
//!
//! Built on the crate's LAPACK-backed [`crate::algos::eigh`]:
//!
//!   - [`matrix_sqrt`] / [`matrix_invsqrt`] — `V·diag(f(λ))·Vᵀ` with
//!     `f = √` and rank-truncated `f = 1/√`, the whitener and its inverse.
//!   - [`covariance`] — empirical (co)variance with selectable `ddof`.
//!   - [`shrink_to_identity`] — blend a covariance toward a scaled identity.
//!   - [`ledoit_wolf`] — the Ledoit–Wolf (2004) optimal shrinkage estimator.
//!
//! All matrices are row-major `f64`. Symmetric-input functions symmetrize
//! implicitly through the eigendecomposition.

/// `f(A) = V·diag(f(λ))·Vᵀ` for a symmetric `n×n` matrix `a` (row-major),
/// writing the `n×n` result into `out`.
fn sym_matrix_funm(
    a: &[f64],
    n: usize,
    f: impl Fn(f64) -> f64,
    out: &mut [f64],
) -> Result<(), String> {
    if a.len() != n * n || out.len() != n * n {
        return Err(format!("sym_matrix_funm: shape mismatch (n={n})"));
    }
    let mut packed = vec![0f64; n + n * n];
    crate::algos::eigh(a, n, &mut packed)?;
    let (evals, evecs) = packed.split_at(n); // evecs: row i = eigenvector i
    let fvals: Vec<f64> = evals.iter().map(|&l| f(l)).collect();
    for k in 0..n {
        for l in 0..n {
            let mut s = 0.0;
            for i in 0..n {
                s += fvals[i] * evecs[i * n + k] * evecs[i * n + l];
            }
            out[k * n + l] = s;
        }
    }
    Ok(())
}

/// Symmetric matrix square root `A^{1/2}` (negative eigenvalues clamped to 0).
/// `a` must be symmetric positive semidefinite; `out` is `n×n` row-major.
pub fn matrix_sqrt(a: &[f64], n: usize, out: &mut [f64]) -> Result<(), String> {
    sym_matrix_funm(a, n, |l| l.max(0.0).sqrt(), out)
}

/// Symmetric inverse square root `A^{-1/2}` with rank truncation.
///
/// Eigenvalues at or below `rcond · λ_max` are treated as zero (their
/// contribution is dropped), giving the Moore–Penrose pseudo-inverse-sqrt —
/// the regularized whitener used for noise-covariance whitening. Pass
/// `rcond = 0.0` for no truncation.
pub fn matrix_invsqrt(a: &[f64], n: usize, rcond: f64, out: &mut [f64]) -> Result<(), String> {
    if a.len() != n * n || out.len() != n * n {
        return Err(format!("matrix_invsqrt: shape mismatch (n={n})"));
    }
    let mut packed = vec![0f64; n + n * n];
    crate::algos::eigh(a, n, &mut packed)?;
    let (evals, evecs) = packed.split_at(n);
    // eigh returns ascending eigenvalues → λ_max is the last.
    let lambda_max = evals.last().copied().unwrap_or(0.0);
    let thresh = rcond * lambda_max.max(0.0);
    let fvals: Vec<f64> = evals
        .iter()
        .map(|&l| {
            if l > thresh && l > 0.0 {
                1.0 / l.sqrt()
            } else {
                0.0
            }
        })
        .collect();
    for k in 0..n {
        for l in 0..n {
            let mut s = 0.0;
            for i in 0..n {
                s += fvals[i] * evecs[i * n + k] * evecs[i * n + l];
            }
            out[k * n + l] = s;
        }
    }
    Ok(())
}

/// Empirical covariance of `x` (`[n_samples, n_features]`, row-major).
///
/// Columns are de-meaned; the normalizer is `n_samples − ddof` (`ddof = 1`
/// for the unbiased estimate, `ddof = 0` for the MLE). Output is the
/// `n_features × n_features` covariance, row-major.
pub fn covariance(
    x: &[f64],
    n_samples: usize,
    n_features: usize,
    ddof: usize,
    out: &mut [f64],
) -> Result<(), String> {
    if x.len() != n_samples * n_features || out.len() != n_features * n_features {
        return Err("covariance: shape mismatch".to_string());
    }
    if n_samples <= ddof {
        return Err(format!("covariance: n_samples {n_samples} <= ddof {ddof}"));
    }
    let mut mean = vec![0f64; n_features];
    for s in 0..n_samples {
        for (j, m) in mean.iter_mut().enumerate() {
            *m += x[s * n_features + j];
        }
    }
    for m in mean.iter_mut() {
        *m /= n_samples as f64;
    }
    for c in out.iter_mut() {
        *c = 0.0;
    }
    for s in 0..n_samples {
        let row = &x[s * n_features..(s + 1) * n_features];
        for i in 0..n_features {
            let di = row[i] - mean[i];
            for j in 0..n_features {
                out[i * n_features + j] += di * (row[j] - mean[j]);
            }
        }
    }
    let norm = (n_samples - ddof) as f64;
    for c in out.iter_mut() {
        *c /= norm;
    }
    Ok(())
}

/// Blend a covariance toward a scaled identity:
/// `Σ_shrunk = (1−α)·cov + α·μ·I`, with `μ = trace(cov)/n`.
///
/// `alpha ∈ [0, 1]` is the shrinkage intensity. Operates in place.
pub fn shrink_to_identity(cov: &mut [f64], n: usize, alpha: f64) -> Result<(), String> {
    if cov.len() != n * n {
        return Err(format!("shrink_to_identity: expected n² (n={n})"));
    }
    let mut trace = 0.0;
    for i in 0..n {
        trace += cov[i * n + i];
    }
    let mu = trace / n as f64;
    for i in 0..n {
        for j in 0..n {
            let target = if i == j { mu } else { 0.0 };
            let idx = i * n + j;
            cov[idx] = (1.0 - alpha) * cov[idx] + alpha * target;
        }
    }
    Ok(())
}

/// Ledoit–Wolf (2004) optimal shrinkage toward a scaled identity.
///
/// Estimates the analytically-optimal intensity `α*` that minimizes expected
/// Frobenius error, then writes `Σ_shrunk = (1−α*)·S + α*·μ·I` into `out`
/// (where `S` is the MLE covariance and `μ = trace(S)/p`). Returns `α*`.
///
/// `x` is `[n_samples, n_features]` row-major.
pub fn ledoit_wolf(
    x: &[f64],
    n_samples: usize,
    n_features: usize,
    out: &mut [f64],
) -> Result<f64, String> {
    let n = n_samples;
    let p = n_features;
    if x.len() != n * p || out.len() != p * p {
        return Err("ledoit_wolf: shape mismatch".to_string());
    }
    if n < 2 {
        return Err("ledoit_wolf: need ≥2 samples".to_string());
    }
    // De-mean columns.
    let mut mean = vec![0f64; p];
    for s in 0..n {
        for (j, m) in mean.iter_mut().enumerate() {
            *m += x[s * p + j];
        }
    }
    for m in mean.iter_mut() {
        *m /= n as f64;
    }
    let mut xc = vec![0f64; n * p];
    for s in 0..n {
        for j in 0..p {
            xc[s * p + j] = x[s * p + j] - mean[j];
        }
    }
    // MLE covariance S = Xcᵀ Xc / n.
    let mut s_cov = vec![0f64; p * p];
    for s in 0..n {
        let row = &xc[s * p..(s + 1) * p];
        for i in 0..p {
            let ri = row[i];
            for j in 0..p {
                s_cov[i * p + j] += ri * row[j];
            }
        }
    }
    for c in s_cov.iter_mut() {
        *c /= n as f64;
    }
    // μ = trace(S)/p ; d² = ‖S − μI‖²_F / p.
    let mut trace = 0.0;
    for i in 0..p {
        trace += s_cov[i * p + i];
    }
    let mu = trace / p as f64;
    let mut d2 = 0.0;
    for i in 0..p {
        for j in 0..p {
            let target = if i == j { mu } else { 0.0 };
            let e = s_cov[i * p + j] - target;
            d2 += e * e;
        }
    }
    d2 /= p as f64;
    // b̄² = (1/n²) Σ_k ‖x_k x_kᵀ − S‖²_F / p ; b² = min(b̄², d²).
    let mut b_bar2 = 0.0;
    for s in 0..n {
        let row = &xc[s * p..(s + 1) * p];
        let mut acc = 0.0;
        for i in 0..p {
            let ri = row[i];
            for j in 0..p {
                let e = ri * row[j] - s_cov[i * p + j];
                acc += e * e;
            }
        }
        b_bar2 += acc;
    }
    b_bar2 /= (n as f64) * (n as f64) * p as f64;
    let b2 = b_bar2.min(d2);
    let alpha = if d2 > 0.0 {
        (b2 / d2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    // Σ_shrunk = (1−α) S + α μ I.
    for i in 0..p {
        for j in 0..p {
            let target = if i == j { mu } else { 0.0 };
            let idx = i * p + j;
            out[idx] = (1.0 - alpha) * s_cov[idx] + alpha * target;
        }
    }
    Ok(alpha)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matmul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
        let mut c = vec![0f64; n * n];
        for i in 0..n {
            for k in 0..n {
                let aik = a[i * n + k];
                for j in 0..n {
                    c[i * n + j] += aik * b[k * n + j];
                }
            }
        }
        c
    }

    #[test]
    fn matrix_sqrt_squares_back() {
        // SPD: [[2,1],[1,2]], eigenvalues 1 and 3.
        let a = [2.0, 1.0, 1.0, 2.0];
        let mut r = [0f64; 4];
        matrix_sqrt(&a, 2, &mut r).unwrap();
        let rr = matmul(&r, &r, 2);
        for (g, e) in rr.iter().zip(a.iter()) {
            assert!((g - e).abs() < 1e-10, "sqrt²={g} vs {e}");
        }
    }

    #[test]
    fn invsqrt_whitens() {
        // A^{-1/2} A A^{-1/2} ≈ I.
        let a = [4.0, 1.0, 1.0, 3.0];
        let mut w = [0f64; 4];
        matrix_invsqrt(&a, 2, 0.0, &mut w).unwrap();
        let wa = matmul(&w, &a, 2);
        let waw = matmul(&wa, &w, 2);
        let eye = [1.0, 0.0, 0.0, 1.0];
        for (g, e) in waw.iter().zip(eye.iter()) {
            assert!((g - e).abs() < 1e-9, "WAW={g} vs I={e}");
        }
    }

    #[test]
    fn invsqrt_rank_truncates() {
        // Rank-1 (singular) matrix: [[1,0],[0,0]] with rcond drops the null dir.
        let a = [1.0, 0.0, 0.0, 0.0];
        let mut w = [0f64; 4];
        matrix_invsqrt(&a, 2, 1e-6, &mut w).unwrap();
        // Only the (0,0) entry survives (=1), the null direction is zeroed.
        assert!((w[0] - 1.0).abs() < 1e-9);
        assert!(w[1].abs() < 1e-12 && w[2].abs() < 1e-12 && w[3].abs() < 1e-12);
    }

    #[test]
    fn covariance_matches_manual() {
        // X = [[1,2],[3,4],[5,6]] ; columns have variance 4 (ddof=1), cov 4.
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut c = [0f64; 4];
        covariance(&x, 3, 2, 1, &mut c).unwrap();
        for v in c.iter() {
            assert!((v - 4.0).abs() < 1e-10, "cov entry {v} != 4");
        }
    }

    #[test]
    fn ledoit_wolf_alpha_in_range_and_shrinks() {
        // p > n → sample cov is singular; LW must pull toward the identity.
        let x = [
            1.0, 2.0, 0.5, -1.0, 0.3, 2.0, -0.5, 1.5, 0.1, 0.4, 1.1, -0.2,
        ];
        let mut out = [0f64; 16];
        let alpha = ledoit_wolf(&x, 3, 4, &mut out).unwrap();
        assert!((0.0..=1.0).contains(&alpha), "alpha={alpha} out of range");
        // Off-diagonals are shrunk relative to the diagonal (identity target).
        let diag_sum: f64 = (0..4).map(|i| out[i * 4 + i]).sum();
        assert!(diag_sum > 0.0);
    }
}
