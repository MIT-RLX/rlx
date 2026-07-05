//! SPD-manifold (symmetric positive-definite) kernels — the core primitives of Riemannian
//! machine learning on covariance matrices (EEG BCI, diffusion tensors, Gaussian embeddings).
//!
//! Built on the `blas::dsyevd` symmetric eigendecomposition:
//!   - matrix functions `logm` / `expm` / `sqrtm` / `invsqrtm` (spectral: `V·diag(f(λ))·Vᵀ`),
//!   - the affine-invariant Riemannian metric distance `airm_dist2`,
//!   - the Fréchet / Karcher mean `karcher_mean` (iterative tangent averaging),
//!   - and a rayon-parallel **batched** eigendecomposition `eigh_batch` (the hot path when you
//!     have thousands of small covariances — one per band × recording).
//!
//! These make `rlx-spdnet` / `rlx-tsmnet` / `rlx-tensorcspnet` (and tangent-space MDM / EA
//! alignment) fast and reusable instead of hand-rolled per project. All matrices are row-major
//! `n×n` and symmetric (row-major == column-major, so LAPACK ingests them directly).

/// Symmetric eigendecomposition: returns `(eigenvalues ascending, eigenvectors column-major)`
/// where `evecs[k*n + i]` is component `i` of eigenvector `k`.
pub fn eigh(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut am = a.to_vec();
    let mut w = vec![0f64; n];
    let _info = crate::blas::dsyevd(&mut am, &mut w, n);
    (w, am)
}

/// Batched symmetric eigendecomposition — rayon-parallel over many small SPD matrices. This is the
/// EEG hot path (a covariance per frequency band per recording); one call, all cores.
pub fn eigh_batch(mats: &[Vec<f64>], n: usize) -> Vec<(Vec<f64>, Vec<f64>)> {
    use rayon::prelude::*;
    mats.par_iter().map(|a| eigh(a, n)).collect()
}

/// `V·diag(f(λ))·Vᵀ` — apply a scalar function to the eigenvalues of a symmetric matrix.
pub fn matrix_fn(a: &[f64], n: usize, f: impl Fn(f64) -> f64) -> Vec<f64> {
    let (w, v) = eigh(a, n);
    let fl: Vec<f64> = w.iter().map(|&l| f(l)).collect();
    let mut out = vec![0f64; n * n];
    for k in 0..n {
        let fk = fl[k];
        for i in 0..n {
            let vik = fk * v[k * n + i];
            for j in 0..n {
                out[i * n + j] += vik * v[k * n + j];
            }
        }
    }
    out
}

/// Matrix logarithm of an SPD matrix.
pub fn logm(a: &[f64], n: usize) -> Vec<f64> {
    matrix_fn(a, n, |l| l.max(1e-12).ln())
}
/// Matrix exponential of a symmetric matrix.
pub fn expm(a: &[f64], n: usize) -> Vec<f64> {
    matrix_fn(a, n, |l| l.exp())
}
/// Matrix square root of an SPD matrix.
pub fn sqrtm(a: &[f64], n: usize) -> Vec<f64> {
    matrix_fn(a, n, |l| l.max(0.0).sqrt())
}
/// Inverse matrix square root of an SPD matrix (`A^{-1/2}`).
pub fn invsqrtm(a: &[f64], n: usize) -> Vec<f64> {
    matrix_fn(a, n, |l| 1.0 / l.max(1e-12).sqrt())
}

fn matmul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut o = vec![0f64; n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..n {
                o[i * n + j] += aik * b[k * n + j];
            }
        }
    }
    o
}

/// Squared AIRM (affine-invariant Riemannian) geodesic distance:
/// `δ²(A,B) = ‖log(A^{-1/2} B A^{-1/2})‖_F² = Σ log²(λ_i)` of `A^{-1/2} B A^{-1/2}`.
pub fn airm_dist2(a: &[f64], b: &[f64], n: usize) -> f64 {
    let w = invsqrtm(a, n);
    let m = matmul(&matmul(&w, b, n), &w, n);
    let (evals, _) = eigh(&m, n);
    evals.iter().map(|&l| l.max(1e-12).ln().powi(2)).sum()
}

/// Karcher / Fréchet mean of SPD matrices under the AIRM metric — iterative tangent-space
/// averaging: `M ← M^{1/2} · exp(mean_i log(M^{-1/2} C_i M^{-1/2})) · M^{1/2}`, initialised at the
/// arithmetic mean, until the mean tangent norm falls below `tol` (or `iters` reached).
pub fn karcher_mean(covs: &[Vec<f64>], n: usize, iters: usize, tol: f64) -> Vec<f64> {
    let k = covs.len().max(1) as f64;
    let mut m = vec![0f64; n * n];
    for c in covs {
        for i in 0..n * n {
            m[i] += c[i] / k;
        }
    }
    for _ in 0..iters {
        let msqrt = sqrtm(&m, n);
        let minv = invsqrtm(&m, n);
        let mut sbar = vec![0f64; n * n];
        for c in covs {
            let wcw = matmul(&matmul(&minv, c, n), &minv, n);
            let l = logm(&wcw, n);
            for i in 0..n * n {
                sbar[i] += l[i] / k;
            }
        }
        let e = expm(&sbar, n);
        m = matmul(&matmul(&msqrt, &e, n), &msqrt, n);
        let norm: f64 = sbar.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < tol {
            break;
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ident(n: usize) -> Vec<f64> {
        let mut a = vec![0f64; n * n];
        for i in 0..n {
            a[i * n + i] = 1.0;
        }
        a
    }

    #[test]
    fn logm_expm_roundtrip() {
        // A SPD; expm(logm(A)) ≈ A.
        let n = 4;
        let mut a = vec![0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = if i == j { 2.0 + i as f64 } else { 0.3 };
            }
        }
        let back = expm(&logm(&a, n), n);
        let err: f64 = a
            .iter()
            .zip(&back)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-6, "logm/expm roundtrip err {err}");
    }

    #[test]
    fn sqrtm_squares_back() {
        let n = 3;
        let a = vec![4.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 16.0];
        let s = sqrtm(&a, n);
        assert!(
            (s[0] - 2.0).abs() < 1e-9 && (s[4] - 3.0).abs() < 1e-9 && (s[8] - 4.0).abs() < 1e-9
        );
    }

    #[test]
    fn airm_dist_zero_to_self_and_invariant() {
        let n = 3;
        let a = vec![2.0, 0.1, 0.0, 0.1, 3.0, 0.2, 0.0, 0.2, 1.5];
        assert!(airm_dist2(&a, &a, n) < 1e-9, "distance to self must be 0");
        // Affine invariance: δ(A,I) = δ(kA,kI) for scalar k (both are congruence by √k·I).
        let k = 5.0;
        let ka: Vec<f64> = a.iter().map(|x| x * k).collect();
        let ki: Vec<f64> = ident(n).iter().map(|x| x * k).collect();
        let d1 = airm_dist2(&a, &ident(n), n);
        let d2 = airm_dist2(&ka, &ki, n);
        assert!(
            (d1 - d2).abs() < 1e-6,
            "AIRM not affine-invariant: {d1} vs {d2}"
        );
    }

    #[test]
    fn karcher_mean_of_identicals_is_the_matrix() {
        let n = 3;
        let a = vec![2.0, 0.1, 0.0, 0.1, 3.0, 0.2, 0.0, 0.2, 1.5];
        let m = karcher_mean(&[a.clone(), a.clone(), a.clone()], n, 20, 1e-10);
        let err: f64 = a
            .iter()
            .zip(&m)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-6, "Karcher mean of identicals err {err}");
    }
}
