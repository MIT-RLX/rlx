//! SPD-manifold (symmetric positive-definite) kernels — the core primitives of Riemannian
//! machine learning on covariance matrices (EEG BCI, diffusion tensors, Gaussian embeddings).
//!
//! Built on the `blas::dsyevd` symmetric eigendecomposition:
//!   - matrix functions `logm` / `expm` / `sqrtm` / `invsqrtm` (spectral: `V·diag(f(λ))·Vᵀ`),
//!     with rayon-parallel **batched** counterparts `logm_batch` / … (thousands of small
//!     covariances in one call — the shared seam a GPU spd backend can route through),
//!   - the affine-invariant Riemannian metric distance `airm_dist2`,
//!   - the Fréchet / Karcher mean `karcher_mean` and its `karcher_mean_weighted` barycentre
//!     (iterative tangent averaging; the weighted form is the true AIRM barycentre a
//!     barycentric-OT projection / soft-clustering / weighted-MDM needs),
//!   - the arbitrary-base AIRM `log_map` / `exp_map` and `parallel_transport` (the
//!     Riemannian OT / domain-adaptation primitives of Yair et al. 2019 — `geodesic_interp`
//!     is `exp_map(A, t·log_map(A, B))`),
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

/// `(A^{1/2}, A^{-1/2})` from a **single** eigendecomposition of SPD `A`. The
/// AIRM maps and their VJPs need both, so sharing the `eigh` (the dominant cost)
/// is ~a third cheaper than two separate `sqrtm`/`invsqrtm` calls.
fn sqrt_invsqrt(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let (w, v) = eigh(a, n);
    let sh: Vec<f64> = w.iter().map(|&l| l.max(0.0).sqrt()).collect();
    let ish: Vec<f64> = w.iter().map(|&l| 1.0 / l.max(1e-12).sqrt()).collect();
    (reconstruct(&sh, &v, n), reconstruct(&ish, &v, n))
}

/// Apply [`matrix_fn`] to a batch of symmetric matrices in parallel (rayon) — the
/// throughput counterpart of the scalar matrix functions. Each `mats[b]` is a
/// row-major `n×n` symmetric matrix; output order is preserved. Mirrors
/// [`eigh_batch`]: one call amortises setup across thousands of small covariances,
/// and the batched signature is the seam a metal/cuda/rocm spd backend routes through.
fn matrix_fn_batch(mats: &[Vec<f64>], n: usize, f: impl Fn(f64) -> f64 + Sync) -> Vec<Vec<f64>> {
    use rayon::prelude::*;
    mats.par_iter().map(|a| matrix_fn(a, n, &f)).collect()
}

/// Batched [`logm`] — rayon-parallel over `covs` (each row-major `n×n` SPD).
pub fn logm_batch(covs: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    matrix_fn_batch(covs, n, |l| l.max(1e-12).ln())
}
/// Batched [`expm`] — rayon-parallel over `mats` (each row-major `n×n` symmetric).
pub fn expm_batch(mats: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    matrix_fn_batch(mats, n, |l| l.exp())
}
/// Batched [`sqrtm`] — rayon-parallel over `covs` (each row-major `n×n` SPD).
pub fn sqrtm_batch(covs: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    matrix_fn_batch(covs, n, |l| l.max(0.0).sqrt())
}
/// Batched [`invsqrtm`] — rayon-parallel over `covs` (each row-major `n×n` SPD).
pub fn invsqrtm_batch(covs: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    matrix_fn_batch(covs, n, |l| 1.0 / l.max(1e-12).sqrt())
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

/// Weighted Karcher / Fréchet mean of SPD matrices under the AIRM metric — the
/// barycentre `argmin_M Σᵢ wᵢ · δ²(M, Cᵢ)`. Same iterative tangent-space averaging
/// as [`karcher_mean`], but each `Cᵢ` contributes with the **normalised** weight
/// `w̄ᵢ = wᵢ / Σⱼ wⱼ` in place of `1/k`:
///
/// ```text
///   M₀ = Σᵢ w̄ᵢ Cᵢ                                    (weighted arithmetic init)
///   S̄  = Σᵢ w̄ᵢ · log(M^{-1/2} Cᵢ M^{-1/2})            (weighted tangent mean)
///   M ← M^{1/2} · exp(S̄) · M^{1/2}                    until ‖S̄‖_F < tol (or `iters`)
/// ```
///
/// This is the exact barycentre that a barycentric OT projection (the Fréchet mean
/// of an entropic coupling), soft-clustering, or weighted MDM needs — the *true*
/// AIRM mean, not the log-Euclidean shortcut `expm(Σᵢ w̄ᵢ logm(Cᵢ))`. With uniform
/// weights it coincides with [`karcher_mean`]; results depend only on the weight
/// *ratios* (an all-zero weight vector falls back to a uniform mean). `weights`
/// must be non-negative and align 1:1 with `covs`.
pub fn karcher_mean_weighted(
    covs: &[Vec<f64>],
    weights: &[f64],
    n: usize,
    iters: usize,
    tol: f64,
) -> Vec<f64> {
    assert_eq!(
        covs.len(),
        weights.len(),
        "karcher_mean_weighted: {} covs but {} weights",
        covs.len(),
        weights.len()
    );
    // Normalise to a convex combination (Σ w̄ᵢ = 1); degenerate all-zero → uniform.
    let wsum: f64 = weights.iter().sum();
    let wbar: Vec<f64> = if wsum.abs() < 1e-300 {
        let u = 1.0 / covs.len().max(1) as f64;
        vec![u; covs.len()]
    } else {
        weights.iter().map(|&w| w / wsum).collect()
    };
    // Weighted arithmetic mean as the initial guess.
    let mut m = vec![0f64; n * n];
    for (c, &w) in covs.iter().zip(&wbar) {
        for i in 0..n * n {
            m[i] += w * c[i];
        }
    }
    for _ in 0..iters {
        let msqrt = sqrtm(&m, n);
        let minv = invsqrtm(&m, n);
        let mut sbar = vec![0f64; n * n];
        for (c, &w) in covs.iter().zip(&wbar) {
            let wcw = matmul(&matmul(&minv, c, n), &minv, n);
            let l = logm(&wcw, n);
            for i in 0..n * n {
                sbar[i] += w * l[i];
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

// ── SPDNet / Riemannian layer kernels ────────────────────────────
//
// Forward + backward for the BiMap / ReEig / LogEig layers of SPDNet
// (Huang & Van Gool, AAAI 2017) and the SPD batch-norm transport of
// Brooks et al. (NeurIPS 2019). These are the compute bodies the core
// `Op::BiMap` / `Op::ReEig` / `Op::LogEig` / `Op::SpdBatchNorm` CPU
// thunks call — kept here (host-testable, LAPACK-backed) so the thunk
// layer stays a thin arena-offset shim.

/// Transpose an `n×n` row-major matrix.
fn transpose(a: &[f64], n: usize) -> Vec<f64> {
    let mut t = vec![0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            t[j * n + i] = a[i * n + j];
        }
    }
    t
}

/// Symmetrize an `n×n` matrix in place-style: `½(A + Aᵀ)`.
fn symmetrize(a: &[f64], n: usize) -> Vec<f64> {
    let mut o = vec![0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            o[i * n + j] = 0.5 * (a[i * n + j] + a[j * n + i]);
        }
    }
    o
}

/// BiMap (bilinear mapping) layer forward: `Y = W · X · Wᵀ`.
/// `W` is `[m, n]`, `X` is `[n, n]` (symmetric SPD), `Y` is `[m, m]`.
/// Maps `SPD_n → SPD_m` when `W` has full row rank (the Stiefel /
/// semi-orthogonality constraint on `W` is enforced by the optimizer,
/// not here).
pub fn bimap(w: &[f64], x: &[f64], m: usize, n: usize) -> Vec<f64> {
    // T = W·X  [m, n]
    let mut t = vec![0f64; m * n];
    crate::blas::dgemm(w, x, &mut t, m, n, n);
    // Y = T·Wᵀ  [m, m]
    let mut wt = vec![0f64; n * m];
    for i in 0..m {
        for j in 0..n {
            wt[j * m + i] = w[i * n + j];
        }
    }
    let mut y = vec![0f64; m * m];
    crate::blas::dgemm(&t, &wt, &mut y, m, n, m);
    y
}

/// ReEig (eigenvalue rectification) forward: `Y = U · max(ε, Σ) · Uᵀ`
/// where `X = U Σ Uᵀ`. The SPD analogue of ReLU — floors the spectrum
/// at `eps` to keep the output SPD and well-conditioned.
pub fn reeig(x: &[f64], n: usize, eps: f64) -> Vec<f64> {
    matrix_fn(x, n, |l| l.max(eps))
}

/// LogEig forward: `Y = U · log(Σ) · Uᵀ = logm(X)`. Maps the SPD
/// manifold to the tangent space at the identity (symmetric matrices)
/// so a Euclidean classifier can consume it. `eps` floors the spectrum
/// before the log for numerical safety.
pub fn logeig(x: &[f64], n: usize, eps: f64) -> Vec<f64> {
    matrix_fn(x, n, |l| l.max(eps).ln())
}

/// Adjoint (reverse-mode VJP) of a symmetric matrix function
/// `Y = U f(Σ) Uᵀ` via the Daleckii–Krein / Loewner formula:
///
/// ```text
///   (λ, U) = eigh(X)                       // U columns = eigenvectors
///   Ḡ      = ½(dY + dYᵀ)                   // project to symmetric cotangent
///   C      = Uᵀ · Ḡ · U
///   P[a,b] = (f(λ_a) − f(λ_b)) / (λ_a − λ_b)  (a≠b, else f′(λ_a))   // Loewner
///   dX     = sym( U · (P ⊙ C) · Uᵀ )
/// ```
///
/// `P` is symmetric so the differential is self-adjoint; the same
/// kernel serves ReEig (`f = max(·,ε)`), LogEig (`f = log`), and the
/// matrix-√ used by SPD batch-norm's `G` gradient.
pub fn spectral_backward(
    x: &[f64],
    dy: &[f64],
    n: usize,
    f: impl Fn(f64) -> f64,
    fprime: impl Fn(f64) -> f64,
) -> Vec<f64> {
    let (lam, v) = eigh(x, n);
    spectral_backward_precomputed(&lam, &v, dy, n, f, fprime)
}

/// Square `n×n` matmul via BLAS `dgemm` (Accelerate/MKL/OpenBLAS).
fn mm(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut o = vec![0f64; n * n];
    crate::blas::dgemm(a, b, &mut o, n, n, n);
    o
}

/// Reconstruct `Σ_k fk[k] · u_k u_kᵀ` from row-major eigenvectors `v`
/// (row `k` = eigenvector `k`, matching [`eigh`]) and per-eigenvalue
/// weights `fk`. Computes `(diag(fk)·V)ᵀ · V` via `dgemm`.
fn reconstruct(fk: &[f64], v: &[f64], n: usize) -> Vec<f64> {
    let mut sv = vec![0f64; n * n];
    for k in 0..n {
        for i in 0..n {
            sv[k * n + i] = fk[k] * v[k * n + i];
        }
    }
    mm(&transpose(&sv, n), v, n)
}

/// Forward of a symmetric matrix function `Y = U f(Σ) Uᵀ`, packed as
/// `[Y (n²), λ (n), U (n²)]` so the backward can **reuse** the
/// eigendecomposition instead of recomputing it. `U` is row-major
/// (row `k` = eigenvector `k`), matching [`eigh`]. Consumed by the
/// `Op::ReEig` / `Op::LogEig` CPU kernels.
pub fn spectral_forward_packed(x: &[f64], n: usize, f: impl Fn(f64) -> f64) -> Vec<f64> {
    let (lam, v) = eigh(x, n);
    let fk: Vec<f64> = lam.iter().map(|&l| f(l)).collect();
    let y = reconstruct(&fk, &v, n);
    let mut out = vec![0f64; 2 * n * n + n];
    out[..n * n].copy_from_slice(&y);
    out[n * n..n * n + n].copy_from_slice(&lam);
    out[n * n + n..].copy_from_slice(&v);
    out
}

/// [`spectral_backward`] with a **precomputed** eigendecomposition
/// (`lam`, `v` = row-major eigenvectors). This is the hot path: the
/// forward already produced `(λ, U)`, so the backward skips a redundant
/// `eigh`. All three matrix products go through `dgemm`.
pub fn spectral_backward_precomputed(
    lam: &[f64],
    v: &[f64],
    dy: &[f64],
    n: usize,
    f: impl Fn(f64) -> f64,
    fprime: impl Fn(f64) -> f64,
) -> Vec<f64> {
    let g = symmetrize(dy, n);
    let vt = transpose(v, n);
    // C = Uᵀ·Ḡ·U = V·Ḡ·Vᵀ   (V row-major = rows-are-eigenvectors).
    let c = mm(&mm(v, &g, n), &vt, n);
    // Loewner-weighted middle matrix M = P ⊙ C.
    let fl: Vec<f64> = lam.iter().map(|&l| f(l)).collect();
    let scale = lam.iter().fold(0f64, |acc, &l| acc.max(l.abs())).max(1.0);
    let tol = scale * 1e-9;
    let mut mmat = vec![0f64; n * n];
    for a in 0..n {
        for b in 0..n {
            let d = lam[a] - lam[b];
            let p = if d.abs() > tol {
                (fl[a] - fl[b]) / d
            } else {
                fprime(lam[a])
            };
            mmat[a * n + b] = p * c[a * n + b];
        }
    }
    // dX = U·M·Uᵀ = Vᵀ·M·V.
    let dx = mm(&mm(&vt, &mmat, n), v, n);
    symmetrize(&dx, n)
}

/// ReEig backward: `dX` for `Y = U max(ε, Σ) Uᵀ`. `f(λ)=max(λ,ε)`,
/// `f′(λ)=[λ>ε]` (the straddling case is handled exactly by the
/// Loewner divided difference).
pub fn reeig_backward(x: &[f64], dy: &[f64], n: usize, eps: f64) -> Vec<f64> {
    let (lam, v) = eigh(x, n);
    reeig_backward_precomputed(&lam, &v, dy, n, eps)
}

/// [`reeig_backward`] with a precomputed eigendecomposition.
pub fn reeig_backward_precomputed(
    lam: &[f64],
    v: &[f64],
    dy: &[f64],
    n: usize,
    eps: f64,
) -> Vec<f64> {
    spectral_backward_precomputed(
        lam,
        v,
        dy,
        n,
        |l| l.max(eps),
        |l| {
            if l > eps { 1.0 } else { 0.0 }
        },
    )
}

/// LogEig backward: `dX` for `Y = logm(X)`. `f=log`, `f′=1/λ`.
pub fn logeig_backward(x: &[f64], dy: &[f64], n: usize, eps: f64) -> Vec<f64> {
    let (lam, v) = eigh(x, n);
    logeig_backward_precomputed(&lam, &v, dy, n, eps)
}

/// [`logeig_backward`] with a precomputed eigendecomposition.
pub fn logeig_backward_precomputed(
    lam: &[f64],
    v: &[f64],
    dy: &[f64],
    n: usize,
    eps: f64,
) -> Vec<f64> {
    spectral_backward_precomputed(lam, v, dy, n, |l| l.max(eps).ln(), |l| 1.0 / l.max(eps))
}

/// SPD batch-norm transport forward. For a batch of SPD matrices
/// `X_i` (`x` packed `[batch, n, n]`), a (detached) Fréchet-mean
/// `mean` `[n,n]`, and a learnable SPD bias `g` `[n,n]`:
///
/// ```text
///   Y_i = G^{1/2} · (M^{-1/2} X_i M^{-1/2}) · G^{1/2}
/// ```
///
/// i.e. parallel-transport each `X_i` to the identity (centering by the
/// batch mean) then bias toward `G`. `eps` floors the spectra of `M`/`G`.
pub fn spd_bn_transport(
    x: &[f64],
    mean: &[f64],
    g: &[f64],
    batch: usize,
    n: usize,
    eps: f64,
) -> Vec<f64> {
    use rayon::prelude::*;
    let ms = matrix_fn(mean, n, |l| 1.0 / l.max(eps).sqrt()); // M^{-1/2}
    let gs = matrix_fn(g, n, |l| l.max(eps).sqrt()); // G^{1/2}
    // Per-slice work is independent → rayon over the batch (the hot
    // path: thousands of small covariances). Each slice's matmuls go
    // through BLAS dgemm.
    let slices: Vec<Vec<f64>> = (0..batch)
        .into_par_iter()
        .map(|bi| {
            let xi = &x[bi * n * n..(bi + 1) * n * n];
            let ci = mm(&mm(&ms, xi, n), &ms, n); // Ms Xi Ms
            mm(&mm(&gs, &ci, n), &gs, n) // Gs Ci Gs
        })
        .collect();
    slices.concat()
}

/// SPD batch-norm transport backward w.r.t. the inputs `X_i`
/// (mean detached). With `Y_i = P X_i Pᵀ`, `P = G^{1/2} M^{-1/2}`,
/// `dX_i = Pᵀ · Ḡ_i · P = M^{-1/2} G^{1/2} Ḡ_i G^{1/2} M^{-1/2}`.
pub fn spd_bn_backward_x(
    mean: &[f64],
    g: &[f64],
    dy: &[f64],
    batch: usize,
    n: usize,
    eps: f64,
) -> Vec<f64> {
    use rayon::prelude::*;
    let ms = matrix_fn(mean, n, |l| 1.0 / l.max(eps).sqrt());
    let gs = matrix_fn(g, n, |l| l.max(eps).sqrt());
    let q = mm(&ms, &gs, n); // Q = Ms Gs  (= Pᵀ)
    let qt = transpose(&q, n); // Qᵀ = Gs Ms
    let slices: Vec<Vec<f64>> = (0..batch)
        .into_par_iter()
        .map(|bi| {
            let gi = &dy[bi * n * n..(bi + 1) * n * n];
            symmetrize(&mm(&mm(&q, gi, n), &qt, n), n) // sym(Q Ḡ_i Qᵀ)
        })
        .collect();
    slices.concat()
}

/// SPD batch-norm transport backward w.r.t. the learnable bias `G`
/// (mean detached). `dL/dG^{1/2} = Σ_i (Ḡ_i G^{1/2} C_i + C_i G^{1/2} Ḡ_i)`
/// with `C_i = M^{-1/2} X_i M^{-1/2}`, then pushed through the matrix-√
/// VJP (`spectral_backward` with `f=√`, `f′=1/(2√)`).
pub fn spd_bn_backward_g(
    x: &[f64],
    mean: &[f64],
    g: &[f64],
    dy: &[f64],
    batch: usize,
    n: usize,
    eps: f64,
) -> Vec<f64> {
    use rayon::prelude::*;
    let ms = matrix_fn(mean, n, |l| 1.0 / l.max(eps).sqrt());
    let gs = matrix_fn(g, n, |l| l.max(eps).sqrt());
    // Map each slice to its dL/dGs contribution, then reduce-sum.
    let dgs = (0..batch)
        .into_par_iter()
        .map(|bi| {
            let xi = &x[bi * n * n..(bi + 1) * n * n];
            let gi = &dy[bi * n * n..(bi + 1) * n * n];
            let ci = mm(&mm(&ms, xi, n), &ms, n); // C_i (symmetric)
            let t1 = mm(&mm(gi, &gs, n), &ci, n); // Ḡ_i Gs C_i
            let t2 = mm(&mm(&ci, &gs, n), gi, n); // C_i Gs Ḡ_i
            let mut s = t1;
            for k in 0..n * n {
                s[k] += t2[k];
            }
            s
        })
        .reduce(
            || vec![0f64; n * n],
            |mut a, b| {
                for k in 0..n * n {
                    a[k] += b[k];
                }
                a
            },
        );
    // Push dL/dGs through Gs = G^{1/2} (reuses eigh of G once).
    spectral_backward(
        g,
        &dgs,
        n,
        |l| l.max(eps).sqrt(),
        |l| 0.5 / l.max(eps).sqrt(),
    )
}

/// Geodesic interpolation between two SPD matrices under the AIRM
/// metric: `A #_t B = A^{1/2} (A^{-1/2} B A^{-1/2})^t A^{1/2}`. Used by
/// the trainer to update the running Fréchet mean of SPD batch-norm
/// (`running ← running #_momentum batch_mean`) — a non-differentiable
/// buffer update, mirroring how Euclidean batch-norm tracks running
/// stats outside the autodiff graph.
pub fn geodesic_interp(a: &[f64], b: &[f64], t: f64, n: usize) -> Vec<f64> {
    let (asqrt, ainv) = sqrt_invsqrt(a, n);
    let m = matmul(&matmul(&ainv, b, n), &ainv, n);
    let mt = matrix_fn(&m, n, |l| l.max(1e-12).powf(t));
    matmul(&matmul(&asqrt, &mt, n), &asqrt, n)
}

/// AIRM Riemannian **logarithm** at base point `base`: maps the SPD point `x` to
/// the tangent vector `V ∈ T_base` whose geodesic reaches `x`:
///
/// ```text
///   Log_P(X) = P^{1/2} · logm(P^{-1/2} X P^{-1/2}) · P^{1/2}
/// ```
///
/// The arbitrary-base generalisation of [`logm`] (which is `Log_I`, the
/// log-Euclidean tangent at the identity), and the inverse of [`exp_map`]. Its
/// AIRM norm equals the geodesic distance, `‖Log_P(X)‖_P = δ_AIRM(P, X)`. `base`
/// and `x` are row-major `n×n` SPD; the result is symmetric.
pub fn log_map(base: &[f64], x: &[f64], n: usize) -> Vec<f64> {
    let (phalf, pinv) = sqrt_invsqrt(base, n);
    let m = matmul(&matmul(&pinv, x, n), &pinv, n); // P^{-1/2} X P^{-1/2}
    let lm = logm(&m, n);
    matmul(&matmul(&phalf, &lm, n), &phalf, n)
}

/// AIRM Riemannian **exponential** at base point `base`: maps the tangent vector
/// `v ∈ T_base` to the SPD point reached by the geodesic leaving `base` with
/// initial velocity `v`:
///
/// ```text
///   Exp_P(V) = P^{1/2} · expm(P^{-1/2} V P^{-1/2}) · P^{1/2}
/// ```
///
/// Inverse of [`log_map`] (`Exp_P(Log_P(X)) = X`) and the arbitrary-base
/// generalisation of [`expm`] (`Exp_I`). Note `geodesic_interp(A, B, t)` is
/// exactly `exp_map(A, t · log_map(A, B))`. `base` is SPD and `v` symmetric
/// (row-major `n×n`); the result is SPD.
pub fn exp_map(base: &[f64], v: &[f64], n: usize) -> Vec<f64> {
    let (phalf, pinv) = sqrt_invsqrt(base, n);
    let m = matmul(&matmul(&pinv, v, n), &pinv, n); // P^{-1/2} V P^{-1/2}
    let em = expm(&m, n);
    matmul(&matmul(&phalf, &em, n), &phalf, n)
}

/// AIRM **parallel transport** of a tangent vector `v ∈ T_from` to `T_to` along
/// the connecting geodesic:
///
/// ```text
///   Γ_{P→Q}(V) = E · V · Eᵀ,   E = (Q P^{-1})^{1/2} = P^{1/2} (P^{-1/2} Q P^{-1/2})^{1/2} P^{-1/2}
/// ```
///
/// The right-hand symmetric-SPD form is what's computed — every factor is a
/// matrix function of an SPD matrix, so no non-symmetric eigendecomposition of
/// `Q P^{-1}` is needed. Transport is an AIRM isometry (it preserves the metric
/// inner product), making it the operator that moves tangent vectors between base
/// points for Riemannian OT / domain adaptation (Yair et al., 2019) and SPD
/// batch-norm. `from`/`to` are SPD and `v` symmetric (row-major `n×n`); the result
/// is a symmetric tangent vector in `T_to`.
pub fn parallel_transport(from: &[f64], to: &[f64], v: &[f64], n: usize) -> Vec<f64> {
    let (phalf, pinv) = sqrt_invsqrt(from, n);
    let wq = matmul(&matmul(&pinv, to, n), &pinv, n); // P^{-1/2} Q P^{-1/2}  (SPD)
    let wq_half = sqrtm(&wq, n);
    let e = matmul(&matmul(&phalf, &wq_half, n), &pinv, n); // E = P^{1/2} (…)^{1/2} P^{-1/2}
    let et = transpose(&e, n);
    let evet = matmul(&matmul(&e, v, n), &et, n);
    symmetrize(&evet, n) // E V Eᵀ is symmetric for symmetric V — kill roundoff drift
}

// ── Riemannian map VJPs (reverse-mode) ───────────────────────────
//
// Analytic adjoints for `log_map` / `exp_map` / `parallel_transport`, so the
// AIRM geometric maps are differentiable end-to-end. Every gradient reduces to
// matmuls plus the Daleckii–Krein spectral adjoint [`spectral_backward`] of the
// three matrix functions involved (√, ·^{-1/2}, and log or exp). Base points get
// gradients too (they enter through `P^{1/2}` and `P^{-1/2}`), so a learned base
// trains correctly — not just the moving argument. All are finite-difference
// checked in the tests.

/// `A + B` for two row-major `n×n` matrices.
fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

// Spectral `(f, f′)` pairs used by the map VJPs — floored to match the forward
// maps. Named fns (not closures) so a cached eigendecomposition can be reused by
// both `reconstruct_fn` (forward) and `spectral_backward_precomputed` (adjoint).
fn sqrt_ff(l: f64) -> f64 {
    l.max(0.0).sqrt()
}
fn sqrt_fp(l: f64) -> f64 {
    0.5 / l.max(1e-12).sqrt()
}
fn invsqrt_ff(l: f64) -> f64 {
    1.0 / l.max(1e-12).sqrt()
}
fn invsqrt_fp(l: f64) -> f64 {
    -0.5 * l.max(1e-12).powf(-1.5)
}
fn log_ff(l: f64) -> f64 {
    l.max(1e-12).ln()
}
fn log_fp(l: f64) -> f64 {
    1.0 / l.max(1e-12)
}

/// `V·diag(f(λ))·Vᵀ` from a precomputed eigendecomposition `(lam, v)` (row-major
/// eigenvectors, as [`eigh`]) — the matrix-function partner of
/// [`spectral_backward_precomputed`], so one cached `eigh` drives both the value
/// and its adjoint.
fn reconstruct_fn(lam: &[f64], v: &[f64], n: usize, f: impl Fn(f64) -> f64) -> Vec<f64> {
    let fk: Vec<f64> = lam.iter().map(|&l| f(l)).collect();
    reconstruct(&fk, v, n)
}

/// VJP of [`log_map`]: given the cotangent `dy` on `Log_P(X)`, returns
/// `(∂L/∂base, ∂L/∂x)`. With `A = P^{1/2}`, `W = P^{-1/2}`, `M = W X W`:
/// `X̄ = W · DK_log(M, A ḡ A) · W`, and the base receives the √ / ·^{-1/2}
/// adjoints of `Ā = ḡAL + (ḡAL)ᵀ` and `W̄ = M̄WX + (M̄WX)ᵀ`. The two SPD
/// eigendecompositions (`base`, `M`) are each computed once and reused for their
/// matrix functions *and* Daleckii–Krein adjoints.
pub fn log_map_backward(base: &[f64], x: &[f64], dy: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let (lam_b, v_b) = eigh(base, n);
    let a = reconstruct_fn(&lam_b, &v_b, n, sqrt_ff); // P^{1/2}
    let w = reconstruct_fn(&lam_b, &v_b, n, invsqrt_ff); // P^{-1/2}
    let m = matmul(&matmul(&w, x, n), &w, n); // M = W X W
    let (lam_m, v_m) = eigh(&m, n);
    let g = symmetrize(dy, n);
    let lbar = matmul(&matmul(&a, &g, n), &a, n); // L̄ = A ḡ A
    let mbar = spectral_backward_precomputed(&lam_m, &v_m, &lbar, n, log_ff, log_fp);
    let d_x = symmetrize(&matmul(&matmul(&w, &mbar, n), &w, n), n); // W M̄ W
    // Base gradient: A = P^{1/2} and W = P^{-1/2} both depend on P.
    let l = reconstruct_fn(&lam_m, &v_m, n, log_ff); // logm(M)
    let gal = matmul(&matmul(&g, &a, n), &l, n); // ḡ A L
    let abar = add(&gal, &transpose(&gal, n)); // Ā = ḡAL + (ḡAL)ᵀ
    let mwx = matmul(&matmul(&mbar, &w, n), x, n); // M̄ W X
    let wbar = add(&mwx, &transpose(&mwx, n)); // W̄ = M̄WX + (M̄WX)ᵀ
    let d_base_a = spectral_backward_precomputed(&lam_b, &v_b, &abar, n, sqrt_ff, sqrt_fp);
    let d_base_w = spectral_backward_precomputed(&lam_b, &v_b, &wbar, n, invsqrt_ff, invsqrt_fp);
    let d_base = symmetrize(&add(&d_base_a, &d_base_w), n);
    (d_base, d_x)
}

/// VJP of [`exp_map`]: given the cotangent `dy` on `Exp_P(V)`, returns
/// `(∂L/∂base, ∂L/∂v)`. Identical structure to [`log_map_backward`] with the
/// `exp` spectral adjoint (and `E = expm(M)` in the base term) instead of `log`.
pub fn exp_map_backward(base: &[f64], v: &[f64], dy: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let (lam_b, v_b) = eigh(base, n);
    let a = reconstruct_fn(&lam_b, &v_b, n, sqrt_ff);
    let w = reconstruct_fn(&lam_b, &v_b, n, invsqrt_ff);
    let m = matmul(&matmul(&w, v, n), &w, n); // M = W V W
    let (lam_m, v_m) = eigh(&m, n);
    let g = symmetrize(dy, n);
    let ebar = matmul(&matmul(&a, &g, n), &a, n); // Ē = A ḡ A
    let mbar = spectral_backward_precomputed(&lam_m, &v_m, &ebar, n, f64::exp, f64::exp);
    let d_v = symmetrize(&matmul(&matmul(&w, &mbar, n), &w, n), n); // W M̄ W
    let e = reconstruct_fn(&lam_m, &v_m, n, f64::exp); // expm(M)
    let gae = matmul(&matmul(&g, &a, n), &e, n); // ḡ A E
    let abar = add(&gae, &transpose(&gae, n));
    let mwv = matmul(&matmul(&mbar, &w, n), v, n); // M̄ W V
    let wbar = add(&mwv, &transpose(&mwv, n));
    let d_base_a = spectral_backward_precomputed(&lam_b, &v_b, &abar, n, sqrt_ff, sqrt_fp);
    let d_base_w = spectral_backward_precomputed(&lam_b, &v_b, &wbar, n, invsqrt_ff, invsqrt_fp);
    let d_base = symmetrize(&add(&d_base_a, &d_base_w), n);
    (d_base, d_v)
}

/// VJP of [`parallel_transport`]: given the cotangent `dy` on `Γ_{P→Q}(V)`,
/// returns `(∂L/∂from, ∂L/∂to, ∂L/∂v)`. With `E = A S W`, `S = (W Q W)^{1/2}`,
/// `A = P^{1/2}`, `W = P^{-1/2}`: `V̄ = Eᵀ ḡ E`, `Q̄ = W · DK_√(M_q, S̄) · W`,
/// and `P̄` collects the √ / ·^{-1/2} adjoints of the `A`/`W` cotangents. The
/// `from` and `M_q` eigendecompositions are each computed once and reused.
pub fn parallel_transport_backward(
    from: &[f64],
    to: &[f64],
    v: &[f64],
    dy: &[f64],
    n: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (lam_f, v_f) = eigh(from, n);
    let a = reconstruct_fn(&lam_f, &v_f, n, sqrt_ff); // P^{1/2}
    let w = reconstruct_fn(&lam_f, &v_f, n, invsqrt_ff); // P^{-1/2}
    let mq = matmul(&matmul(&w, to, n), &w, n); // M_q = W Q W
    let (lam_mq, v_mq) = eigh(&mq, n);
    let s = reconstruct_fn(&lam_mq, &v_mq, n, sqrt_ff); // S = M_q^{1/2}
    let e = matmul(&matmul(&a, &s, n), &w, n); // E = A S W
    let et = transpose(&e, n);
    let g = symmetrize(dy, n);
    // grad_v = Eᵀ ḡ E.
    let d_v = symmetrize(&matmul(&matmul(&et, &g, n), &e, n), n);
    // Ē = 2 ḡ E V (cotangent on the non-symmetric E).
    let gev = matmul(&matmul(&g, &e, n), v, n);
    let ebar: Vec<f64> = gev.iter().map(|x| 2.0 * x).collect();
    // E = A S W: split Ē across A, S, W.
    let abar = matmul(&matmul(&ebar, &w, n), &s, n); // Ā = Ē (SW)ᵀ = Ē W S
    let sbar = matmul(&matmul(&a, &ebar, n), &w, n); // S̄ = A Ē W
    let wbar1 = matmul(&matmul(&s, &a, n), &ebar, n); // W̄₁ = (AS)ᵀ Ē = S A Ē
    // S = (M_q)^{1/2}: adjoint back to M_q (reusing its eigendecomp), then Q, W.
    let mqbar = spectral_backward_precomputed(&lam_mq, &v_mq, &sbar, n, sqrt_ff, sqrt_fp);
    let d_to = symmetrize(&matmul(&matmul(&w, &mqbar, n), &w, n), n); // Q̄ = W M̄_q W
    let mqwq = matmul(&matmul(&mqbar, &w, n), to, n); // M̄_q W Q
    let wbar = add(&wbar1, &add(&mqwq, &transpose(&mqwq, n))); // W̄ = W̄₁ + (M̄_qWQ + ·ᵀ)
    let d_from_a = spectral_backward_precomputed(&lam_f, &v_f, &abar, n, sqrt_ff, sqrt_fp);
    let d_from_w = spectral_backward_precomputed(&lam_f, &v_f, &wbar, n, invsqrt_ff, invsqrt_fp);
    let d_from = symmetrize(&add(&d_from_a, &d_from_w), n);
    (d_from, d_to, d_v)
}

/// VJP of [`logm_batch`]/[`expm_batch`]/[`sqrtm_batch`]/[`invsqrtm_batch`]:
/// per-slice Daleckii–Krein adjoint of the chosen matrix function. `x`/`dy` are
/// `[batch, n, n]`; `(f, fprime)` is the scalar spectral function and its
/// derivative (matching the batch forward). Rayon-parallel over the batch.
pub fn matrix_fn_batch_backward(
    x: &[f64],
    dy: &[f64],
    n: usize,
    f: impl Fn(f64) -> f64 + Sync,
    fprime: impl Fn(f64) -> f64 + Sync,
) -> Vec<f64> {
    use rayon::prelude::*;
    let batch = x.len() / (n * n);
    (0..batch)
        .into_par_iter()
        .flat_map(|bi| {
            let xi = &x[bi * n * n..(bi + 1) * n * n];
            let dyi = &dy[bi * n * n..(bi + 1) * n * n];
            spectral_backward(xi, dyi, n, &f, &fprime)
        })
        .collect()
}

// ── Symmetric eigendecomposition as a differentiable op ──────────
//
// `eigh` exposed as a first-class op: forward packs `[λ ∥ U]` (ascending λ,
// `U` with column j = eigenvector j, so `A = U diag(λ) Uᵀ`), and the reverse
// mode is the standard symmetric-eigendecomposition adjoint. This is the
// differentiable primitive the SPD spectral functions can be built on, and the
// seam a native batched eigensolver (cuSOLVER `syevjBatched` &c.) slots into.

/// Symmetric eigendecomposition packed as `[λ (n) ∥ U (n²)]` (row-major `U`,
/// column `j` = eigenvector `j`, `A = U diag(λ) Uᵀ`; `λ` ascending).
pub fn eigh_packed(a: &[f64], n: usize) -> Vec<f64> {
    let (w, vrow) = eigh(a, n); // vrow: row k = eigenvector k  (= Uᵀ)
    let u = transpose(&vrow, n); // column j = eigenvector j
    let mut out = vec![0f64; n + n * n];
    out[..n].copy_from_slice(&w);
    out[n..].copy_from_slice(&u);
    out
}

/// VJP of the symmetric eigendecomposition. `fwd = [λ ∥ U]` (from
/// [`eigh_packed`]); `bar = [λ̄ ∥ Ū]` is the upstream cotangent (same layout).
/// Returns the symmetric `Ā [n,n]`:
///
/// ```text
///   Ā = sym( U (diag(λ̄) + F ∘ (Uᵀ Ū)) Uᵀ ),   F_ij = 1/(λ_j − λ_i)  (i≠j)
/// ```
///
/// Degenerate pairs (`|λ_j − λ_i| ≤ tol`) zero the off-diagonal coupling — the
/// eigenvector gradient is undefined across a repeated eigenvalue.
pub fn eigh_backward_packed(fwd: &[f64], bar: &[f64], n: usize) -> Vec<f64> {
    let lam = &fwd[..n];
    let u = &fwd[n..];
    let lbar = &bar[..n];
    let ubar = &bar[n..];
    let ut = transpose(u, n);
    let c = mm(&ut, ubar, n); // C = Uᵀ Ū
    let scale = lam.iter().fold(0f64, |a, &l| a.max(l.abs())).max(1.0);
    let tol = scale * 1e-9;
    let mut mid = vec![0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            mid[i * n + j] = if i == j {
                lbar[i]
            } else {
                let d = lam[j] - lam[i];
                if d.abs() > tol { c[i * n + j] / d } else { 0.0 }
            };
        }
    }
    let abar = mm(&mm(u, &mid, n), &ut, n); // U · mid · Uᵀ
    symmetrize(&abar, n)
}

/// Batched [`eigh_packed`] — rayon-parallel over `mats` (each row-major `n×n`
/// symmetric). Output slice `b` is the packed `[λ ∥ U]` for `mats[b]`.
pub fn eigh_batch_packed(mats: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    use rayon::prelude::*;
    mats.par_iter().map(|a| eigh_packed(a, n)).collect()
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

    // ── SPDNet layer kernels ─────────────────────────────────────

    /// Deterministic symmetric matrix (no RNG — reproducible).
    fn sym(n: usize, seed: f64) -> Vec<f64> {
        let mut a = vec![0f64; n * n];
        for i in 0..n {
            for j in i..n {
                let v = ((i as f64 * 3.0 + j as f64 * 1.7 + seed).sin()) * 0.5;
                a[i * n + j] = v;
                a[j * n + i] = v;
            }
        }
        a
    }

    /// Deterministic SPD matrix `MᵀM + (n+1)·I` (well-conditioned).
    fn spd(n: usize, seed: f64) -> Vec<f64> {
        let m = sym(n, seed);
        let mut a = matmul(&m, &m, n); // MᵀM = M·M (M symmetric) — PSD
        for i in 0..n {
            a[i * n + i] += (n + 1) as f64;
        }
        a
    }

    fn frob(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn bimap_matches_manual() {
        let (m, n) = (2, 3);
        // W [2,3], X [3,3] SPD.
        let w = vec![1.0, 0.5, -0.25, 2.0, -1.0, 0.75];
        let x = spd(n, 0.3);
        let y = bimap(&w, &x, m, n);
        // Manual W·X·Wᵀ.
        let mut wx = vec![0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += w[i * n + k] * x[k * n + j];
                }
                wx[i * n + j] = s;
            }
        }
        let mut expect = vec![0f64; m * m];
        for i in 0..m {
            for j in 0..m {
                let mut s = 0.0;
                for k in 0..n {
                    s += wx[i * n + k] * w[j * n + k];
                }
                expect[i * m + j] = s;
            }
        }
        let err = y
            .iter()
            .zip(&expect)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-10, "bimap err {err}");
    }

    #[test]
    fn reeig_floors_spectrum() {
        let n = 4;
        // A diagonal-ish SPD with a small eigenvalue.
        let mut x = vec![0f64; n * n];
        for i in 0..n {
            x[i * n + i] = [0.01, 0.5, 2.0, 5.0][i];
        }
        let eps = 0.1;
        let y = reeig(&x, n, eps);
        let (evals, _) = eigh(&y, n);
        for &l in &evals {
            assert!(l >= eps - 1e-9, "reeig eigenvalue {l} below eps {eps}");
        }
        // The 0.01 eigenvalue must be lifted to eps; others unchanged.
        assert!((y[0] - eps).abs() < 1e-9, "smallest not floored: {}", y[0]);
        assert!((y[n + 1] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn logeig_expm_roundtrip() {
        let n = 4;
        let x = spd(n, 1.1);
        let back = expm(&logeig(&x, n, 1e-12), n);
        let err = x
            .iter()
            .zip(&back)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-8, "expm(logeig) roundtrip err {err}");
    }

    /// Finite-difference gradient check for a symmetric matrix function.
    /// `L(X) = <C, layer(X)>`, so `dL/dX = layer_backward(X, C)`. Compare
    /// its directional derivative along a symmetric `V` to central FD.
    fn fd_spectral_check(
        layer: impl Fn(&[f64], usize) -> Vec<f64>,
        backward: impl Fn(&[f64], &[f64], usize) -> Vec<f64>,
        n: usize,
        seed: f64,
    ) {
        let x = spd(n, seed);
        let c = sym(n, seed + 10.0); // cotangent dL/dY
        let v = sym(n, seed + 20.0); // symmetric direction
        let grad = backward(&x, &c, n);
        let analytic = frob(&grad, &v);
        let h = 1e-6;
        let xp: Vec<f64> = x.iter().zip(&v).map(|(a, b)| a + h * b).collect();
        let xm: Vec<f64> = x.iter().zip(&v).map(|(a, b)| a - h * b).collect();
        let fd = (frob(&c, &layer(&xp, n)) - frob(&c, &layer(&xm, n))) / (2.0 * h);
        let err = (analytic - fd).abs() / (fd.abs() + 1e-6);
        assert!(
            err < 1e-4,
            "spectral backward FD mismatch: analytic {analytic}, fd {fd}, rel {err}"
        );
    }

    #[test]
    fn logeig_backward_matches_fd() {
        fd_spectral_check(
            |x, n| logeig(x, n, 1e-12),
            |x, dy, n| logeig_backward(x, dy, n, 1e-12),
            5,
            0.7,
        );
    }

    #[test]
    fn reeig_backward_matches_fd() {
        // eps chosen inside the spectrum so some eigenvalues are floored
        // and some are not (exercises both f′ branches + the straddling
        // divided difference).
        let eps = 3.0;
        fd_spectral_check(
            move |x, n| reeig(x, n, eps),
            move |x, dy, n| reeig_backward(x, dy, n, eps),
            5,
            0.42,
        );
    }

    #[test]
    fn spd_bn_transport_normalizes() {
        // With mean = batch Fréchet mean and G = I, the transported
        // batch has Fréchet mean ≈ I (that's the point of the layer).
        let n = 3;
        let batch = 4;
        let mut x = vec![0f64; batch * n * n];
        let mut slices = Vec::new();
        for bi in 0..batch {
            let s = spd(n, bi as f64 * 0.9 + 0.2);
            x[bi * n * n..(bi + 1) * n * n].copy_from_slice(&s);
            slices.push(s);
        }
        let mean = karcher_mean(&slices, n, 50, 1e-12);
        let mut g = vec![0f64; n * n];
        for i in 0..n {
            g[i * n + i] = 1.0;
        }
        let y = spd_bn_transport(&x, &mean, &g, batch, n, 1e-12);
        let yslices: Vec<Vec<f64>> = (0..batch)
            .map(|bi| y[bi * n * n..(bi + 1) * n * n].to_vec())
            .collect();
        let ymean = karcher_mean(&yslices, n, 50, 1e-12);
        // ymean ≈ I.
        let mut ident = vec![0f64; n * n];
        for i in 0..n {
            ident[i * n + i] = 1.0;
        }
        let err = ymean
            .iter()
            .zip(&ident)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-4, "SPD-BN did not normalize to I: err {err}");
    }

    #[test]
    fn spd_bn_backward_x_matches_fd() {
        let n = 3;
        let batch = 2;
        let mean = spd(n, 5.0);
        let g = spd(n, 6.0);
        let mut x = vec![0f64; batch * n * n];
        for bi in 0..batch {
            let s = spd(n, bi as f64 + 0.3);
            x[bi * n * n..(bi + 1) * n * n].copy_from_slice(&s);
        }
        // cotangent (per-slice symmetric)
        let mut c = vec![0f64; batch * n * n];
        for bi in 0..batch {
            let s = sym(n, bi as f64 + 7.0);
            c[bi * n * n..(bi + 1) * n * n].copy_from_slice(&s);
        }
        let eps = 1e-12;
        let grad = spd_bn_backward_x(&mean, &g, &c, batch, n, eps);
        // symmetric direction over the whole batch
        let mut v = vec![0f64; batch * n * n];
        for bi in 0..batch {
            let s = sym(n, bi as f64 + 8.0);
            v[bi * n * n..(bi + 1) * n * n].copy_from_slice(&s);
        }
        let analytic = frob(&grad, &v);
        let h = 1e-6;
        let xp: Vec<f64> = x.iter().zip(&v).map(|(a, b)| a + h * b).collect();
        let xm: Vec<f64> = x.iter().zip(&v).map(|(a, b)| a - h * b).collect();
        let fd = (frob(&c, &spd_bn_transport(&xp, &mean, &g, batch, n, eps))
            - frob(&c, &spd_bn_transport(&xm, &mean, &g, batch, n, eps)))
            / (2.0 * h);
        let err = (analytic - fd).abs() / (fd.abs() + 1e-6);
        assert!(err < 1e-4, "SPD-BN dX FD mismatch: {analytic} vs {fd}");
    }

    #[test]
    fn spd_bn_backward_g_matches_fd() {
        let n = 3;
        let batch = 2;
        let mean = spd(n, 5.0);
        let g = spd(n, 6.0);
        let mut x = vec![0f64; batch * n * n];
        for bi in 0..batch {
            let s = spd(n, bi as f64 + 0.3);
            x[bi * n * n..(bi + 1) * n * n].copy_from_slice(&s);
        }
        let mut c = vec![0f64; batch * n * n];
        for bi in 0..batch {
            let s = sym(n, bi as f64 + 7.0);
            c[bi * n * n..(bi + 1) * n * n].copy_from_slice(&s);
        }
        let eps = 1e-12;
        let grad = spd_bn_backward_g(&x, &mean, &g, &c, batch, n, eps);
        let vg = sym(n, 9.0); // symmetric direction in G
        let analytic = frob(&grad, &vg);
        let h = 1e-6;
        let gp: Vec<f64> = g.iter().zip(&vg).map(|(a, b)| a + h * b).collect();
        let gm: Vec<f64> = g.iter().zip(&vg).map(|(a, b)| a - h * b).collect();
        let fd = (frob(&c, &spd_bn_transport(&x, &mean, &gp, batch, n, eps))
            - frob(&c, &spd_bn_transport(&x, &mean, &gm, batch, n, eps)))
            / (2.0 * h);
        let err = (analytic - fd).abs() / (fd.abs() + 1e-6);
        assert!(err < 1e-4, "SPD-BN dG FD mismatch: {analytic} vs {fd}");
    }

    #[test]
    fn spectral_backward_handles_degenerate_eigenvalues() {
        // X = 2·I + v·vᵀ has eigenvalues {2, 2, 2+‖v‖²} — a genuinely
        // degenerate pair (2, multiplicity n−1). The Loewner mask must
        // fall back to f′ on the equal-eigenvalue block; logm is smooth
        // there, so the analytic gradient must still match central FD.
        let n = 3;
        let vv = [1.0, 0.5, -0.3];
        let mut x = vec![0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let d = if i == j { 2.0 } else { 0.0 };
                x[i * n + j] = d + vv[i] * vv[j];
            }
        }
        let c = sym(n, 3.0);
        let dir = sym(n, 4.0);
        let grad = logeig_backward(&x, &c, n, 1e-12);
        let analytic = frob(&grad, &dir);
        let h = 1e-6;
        let xp: Vec<f64> = x.iter().zip(&dir).map(|(a, b)| a + h * b).collect();
        let xm: Vec<f64> = x.iter().zip(&dir).map(|(a, b)| a - h * b).collect();
        let fd = (frob(&c, &logeig(&xp, n, 1e-12)) - frob(&c, &logeig(&xm, n, 1e-12))) / (2.0 * h);
        let err = (analytic - fd).abs() / (fd.abs() + 1e-6);
        assert!(
            err < 1e-4,
            "degenerate spectral backward FD: analytic {analytic} vs fd {fd}"
        );
    }

    #[test]
    fn geodesic_interp_endpoints_and_midpoint() {
        let n = 3;
        let a = spd(n, 1.0);
        let b = spd(n, 2.0);
        let maxdiff = |p: &[f64], q: &[f64]| {
            p.iter()
                .zip(q)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f64::max)
        };
        // t=0 → A, t=1 → B.
        assert!(maxdiff(&geodesic_interp(&a, &b, 0.0, n), &a) < 1e-8);
        assert!(maxdiff(&geodesic_interp(&a, &b, 1.0, n), &b) < 1e-8);
        // Midpoint stays SPD (all eigenvalues > 0).
        let mid = geodesic_interp(&a, &b, 0.5, n);
        let (ev, _) = eigh(&mid, n);
        assert!(
            ev.iter().all(|&l| l > 1e-9),
            "geodesic midpoint not SPD: {ev:?}"
        );
    }

    // ── Weighted Karcher mean / arbitrary-base maps / batched fns ─

    fn maxerr(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }

    /// Max |A - Aᵀ| — asymmetry of a row-major `n×n` matrix.
    fn asymmetry(a: &[f64], n: usize) -> f64 {
        let mut m = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                m = m.max((a[i * n + j] - a[j * n + i]).abs());
            }
        }
        m
    }

    /// AIRM tangent inner product at `P`: `⟨U, V⟩_P = tr(P^{-1} U P^{-1} V)`.
    fn airm_inner(p: &[f64], u: &[f64], v: &[f64], n: usize) -> f64 {
        let pinv = matrix_fn(p, n, |l| 1.0 / l.max(1e-12));
        let m = matmul(&matmul(&pinv, u, n), &pinv, n); // P^{-1} U P^{-1}
        let mut tr = 0.0;
        for i in 0..n {
            for j in 0..n {
                tr += m[i * n + j] * v[j * n + i];
            }
        }
        tr
    }

    #[test]
    fn karcher_mean_weighted_uniform_matches_unweighted() {
        let n = 3;
        let covs = vec![spd(n, 0.2), spd(n, 1.3), spd(n, 2.7), spd(n, 3.1)];
        let unw = karcher_mean(&covs, n, 60, 1e-13);
        let wtd = karcher_mean_weighted(&covs, &[1.0; 4], n, 60, 1e-13);
        assert!(maxerr(&unw, &wtd) < 1e-9, "uniform weighted != unweighted");
        // Only weight *ratios* matter: scaling all weights leaves the mean fixed.
        let scaled = karcher_mean_weighted(&covs, &[7.5; 4], n, 60, 1e-13);
        assert!(maxerr(&wtd, &scaled) < 1e-12, "weight scale changed result");
    }

    #[test]
    fn karcher_mean_weighted_dominant_weight_selects_matrix() {
        let n = 3;
        let covs = vec![spd(n, 0.5), spd(n, 1.5), spd(n, 2.5)];
        // Nearly all mass on covs[1] → barycentre ≈ covs[1].
        let m = karcher_mean_weighted(&covs, &[1e-9, 1.0, 1e-9], n, 80, 1e-14);
        assert!(
            maxerr(&covs[1], &m) < 1e-6,
            "dominant-weight barycentre should ≈ that cov"
        );
    }

    #[test]
    fn karcher_mean_weighted_is_stationary() {
        // At the weighted barycentre the Riemannian gradient of
        // Σ wᵢ δ²(M, Cᵢ), namely -2 Σ w̄ᵢ Log_M(Cᵢ), must vanish. This
        // cross-validates karcher_mean_weighted *and* log_map together.
        let n = 4;
        let covs = vec![
            spd(n, 0.3),
            spd(n, 1.1),
            spd(n, 2.2),
            spd(n, 3.9),
            spd(n, 5.0),
        ];
        let w = [0.4, 0.1, 0.25, 0.05, 0.2];
        let m = karcher_mean_weighted(&covs, &w, n, 200, 1e-15);
        let wsum: f64 = w.iter().sum();
        let mut g = vec![0f64; n * n];
        for (c, &wi) in covs.iter().zip(&w) {
            let l = log_map(&m, c, n);
            for k in 0..n * n {
                g[k] += (wi / wsum) * l[k];
            }
        }
        let gnorm: f64 = g.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            gnorm < 1e-6,
            "weighted barycentre not stationary: |g|={gnorm}"
        );
    }

    #[test]
    fn log_exp_map_roundtrip_and_identity_base() {
        let n = 4;
        let base = spd(n, 0.7);
        let x = spd(n, 2.4);
        // Exp_P(Log_P(X)) = X.
        let v = log_map(&base, &x, n);
        let back = exp_map(&base, &v, n);
        assert!(maxerr(&x, &back) < 1e-8, "exp∘log roundtrip");
        // log_map output is a symmetric tangent vector.
        assert!(asymmetry(&v, n) < 1e-10, "log_map output not symmetric");
        // Base = I collapses to plain logm / expm.
        let i = ident(n);
        assert!(
            maxerr(&log_map(&i, &x, n), &logm(&x, n)) < 1e-9,
            "log_map(I, ·) != logm"
        );
        let s = sym(n, 3.0);
        assert!(
            maxerr(&exp_map(&i, &s, n), &expm(&s, n)) < 1e-9,
            "exp_map(I, ·) != expm"
        );
    }

    #[test]
    fn exp_log_maps_reproduce_geodesic() {
        // geodesic_interp(A, B, t) == exp_map(A, t · log_map(A, B)).
        let n = 3;
        let a = spd(n, 1.0);
        let b = spd(n, 2.6);
        let v = log_map(&a, &b, n);
        for &t in &[0.0, 0.25, 0.5, 0.9, 1.0] {
            let tv: Vec<f64> = v.iter().map(|x| x * t).collect();
            let via_exp = exp_map(&a, &tv, n);
            let gi = geodesic_interp(&a, &b, t, n);
            assert!(
                maxerr(&via_exp, &gi) < 1e-8,
                "exp_map(t·log) != geodesic_interp at t={t}"
            );
        }
    }

    #[test]
    fn parallel_transport_is_airm_isometry() {
        let n = 4;
        let p = spd(n, 0.9);
        let q = spd(n, 3.3);
        let v = sym(n, 1.2);
        let w = sym(n, 2.4);
        let gv = parallel_transport(&p, &q, &v, n);
        let gw = parallel_transport(&p, &q, &w, n);
        // ⟨v, w⟩_P == ⟨Γv, Γw⟩_Q  (transport preserves the metric).
        let before = airm_inner(&p, &v, &w, n);
        let after = airm_inner(&q, &gv, &gw, n);
        let rel = (before - after).abs() / (before.abs() + 1e-9);
        assert!(
            rel < 1e-7,
            "transport not an isometry: ⟨v,w⟩_P={before} vs ⟨Γv,Γw⟩_Q={after}"
        );
        // Transported vector stays symmetric; P→P is the identity map.
        assert!(asymmetry(&gv, n) < 1e-9, "transported vector not symmetric");
        let same = parallel_transport(&p, &p, &v, n);
        assert!(maxerr(&same, &v) < 1e-8, "transport P→P must be identity");
    }

    #[test]
    fn batched_matrix_fns_match_scalar() {
        let n = 3;
        let covs: Vec<Vec<f64>> = (0..6).map(|i| spd(n, i as f64 * 0.6 + 0.1)).collect();
        let syms: Vec<Vec<f64>> = (0..6).map(|i| sym(n, i as f64 * 0.4 + 0.2)).collect();
        let lb = logm_batch(&covs, n);
        let sb = sqrtm_batch(&covs, n);
        let ib = invsqrtm_batch(&covs, n);
        for (i, c) in covs.iter().enumerate() {
            assert!(maxerr(&lb[i], &logm(c, n)) < 1e-12, "logm_batch[{i}]");
            assert!(maxerr(&sb[i], &sqrtm(c, n)) < 1e-12, "sqrtm_batch[{i}]");
            assert!(
                maxerr(&ib[i], &invsqrtm(c, n)) < 1e-12,
                "invsqrtm_batch[{i}]"
            );
        }
        let eb = expm_batch(&syms, n);
        for (i, s) in syms.iter().enumerate() {
            assert!(maxerr(&eb[i], &expm(s, n)) < 1e-12, "expm_batch[{i}]");
        }
        // Empty batch → empty output (no panic).
        assert!(logm_batch(&[], n).is_empty());
    }

    // ── Riemannian map VJPs (finite-difference checked) ──────────

    /// Central-FD check that `⟨grad, dir⟩ ≈ d/dh ⟨c, forward(base + h·dir)⟩`.
    fn fd_dir_check(
        grad: &[f64],
        dir: &[f64],
        c: &[f64],
        forward: impl Fn(&[f64]) -> Vec<f64>,
        at: &[f64],
        name: &str,
    ) {
        let h = 1e-6;
        let xp: Vec<f64> = at.iter().zip(dir).map(|(a, b)| a + h * b).collect();
        let xm: Vec<f64> = at.iter().zip(dir).map(|(a, b)| a - h * b).collect();
        let fd = (frob(c, &forward(&xp)) - frob(c, &forward(&xm))) / (2.0 * h);
        let an = frob(grad, dir);
        assert!(
            (an - fd).abs() / (fd.abs() + 1e-6) < 1e-4,
            "{name}: analytic {an} vs fd {fd}"
        );
    }

    #[test]
    fn log_map_backward_matches_fd() {
        let n = 3;
        let base = spd(n, 0.7);
        let x = spd(n, 2.1);
        let c = sym(n, 3.0); // cotangent dL/dY
        let (d_base, d_x) = log_map_backward(&base, &x, &c, n);
        fd_dir_check(
            &d_x,
            &sym(n, 4.0),
            &c,
            |x| log_map(&base, x, n),
            &x,
            "log_map d_x",
        );
        fd_dir_check(
            &d_base,
            &sym(n, 5.0),
            &c,
            |b| log_map(b, &x, n),
            &base,
            "log_map d_base",
        );
    }

    #[test]
    fn exp_map_backward_matches_fd() {
        let n = 3;
        let base = spd(n, 0.8);
        let v = sym(n, 1.4); // symmetric tangent (kept small → well-conditioned Exp)
        let c = sym(n, 3.2);
        let (d_base, d_v) = exp_map_backward(&base, &v, &c, n);
        fd_dir_check(
            &d_v,
            &sym(n, 4.1),
            &c,
            |v| exp_map(&base, v, n),
            &v,
            "exp_map d_v",
        );
        fd_dir_check(
            &d_base,
            &sym(n, 5.1),
            &c,
            |b| exp_map(b, &v, n),
            &base,
            "exp_map d_base",
        );
    }

    #[test]
    fn parallel_transport_backward_matches_fd() {
        let n = 3;
        let from = spd(n, 0.6);
        let to = spd(n, 1.9);
        let v = sym(n, 2.5);
        let c = sym(n, 3.3);
        let (d_from, d_to, d_v) = parallel_transport_backward(&from, &to, &v, &c, n);
        fd_dir_check(
            &d_from,
            &sym(n, 4.0),
            &c,
            |p| parallel_transport(p, &to, &v, n),
            &from,
            "transport d_from",
        );
        fd_dir_check(
            &d_to,
            &sym(n, 5.0),
            &c,
            |q| parallel_transport(&from, q, &v, n),
            &to,
            "transport d_to",
        );
        fd_dir_check(
            &d_v,
            &sym(n, 6.0),
            &c,
            |vv| parallel_transport(&from, &to, vv, n),
            &v,
            "transport d_v",
        );
    }

    #[test]
    fn matrix_fn_batch_backward_matches_fd() {
        let n = 3;
        let batch = 3;
        let mut x = vec![0.0; batch * n * n];
        let mut c = vec![0.0; batch * n * n];
        for bi in 0..batch {
            x[bi * n * n..(bi + 1) * n * n].copy_from_slice(&spd(n, bi as f64 * 0.5 + 0.2));
            c[bi * n * n..(bi + 1) * n * n].copy_from_slice(&sym(n, bi as f64 + 3.0));
        }
        // logm variant: f = ln, f' = 1/λ.
        let grad =
            matrix_fn_batch_backward(&x, &c, n, |l| l.max(1e-12).ln(), |l| 1.0 / l.max(1e-12));
        let mut dir = vec![0.0; batch * n * n];
        for bi in 0..batch {
            dir[bi * n * n..(bi + 1) * n * n].copy_from_slice(&sym(n, bi as f64 + 7.0));
        }
        let logm_flat = |xx: &[f64]| {
            let covs: Vec<Vec<f64>> = (0..batch)
                .map(|bi| xx[bi * n * n..(bi + 1) * n * n].to_vec())
                .collect();
            logm_batch(&covs, n).concat()
        };
        fd_dir_check(
            &grad,
            &dir,
            &c,
            logm_flat,
            &x,
            "matrix_fn_batch(logm) backward",
        );
    }

    // ── Symmetric eigendecomposition op ──────────────────────────

    /// SPD with well-separated eigenvalues (distinct diagonal + tiny coupling),
    /// so the eigenvector gradient is well-conditioned for the FD check.
    fn spd_sep(n: usize) -> Vec<f64> {
        let mut a = vec![0f64; n * n];
        for i in 0..n {
            for j in i..n {
                let v = if i == j {
                    (i as f64 + 1.0) * 2.0
                } else {
                    0.05 * ((i + j) as f64).cos()
                };
                a[i * n + j] = v;
                a[j * n + i] = v;
            }
        }
        a
    }

    #[test]
    fn eigh_packed_reconstructs() {
        let n = 4;
        let a = spd_sep(n);
        let fwd = eigh_packed(&a, n);
        let (lam, u) = (&fwd[..n], &fwd[n..]);
        // A = U diag(λ) Uᵀ.
        let mut recon = vec![0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += u[i * n + k] * lam[k] * u[j * n + k];
                }
                recon[i * n + j] = s;
            }
        }
        assert!(maxerr(&recon, &a) < 1e-9, "eigh_packed A = U Λ Uᵀ failed");
        // Eigenvalues ascending.
        assert!(lam.windows(2).all(|w| w[0] <= w[1] + 1e-12));
    }

    #[test]
    fn eigh_backward_matches_fd() {
        let n = 4;
        let a = spd_sep(n);
        let fwd = eigh_packed(&a, n);
        let u_base = fwd[n..].to_vec();
        // Arbitrary cotangents on λ and U.
        let lbar: Vec<f64> = (0..n).map(|i| ((i as f64 + 1.0) * 0.7).sin()).collect();
        let ubar = sym(n, 5.0);
        let mut bar = vec![0f64; n + n * n];
        bar[..n].copy_from_slice(&lbar);
        bar[n..].copy_from_slice(&ubar);
        let abar = eigh_backward_packed(&fwd, &bar, n);
        // Loss ⟨λ̄, λ⟩ + ⟨Ū, U⟩ with perturbed U sign-aligned to the base U
        // (eigenvectors carry an arbitrary sign the solver may flip).
        let loss = |mat: &[f64]| -> f64 {
            let fw = eigh_packed(mat, n);
            let (l, mut uu) = (fw[..n].to_vec(), fw[n..].to_vec());
            for j in 0..n {
                let mut dot = 0.0;
                for i in 0..n {
                    dot += uu[i * n + j] * u_base[i * n + j];
                }
                if dot < 0.0 {
                    for i in 0..n {
                        uu[i * n + j] = -uu[i * n + j];
                    }
                }
            }
            let mut s = 0.0;
            for i in 0..n {
                s += lbar[i] * l[i];
            }
            for k in 0..n * n {
                s += ubar[k] * uu[k];
            }
            s
        };
        // Scalar loss ⇒ central difference directly (not via fd_dir_check).
        let dir = sym(n, 4.0);
        let h = 1e-6;
        let ap: Vec<f64> = a.iter().zip(&dir).map(|(x, y)| x + h * y).collect();
        let am: Vec<f64> = a.iter().zip(&dir).map(|(x, y)| x - h * y).collect();
        let fd = (loss(&ap) - loss(&am)) / (2.0 * h);
        let an = frob(&abar, &dir);
        assert!(
            (an - fd).abs() / (fd.abs() + 1e-6) < 1e-4,
            "eigh backward FD: analytic {an} vs fd {fd}"
        );
    }

    #[test]
    fn eigh_batch_packed_matches_scalar() {
        let n = 3;
        let mats: Vec<Vec<f64>> = (0..4).map(|i| spd(n, i as f64 * 0.4 + 0.2)).collect();
        let batched = eigh_batch_packed(&mats, n);
        for (i, m) in mats.iter().enumerate() {
            assert!(
                maxerr(&batched[i], &eigh_packed(m, n)) < 1e-12,
                "eigh_batch[{i}]"
            );
        }
    }
}
