// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared helpers (grad clipping, state-buffer creation).

use std::collections::HashMap;

/// Tensor-size threshold above which we'll dispatch the inner loop to
/// rayon (only when the `parallel` feature is on). Below this size the
/// fork-join overhead dominates.
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) const PARALLEL_THRESHOLD: usize = 64 * 1024;

/// Iterate `(param, m, v, grad)` in lockstep, calling
/// `body(p_i, m_i, v_i, g_i, i)` for each index. When the `parallel`
/// feature is on and the tensor is at least `PARALLEL_THRESHOLD`
/// elements, splits the work across rayon's thread pool — the four
/// buffers are partitioned with `par_chunks_mut` / `par_chunks`, so
/// each thread sees disjoint regions.
#[inline]
pub(crate) fn zip4_for_each<F>(
    param: &mut [f32],
    m: &mut [f32],
    v: &mut [f32],
    grad: &[f32],
    body: F,
) where
    F: Fn(&mut f32, &mut f32, &mut f32, f32) + Sync + Send,
{
    debug_assert_eq!(param.len(), grad.len());
    debug_assert_eq!(param.len(), m.len());
    debug_assert_eq!(param.len(), v.len());
    #[cfg(feature = "parallel")]
    if param.len() >= PARALLEL_THRESHOLD {
        use rayon::prelude::*;
        let n_threads = rayon::current_num_threads().max(1);
        let chunk = param.len().div_ceil(n_threads).max(1);
        param
            .par_chunks_mut(chunk)
            .zip(m.par_chunks_mut(chunk))
            .zip(v.par_chunks_mut(chunk))
            .zip(grad.par_chunks(chunk))
            .for_each(|(((pc, mc), vc), gc)| {
                for ((pi, mi), (vi, gi)) in pc.iter_mut().zip(mc).zip(vc.iter_mut().zip(gc)) {
                    body(pi, mi, vi, *gi);
                }
            });
        return;
    }
    for (((pi, mi), vi), gi) in param.iter_mut().zip(m).zip(v).zip(grad) {
        body(pi, mi, vi, *gi);
    }
}

/// Two-buffer variant for optimizers that maintain a single momentum
/// (Lion, SGD-momentum, Muon-fallback). `(param, m, grad)`.
#[inline]
pub(crate) fn zip3_for_each<F>(param: &mut [f32], m: &mut [f32], grad: &[f32], body: F)
where
    F: Fn(&mut f32, &mut f32, f32) + Sync + Send,
{
    debug_assert_eq!(param.len(), grad.len());
    debug_assert_eq!(param.len(), m.len());
    #[cfg(feature = "parallel")]
    if param.len() >= PARALLEL_THRESHOLD {
        use rayon::prelude::*;
        let n_threads = rayon::current_num_threads().max(1);
        let chunk = param.len().div_ceil(n_threads).max(1);
        param
            .par_chunks_mut(chunk)
            .zip(m.par_chunks_mut(chunk))
            .zip(grad.par_chunks(chunk))
            .for_each(|((pc, mc), gc)| {
                for ((pi, mi), gi) in pc.iter_mut().zip(mc).zip(gc) {
                    body(pi, mi, *gi);
                }
            });
        return;
    }
    for ((pi, mi), gi) in param.iter_mut().zip(m).zip(grad) {
        body(pi, mi, *gi);
    }
}

/// L2 norm across a slice (skipping non-finite entries).
pub fn l2_norm(xs: &[f32]) -> f32 {
    let mut s = 0.0f64;
    for &x in xs {
        if x.is_finite() {
            s += (x as f64) * (x as f64);
        }
    }
    s.sqrt() as f32
}

/// Global L2-norm clip across many tensors. Returns the scale factor
/// (`<= 1.0`) to multiply every gradient by; callers can pre-scale
/// before passing to [`Optimizer::step`]. Identical to
/// `rlx_umap::adam::global_grad_clip_scale` but generic over any
/// iterator yielding slices.
pub fn global_grad_clip_scale<'a, I>(grads: I, max_norm: f32) -> f32
where
    I: IntoIterator<Item = &'a [f32]>,
{
    let mut norm_sq = 0.0f64;
    for g in grads {
        for &gi in g {
            if gi.is_finite() {
                norm_sq += (gi as f64) * (gi as f64);
            }
        }
    }
    let max_sq = (max_norm as f64) * (max_norm as f64);
    if norm_sq > max_sq && norm_sq > 0.0 {
        (max_norm / (norm_sq.sqrt() as f32)).min(1.0)
    } else {
        1.0
    }
}

/// `HashMap` helper used by every algorithm: lazily allocate a zero
/// buffer of the requested length for `name`. Returns `&mut Vec<f32>`.
pub(crate) fn zeros_entry<'a>(
    map: &'a mut HashMap<String, Vec<f32>>,
    name: &str,
    len: usize,
) -> &'a mut Vec<f32> {
    map.entry(name.to_owned()).or_insert_with(|| vec![0.0; len])
}

// ── Small dense linear-algebra helpers ──────────────────────────────
//
// SOAP / Shampoo / Kron-PSGD all need eigendecompositions and matmuls
// on `axis × axis` symmetric matrices (e.g. a 768 × 768 left
// covariance for a 768×N projection). For these sizes a plain
// row-major matmul + Jacobi eigh is fine and keeps `rlx-optim`
// dependency-free.

/// Row-major matrix multiply `c = a · b`. Shapes: `a: m×k`, `b: k×n`,
/// `c: m×n`. Written naively; only called from the matrix-aware
/// optimizers, on the small `axis × axis` factor matrices.
pub(crate) fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, c: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for p in 0..k {
                s += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = s;
        }
    }
}

/// Symmetric-eigendecomposition via cyclic Jacobi rotations. `a` is
/// modified in-place; on return its diagonal is the eigenvalues and
/// `v` is the column-stored eigenvector matrix (row-major: row i of
/// `v` is eigenvector i). Iterates until off-diagonal mass falls below
/// `tol` or `max_sweeps` sweeps elapse. Suitable for `n ≲ 1024`.
pub(crate) fn jacobi_eigh_sym(a: &mut [f32], n: usize, v: &mut [f32], max_sweeps: u32, tol: f32) {
    // Initialize V = I.
    for i in 0..n {
        for j in 0..n {
            v[i * n + j] = if i == j { 1.0 } else { 0.0 };
        }
    }
    for _ in 0..max_sweeps {
        let mut off = 0.0f32;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        if off.sqrt() < tol {
            return;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < f32::EPSILON {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    1.0 / (theta - (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                // Update rows/cols p,q of A.
                a[p * n + p] = app - t * apq;
                a[q * n + q] = aqq + t * apq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;
                for r in 0..n {
                    if r != p && r != q {
                        let arp = a[r * n + p];
                        let arq = a[r * n + q];
                        a[r * n + p] = c * arp - s * arq;
                        a[r * n + q] = s * arp + c * arq;
                        a[p * n + r] = a[r * n + p];
                        a[q * n + r] = a[r * n + q];
                    }
                }
                // Update V.
                for r in 0..n {
                    let vrp = v[r * n + p];
                    let vrq = v[r * n + q];
                    v[r * n + p] = c * vrp - s * vrq;
                    v[r * n + q] = s * vrp + c * vrq;
                }
            }
        }
    }
}
