// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Apple Accelerate's BLAS / LAPACK as NumPy, SciPy and torch's macOS arm64 wheels
//! run it, modelled in pure Rust** — so code reproducing those libraries is bit-exact
//! on every target, wasm included, with no native link.
//!
//! ## `sgemm_`
//!
//! The `sgemm_` torch 2.10's macOS wheel calls (Fortran ABI) — e.g. for an im2col
//! convolution (`cpublas::gemm('N', 'N', m = out_len, n, k = 9)`, aeon's Hydra).
//!
//! Accelerate is closed source; this is a model measured against it (macOS 26,
//! Apple arm64, `parity/export_accelerate_sgemm.py`): the summation trees were read
//! off with exact-product probes (±2³⁰ against ones — the FPRev method), the fusion
//! and the signed-zero behaviour from random and all-zero data
//! (`parity/export_accelerate_sgemm.py` → `tests/accelerate.rs`). It matched all
//! 3 022 166 outputs of 2 856 calls (m = 1…200, n = 1…256). Every accumulator starts at
//! `+0` (an all-`-0` product column sums to `+0`). Per output element `(i, j)`:
//!
//! * the default — every element of a call with `n == 1`, and otherwise every element
//!   outside the corner below — is one FMA chain over `k` in order;
//! * `m == 1` (`n ≥ 2`): a 4-lane vector over `k`: lane `q` = `fma(x[q+4], y[q+4],
//!   fma(x[q], y[q], 0))`, then lane 3 `+= x[8]·y[8]` (not fused); reduced
//!   `(l0 + l1) + (l2 + l3)`, except the last column of an odd `n`:
//!   `(l0 + l2) + (l1 + l3)`;
//! * the corner — rows past the last multiple of 4 *and* columns past the last
//!   multiple of 4 — of a small call (`m < 16`, `n < 16` or `m·n < 342`; measured: 324
//!   still small, 342 not): two FMA chains, over `k = 0, 1, 3, 5, 7` and `2, 4, 6, 8`,
//!   added.
//!
//! The model covers what was measured: `k = 9`, `alpha = 1`, `beta` 0, or 1 into a
//! zeroed `C` (Hydra's `slow_conv_dilated2d`). Anything else panics rather than
//! return numbers no measurement backs.

#[inline(always)]
fn fma(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}

/// Column-major `C = alpha · A · B + beta · C`, `A` m×k (`lda`), `B` k×n (`ldb`),
/// `C` m×n (`ldc`), no transposes — as Accelerate computes it (see the module docs
/// for the measured scope).
#[allow(clippy::too_many_arguments)]
pub fn sgemm_nn(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    assert!(
        k == 9 && alpha == 1.0,
        "Accelerate's sgemm_ is modelled for k = 9, alpha = 1 (got k = {k}, alpha = {alpha})"
    );
    assert!(
        m == 0
            || n == 0
            || (lda >= m
                && ldb >= k
                && ldc >= m
                && a.len() >= lda * (k - 1) + m
                && b.len() >= ldb * (n - 1) + k
                && c.len() >= ldc * (n - 1) + m),
        "sgemm_nn: buffers too small"
    );
    if beta != 0.0 {
        assert!(
            beta == 1.0 && (0..n).all(|j| c[j * ldc..j * ldc + m].iter().all(|&v| v == 0.0)),
            "Accelerate's sgemm_ is modelled for beta = 0, or 1 into a zeroed C"
        );
    }
    let small = m < 16 || n < 16 || m * n < 342;
    let (m4, n4) = (m / 4 * 4, n / 4 * 4);
    for j in 0..n {
        let y = &b[j * ldb..j * ldb + k];
        for i in 0..m {
            let x = |p: usize| a[p * lda + i];
            let v = if n == 1 {
                (0..k).fold(0.0f32, |s, p| fma(x(p), y[p], s))
            } else if m == 1 {
                let mut l = [0.0f32; 4];
                for (q, lq) in l.iter_mut().enumerate() {
                    *lq = fma(x(q + 4), y[q + 4], fma(x(q), y[q], 0.0));
                }
                l[3] += x(8) * y[8];
                if n % 2 == 1 && j == n - 1 {
                    (l[0] + l[2]) + (l[1] + l[3])
                } else {
                    (l[0] + l[1]) + (l[2] + l[3])
                }
            } else if small && i >= m4 && j >= n4 {
                let e = [0, 1, 3, 5, 7]
                    .iter()
                    .fold(0.0f32, |s, &p| fma(x(p), y[p], s));
                let o = [2, 4, 6, 8].iter().fold(0.0f32, |s, &p| fma(x(p), y[p], s));
                e + o
            } else {
                (0..k).fold(0.0f32, |s, p| fma(x(p), y[p], s))
            };
            let slot = &mut c[j * ldc + i];
            *slot = if beta == 0.0 { v } else { v + *slot };
        }
    }
}
