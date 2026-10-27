// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reference LAPACK 3.12's symmetric eigensolver `DSYEVR` (`JOBZ = 'V'`,
//! `RANGE = 'A'`, `UPLO = 'L'`, `ABSTOL = 0` — what `scipy.linalg.eigh` runs):
//! `DSYTRD` (blocked `DLATRD` + `DSYR2K`, then `DSYTD2`) on OpenBLAS's Haswell
//! kernels, `DSTEMR` ([`crate::lapack_mrrr`]), `DORMTR` → `DORMQR`. Column-major,
//! `A(I, J)` = `a[(i-1) + (j-1)*lda]`.

use crate::lapack::{PREC as EPS, SFMIN, dlarfg, dormxx_left};
use crate::lapack_mrrr::{dstemr_all, sort_pairs};
use crate::openblas as ob;
use crate::{COL_MAJOR, NO_TRANS, TRANS};

/// `y` (contiguous) `= alpha·op(A)·x + beta·y` through the OpenBLAS model; `x`
/// read at stride `incx` from `xs`.
#[allow(clippy::too_many_arguments)]
fn gemv(
    trans: bool,
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    xs: &[f64],
    incx: usize,
    beta: f64,
    y: &mut [f64],
) {
    let t = if trans { TRANS } else { NO_TRANS };
    ob::dgemv_ab(COL_MAJOR, t, m, n, alpha, a, lda, xs, incx, beta, y, 1);
}

/// `len` elements of `src` from `off` at stride `inc`, keeping the stride.
fn strided(src: &[f64], off: usize, len: usize, inc: usize) -> Vec<f64> {
    if len == 0 {
        return vec![];
    }
    src[off..off + (len - 1) * inc + 1].to_vec()
}

/// `DSYTD2('L', n, A, lda, d, e, tau)`.
fn dsytd2_lower(
    n: usize,
    a: &mut [f64],
    lda: usize,
    d: &mut [f64],
    e: &mut [f64],
    tau: &mut [f64],
) {
    if n == 0 {
        return;
    }
    let ix = |i: usize, j: usize| (i - 1) + (j - 1) * lda;
    for i in 1..n {
        let mut alpha = a[ix(i + 1, i)];
        let xo = ix((i + 2).min(n), i);
        let taui = dlarfg(n - i, &mut alpha, &mut a[xo..], 1);
        a[ix(i + 1, i)] = alpha;
        e[i - 1] = alpha;
        if taui != 0.0 {
            a[ix(i + 1, i)] = 1.0;
            let v = a[ix(i + 1, i)..ix(i + 1, i) + (n - i)].to_vec();
            let t = &mut tau[i - 1..i - 1 + (n - i)];
            ob::dsymv_lower(n - i, taui, &a[ix(i + 1, i + 1)..], lda, &v, 0.0, t);
            let alpha2 = -0.5 * taui * ob::ddot(n - i, t, 1, &v, 1);
            ob::daxpy_unit(n - i, alpha2, &v, t);
            let t = t.to_vec();
            ob::dsyr2_lower(n - i, -1.0, &v, &t, &mut a[ix(i + 1, i + 1)..], lda);
            a[ix(i + 1, i)] = e[i - 1];
        }
        d[i - 1] = a[ix(i, i)];
        tau[i - 1] = taui;
    }
    d[n - 1] = a[ix(n, n)];
}

/// `DLATRD('L', n, nb, A, lda, e, tau, W, ldw)`.
#[allow(clippy::too_many_arguments)]
fn dlatrd_lower(
    n: usize,
    nb: usize,
    a: &mut [f64],
    lda: usize,
    e: &mut [f64],
    tau: &mut [f64],
    w: &mut [f64],
    ldw: usize,
) {
    let ia = |i: usize, j: usize| (i - 1) + (j - 1) * lda;
    let iw = |i: usize, j: usize| (i - 1) + (j - 1) * ldw;
    for i in 1..=nb {
        // A(i:n, i) -= A(i:n, 1:i-1)·W(i, 1:i-1)ᵀ + W(i:n, 1:i-1)·A(i, 1:i-1)ᵀ
        let m = n - i + 1;
        if i > 1 {
            let mut y = a[ia(i, i)..ia(i, i) + m].to_vec();
            let xs = strided(w, iw(i, 1), i - 1, ldw);
            gemv(
                false,
                m,
                i - 1,
                -1.0,
                &a[ia(i, 1)..],
                lda,
                &xs,
                ldw,
                1.0,
                &mut y,
            );
            let xs = strided(a, ia(i, 1), i - 1, lda);
            gemv(
                false,
                m,
                i - 1,
                -1.0,
                &w[iw(i, 1)..],
                ldw,
                &xs,
                lda,
                1.0,
                &mut y,
            );
            a[ia(i, i)..ia(i, i) + m].copy_from_slice(&y);
        }
        if i < n {
            let mut alpha = a[ia(i + 1, i)];
            let xo = ia((i + 2).min(n), i);
            tau[i - 1] = dlarfg(n - i, &mut alpha, &mut a[xo..], 1);
            e[i - 1] = alpha;
            a[ia(i + 1, i)] = 1.0;
            let k = n - i;
            let v = a[ia(i + 1, i)..ia(i + 1, i) + k].to_vec();
            let mut wi = vec![0.0f64; k];
            ob::dsymv_lower(k, 1.0, &a[ia(i + 1, i + 1)..], lda, &v, 0.0, &mut wi);
            if i > 1 {
                let mut w1 = w[iw(1, i)..iw(1, i) + (i - 1)].to_vec();
                gemv(
                    true,
                    k,
                    i - 1,
                    1.0,
                    &w[iw(i + 1, 1)..],
                    ldw,
                    &v,
                    1,
                    0.0,
                    &mut w1,
                );
                gemv(
                    false,
                    k,
                    i - 1,
                    -1.0,
                    &a[ia(i + 1, 1)..],
                    lda,
                    &w1,
                    1,
                    1.0,
                    &mut wi,
                );
                gemv(
                    true,
                    k,
                    i - 1,
                    1.0,
                    &a[ia(i + 1, 1)..],
                    lda,
                    &v,
                    1,
                    0.0,
                    &mut w1,
                );
                gemv(
                    false,
                    k,
                    i - 1,
                    -1.0,
                    &w[iw(i + 1, 1)..],
                    ldw,
                    &w1,
                    1,
                    1.0,
                    &mut wi,
                );
                w[iw(1, i)..iw(1, i) + (i - 1)].copy_from_slice(&w1);
            }
            ob::dscal(k, tau[i - 1], &mut wi, 1);
            let alpha2 = -0.5 * tau[i - 1] * ob::ddot(k, &wi, 1, &v, 1);
            ob::daxpy_unit(k, alpha2, &v, &mut wi);
            w[iw(i + 1, i)..iw(i + 1, i) + k].copy_from_slice(&wi);
        }
    }
}

/// `DSYTRD('L', n, A, lda, d, e, tau, work, lwork)`: `NB = ILAENV = 32` (cut to
/// `lwork / n` when the workspace is short), `NX = 32`.
fn dsytrd_lower(
    n: usize,
    a: &mut [f64],
    lda: usize,
    d: &mut [f64],
    e: &mut [f64],
    tau: &mut [f64],
    lwork: i64,
) {
    if n == 0 {
        return;
    }
    let mut nb: usize = 32;
    let mut nx = n;
    let ldwork = n;
    if nb > 1 && nb < n {
        nx = nb.max(32);
        if nx < n {
            let iws = (ldwork * nb) as i64;
            if lwork < iws {
                nb = ((lwork / ldwork as i64).max(1)) as usize;
                if nb < 2 {
                    nx = n;
                }
            }
        } else {
            nx = n;
        }
    } else {
        nb = 1;
    }
    let ia = |i: usize, j: usize| (i - 1) + (j - 1) * lda;
    let mut i = 1;
    if n > nx {
        let mut work = vec![0.0f64; ldwork * nb];
        while i <= n - nx {
            let off = ia(i, i);
            dlatrd_lower(
                n - i + 1,
                nb,
                &mut a[off..],
                lda,
                &mut e[i - 1..],
                &mut tau[i - 1..],
                &mut work,
                ldwork,
            );
            // A(i+nb:n, i+nb:n) -= V·Wᵀ + W·Vᵀ
            let rows = n - i - nb + 1;
            let v = a[ia(i + nb, i)..].to_vec();
            let ok = ob::dsyr2k_lower(
                rows,
                nb,
                -1.0,
                &v,
                lda,
                &work[nb..],
                ldwork,
                &mut a[ia(i + nb, i + nb)..],
                lda,
            );
            assert!(ok, "dsyr2k past one GEMM_P block is not modelled");
            for j in i..i + nb {
                a[ia(j + 1, j)] = e[j - 1];
                d[j - 1] = a[ia(j, j)];
            }
            i += nb;
        }
    }
    let off = ia(i, i);
    dsytd2_lower(
        n - i + 1,
        &mut a[off..],
        lda,
        &mut d[i - 1..],
        &mut e[i - 1..],
        &mut tau[i - 1..],
    );
}

/// `scipy.linalg.lapack.dsytrd(A, lower=1, lwork)`: `A` reduced in place
/// (column-major), returns `(d, e, tau)` (`e`, `tau` of length `n`, last unused).
pub fn dsytrd_lower_pub(
    n: usize,
    a: &mut [f64],
    lda: usize,
    lwork: i64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut d, mut e, mut tau) = (vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]);
    dsytrd_lower(n, a, lda, &mut d, &mut e, &mut tau, lwork);
    (d, e, tau)
}

/// `DLANSY('M', 'L', n, A)`.
fn dlansy_max_lower(n: usize, a: &[f64], lda: usize) -> f64 {
    let mut value = 0.0f64;
    for j in 0..n {
        for i in j..n {
            let s = a[i + j * lda].abs();
            if value < s || s.is_nan() {
                value = s;
            }
        }
    }
    value
}

/// The largest `n` [`dsyevr_lower`] models: above it the trailing `dsyr2k` of
/// `DSYTRD` spans more than one `GEMM_P` block (`n − 28 > 512`), not modelled.
/// (The fixtures check `n ≤ 300`.)
pub const DSYEVR_MAX_N: usize = 540;

/// `scipy.linalg.eigh(a)` on the Linux reference: `DSYEVR('V', 'A', 'L', n, A,
/// lda, …, ABSTOL = 0, …, LWORK)` with `lwork` as SciPy's query returns it
/// (`max((NB+1)·n, 26n)` = `33n`). `a` is column-major `n × n` (lower triangle
/// read, overwritten). Returns the ascending eigenvalues and the eigenvectors
/// column-major (column `j` = vector `j`), or `INFO`.
pub fn dsyevr_lower(
    n: usize,
    a: &mut [f64],
    lda: usize,
    lwork: i64,
) -> Result<(Vec<f64>, Vec<f64>), i32> {
    let mut w = vec![0.0f64; n];
    let mut z = vec![0.0f64; n * n];
    if n == 0 {
        return Ok((w, z));
    }
    if n == 1 {
        w[0] = a[0];
        z[0] = 1.0;
        return Ok((w, z));
    }
    let safmin = SFMIN;
    let smlnum = safmin / EPS;
    let bignum = 1.0 / smlnum;
    let rmin = smlnum.sqrt();
    let rmax = (bignum.sqrt()).min(1.0 / safmin.sqrt().sqrt());
    let anrm = dlansy_max_lower(n, a, lda);
    let mut sigma = 1.0;
    let iscale = if anrm > 0.0 && anrm < rmin {
        sigma = rmin / anrm;
        true
    } else if anrm > rmax {
        sigma = rmax / anrm;
        true
    } else {
        false
    };
    if iscale {
        for j in 0..n {
            ob::dscal(n - j, sigma, &mut a[j + j * lda..], 1);
        }
    }
    let mut tau = vec![0.0f64; n];
    let mut d = vec![0.0f64; n];
    let mut e = vec![0.0f64; n];
    // WORK(INDWK) holds LWORK − 5n
    dsytrd_lower(n, a, lda, &mut d, &mut e, &mut tau, lwork - 5 * n as i64);
    // ABSTOL = 0 ≤ 2·n·eps: TRYRAC
    let mut ee = e.clone();
    let mut dd = d.clone();
    let info = dstemr_all(n, &mut dd, &mut ee, &mut w, &mut z, n, true);
    if info != 0 {
        // DSTEBZ + DSTEIN fallback: not modelled
        return Err(info);
    }
    // DORMTR('L', 'L', 'N', n, n, A, tau, Z): DORMQR on rows 2..n with
    // LWORK − 2n (WORK(INDE) onwards)
    let mut z2 = z.clone();
    dormxx_left(
        false,
        true,
        n - 1,
        n,
        n - 1,
        &mut a[1..],
        lda,
        &tau,
        &mut z2[1..],
        n,
        lwork - 2 * n as i64,
    );
    z = z2;
    if iscale {
        for v in w.iter_mut() {
            *v *= 1.0 / sigma;
        }
    }
    sort_pairs(n, n, &mut w, &mut z, n);
    Ok((w, z))
}
