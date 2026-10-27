// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The LAPACK NumPy and SciPy run on the Linux reference: reference LAPACK 3.12 as
//! OpenBLAS 0.3.30 bundles it (`lapack-netlib`, built by gfortran for the x86-64
//! baseline — measured: no FMA instruction in any of these routines), calling
//! OpenBLAS's Haswell BLAS kernels ([`crate::openblas`]). Translated statement
//! for statement from the Fortran (BSD-3-Clause, see `LICENSE-LAPACK`), so every
//! product and sum rounds as gfortran evaluates the source: left to right, inside
//! parentheses first, `x**2` as `x*x`, `SIGN` as `copysign`.
//!
//! Arrays are column-major; `A(I, J)` of the Fortran is `a[(i-1) + (j-1)*lda]`.

use crate::openblas as ob;

// ── machine constants (`DLAMCH`, `la_constants`) ─────────────────────────────

/// `DLAMCH('E')`: relative machine epsilon with rounding, `2^-53`.
pub(crate) const EPS: f64 = f64::EPSILON * 0.5;
/// `DLAMCH('P')`: `eps · base`, `2^-52`.
pub(crate) const PREC: f64 = f64::EPSILON;
/// `DLAMCH('S')`: safe minimum, `2^-1022`.
pub(crate) const SFMIN: f64 = f64::MIN_POSITIVE;
/// `DLAMCH('O')`.
const HUGE: f64 = f64::MAX;

/// `ILAENV(9, 'DGELSD')` (`SMLSIZ`).
pub(crate) const SMLSIZ: usize = 25;

pub(crate) fn sign(a: f64, b: f64) -> f64 {
    a.abs().copysign(b)
}

// ── auxiliaries ───────────────────────────────────────────────────────────────

/// `DLAPY2(x, y)`: `sqrt(x² + y²)` avoiding overflow.
pub(crate) fn dlapy2(x: f64, y: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if y.is_nan() {
        return y;
    }
    let (xa, ya) = (x.abs(), y.abs());
    let w = xa.max(ya);
    let z = xa.min(ya);
    if z == 0.0 || w > HUGE {
        w
    } else {
        let q = z / w;
        w * (1.0 + q * q).sqrt()
    }
}

/// `DLARTG(f, g) → (c, s, r)` (LAPACK 3.10+ `dlartg.f90`).
fn dlartg(f: f64, g: f64) -> (f64, f64, f64) {
    let safmin = SFMIN;
    let safmax = 1.0 / safmin;
    let rtmin = safmin.sqrt();
    let rtmax = (safmax / 2.0).sqrt();
    let (f1, g1) = (f.abs(), g.abs());
    if g == 0.0 {
        (1.0, 0.0, f)
    } else if f == 0.0 {
        (0.0, sign(1.0, g), g1)
    } else if f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax {
        let d = (f * f + g * g).sqrt();
        let c = f1 / d;
        let r = sign(d, f);
        (c, g / r, r)
    } else {
        let u = safmax.min(safmin.max(f1).max(g1));
        let (fs, gs) = (f / u, g / u);
        let d = (fs * fs + gs * gs).sqrt();
        let c = fs.abs() / d;
        let r = sign(d, f);
        let s = gs / r;
        (c, s, r * u)
    }
}

/// `DLAS2(f, g, h) → (ssmin, ssmax)`.
fn dlas2(f: f64, g: f64, h: f64) -> (f64, f64) {
    let (fa, ga, ha) = (f.abs(), g.abs(), h.abs());
    let fhmn = fa.min(ha);
    let fhmx = fa.max(ha);
    if fhmn == 0.0 {
        let ssmax = if fhmx == 0.0 {
            ga
        } else {
            let q = fhmx.min(ga) / fhmx.max(ga);
            fhmx.max(ga) * (1.0 + q * q).sqrt()
        };
        (0.0, ssmax)
    } else if ga < fhmx {
        let as_ = 1.0 + fhmn / fhmx;
        let at = (fhmx - fhmn) / fhmx;
        let q = ga / fhmx;
        let au = q * q;
        let c = 2.0 / ((as_ * as_ + au).sqrt() + (at * at + au).sqrt());
        (fhmn * c, fhmx / c)
    } else {
        let au = fhmx / ga;
        if au == 0.0 {
            ((fhmn * fhmx) / ga, ga)
        } else {
            let as_ = 1.0 + fhmn / fhmx;
            let at = (fhmx - fhmn) / fhmx;
            let (p, q) = (as_ * au, at * au);
            let c = 1.0 / ((1.0 + p * p).sqrt() + (1.0 + q * q).sqrt());
            let ssmin = (fhmn * c) * au;
            (ssmin + ssmin, ga / (c + c))
        }
    }
}

/// `DLASV2(f, g, h) → (ssmin, ssmax, snr, csr, snl, csl)`.
fn dlasv2(f: f64, g: f64, h: f64) -> (f64, f64, f64, f64, f64, f64) {
    let (mut ft, mut fa, mut ht, mut ha) = (f, f.abs(), h, h.abs());
    let mut pmax = 1;
    let swap = ha > fa;
    if swap {
        pmax = 3;
        std::mem::swap(&mut ft, &mut ht);
        std::mem::swap(&mut fa, &mut ha);
    }
    let gt = g;
    let ga = gt.abs();
    let (ssmin, ssmax, clt, crt, slt, srt);
    if ga == 0.0 {
        ssmin = ha;
        ssmax = fa;
        clt = 1.0;
        crt = 1.0;
        slt = 0.0;
        srt = 0.0;
    } else {
        let mut gasmal = true;
        let mut out = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        if ga > fa {
            pmax = 2;
            if (fa / ga) < EPS {
                gasmal = false;
                let smin = if ha > 1.0 {
                    fa / (ga / ha)
                } else {
                    (fa / ga) * ha
                };
                out = (smin, ga, 1.0, ft / gt, ht / gt, 1.0);
            }
        }
        if gasmal {
            let d = fa - ha;
            let mut l = if d == fa { 1.0 } else { d / fa };
            let m = gt / ft;
            let mut t = 2.0 - l;
            let mm = m * m;
            let tt = t * t;
            let s = (tt + mm).sqrt();
            let r = if l == 0.0 {
                m.abs()
            } else {
                (l * l + mm).sqrt()
            };
            let a = 0.5 * (s + r);
            let smin = ha / a;
            let smax = fa * a;
            if mm == 0.0 {
                t = if l == 0.0 {
                    sign(2.0, ft) * sign(1.0, gt)
                } else {
                    gt / sign(d, ft) + m / t
                };
            } else {
                t = (m / (s + t) + m / (r + l)) * (1.0 + a);
            }
            l = (t * t + 4.0).sqrt();
            let crt_ = 2.0 / l;
            let srt_ = t / l;
            let clt_ = (crt_ + srt_ * m) / a;
            let slt_ = (ht / ft) * srt_ / a;
            out = (smin, smax, clt_, crt_, slt_, srt_);
        }
        (ssmin, ssmax, clt, crt, slt, srt) = out;
    }
    let (csl, snl, csr, snr) = if swap {
        (srt, crt, slt, clt)
    } else {
        (clt, slt, crt, srt)
    };
    let tsign = match pmax {
        1 => sign(1.0, csr) * sign(1.0, csl) * sign(1.0, f),
        2 => sign(1.0, snr) * sign(1.0, csl) * sign(1.0, g),
        _ => sign(1.0, snr) * sign(1.0, snl) * sign(1.0, h),
    };
    let ssmax = sign(ssmax, tsign);
    let ssmin = sign(ssmin, tsign * sign(1.0, f) * sign(1.0, h));
    (ssmin, ssmax, snr, csr, snl, csl)
}

/// `DLASCL('G', 0, 0, cfrom, cto, m, n, A, lda)`: multiply by `cto/cfrom` in steps
/// that cannot over/underflow.
pub(crate) fn dlascl(cfrom: f64, cto: f64, m: usize, n: usize, a: &mut [f64], lda: usize) {
    let smlnum = SFMIN;
    let bignum = 1.0 / smlnum;
    let (mut cfromc, mut ctoc) = (cfrom, cto);
    loop {
        let cfrom1 = cfromc * smlnum;
        let (mul, done);
        if cfrom1 == cfromc {
            mul = ctoc / cfromc;
            done = true;
        } else {
            let cto1 = ctoc / bignum;
            if cto1 == ctoc {
                mul = ctoc;
                done = true;
                cfromc = 1.0;
            } else if cfrom1.abs() > ctoc.abs() && ctoc != 0.0 {
                mul = smlnum;
                done = false;
                cfromc = cfrom1;
            } else if cto1.abs() > cfromc.abs() {
                mul = bignum;
                done = false;
                ctoc = cto1;
            } else {
                mul = ctoc / cfromc;
                done = true;
                if mul == 1.0 {
                    return;
                }
            }
        }
        for j in 0..n {
            for i in 0..m {
                a[i + j * lda] *= mul;
            }
        }
        if done {
            return;
        }
    }
}

/// `DLANGE('M')`: the largest `|a(i,j)|` (NaN propagates).
fn dlange_max(m: usize, n: usize, a: &[f64], lda: usize) -> f64 {
    let mut v = 0.0f64;
    for j in 0..n {
        for i in 0..m {
            let t = a[i + j * lda].abs();
            if v < t || t.is_nan() {
                v = t;
            }
        }
    }
    v
}

/// `DLANST('M', n, d, e)`.
pub(crate) fn dlanst_max(n: usize, d: &[f64], e: &[f64]) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let mut anorm = d[n - 1].abs();
    for i in 0..n - 1 {
        for v in [d[i].abs(), e[i].abs()] {
            if anorm < v || v.is_nan() {
                anorm = v;
            }
        }
    }
    anorm
}

/// `ILADLC(m, n, A)`: the last non-zero column (1-based; 0 if none).
fn iladlc(m: usize, n: usize, a: &[f64], lda: usize) -> usize {
    if n == 0 {
        return 0;
    }
    if a[(n - 1) * lda] != 0.0 || a[(m - 1) + (n - 1) * lda] != 0.0 {
        return n;
    }
    for j in (1..=n).rev() {
        for i in 0..m {
            if a[i + (j - 1) * lda] != 0.0 {
                return j;
            }
        }
    }
    0
}

/// `ILADLR(m, n, A)`: the last non-zero row (1-based; 0 if none).
fn iladlr(m: usize, n: usize, a: &[f64], lda: usize) -> usize {
    if m == 0 {
        return 0;
    }
    if a[m - 1] != 0.0 || a[(m - 1) + (n - 1) * lda] != 0.0 {
        return m;
    }
    let mut r = 0;
    for j in 0..n {
        let mut i = m;
        while i >= 1 && a[(i.max(1) - 1) + j * lda] == 0.0 {
            i -= 1;
        }
        r = r.max(i);
    }
    r
}

// ── Householder reflectors ────────────────────────────────────────────────────

/// `DLARFG(n, alpha, x, incx) → tau` (`alpha` and `x` overwritten).
pub(crate) fn dlarfg(n: usize, alpha: &mut f64, x: &mut [f64], incx: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let mut xnorm = ob::dnrm2(n - 1, x, incx);
    if xnorm == 0.0 {
        return 0.0;
    }
    let mut beta = -sign(dlapy2(*alpha, xnorm), *alpha);
    let safmin = SFMIN / EPS;
    let mut knt = 0;
    if beta.abs() < safmin {
        let rsafmn = 1.0 / safmin;
        loop {
            knt += 1;
            ob::dscal(n - 1, rsafmn, x, incx);
            beta *= rsafmn;
            *alpha *= rsafmn;
            if !(beta.abs() < safmin && knt < 20) {
                break;
            }
        }
        xnorm = ob::dnrm2(n - 1, x, incx);
        beta = -sign(dlapy2(*alpha, xnorm), *alpha);
    }
    let tau = (beta - *alpha) / beta;
    ob::dscal(n - 1, 1.0 / (*alpha - beta), x, incx);
    for _ in 0..knt {
        beta *= safmin;
    }
    *alpha = beta;
    tau
}

/// `DLARF('L', m, n, v, incv, tau, C, ldc)`: `C = (I - tau·v·vᵀ)·C`.
pub(crate) fn dlarf_left(
    m: usize,
    n: usize,
    v: &[f64],
    incv: usize,
    tau: f64,
    c: &mut [f64],
    ldc: usize,
) {
    if tau == 0.0 {
        return;
    }
    // `v` is a contiguous copy; the BLAS calls see the caller's stride (`dgemv_n` picks
    // its kernels by it)
    let v = &spread(v, incv);
    let mut lastv = m;
    while lastv > 0 && v[(lastv - 1) * incv] == 0.0 {
        lastv -= 1;
    }
    if lastv == 0 {
        return;
    }
    let lastc = iladlc(lastv, n, c, ldc);
    let mut work = vec![0.0; lastc];
    gemv(102, 112, lastv, lastc, c, ldc, v, incv, &mut work);
    ob::dger(lastv, lastc, -tau, v, incv, &work, 1, c, ldc);
}

/// `DLARF('R', m, n, v, incv, tau, C, ldc)`: `C = C·(I - tau·v·vᵀ)`.
fn dlarf_right(m: usize, n: usize, v: &[f64], incv: usize, tau: f64, c: &mut [f64], ldc: usize) {
    if tau == 0.0 {
        return;
    }
    // `v` is a contiguous copy; the BLAS calls see the caller's stride (`dgemv_n` picks
    // its kernels by it)
    let v = &spread(v, incv);
    let mut lastv = n;
    while lastv > 0 && v[(lastv - 1) * incv] == 0.0 {
        lastv -= 1;
    }
    if lastv == 0 {
        return;
    }
    let lastc = iladlr(m, lastv, c, ldc);
    let mut work = vec![0.0; lastc];
    gemv(102, 111, lastc, lastv, c, ldc, v, incv, &mut work);
    ob::dger(lastc, lastv, -tau, &work, 1, v, incv, c, ldc);
}

#[allow(clippy::too_many_arguments)]
fn gemv(
    order: i32,
    trans: i32,
    m: usize,
    n: usize,
    a: &[f64],
    lda: usize,
    x: &[f64],
    incx: usize,
    y: &mut [f64],
) {
    if m == 0 || n == 0 {
        y.iter_mut().for_each(|v| *v = 0.0);
        return;
    }
    let ok = ob::dgemv(order, trans, m, n, a, lda, x, incx, y, 1);
    assert!(
        ok,
        "dgemv {m}×{n} above OpenBLAS's threading threshold is not modelled"
    );
}

/// A contiguous copy of a strided vector out of `a` (Fortran passes it by reference
/// into the array the reflector updates; the values it reads are not the ones
/// written). OpenBLAS's `dnrm2`/`dscal` round the same at any stride, so the copy is
/// handed on with stride 1.
fn take(a: &[f64], start: usize, n: usize, inc: usize) -> Vec<f64> {
    (0..n).map(|i| a[start + i * inc]).collect()
}

/// `v` laid out at stride `inc` (gaps zero).
fn spread(v: &[f64], inc: usize) -> Vec<f64> {
    if inc == 1 {
        return v.to_vec();
    }
    let mut out = vec![0.0; v.len().saturating_sub(1) * inc + 1];
    for (i, x) in v.iter().enumerate() {
        out[i * inc] = *x;
    }
    out
}

// ── QR / LQ / bidiagonalisation (unblocked; ILAENV NB = 32, NX = 128) ─────────

/// `DGEQR2` (what `DGEQRF` runs for `min(m,n) ≤ 32`).
fn dgeqr2(m: usize, n: usize, a: &mut [f64], lda: usize, tau: &mut [f64]) {
    let k = m.min(n);
    for i in 1..=k {
        let ii = (i - 1) + (i - 1) * lda;
        let xs = (i + 1).min(m) - 1 + (i - 1) * lda; // A(MIN(I+1,M), I)
        let mut alpha = a[ii];
        let t = {
            let (x_start, len) = (xs, m - i);
            let mut x = take(a, x_start, len, 1);
            let t = dlarfg(m - i + 1, &mut alpha, &mut x, 1);
            for (k2, v) in x.iter().enumerate() {
                a[x_start + k2] = *v;
            }
            t
        };
        a[ii] = alpha;
        tau[i - 1] = t;
        if i < n {
            let aii = a[ii];
            a[ii] = 1.0;
            let v = take(a, ii, m - i + 1, 1);
            dlarf_left(m - i + 1, n - i, &v, 1, t, &mut a[ii + lda..], lda);
            a[ii] = aii;
        }
    }
}

/// `DGELQ2`.
fn dgelq2(m: usize, n: usize, a: &mut [f64], lda: usize, tau: &mut [f64]) {
    let k = m.min(n);
    for i in 1..=k {
        let ii = (i - 1) + (i - 1) * lda;
        let xs = (i - 1) + (i.min(n - 1)) * lda; // A(I, MIN(I+1,N))
        let mut alpha = a[ii];
        let mut x = take(a, xs, n - i, lda);
        let t = dlarfg(n - i + 1, &mut alpha, &mut x, 1);
        for (k2, v) in x.iter().enumerate() {
            a[xs + k2 * lda] = *v;
        }
        a[ii] = alpha;
        tau[i - 1] = t;
        if i < m {
            let aii = a[ii];
            a[ii] = 1.0;
            let v = take(a, ii, n - i + 1, lda);
            dlarf_right(m - i, n - i + 1, &v, lda, t, &mut a[ii + 1..], lda);
            a[ii] = aii;
        }
    }
}

/// `DORM2R(side, trans, m, n, k, A, tau, C)` (unblocked `DORMQR`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dorm2r(
    left: bool,
    notran: bool,
    m: usize,
    n: usize,
    k: usize,
    a: &mut [f64],
    lda: usize,
    tau: &[f64],
    c: &mut [f64],
    ldc: usize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    let forward = (left && !notran) || (!left && notran);
    let order: Vec<usize> = if forward {
        (1..=k).collect()
    } else {
        (1..=k).rev().collect()
    };
    for i in order {
        let ii = (i - 1) + (i - 1) * lda;
        let aii = a[ii];
        a[ii] = 1.0;
        if left {
            let v = take(a, ii, m - i + 1, 1);
            dlarf_left(m - i + 1, n, &v, 1, tau[i - 1], &mut c[i - 1..], ldc);
        } else {
            let v = take(a, ii, n - i + 1, 1);
            dlarf_right(
                m,
                n - i + 1,
                &v,
                1,
                tau[i - 1],
                &mut c[(i - 1) * ldc..],
                ldc,
            );
        }
        a[ii] = aii;
    }
}

/// `DORML2(side, trans, m, n, k, A, tau, C)` (unblocked `DORMLQ`).
#[allow(clippy::too_many_arguments)]
fn dorml2(
    left: bool,
    notran: bool,
    m: usize,
    n: usize,
    k: usize,
    a: &mut [f64],
    lda: usize,
    tau: &[f64],
    c: &mut [f64],
    ldc: usize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    let forward = (left && notran) || (!left && !notran);
    let order: Vec<usize> = if forward {
        (1..=k).collect()
    } else {
        (1..=k).rev().collect()
    };
    for i in order {
        let ii = (i - 1) + (i - 1) * lda;
        let aii = a[ii];
        a[ii] = 1.0;
        if left {
            let v = take(a, ii, m - i + 1, lda);
            dlarf_left(m - i + 1, n, &v, lda, tau[i - 1], &mut c[i - 1..], ldc);
        } else {
            let v = take(a, ii, n - i + 1, lda);
            dlarf_right(
                m,
                n - i + 1,
                &v,
                lda,
                tau[i - 1],
                &mut c[(i - 1) * ldc..],
                ldc,
            );
        }
        a[ii] = aii;
    }
}

/// `DGEBD2(m, n, A) → (d, e, tauq, taup)` (what `DGEBRD` runs for `min(m,n) < 128`).
#[allow(clippy::too_many_arguments)]
fn dgebd2(
    m: usize,
    n: usize,
    a: &mut [f64],
    lda: usize,
    d: &mut [f64],
    e: &mut [f64],
    tauq: &mut [f64],
    taup: &mut [f64],
) {
    let at = |i: usize, j: usize| (i - 1) + (j - 1) * lda;
    if m >= n {
        for i in 1..=n {
            let xs = at((i + 1).min(m), i);
            let mut alpha = a[at(i, i)];
            let mut x = take(a, xs, m - i, 1);
            let t = dlarfg(m - i + 1, &mut alpha, &mut x, 1);
            put(a, xs, &x, 1);
            a[at(i, i)] = alpha;
            tauq[i - 1] = t;
            d[i - 1] = a[at(i, i)];
            a[at(i, i)] = 1.0;
            if i < n {
                let v = take(a, at(i, i), m - i + 1, 1);
                dlarf_left(m - i + 1, n - i, &v, 1, t, &mut a[at(i, i + 1)..], lda);
            }
            a[at(i, i)] = d[i - 1];
            if i < n {
                let xs = at(i, (i + 2).min(n));
                let mut alpha = a[at(i, i + 1)];
                let mut x = take(a, xs, n - i - 1, lda);
                let t = dlarfg(n - i, &mut alpha, &mut x, 1);
                put(a, xs, &x, lda);
                a[at(i, i + 1)] = alpha;
                taup[i - 1] = t;
                e[i - 1] = a[at(i, i + 1)];
                a[at(i, i + 1)] = 1.0;
                let v = take(a, at(i, i + 1), n - i, lda);
                if m > i {
                    dlarf_right(m - i, n - i, &v, lda, t, &mut a[at(i + 1, i + 1)..], lda);
                }
                a[at(i, i + 1)] = e[i - 1];
            } else {
                taup[i - 1] = 0.0;
            }
        }
    } else {
        for i in 1..=m {
            let xs = at(i, (i + 1).min(n));
            let mut alpha = a[at(i, i)];
            let mut x = take(a, xs, n - i, lda);
            let t = dlarfg(n - i + 1, &mut alpha, &mut x, 1);
            put(a, xs, &x, lda);
            a[at(i, i)] = alpha;
            taup[i - 1] = t;
            d[i - 1] = a[at(i, i)];
            a[at(i, i)] = 1.0;
            if i < m {
                let v = take(a, at(i, i), n - i + 1, lda);
                dlarf_right(m - i, n - i + 1, &v, lda, t, &mut a[at(i + 1, i)..], lda);
            }
            a[at(i, i)] = d[i - 1];
            if i < m {
                let xs = at((i + 2).min(m), i);
                let mut alpha = a[at(i + 1, i)];
                let mut x = take(a, xs, m - i - 1, 1);
                let t = dlarfg(m - i, &mut alpha, &mut x, 1);
                put(a, xs, &x, 1);
                a[at(i + 1, i)] = alpha;
                tauq[i - 1] = t;
                e[i - 1] = a[at(i + 1, i)];
                a[at(i + 1, i)] = 1.0;
                let v = take(a, at(i + 1, i), m - i, 1);
                if n > i {
                    dlarf_left(m - i, n - i, &v, 1, t, &mut a[at(i + 1, i + 1)..], lda);
                }
                a[at(i + 1, i)] = e[i - 1];
            } else {
                tauq[i - 1] = 0.0;
            }
        }
    }
}

fn put(a: &mut [f64], start: usize, x: &[f64], inc: usize) {
    for (k, v) in x.iter().enumerate() {
        a[start + k * inc] = *v;
    }
}

/// `DORMBR(vect, side='L', trans, m, n, k, A, tau, C, lwork)`.
#[allow(clippy::too_many_arguments)]
fn dormbr_left(
    apply_q: bool,
    notran: bool,
    m: usize,
    n: usize,
    k: usize,
    a: &mut [f64],
    lda: usize,
    tau: &[f64],
    c: &mut [f64],
    ldc: usize,
    lwork: i64,
) {
    let nq = m;
    if m == 0 || n == 0 {
        return;
    }
    if apply_q {
        if nq >= k {
            dormxx_left(false, notran, m, n, k, a, lda, tau, c, ldc, lwork);
        } else if nq > 1 {
            dormxx_left(
                false,
                notran,
                m - 1,
                n,
                nq - 1,
                &mut a[1..],
                lda,
                tau,
                &mut c[1..],
                ldc,
                lwork,
            );
        }
    } else {
        let transt_notran = !notran;
        if nq > k {
            dormxx_left(true, transt_notran, m, n, k, a, lda, tau, c, ldc, lwork);
        } else if nq > 1 {
            dormxx_left(
                true,
                transt_notran,
                m - 1,
                n,
                nq - 1,
                &mut a[lda..],
                lda,
                tau,
                &mut c[1..],
                ldc,
                lwork,
            );
        }
    }
}

// ── bidiagonal SVD ────────────────────────────────────────────────────────────

/// `DLASR(side, 'V', direct, m, n, c, s, A)`.
#[allow(clippy::too_many_arguments)]
fn dlasr(
    left: bool,
    forward: bool,
    m: usize,
    n: usize,
    c: &[f64],
    s: &[f64],
    a: &mut [f64],
    lda: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    if left {
        let js: Vec<usize> = if forward {
            (1..m).collect()
        } else {
            (1..m).rev().collect()
        };
        for j in js {
            let (ct, st) = (c[j - 1], s[j - 1]);
            if ct != 1.0 || st != 0.0 {
                for i in 0..n {
                    let (r0, r1) = ((j - 1) + i * lda, j + i * lda);
                    let temp = a[r1];
                    a[r1] = ct * temp - st * a[r0];
                    a[r0] = st * temp + ct * a[r0];
                }
            }
        }
    } else {
        let js: Vec<usize> = if forward {
            (1..n).collect()
        } else {
            (1..n).rev().collect()
        };
        for j in js {
            let (ct, st) = (c[j - 1], s[j - 1]);
            if ct != 1.0 || st != 0.0 {
                for i in 0..m {
                    let (c0, c1) = (i + (j - 1) * lda, i + j * lda);
                    let temp = a[c1];
                    a[c1] = ct * temp - st * a[c0];
                    a[c0] = st * temp + ct * a[c0];
                }
            }
        }
    }
}

/// `DSWAP` of two rows (stride `ld`) or columns (stride 1).
fn swap_strided(a: &mut [f64], i0: usize, i1: usize, n: usize, inc: usize) {
    for k in 0..n {
        a.swap(i0 + k * inc, i1 + k * inc);
    }
}

/// `DBDSQR('U', n, ncvt, nru, ncc, d, e, VT, U, C)` with rotations (`ncvt + nru +
/// ncc > 0`; the `DLASQ1` path is not ported).
/// Returns `info`.
#[allow(clippy::too_many_arguments)]
fn dbdsqr_upper(
    n: usize,
    ncvt: usize,
    nru: usize,
    ncc: usize,
    d: &mut [f64],
    e: &mut [f64],
    vt: &mut [f64],
    ldvt: usize,
    u: &mut [f64],
    ldu: usize,
    c: &mut [f64],
    ldc: usize,
) -> usize {
    const MAXITR: usize = 6;
    const MEIGTH: f64 = -0.125;
    const HNDRTH: f64 = 0.01;
    const TEN: f64 = 10.0;
    const HNDRD: f64 = 100.0;
    if n == 0 {
        return 0;
    }
    let mut info = 0;
    if n > 1 {
        let nm1 = n - 1;
        let nm12 = nm1 + nm1;
        let nm13 = nm12 + nm1;
        let mut work = vec![0.0f64; 4 * n];
        let mut idir = 0;
        let eps = EPS;
        let unfl = SFMIN;
        // EPS**MEIGTH: gfortran calls libm `pow` (glibc-exact on every target)
        let tolmul = TEN.max(HNDRD.min(rlx_libm::pow(eps, MEIGTH)));
        let tol = tolmul * eps;
        let mut smax = 0.0f64;
        for v in d.iter().take(n) {
            smax = smax.max(v.abs());
        }
        for v in e.iter().take(n - 1) {
            smax = smax.max(v.abs());
        }
        let mut smin;
        // tol ≥ 0 always here
        let mut sminoa = d[0].abs();
        if sminoa != 0.0 {
            let mut mu = sminoa;
            for i in 2..=n {
                mu = d[i - 1].abs() * (mu / (mu + e[i - 2].abs()));
                sminoa = sminoa.min(mu);
                if sminoa == 0.0 {
                    break;
                }
            }
        }
        sminoa /= (n as f64).sqrt();
        let thresh = (tol * sminoa).max(MAXITR as f64 * (n as f64 * (n as f64 * unfl)));
        let maxitdivn = MAXITR * n;
        let mut iterdivn = 0usize;
        let mut iter: isize = -1;
        let mut oldll: isize = -1;
        let mut oldm: isize = -1;
        let mut m = n;
        let dd = |i: usize| i - 1; // 1-based → 0-based
        'outer: loop {
            // 60
            if m <= 1 {
                break;
            }
            if iter >= n as isize {
                iter -= n as isize;
                iterdivn += 1;
                if iterdivn >= maxitdivn {
                    // 200: failure to converge
                    info = (0..n - 1).filter(|&i| e[i] != 0.0).count();
                    return info;
                }
            }
            smax = d[dd(m)].abs();
            let mut ll = 0usize;
            let mut found = false;
            for lll in 1..m {
                let l = m - lll;
                let abss = d[dd(l)].abs();
                let abse = e[dd(l)].abs();
                if abse <= thresh {
                    ll = l;
                    found = true;
                    break;
                }
                smax = smax.max(abss).max(abse);
            }
            if found {
                // 80
                e[dd(ll)] = 0.0;
                if ll == m - 1 {
                    m -= 1;
                    continue 'outer;
                }
            } else {
                ll = 0;
            }
            // 90
            ll += 1;
            if ll == m - 1 {
                let (sigmn, sigmx, sinr, cosr, sinl, cosl) =
                    dlasv2(d[dd(m - 1)], e[dd(m - 1)], d[dd(m)]);
                d[dd(m - 1)] = sigmx;
                e[dd(m - 1)] = 0.0;
                d[dd(m)] = sigmn;
                if ncvt > 0 {
                    let (r0, r1) = (m - 2, m - 1);
                    rot_rows(vt, ldvt, r0, r1, ncvt, cosr, sinr);
                }
                if nru > 0 {
                    rot_cols(u, ldu, m - 2, m - 1, nru, cosl, sinl);
                }
                if ncc > 0 {
                    rot_rows(c, ldc, m - 2, m - 1, ncc, cosl, sinl);
                }
                m -= 2;
                continue 'outer;
            }
            if ll as isize > oldm || (m as isize) < oldll {
                idir = if d[dd(ll)].abs() >= d[dd(m)].abs() {
                    1
                } else {
                    2
                };
            }
            if idir == 1 {
                if e[dd(m - 1)].abs() <= tol.abs() * d[dd(m)].abs() {
                    e[dd(m - 1)] = 0.0;
                    continue 'outer;
                }
                let mut mu = d[dd(ll)].abs();
                smin = mu;
                for lll in ll..m {
                    if e[dd(lll)].abs() <= tol * mu {
                        e[dd(lll)] = 0.0;
                        continue 'outer;
                    }
                    mu = d[dd(lll + 1)].abs() * (mu / (mu + e[dd(lll)].abs()));
                    smin = smin.min(mu);
                }
            } else {
                if e[dd(ll)].abs() <= tol.abs() * d[dd(ll)].abs() {
                    e[dd(ll)] = 0.0;
                    continue 'outer;
                }
                let mut mu = d[dd(m)].abs();
                smin = mu;
                for lll in (ll..m).rev() {
                    if e[dd(lll)].abs() <= tol * mu {
                        e[dd(lll)] = 0.0;
                        continue 'outer;
                    }
                    mu = d[dd(lll)].abs() * (mu / (mu + e[dd(lll)].abs()));
                    smin = smin.min(mu);
                }
            }
            oldll = ll as isize;
            oldm = m as isize;
            let mut shift;
            if n as f64 * tol * (smin / smax) <= eps.max(HNDRTH * tol) {
                shift = 0.0;
            } else {
                let (sll, r);
                if idir == 1 {
                    sll = d[dd(ll)].abs();
                    (shift, r) = dlas2(d[dd(m - 1)], e[dd(m - 1)], d[dd(m)]);
                } else {
                    sll = d[dd(m)].abs();
                    (shift, r) = dlas2(d[dd(ll)], e[dd(ll)], d[dd(ll + 1)]);
                }
                let _ = r;
                if sll > 0.0 {
                    let q = shift / sll;
                    if q * q < eps {
                        shift = 0.0;
                    }
                }
            }
            iter += (m - ll) as isize;
            let w = |k: usize| k - 1; // WORK(k), 1-based
            if shift == 0.0 {
                if idir == 1 {
                    let (mut cs, mut oldcs, mut oldsn) = (1.0, 1.0, 0.0);
                    let mut sn;
                    for i in ll..m {
                        let r;
                        (cs, sn, r) = dlartg(d[dd(i)] * cs, e[dd(i)]);
                        if i > ll {
                            e[dd(i - 1)] = oldsn * r;
                        }
                        let dnew;
                        (oldcs, oldsn, dnew) = dlartg(oldcs * r, d[dd(i + 1)] * sn);
                        d[dd(i)] = dnew;
                        work[w(i - ll + 1)] = cs;
                        work[w(i - ll + 1 + nm1)] = sn;
                        work[w(i - ll + 1 + nm12)] = oldcs;
                        work[w(i - ll + 1 + nm13)] = oldsn;
                    }
                    let h = d[dd(m)] * cs;
                    d[dd(m)] = h * oldcs;
                    e[dd(m - 1)] = h * oldsn;
                    if ncvt > 0 {
                        let (cw, sw) = (work[w(1)..].to_vec(), work[w(n)..].to_vec());
                        dlasr(
                            true,
                            true,
                            m - ll + 1,
                            ncvt,
                            &cw,
                            &sw,
                            &mut vt[ll - 1..],
                            ldvt,
                        );
                    }
                    if nru > 0 {
                        let (cw, sw) = (work[w(nm12 + 1)..].to_vec(), work[w(nm13 + 1)..].to_vec());
                        dlasr(
                            false,
                            true,
                            nru,
                            m - ll + 1,
                            &cw,
                            &sw,
                            &mut u[(ll - 1) * ldu..],
                            ldu,
                        );
                    }
                    if ncc > 0 {
                        let (cw, sw) = (work[w(nm12 + 1)..].to_vec(), work[w(nm13 + 1)..].to_vec());
                        dlasr(true, true, m - ll + 1, ncc, &cw, &sw, &mut c[ll - 1..], ldc);
                    }
                    if e[dd(m - 1)].abs() <= thresh {
                        e[dd(m - 1)] = 0.0;
                    }
                } else {
                    let (mut cs, mut oldcs, mut oldsn) = (1.0, 1.0, 0.0);
                    let mut sn;
                    for i in (ll + 1..=m).rev() {
                        let r;
                        (cs, sn, r) = dlartg(d[dd(i)] * cs, e[dd(i - 1)]);
                        if i < m {
                            e[dd(i)] = oldsn * r;
                        }
                        let dnew;
                        (oldcs, oldsn, dnew) = dlartg(oldcs * r, d[dd(i - 1)] * sn);
                        d[dd(i)] = dnew;
                        work[w(i - ll)] = cs;
                        work[w(i - ll + nm1)] = -sn;
                        work[w(i - ll + nm12)] = oldcs;
                        work[w(i - ll + nm13)] = -oldsn;
                    }
                    let h = d[dd(ll)] * cs;
                    d[dd(ll)] = h * oldcs;
                    e[dd(ll)] = h * oldsn;
                    if ncvt > 0 {
                        let (cw, sw) = (work[w(nm12 + 1)..].to_vec(), work[w(nm13 + 1)..].to_vec());
                        dlasr(
                            true,
                            false,
                            m - ll + 1,
                            ncvt,
                            &cw,
                            &sw,
                            &mut vt[ll - 1..],
                            ldvt,
                        );
                    }
                    if nru > 0 {
                        let (cw, sw) = (work[w(1)..].to_vec(), work[w(n)..].to_vec());
                        dlasr(
                            false,
                            false,
                            nru,
                            m - ll + 1,
                            &cw,
                            &sw,
                            &mut u[(ll - 1) * ldu..],
                            ldu,
                        );
                    }
                    if ncc > 0 {
                        let (cw, sw) = (work[w(1)..].to_vec(), work[w(n)..].to_vec());
                        dlasr(
                            true,
                            false,
                            m - ll + 1,
                            ncc,
                            &cw,
                            &sw,
                            &mut c[ll - 1..],
                            ldc,
                        );
                    }
                    if e[dd(ll)].abs() <= thresh {
                        e[dd(ll)] = 0.0;
                    }
                }
            } else if idir == 1 {
                let mut f = (d[dd(ll)].abs() - shift) * (sign(1.0, d[dd(ll)]) + shift / d[dd(ll)]);
                let mut g = e[dd(ll)];
                for i in ll..m {
                    let (cosr, sinr, r) = dlartg(f, g);
                    if i > ll {
                        e[dd(i - 1)] = r;
                    }
                    f = cosr * d[dd(i)] + sinr * e[dd(i)];
                    e[dd(i)] = cosr * e[dd(i)] - sinr * d[dd(i)];
                    g = sinr * d[dd(i + 1)];
                    d[dd(i + 1)] = cosr * d[dd(i + 1)];
                    let (cosl, sinl, r) = dlartg(f, g);
                    d[dd(i)] = r;
                    f = cosl * e[dd(i)] + sinl * d[dd(i + 1)];
                    d[dd(i + 1)] = cosl * d[dd(i + 1)] - sinl * e[dd(i)];
                    if i < m - 1 {
                        g = sinl * e[dd(i + 1)];
                        e[dd(i + 1)] = cosl * e[dd(i + 1)];
                    }
                    work[w(i - ll + 1)] = cosr;
                    work[w(i - ll + 1 + nm1)] = sinr;
                    work[w(i - ll + 1 + nm12)] = cosl;
                    work[w(i - ll + 1 + nm13)] = sinl;
                }
                e[dd(m - 1)] = f;
                if ncvt > 0 {
                    let (cw, sw) = (work[w(1)..].to_vec(), work[w(n)..].to_vec());
                    dlasr(
                        true,
                        true,
                        m - ll + 1,
                        ncvt,
                        &cw,
                        &sw,
                        &mut vt[ll - 1..],
                        ldvt,
                    );
                }
                if nru > 0 {
                    let (cw, sw) = (work[w(nm12 + 1)..].to_vec(), work[w(nm13 + 1)..].to_vec());
                    dlasr(
                        false,
                        true,
                        nru,
                        m - ll + 1,
                        &cw,
                        &sw,
                        &mut u[(ll - 1) * ldu..],
                        ldu,
                    );
                }
                if ncc > 0 {
                    let (cw, sw) = (work[w(nm12 + 1)..].to_vec(), work[w(nm13 + 1)..].to_vec());
                    dlasr(true, true, m - ll + 1, ncc, &cw, &sw, &mut c[ll - 1..], ldc);
                }
                if e[dd(m - 1)].abs() <= thresh {
                    e[dd(m - 1)] = 0.0;
                }
            } else {
                let mut f = (d[dd(m)].abs() - shift) * (sign(1.0, d[dd(m)]) + shift / d[dd(m)]);
                let mut g = e[dd(m - 1)];
                for i in (ll + 1..=m).rev() {
                    let (cosr, sinr, r) = dlartg(f, g);
                    if i < m {
                        e[dd(i)] = r;
                    }
                    f = cosr * d[dd(i)] + sinr * e[dd(i - 1)];
                    e[dd(i - 1)] = cosr * e[dd(i - 1)] - sinr * d[dd(i)];
                    g = sinr * d[dd(i - 1)];
                    d[dd(i - 1)] = cosr * d[dd(i - 1)];
                    let (cosl, sinl, r) = dlartg(f, g);
                    d[dd(i)] = r;
                    f = cosl * e[dd(i - 1)] + sinl * d[dd(i - 1)];
                    d[dd(i - 1)] = cosl * d[dd(i - 1)] - sinl * e[dd(i - 1)];
                    if i > ll + 1 {
                        g = sinl * e[dd(i - 2)];
                        e[dd(i - 2)] = cosl * e[dd(i - 2)];
                    }
                    work[w(i - ll)] = cosr;
                    work[w(i - ll + nm1)] = -sinr;
                    work[w(i - ll + nm12)] = cosl;
                    work[w(i - ll + nm13)] = -sinl;
                }
                e[dd(ll)] = f;
                if e[dd(ll)].abs() <= thresh {
                    e[dd(ll)] = 0.0;
                }
                if ncvt > 0 {
                    let (cw, sw) = (work[w(nm12 + 1)..].to_vec(), work[w(nm13 + 1)..].to_vec());
                    dlasr(
                        true,
                        false,
                        m - ll + 1,
                        ncvt,
                        &cw,
                        &sw,
                        &mut vt[ll - 1..],
                        ldvt,
                    );
                }
                if nru > 0 {
                    let (cw, sw) = (work[w(1)..].to_vec(), work[w(n)..].to_vec());
                    dlasr(
                        false,
                        false,
                        nru,
                        m - ll + 1,
                        &cw,
                        &sw,
                        &mut u[(ll - 1) * ldu..],
                        ldu,
                    );
                }
                if ncc > 0 {
                    let (cw, sw) = (work[w(1)..].to_vec(), work[w(n)..].to_vec());
                    dlasr(
                        true,
                        false,
                        m - ll + 1,
                        ncc,
                        &cw,
                        &sw,
                        &mut c[ll - 1..],
                        ldc,
                    );
                }
            }
        }
    }
    // 160: make singular values non-negative, then sort decreasing (selection)
    for i in 0..n {
        if d[i] < 0.0 {
            d[i] = -d[i];
            if ncvt > 0 {
                for k in 0..ncvt {
                    vt[i + k * ldvt] *= -1.0;
                }
            }
        }
    }
    for i in 1..n {
        let mut isub = 1;
        let mut smin = d[0];
        for j in 2..=n + 1 - i {
            if d[j - 1] <= smin {
                isub = j;
                smin = d[j - 1];
            }
        }
        let tgt = n + 1 - i;
        if isub != tgt {
            d[isub - 1] = d[tgt - 1];
            d[tgt - 1] = smin;
            if ncvt > 0 {
                swap_strided(vt, isub - 1, tgt - 1, ncvt, ldvt);
            }
            if nru > 0 {
                swap_strided(u, (isub - 1) * ldu, (tgt - 1) * ldu, nru, 1);
            }
            if ncc > 0 {
                swap_strided(c, isub - 1, tgt - 1, ncc, ldc);
            }
        }
    }
    info
}

/// `DROT(n, X(r0,1), ld, X(r1,1), ld, c, s)` on two rows.
fn rot_rows(x: &mut [f64], ld: usize, r0: usize, r1: usize, n: usize, c: f64, s: f64) {
    for k in 0..n {
        let (a, b) = (x[r0 + k * ld], x[r1 + k * ld]);
        x[r0 + k * ld] = c.mul_add(a, s * b);
        x[r1 + k * ld] = c.mul_add(b, -(s * a));
    }
}

/// `DROT(n, X(1,c0), 1, X(1,c1), 1, c, s)` on two columns.
fn rot_cols(x: &mut [f64], ld: usize, c0: usize, c1: usize, n: usize, c: f64, s: f64) {
    for k in 0..n {
        let (a, b) = (x[k + c0 * ld], x[k + c1 * ld]);
        x[k + c0 * ld] = c.mul_add(a, s * b);
        x[k + c1 * ld] = c.mul_add(b, -(s * a));
    }
}

/// `DLAMRG(n1, n2, a, dtrd1, dtrd2) → index` (1-based indices into `a`).
pub(crate) fn dlamrg(n1: usize, n2: usize, a: &[f64], dtrd1: i64, dtrd2: i64, index: &mut [usize]) {
    let (mut n1sv, mut n2sv) = (n1, n2);
    let mut ind1: i64 = if dtrd1 > 0 { 1 } else { n1 as i64 };
    let mut ind2: i64 = if dtrd2 > 0 {
        1 + n1 as i64
    } else {
        (n1 + n2) as i64
    };
    let mut i = 0;
    while n1sv > 0 && n2sv > 0 {
        if a[(ind1 - 1) as usize] <= a[(ind2 - 1) as usize] {
            index[i] = ind1 as usize;
            ind1 += dtrd1;
            n1sv -= 1;
        } else {
            index[i] = ind2 as usize;
            ind2 += dtrd2;
            n2sv -= 1;
        }
        i += 1;
    }
    if n1sv == 0 {
        for _ in 0..n2sv {
            index[i] = ind2 as usize;
            i += 1;
            ind2 += dtrd2;
        }
    } else {
        for _ in 0..n1sv {
            index[i] = ind1 as usize;
            i += 1;
            ind1 += dtrd1;
        }
    }
}

/// `DLASDQ('U', sqre, n, ncvt, nru, ncc, d, e, VT, U, C)`: `DBDSQR` (after rotating a
/// `sqre = 1` matrix to upper bidiagonal), then the singular values sorted with
/// their vectors. Returns `info`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dlasdq(
    sqre: usize,
    n: usize,
    ncvt: usize,
    nru: usize,
    ncc: usize,
    d: &mut [f64],
    e: &mut [f64],
    vt: &mut [f64],
    ldvt: usize,
    u: &mut [f64],
    ldu: usize,
    c: &mut [f64],
    ldc: usize,
) -> usize {
    if n == 0 {
        return 0;
    }
    let rotate = ncvt > 0 || nru > 0 || ncc > 0;
    let np1 = n + 1;
    let mut sqre1 = sqre;
    let mut work = vec![0.0f64; 2 * n.max(1)];
    let mut lower = false;
    if sqre1 == 1 {
        // IUPLO = 1, SQRE1 = 1: rotate to lower bidiagonal
        for i in 1..n {
            let (cs, sn, r) = dlartg(d[i - 1], e[i - 1]);
            d[i - 1] = r;
            e[i - 1] = sn * d[i];
            d[i] *= cs;
            if rotate {
                work[i - 1] = cs;
                work[n + i - 1] = sn;
            }
        }
        let (cs, sn, r) = dlartg(d[n - 1], e[n - 1]);
        d[n - 1] = r;
        e[n - 1] = 0.0;
        if rotate {
            work[n - 1] = cs;
            work[2 * n - 1] = sn;
        }
        lower = true;
        sqre1 = 0;
        if ncvt > 0 {
            let (cw, sw) = (work[..].to_vec(), work[n..].to_vec());
            dlasr(true, true, np1, ncvt, &cw, &sw, vt, ldvt);
        }
    }
    if lower {
        // IUPLO = 2: rotate lower to upper bidiagonal
        for i in 1..n {
            let (cs, sn, r) = dlartg(d[i - 1], e[i - 1]);
            d[i - 1] = r;
            e[i - 1] = sn * d[i];
            d[i] *= cs;
            if rotate {
                work[i - 1] = cs;
                work[n + i - 1] = sn;
            }
        }
        if sqre1 == 1 {
            unreachable!("SQRE1 is 0 after the first rotation");
        }
        if nru > 0 {
            let (cw, sw) = (work[..].to_vec(), work[n..].to_vec());
            dlasr(false, true, nru, n, &cw, &sw, u, ldu);
        }
        if ncc > 0 {
            let (cw, sw) = (work[..].to_vec(), work[n..].to_vec());
            dlasr(true, true, n, ncc, &cw, &sw, c, ldc);
        }
    }
    let info = dbdsqr_upper(n, ncvt, nru, ncc, d, e, vt, ldvt, u, ldu, c, ldc);
    for i in 1..=n {
        let mut isub = i;
        let mut smin = d[i - 1];
        for j in i + 1..=n {
            if d[j - 1] < smin {
                isub = j;
                smin = d[j - 1];
            }
        }
        if isub != i {
            d[isub - 1] = d[i - 1];
            d[i - 1] = smin;
            if ncvt > 0 {
                swap_strided(vt, isub - 1, i - 1, ncvt, ldvt);
            }
            if nru > 0 {
                swap_strided(u, (isub - 1) * ldu, (i - 1) * ldu, nru, 1);
            }
            if ncc > 0 {
                swap_strided(c, isub - 1, i - 1, ncc, ldc);
            }
        }
    }
    info
}

/// `DLALSD(uplo, smlsiz, n, nrhs, d, e, B, rcond) → (rank, info)`; only the
/// `n ≤ SMLSIZ` branch is ported (`None` above it — LAPACK's divide and conquer).
#[allow(clippy::too_many_arguments)]
fn dlalsd(
    upper: bool,
    n: usize,
    nrhs: usize,
    d: &mut [f64],
    e: &mut [f64],
    b: &mut [f64],
    ldb: usize,
    rcond: f64,
) -> Option<(usize, usize)> {
    let eps = EPS;
    let rcnd = if rcond <= 0.0 || rcond >= 1.0 {
        eps
    } else {
        rcond
    };
    let mut rank = 0;
    if n == 0 {
        return Some((0, 0));
    }
    if n == 1 {
        if d[0] == 0.0 {
            for j in 0..nrhs {
                b[j * ldb] = 0.0;
            }
        } else {
            rank = 1;
            dlascl(d[0], 1.0, 1, nrhs, b, ldb);
            d[0] = d[0].abs();
        }
        return Some((rank, 0));
    }
    if !upper {
        let mut work = vec![0.0; 2 * n];
        for i in 1..n {
            let (cs, sn, r) = dlartg(d[i - 1], e[i - 1]);
            d[i - 1] = r;
            e[i - 1] = sn * d[i];
            d[i] *= cs;
            if nrhs == 1 {
                rot_rows(b, ldb, i - 1, i, 1, cs, sn);
            } else {
                work[2 * i - 2] = cs;
                work[2 * i - 1] = sn;
            }
        }
        if nrhs > 1 {
            for col in 0..nrhs {
                for j in 1..n {
                    let (cs, sn) = (work[2 * j - 2], work[2 * j - 1]);
                    rot_rows(&mut b[col * ldb..], ldb, j - 1, j, 1, cs, sn);
                }
            }
        }
    }
    let nm1 = n - 1;
    let orgnrm = dlanst_max(n, d, e);
    if orgnrm == 0.0 {
        for j in 0..nrhs {
            for i in 0..n {
                b[i + j * ldb] = 0.0;
            }
        }
        return Some((0, 0));
    }
    dlascl(orgnrm, 1.0, n, 1, d, n);
    dlascl(orgnrm, 1.0, nm1, 1, e, nm1);
    if n > SMLSIZ {
        return Some(dlalsd_dc(n, nrhs, d, e, b, ldb, rcnd, orgnrm));
    }
    // VT = I (n×n), U unused
    let mut vt = vec![0.0; n * n];
    for i in 0..n {
        vt[i + i * n] = 1.0;
    }
    let info = dlasdq(0, n, n, 0, nrhs, d, e, &mut vt, n, &mut [], 1, b, ldb);
    if info != 0 {
        return Some((0, info));
    }
    let imax = ob_idamax(&d[..n]);
    let tol = rcnd * d[imax].abs();
    for i in 0..n {
        if d[i] <= tol {
            for j in 0..nrhs {
                b[i + j * ldb] = 0.0;
            }
        } else {
            dlascl(d[i], 1.0, 1, nrhs, &mut b[i..], ldb);
            rank += 1;
        }
    }
    // WORK(NWORK) = VTᵀ · B, then B ← it
    let mut bb = vec![0.0; n * nrhs];
    for j in 0..nrhs {
        for i in 0..n {
            bb[i + j * n] = b[i + j * ldb];
        }
    }
    let mut out = vec![0.0; n * nrhs];
    ob::dgemm(102, 112, 111, n, nrhs, n, &vt, n, &bb, n, &mut out, n);
    for j in 0..nrhs {
        for i in 0..n {
            b[i + j * ldb] = out[i + j * n];
        }
    }
    dlascl(1.0, orgnrm, n, 1, d, n);
    // DLASRT('D'): decreasing
    d[..n].sort_by(|x, y| y.partial_cmp(x).unwrap());
    dlascl(orgnrm, 1.0, n, nrhs, b, ldb);
    Some((rank, 0))
}

/// `DLALSD`'s divide-and-conquer branch (`n > SMLSIZ`): split at negligible `e(i)`,
/// solve each piece (`DLASDQ` when small, else `DLASDA` + `DLALSA`), scale by the
/// singular values, and apply the right singular vectors back. `(rank, info)`.
#[allow(clippy::too_many_arguments)]
fn dlalsd_dc(
    n: usize,
    nrhs: usize,
    d: &mut [f64],
    e: &mut [f64],
    b: &mut [f64],
    ldb: usize,
    rcnd: f64,
    orgnrm: f64,
) -> (usize, usize) {
    use crate::lapack_dc::{Factors, dlalsa, dlasda};
    let eps = EPS;
    let nm1 = n - 1;
    let nlvl = ((n as f64 / (SMLSIZ + 1) as f64).ln() / 2f64.ln()) as i64 as usize + 1;
    let smlszp = SMLSIZ + 1;
    let ld = n;
    let mut u = vec![0.0f64; ld * SMLSIZ];
    let mut vt = vec![0.0f64; ld * smlszp];
    let mut difl = vec![0.0f64; ld * nlvl];
    let mut difr = vec![0.0f64; ld * 2 * nlvl];
    let mut z = vec![0.0f64; ld * nlvl];
    let mut c = vec![0.0f64; n];
    let mut s = vec![0.0f64; n];
    let mut poles = vec![0.0f64; ld * 2 * nlvl];
    let mut givnum = vec![0.0f64; ld * 2 * nlvl];
    let mut bx = vec![0.0f64; ld * nrhs];
    let mut kk = vec![0usize; n];
    let mut givptr = vec![0usize; n];
    let mut perm = vec![0usize; ld * nlvl];
    let mut givcol = vec![0usize; ld * 2 * nlvl];
    let mut starts = vec![];
    let mut sizes = vec![];
    for v in d.iter_mut().take(n) {
        if v.abs() < eps {
            *v = sign(eps, *v);
        }
    }
    let mut st = 1usize;
    for i in 1..=nm1 {
        if e[i - 1].abs() < eps || i == nm1 {
            starts.push(st);
            let nsize;
            if i < nm1 {
                nsize = i - st + 1;
                sizes.push(nsize);
            } else if e[i - 1].abs() >= eps {
                nsize = n - st + 1;
                sizes.push(nsize);
            } else {
                nsize = i - st + 1;
                sizes.push(nsize);
                starts.push(n);
                sizes.push(1);
                for col in 0..nrhs {
                    bx[nm1 + col * ld] = b[(n - 1) + col * ldb];
                }
            }
            let st1 = st - 1;
            if nsize == 1 {
                for col in 0..nrhs {
                    bx[st1 + col * ld] = b[st1 + col * ldb];
                }
            } else if nsize <= SMLSIZ {
                for jj in 0..nsize {
                    for ii in 0..nsize {
                        vt[st1 + ii + jj * ld] = if ii == jj { 1.0 } else { 0.0 };
                    }
                }
                let info = dlasdq(
                    0,
                    nsize,
                    nsize,
                    0,
                    nrhs,
                    &mut d[st1..],
                    &mut e[st1..],
                    &mut vt[st1..],
                    ld,
                    &mut [],
                    1,
                    &mut b[st1..],
                    ldb,
                );
                if info != 0 {
                    return (0, info);
                }
                for col in 0..nrhs {
                    for r in 0..nsize {
                        bx[st1 + r + col * ld] = b[st1 + r + col * ldb];
                    }
                }
            } else {
                let mut f = Factors {
                    u: &mut u[st1..],
                    vt: &mut vt[st1..],
                    ldu: ld,
                    k: &mut kk[st1..],
                    difl: &mut difl[st1..],
                    difr: &mut difr[st1..],
                    z: &mut z[st1..],
                    poles: &mut poles[st1..],
                    givptr: &mut givptr[st1..],
                    givcol: &mut givcol[st1..],
                    ldgcol: ld,
                    perm: &mut perm[st1..],
                    givnum: &mut givnum[st1..],
                    c: &mut c[st1..],
                    s: &mut s[st1..],
                };
                let info = dlasda(
                    SMLSIZ,
                    nsize,
                    0,
                    &mut d[st1..st1 + nsize],
                    &mut e[st1..],
                    &mut f,
                );
                if info != 0 {
                    return (0, info);
                }
                dlalsa(
                    0,
                    SMLSIZ,
                    nsize,
                    nrhs,
                    &mut b[st1..],
                    ldb,
                    &mut bx[st1..],
                    ld,
                    &f,
                );
            }
            st = i + 1;
        }
    }
    let imax = ob_idamax(&d[..n]);
    let tol = rcnd * d[imax].abs();
    let mut rank = 0;
    for i in 1..=n {
        if d[i - 1].abs() <= tol {
            for col in 0..nrhs {
                bx[(i - 1) + col * ld] = 0.0;
            }
        } else {
            rank += 1;
            dlascl(d[i - 1], 1.0, 1, nrhs, &mut bx[i - 1..], ld);
        }
        d[i - 1] = d[i - 1].abs();
    }
    for (idx, &st) in starts.iter().enumerate() {
        let st1 = st - 1;
        let nsize = sizes[idx];
        if nsize == 1 {
            for col in 0..nrhs {
                b[st1 + col * ldb] = bx[st1 + col * ld];
            }
        } else if nsize <= SMLSIZ {
            let vts = vt[st1..].to_vec();
            let bxs = bx[st1..].to_vec();
            ob::dgemm_ab(
                102,
                112,
                111,
                nsize,
                nrhs,
                nsize,
                1.0,
                &vts,
                ld,
                &bxs,
                ld,
                0.0,
                &mut b[st1..],
                ldb,
            );
        } else {
            let f = Factors {
                u: &mut u[st1..],
                vt: &mut vt[st1..],
                ldu: ld,
                k: &mut kk[st1..],
                difl: &mut difl[st1..],
                difr: &mut difr[st1..],
                z: &mut z[st1..],
                poles: &mut poles[st1..],
                givptr: &mut givptr[st1..],
                givcol: &mut givcol[st1..],
                ldgcol: ld,
                perm: &mut perm[st1..],
                givnum: &mut givnum[st1..],
                c: &mut c[st1..],
                s: &mut s[st1..],
            };
            let mut bxs = bx[st1..].to_vec();
            dlalsa(1, SMLSIZ, nsize, nrhs, &mut bxs, ld, &mut b[st1..], ldb, &f);
        }
    }
    dlascl(1.0, orgnrm, n, 1, d, n);
    d[..n].sort_by(|x, y| y.partial_cmp(x).unwrap());
    dlascl(orgnrm, 1.0, n, nrhs, b, ldb);
    (rank, 0)
}

/// `IDAMAX` (0-based, first maximum).
fn ob_idamax(x: &[f64]) -> usize {
    let mut best = 0;
    for (i, v) in x.iter().enumerate() {
        if v.abs() > x[best].abs() {
            best = i;
        }
    }
    best
}

/// The outcome of [`dgelsd`].
pub struct Gelsd {
    pub info: i64,
    pub rank: usize,
    pub s: Vec<f64>,
}

/// `DGELSD(m, n, nrhs, A, lda=m, B, ldb, S, rcond)`: `A` (`m×n`, column-major,
/// overwritten) and `B` (`ldb×nrhs`, `ldb ≥ max(m, n)`, overwritten by the solution).
/// The workspace is the query's (as NumPy and SciPy allocate it), which fixes the
/// block sizes. `None` only for the one unported branch: an underdetermined system
/// with more than 128 rows and `n ≥ 1.6·m` (blocked `DGELQF`).
pub fn dgelsd(
    m: usize,
    n: usize,
    nrhs: usize,
    a: &mut [f64],
    b: &mut [f64],
    ldb: usize,
    rcond: f64,
) -> Option<Gelsd> {
    let lwork = dgelsd_lwork(m, n, nrhs);
    let lda = m.max(1);
    let minmn = m.min(n);
    let mnthr = ((minmn as f32) * 1.6f32) as usize;
    if m == 0 || n == 0 {
        return Some(Gelsd {
            info: 0,
            rank: 0,
            s: vec![],
        });
    }
    let eps = PREC;
    let smlnum = SFMIN / eps;
    let bignum = 1.0 / smlnum;
    let anrm = dlange_max(m, n, a, lda);
    let mut iascl = 0;
    if anrm > 0.0 && anrm < smlnum {
        dlascl(anrm, smlnum, m, n, a, lda);
        iascl = 1;
    } else if anrm > bignum {
        dlascl(anrm, bignum, m, n, a, lda);
        iascl = 2;
    } else if anrm == 0.0 {
        for j in 0..nrhs {
            for i in 0..m.max(n) {
                b[i + j * ldb] = 0.0;
            }
        }
        return Some(Gelsd {
            info: 0,
            rank: 0,
            s: vec![0.0; minmn],
        });
    }
    let bnrm = dlange_max(m, nrhs, b, ldb);
    let mut ibscl = 0;
    if bnrm > 0.0 && bnrm < smlnum {
        dlascl(bnrm, smlnum, m, nrhs, b, ldb);
        ibscl = 1;
    } else if bnrm > bignum {
        dlascl(bnrm, bignum, m, nrhs, b, ldb);
        ibscl = 2;
    }
    if m < n {
        for j in 0..nrhs {
            for i in m..n {
                b[i + j * ldb] = 0.0;
            }
        }
    }
    let mut s = vec![0.0; minmn];
    let rank;
    if m >= n {
        let mut mm = m;
        if m >= mnthr {
            mm = n;
            let mut tau = vec![0.0; n];
            let lw = lwork - n as i64;
            dgeqrf(m, n, a, lda, &mut tau, lw);
            dormxx_left(false, false, m, nrhs, n, a, lda, &tau, b, ldb, lw);
            if n > 1 {
                for j in 0..n - 1 {
                    for i in j + 1..n {
                        a[i + j * lda] = 0.0;
                    }
                }
            }
        }
        let (mut e, mut tauq, mut taup) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let lw = lwork - 3 * n as i64;
        dgebrd(mm, n, a, lda, &mut s, &mut e, &mut tauq, &mut taup, lw);
        dormbr_left(true, false, mm, nrhs, n, a, lda, &tauq, b, ldb, lw);
        let (r, info) = dlalsd(true, n, nrhs, &mut s, &mut e, b, ldb, rcond)?;
        if info != 0 {
            return Some(Gelsd {
                info: info as i64,
                rank: r,
                s,
            });
        }
        rank = r;
        dormbr_left(false, true, n, nrhs, n, a, lda, &taup, b, ldb, lw);
    } else if n >= mnthr {
        // Path 2a: LQ, then the M×M problem
        let mut tau = vec![0.0; m];
        if m > 128 {
            return None; // blocked DGELQF (not ported)
        }
        dgelq2(m, n, a, lda, &mut tau);
        let (m_, n_, nrhs_, lda_) = (m as i64, n as i64, nrhs as i64, lda as i64);
        let wlalsd = {
            let smlsiz = SMLSIZ as i64;
            let nlvl =
                (((m_.max(1) as f64 / (smlsiz + 1) as f64).ln() / 2f64.ln()) as i64 + 1).max(0);
            9 * m_ + 2 * m_ * smlsiz + 8 * m_ * nlvl + m_ * nrhs_ + (smlsiz + 1) * (smlsiz + 1)
        };
        let ldwork = if lwork
            >= (4 * m_ + m_ * lda_ + m_.max(2 * m_ - 4).max(nrhs_).max(n_ - 3 * m_))
                .max(m_ * lda_ + m_ + m_ * nrhs_)
                .max(4 * m_ + m_ * lda_ + wlalsd)
        {
            lda_
        } else {
            m_
        };
        let lw_brd = lwork - (m_ + 1 + ldwork * m_ + 3 * m_) + 1;
        let lw_lq = lwork - m_;
        let mut l = vec![0.0; m * m];
        for j in 0..m {
            for i in j..m {
                l[i + j * m] = a[i + j * lda];
            }
        }
        let (mut e, mut tauq, mut taup) = (vec![0.0; m], vec![0.0; m], vec![0.0; m]);
        dgebrd(
            m, m, &mut l, m, &mut s, &mut e, &mut tauq, &mut taup, lw_brd,
        );
        dormbr_left(true, false, m, nrhs, m, &mut l, m, &tauq, b, ldb, lw_brd);
        let (r, info) = dlalsd(true, m, nrhs, &mut s, &mut e, b, ldb, rcond)?;
        if info != 0 {
            return Some(Gelsd {
                info: info as i64,
                rank: r,
                s,
            });
        }
        rank = r;
        dormbr_left(false, true, m, nrhs, m, &mut l, m, &taup, b, ldb, lw_brd);
        for j in 0..nrhs {
            for i in m..n {
                b[i + j * ldb] = 0.0;
            }
        }
        dormxx_left(true, false, n, nrhs, m, a, lda, &tau, b, ldb, lw_lq);
    } else {
        let (mut e, mut tauq, mut taup) = (vec![0.0; m], vec![0.0; m], vec![0.0; m]);
        let lw = lwork - 3 * m as i64;
        dgebrd(m, n, a, lda, &mut s, &mut e, &mut tauq, &mut taup, lw);
        dormbr_left(true, false, m, nrhs, n, a, lda, &tauq, b, ldb, lw);
        let (r, info) = dlalsd(false, m, nrhs, &mut s, &mut e, b, ldb, rcond)?;
        if info != 0 {
            return Some(Gelsd {
                info: info as i64,
                rank: r,
                s,
            });
        }
        rank = r;
        dormbr_left(false, true, n, nrhs, m, a, lda, &taup, b, ldb, lw);
    }
    if iascl == 1 {
        dlascl(anrm, smlnum, n, nrhs, b, ldb);
        dlascl(smlnum, anrm, minmn, 1, &mut s, minmn);
    } else if iascl == 2 {
        dlascl(anrm, bignum, n, nrhs, b, ldb);
        dlascl(bignum, anrm, minmn, 1, &mut s, minmn);
    }
    if ibscl == 1 {
        dlascl(smlnum, bnrm, n, nrhs, b, ldb);
    } else if ibscl == 2 {
        dlascl(bignum, bnrm, n, nrhs, b, ldb);
    }
    Some(Gelsd { info: 0, rank, s })
}

// ── blocked Householder machinery (DLARFT / DLARFB) and the blocked drivers ───

/// `x` gathered from `buf` at `off`, stride `inc` (laid out at that stride, so the
/// kernel sees the caller's stride).
fn gather(buf: &[f64], off: usize, n: usize, inc: usize) -> Vec<f64> {
    let mut v = vec![0.0; if n == 0 { 0 } else { (n - 1) * inc + 1 }];
    for i in 0..n {
        v[i * inc] = buf[off + i * inc];
    }
    v
}

fn scatter(buf: &mut [f64], off: usize, v: &[f64], n: usize, inc: usize) {
    for i in 0..n {
        buf[off + i * inc] = v[i * inc];
    }
}

/// `DGEMV(trans, m, n, alpha, A(aoff), lda, x, incx, beta, Y(yoff), incy)` where `y`
/// may live in the same array as `A` (disjoint elements).
#[allow(clippy::too_many_arguments)]
fn gemv_into(
    trans: bool,
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    x: &[f64],
    incx: usize,
    beta: f64,
    y: &mut [f64],
    incy: usize,
) {
    let leny = if trans { n } else { m };
    if leny == 0 {
        return;
    }
    if m == 0 || n == 0 {
        // the interface still scales y by beta
        if beta != 1.0 {
            for i in 0..leny {
                y[i * incy] = if beta == 0.0 { 0.0 } else { beta * y[i * incy] };
            }
        }
        return;
    }
    let mut yv = gather(y, 0, leny, incy);
    assert!(
        ob::dgemv_ab(
            102,
            if trans { 112 } else { 111 },
            m,
            n,
            alpha,
            a,
            lda,
            x,
            incx,
            beta,
            &mut yv,
            incy
        ),
        "dgemv above the threading threshold"
    );
    scatter(y, 0, &yv, leny, incy);
}

/// `DLARFT('F', storev, n, k, V, ldv, tau, T, ldt)` (LAPACK 3.12, recursive).
#[allow(clippy::too_many_arguments)]
fn dlarft(
    colwise: bool,
    n: usize,
    k: usize,
    v: &[f64],
    ldv: usize,
    tau: &[f64],
    t: &mut [f64],
    ldt: usize,
) {
    if n == 0 || k == 0 {
        return;
    }
    if n == 1 || k == 1 {
        t[0] = tau[0];
        return;
    }
    let l = k / 2;
    let at = |i: usize, j: usize, ld: usize| (i - 1) + (j - 1) * ld;
    dlarft(colwise, n, l, v, ldv, tau, t, ldt);
    {
        let (tl, tr) = (at(l + 1, l + 1, ldt), at(l + 1, l + 1, ldv));
        dlarft(
            colwise,
            n - l,
            k - l,
            &v[tr..],
            ldv,
            &tau[l..],
            &mut t[tl..],
            ldt,
        );
    }
    let t12 = at(1, l + 1, ldt);
    if colwise {
        for j in 1..=l {
            for i in 1..=k - l {
                t[at(j, l + i, ldt)] = v[at(l + i, j, ldv)];
            }
        }
        ob::dtrmm(
            false,
            true,
            false,
            true,
            l,
            k - l,
            1.0,
            &v[at(l + 1, l + 1, ldv)..],
            ldv,
            &mut t[t12..],
            ldt,
        );
        // N-K = 0: an empty product with beta = 1 leaves T unchanged
        if n > k {
            ob::dgemm_ab(
                102,
                112,
                111,
                l,
                k - l,
                n - k,
                1.0,
                &v[at(k + 1, 1, ldv)..],
                ldv,
                &v[at(k + 1, l + 1, ldv)..],
                ldv,
                1.0,
                &mut t[t12..],
                ldt,
            );
        }
    } else {
        for j in 0..k - l {
            for i in 0..l {
                t[t12 + i + j * ldt] = v[at(1, l + 1, ldv) + i + j * ldv];
            }
        }
        ob::dtrmm(
            false,
            false,
            true,
            true,
            l,
            k - l,
            1.0,
            &v[at(l + 1, l + 1, ldv)..],
            ldv,
            &mut t[t12..],
            ldt,
        );
        if n > k {
            ob::dgemm_ab(
                102,
                111,
                112,
                l,
                k - l,
                n - k,
                1.0,
                &v[at(1, k + 1, ldv)..],
                ldv,
                &v[at(l + 1, k + 1, ldv)..],
                ldv,
                1.0,
                &mut t[t12..],
                ldt,
            );
        }
    }
    let t11 = t.to_vec();
    ob::dtrmm(
        true,
        false,
        false,
        false,
        l,
        k - l,
        -1.0,
        &t11,
        ldt,
        &mut t[t12..],
        ldt,
    );
    let t22 = t.to_vec();
    ob::dtrmm(
        false,
        false,
        false,
        false,
        l,
        k - l,
        1.0,
        &t22[at(l + 1, l + 1, ldt)..],
        ldt,
        &mut t[t12..],
        ldt,
    );
}

/// `DLARFB('L', trans, 'F', storev, m, n, k, V, T, C)`.
#[allow(clippy::too_many_arguments)]
fn dlarfb_left(
    trans_t: bool,
    colwise: bool,
    m: usize,
    n: usize,
    k: usize,
    v: &[f64],
    ldv: usize,
    t: &[f64],
    ldt: usize,
    c: &mut [f64],
    ldc: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    // TRANST: the opposite of TRANS
    let transt = !trans_t;
    let ldw = n.max(1);
    let mut w = vec![0.0f64; ldw * k];
    for j in 0..k {
        for i in 0..n {
            w[i + j * ldw] = c[j + i * ldc];
        }
    }
    if colwise {
        ob::dtrmm(false, true, false, true, n, k, 1.0, v, ldv, &mut w, ldw);
        if m > k {
            ob::dgemm_ab(
                102,
                112,
                111,
                n,
                k,
                m - k,
                1.0,
                &c[k..],
                ldc,
                &v[k..],
                ldv,
                1.0,
                &mut w,
                ldw,
            );
        }
        ob::dtrmm(false, false, transt, false, n, k, 1.0, t, ldt, &mut w, ldw);
        if m > k {
            ob::dgemm_ab(
                102,
                111,
                112,
                m - k,
                n,
                k,
                -1.0,
                &v[k..],
                ldv,
                &w,
                ldw,
                1.0,
                &mut c[k..],
                ldc,
            );
        }
        ob::dtrmm(false, true, true, true, n, k, 1.0, v, ldv, &mut w, ldw);
    } else {
        ob::dtrmm(false, false, true, true, n, k, 1.0, v, ldv, &mut w, ldw);
        if m > k {
            ob::dgemm_ab(
                102,
                112,
                112,
                n,
                k,
                m - k,
                1.0,
                &c[k..],
                ldc,
                &v[k * ldv..],
                ldv,
                1.0,
                &mut w,
                ldw,
            );
        }
        ob::dtrmm(false, false, transt, false, n, k, 1.0, t, ldt, &mut w, ldw);
        if m > k {
            ob::dgemm_ab(
                102,
                112,
                112,
                m - k,
                n,
                k,
                -1.0,
                &v[k * ldv..],
                ldv,
                &w,
                ldw,
                1.0,
                &mut c[k..],
                ldc,
            );
        }
        ob::dtrmm(false, false, false, true, n, k, 1.0, v, ldv, &mut w, ldw);
    }
    for j in 0..k {
        for i in 0..n {
            c[j + i * ldc] -= w[i + j * ldw];
        }
    }
}

/// `DORMQR` / `DORMLQ` (`side = 'L'`) with LAPACK's block-size choice: `NB = 32`
/// unless the workspace `lwork` is short of `NW·NB + TSIZE`, in which case
/// `NB = (lwork - TSIZE) / NW`; unblocked when `NB < 2` or `NB ≥ k`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dormxx_left(
    lq: bool,
    notran: bool,
    m: usize,
    n: usize,
    k: usize,
    a: &mut [f64],
    lda: usize,
    tau: &[f64],
    c: &mut [f64],
    ldc: usize,
    lwork: i64,
) {
    const NBMAX: i64 = 64;
    const TSIZE: i64 = (NBMAX + 1) * NBMAX;
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    let nw = n.max(1) as i64;
    let mut nb: i64 = 32.min(NBMAX);
    let lwkopt = nw * nb + TSIZE;
    let nbmin = 2;
    if nb > 1 && nb < k as i64 && lwork < lwkopt {
        nb = (lwork - TSIZE) / nw;
    }
    if nb < nbmin || nb >= k as i64 {
        if lq {
            dorml2(true, notran, m, n, k, a, lda, tau, c, ldc);
        } else {
            dorm2r(true, notran, m, n, k, a, lda, tau, c, ldc);
        }
        return;
    }
    let nb = nb as usize;
    let forward = if lq { notran } else { !notran };
    let starts: Vec<usize> = if forward {
        (1..=k).step_by(nb).collect()
    } else {
        (0..=((k - 1) / nb)).rev().map(|q| q * nb + 1).collect()
    };
    let nq = m;
    for i in starts {
        let ib = nb.min(k - i + 1);
        let at = (i - 1) + (i - 1) * lda;
        let mut t = vec![0.0f64; 65 * ib.max(1)];
        let vcopy = a.to_vec();
        dlarft(
            !lq,
            nq - i + 1,
            ib,
            &vcopy[at..],
            lda,
            &tau[i - 1..],
            &mut t,
            65,
        );
        // DORMQR passes TRANS to DLARFB; DORMLQ passes TRANST
        let trans_flag = if lq { notran } else { !notran };
        dlarfb_left(
            trans_flag,
            !lq,
            m - i + 1,
            n,
            ib,
            &vcopy[at..],
            lda,
            &t,
            65,
            &mut c[i - 1..],
            ldc,
        );
    }
}

/// `DLABRD(m, n, nb, A, d, e, tauq, taup, X, Y)`.
#[allow(clippy::too_many_arguments)]
fn dlabrd(
    m: usize,
    n: usize,
    nb: usize,
    a: &mut [f64],
    lda: usize,
    d: &mut [f64],
    e: &mut [f64],
    tauq: &mut [f64],
    taup: &mut [f64],
    x: &mut [f64],
    ldx: usize,
    y: &mut [f64],
    ldy: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    let at = |i: usize, j: usize, ld: usize| (i - 1) + (j - 1) * ld;
    // `DGEMV(trans, mm, nn, alpha, SRC(r,c), ld, X-vector, incx, beta, DST(r,c), incy)`
    macro_rules! gv {
        ($tr:expr, $mm:expr, $nn:expr, $al:expr, $src:expr, $sr:expr, $sc:expr, $sld:expr, $xv:expr, $incx:expr, $be:expr, $dst:expr, $dr:expr, $dc:expr, $dld:expr, $incy:expr) => {{
            let xvv: Vec<f64> = $xv;
            let srcv: Vec<f64> = $src.to_vec();
            let off = at($dr, $dc, $dld);
            gemv_into(
                $tr,
                $mm,
                $nn,
                $al,
                &srcv[at($sr, $sc, $sld)..],
                $sld,
                &xvv,
                $incx,
                $be,
                &mut $dst[off..],
                $incy,
            );
        }};
    }
    if m >= n {
        for i in 1..=nb {
            gv!(
                false,
                m - i + 1,
                i - 1,
                -1.0,
                a,
                i,
                1,
                lda,
                gather(y, at(i, 1, ldy), i - 1, ldy),
                ldy,
                1.0,
                a,
                i,
                i,
                lda,
                1
            );
            gv!(
                false,
                m - i + 1,
                i - 1,
                -1.0,
                x,
                i,
                1,
                ldx,
                gather(a, at(1, i, lda), i - 1, 1),
                1,
                1.0,
                a,
                i,
                i,
                lda,
                1
            );
            let xs = at((i + 1).min(m), i, lda);
            let mut alpha = a[at(i, i, lda)];
            let mut xx = take(a, xs, m - i, 1);
            tauq[i - 1] = dlarfg(m - i + 1, &mut alpha, &mut xx, 1);
            put(a, xs, &xx, 1);
            a[at(i, i, lda)] = alpha;
            d[i - 1] = a[at(i, i, lda)];
            if i < n {
                a[at(i, i, lda)] = 1.0;
                let vq = gather(a, at(i, i, lda), m - i + 1, 1);
                gv!(
                    true,
                    m - i + 1,
                    n - i,
                    1.0,
                    a,
                    i,
                    i + 1,
                    lda,
                    vq.clone(),
                    1,
                    0.0,
                    y,
                    i + 1,
                    i,
                    ldy,
                    1
                );
                gv!(
                    true,
                    m - i + 1,
                    i - 1,
                    1.0,
                    a,
                    i,
                    1,
                    lda,
                    vq.clone(),
                    1,
                    0.0,
                    y,
                    1,
                    i,
                    ldy,
                    1
                );
                gv!(
                    false,
                    n - i,
                    i - 1,
                    -1.0,
                    y,
                    i + 1,
                    1,
                    ldy,
                    gather(y, at(1, i, ldy), i - 1, 1),
                    1,
                    1.0,
                    y,
                    i + 1,
                    i,
                    ldy,
                    1
                );
                gv!(
                    true,
                    m - i + 1,
                    i - 1,
                    1.0,
                    x,
                    i,
                    1,
                    ldx,
                    vq.clone(),
                    1,
                    0.0,
                    y,
                    1,
                    i,
                    ldy,
                    1
                );
                gv!(
                    true,
                    i - 1,
                    n - i,
                    -1.0,
                    a,
                    1,
                    i + 1,
                    lda,
                    gather(y, at(1, i, ldy), i - 1, 1),
                    1,
                    1.0,
                    y,
                    i + 1,
                    i,
                    ldy,
                    1
                );
                let yo = at(i + 1, i, ldy);
                ob::dscal(n - i, tauq[i - 1], &mut y[yo..], 1);
                gv!(
                    false,
                    n - i,
                    i,
                    -1.0,
                    y,
                    i + 1,
                    1,
                    ldy,
                    gather(a, at(i, 1, lda), i, lda),
                    lda,
                    1.0,
                    a,
                    i,
                    i + 1,
                    lda,
                    lda
                );
                gv!(
                    true,
                    i - 1,
                    n - i,
                    -1.0,
                    a,
                    1,
                    i + 1,
                    lda,
                    gather(x, at(i, 1, ldx), i - 1, ldx),
                    ldx,
                    1.0,
                    a,
                    i,
                    i + 1,
                    lda,
                    lda
                );
                let xs = at(i, (i + 2).min(n), lda);
                let mut alpha = a[at(i, i + 1, lda)];
                let mut xx = take(a, xs, n - i - 1, lda);
                taup[i - 1] = dlarfg(n - i, &mut alpha, &mut xx, 1);
                put(a, xs, &xx, lda);
                a[at(i, i + 1, lda)] = alpha;
                e[i - 1] = a[at(i, i + 1, lda)];
                a[at(i, i + 1, lda)] = 1.0;
                let vp = gather(a, at(i, i + 1, lda), n - i, lda);
                gv!(
                    false,
                    m - i,
                    n - i,
                    1.0,
                    a,
                    i + 1,
                    i + 1,
                    lda,
                    vp.clone(),
                    lda,
                    0.0,
                    x,
                    i + 1,
                    i,
                    ldx,
                    1
                );
                gv!(
                    true,
                    n - i,
                    i,
                    1.0,
                    y,
                    i + 1,
                    1,
                    ldy,
                    vp.clone(),
                    lda,
                    0.0,
                    x,
                    1,
                    i,
                    ldx,
                    1
                );
                gv!(
                    false,
                    m - i,
                    i,
                    -1.0,
                    a,
                    i + 1,
                    1,
                    lda,
                    gather(x, at(1, i, ldx), i, 1),
                    1,
                    1.0,
                    x,
                    i + 1,
                    i,
                    ldx,
                    1
                );
                gv!(
                    false,
                    i - 1,
                    n - i,
                    1.0,
                    a,
                    1,
                    i + 1,
                    lda,
                    vp.clone(),
                    lda,
                    0.0,
                    x,
                    1,
                    i,
                    ldx,
                    1
                );
                gv!(
                    false,
                    m - i,
                    i - 1,
                    -1.0,
                    x,
                    i + 1,
                    1,
                    ldx,
                    gather(x, at(1, i, ldx), i - 1, 1),
                    1,
                    1.0,
                    x,
                    i + 1,
                    i,
                    ldx,
                    1
                );
                let xo = at(i + 1, i, ldx);
                ob::dscal(m - i, taup[i - 1], &mut x[xo..], 1);
            }
        }
    } else {
        for i in 1..=nb {
            gv!(
                false,
                n - i + 1,
                i - 1,
                -1.0,
                y,
                i,
                1,
                ldy,
                gather(a, at(i, 1, lda), i - 1, lda),
                lda,
                1.0,
                a,
                i,
                i,
                lda,
                lda
            );
            gv!(
                true,
                i - 1,
                n - i + 1,
                -1.0,
                a,
                1,
                i,
                lda,
                gather(x, at(i, 1, ldx), i - 1, ldx),
                ldx,
                1.0,
                a,
                i,
                i,
                lda,
                lda
            );
            let xs = at(i, (i + 1).min(n), lda);
            let mut alpha = a[at(i, i, lda)];
            let mut xx = take(a, xs, n - i, lda);
            taup[i - 1] = dlarfg(n - i + 1, &mut alpha, &mut xx, 1);
            put(a, xs, &xx, lda);
            a[at(i, i, lda)] = alpha;
            d[i - 1] = a[at(i, i, lda)];
            if i < m {
                a[at(i, i, lda)] = 1.0;
                let vp = gather(a, at(i, i, lda), n - i + 1, lda);
                gv!(
                    false,
                    m - i,
                    n - i + 1,
                    1.0,
                    a,
                    i + 1,
                    i,
                    lda,
                    vp.clone(),
                    lda,
                    0.0,
                    x,
                    i + 1,
                    i,
                    ldx,
                    1
                );
                gv!(
                    true,
                    n - i + 1,
                    i - 1,
                    1.0,
                    y,
                    i,
                    1,
                    ldy,
                    vp.clone(),
                    lda,
                    0.0,
                    x,
                    1,
                    i,
                    ldx,
                    1
                );
                gv!(
                    false,
                    m - i,
                    i - 1,
                    -1.0,
                    a,
                    i + 1,
                    1,
                    lda,
                    gather(x, at(1, i, ldx), i - 1, 1),
                    1,
                    1.0,
                    x,
                    i + 1,
                    i,
                    ldx,
                    1
                );
                gv!(
                    false,
                    i - 1,
                    n - i + 1,
                    1.0,
                    a,
                    1,
                    i,
                    lda,
                    vp.clone(),
                    lda,
                    0.0,
                    x,
                    1,
                    i,
                    ldx,
                    1
                );
                gv!(
                    false,
                    m - i,
                    i - 1,
                    -1.0,
                    x,
                    i + 1,
                    1,
                    ldx,
                    gather(x, at(1, i, ldx), i - 1, 1),
                    1,
                    1.0,
                    x,
                    i + 1,
                    i,
                    ldx,
                    1
                );
                let xo = at(i + 1, i, ldx);
                ob::dscal(m - i, taup[i - 1], &mut x[xo..], 1);
                gv!(
                    false,
                    m - i,
                    i - 1,
                    -1.0,
                    a,
                    i + 1,
                    1,
                    lda,
                    gather(y, at(i, 1, ldy), i - 1, ldy),
                    ldy,
                    1.0,
                    a,
                    i + 1,
                    i,
                    lda,
                    1
                );
                gv!(
                    false,
                    m - i,
                    i,
                    -1.0,
                    x,
                    i + 1,
                    1,
                    ldx,
                    gather(a, at(1, i, lda), i, 1),
                    1,
                    1.0,
                    a,
                    i + 1,
                    i,
                    lda,
                    1
                );
                let xs = at((i + 2).min(m), i, lda);
                let mut alpha = a[at(i + 1, i, lda)];
                let mut xx = take(a, xs, m - i - 1, 1);
                tauq[i - 1] = dlarfg(m - i, &mut alpha, &mut xx, 1);
                put(a, xs, &xx, 1);
                a[at(i + 1, i, lda)] = alpha;
                e[i - 1] = a[at(i + 1, i, lda)];
                a[at(i + 1, i, lda)] = 1.0;
                let vq = gather(a, at(i + 1, i, lda), m - i, 1);
                gv!(
                    true,
                    m - i,
                    n - i,
                    1.0,
                    a,
                    i + 1,
                    i + 1,
                    lda,
                    vq.clone(),
                    1,
                    0.0,
                    y,
                    i + 1,
                    i,
                    ldy,
                    1
                );
                gv!(
                    true,
                    m - i,
                    i - 1,
                    1.0,
                    a,
                    i + 1,
                    1,
                    lda,
                    vq.clone(),
                    1,
                    0.0,
                    y,
                    1,
                    i,
                    ldy,
                    1
                );
                gv!(
                    false,
                    n - i,
                    i - 1,
                    -1.0,
                    y,
                    i + 1,
                    1,
                    ldy,
                    gather(y, at(1, i, ldy), i - 1, 1),
                    1,
                    1.0,
                    y,
                    i + 1,
                    i,
                    ldy,
                    1
                );
                gv!(
                    true,
                    m - i,
                    i,
                    1.0,
                    x,
                    i + 1,
                    1,
                    ldx,
                    vq.clone(),
                    1,
                    0.0,
                    y,
                    1,
                    i,
                    ldy,
                    1
                );
                gv!(
                    true,
                    i,
                    n - i,
                    -1.0,
                    a,
                    1,
                    i + 1,
                    lda,
                    gather(y, at(1, i, ldy), i, 1),
                    1,
                    1.0,
                    y,
                    i + 1,
                    i,
                    ldy,
                    1
                );
                let yo = at(i + 1, i, ldy);
                ob::dscal(n - i, tauq[i - 1], &mut y[yo..], 1);
            }
        }
    }
}

/// `DGEBRD(m, n, A, lwork)`: blocked (`DLABRD` + two `DGEMM`s per block) once
/// `min(m,n) > NX = 128`, with the block size the workspace allows; the rest by
/// `DGEBD2`.
#[allow(clippy::too_many_arguments)]
fn dgebrd(
    m: usize,
    n: usize,
    a: &mut [f64],
    lda: usize,
    d: &mut [f64],
    e: &mut [f64],
    tauq: &mut [f64],
    taup: &mut [f64],
    lwork: i64,
) {
    let minmn = m.min(n);
    if minmn == 0 {
        return;
    }
    let mut nb: i64 = 32;
    let mut nx: usize;
    let ldwrkx = m;
    let ldwrky = n;
    if nb > 1 && (nb as usize) < minmn {
        nx = (nb as usize).max(128);
        if nx < minmn {
            let ws = (m + n) as i64 * nb;
            if lwork < ws {
                let nbmin = 2;
                if lwork >= (m + n) as i64 * nbmin {
                    nb = lwork / (m + n) as i64;
                } else {
                    nb = 1;
                    nx = minmn;
                }
            }
        }
    } else {
        nx = minmn;
    }
    let nb = nb as usize;
    let mut i = 1;
    let at = |i: usize, j: usize| (i - 1) + (j - 1) * lda;
    while minmn >= nx + 1 && i <= minmn - nx {
        let mut x = vec![0.0f64; ldwrkx * nb];
        let mut y = vec![0.0f64; ldwrky * nb];
        dlabrd(
            m - i + 1,
            n - i + 1,
            nb,
            &mut a[at(i, i)..],
            lda,
            &mut d[i - 1..],
            &mut e[i - 1..],
            &mut tauq[i - 1..],
            &mut taup[i - 1..],
            &mut x,
            ldwrkx,
            &mut y,
            ldwrky,
        );
        let src = a.to_vec();
        ob::dgemm_ab(
            102,
            111,
            112,
            m - i - nb + 1,
            n - i - nb + 1,
            nb,
            -1.0,
            &src[at(i + nb, i)..],
            lda,
            &y[nb..],
            ldwrky,
            1.0,
            &mut a[at(i + nb, i + nb)..],
            lda,
        );
        let src = a.to_vec();
        ob::dgemm_ab(
            102,
            111,
            111,
            m - i - nb + 1,
            n - i - nb + 1,
            nb,
            -1.0,
            &x[nb..],
            ldwrkx,
            &src[at(i, i + nb)..],
            lda,
            1.0,
            &mut a[at(i + nb, i + nb)..],
            lda,
        );
        for j in i..i + nb {
            a[at(j, j)] = d[j - 1];
            if m >= n {
                a[at(j, j + 1)] = e[j - 1];
            } else {
                a[at(j + 1, j)] = e[j - 1];
            }
        }
        i += nb;
    }
    dgebd2(
        m - i + 1,
        n - i + 1,
        &mut a[at(i, i)..],
        lda,
        &mut d[i - 1..],
        &mut e[i - 1..],
        &mut tauq[i - 1..],
        &mut taup[i - 1..],
    );
}

/// `DGEQRF(m, n, A, lwork)`: blocked once `min(m,n) > NX = 128` (`DGEQR2` on each
/// panel, `DLARFT` + `DLARFB` on the trailing columns), with `NB` cut to what
/// `lwork` allows.
fn dgeqrf(m: usize, n: usize, a: &mut [f64], lda: usize, tau: &mut [f64], lwork: i64) {
    let k = m.min(n);
    if k == 0 {
        return;
    }
    let mut nb: i64 = 32;
    let mut nbmin: i64 = 2;
    let mut nx: usize = 0;
    let ldwork = n;
    if nb > 1 && (nb as usize) < k {
        nx = 128;
        if nx < k {
            let iws = ldwork as i64 * nb;
            if lwork < iws {
                nb = lwork / ldwork as i64;
                nbmin = 2;
            }
        }
    }
    let at = |i: usize, j: usize| (i - 1) + (j - 1) * lda;
    let mut i = 1usize;
    if nb >= nbmin && (nb as usize) < k && nx < k {
        let nb = nb as usize;
        while i <= k - nx {
            let ib = (k - i + 1).min(nb);
            dgeqr2(m - i + 1, ib, &mut a[at(i, i)..], lda, &mut tau[i - 1..]);
            if i + ib <= n {
                let mut t = vec![0.0f64; ldwork * ib];
                let v = a.to_vec();
                dlarft(
                    true,
                    m - i + 1,
                    ib,
                    &v[at(i, i)..],
                    lda,
                    &tau[i - 1..],
                    &mut t,
                    ldwork,
                );
                dlarfb_left(
                    true,
                    true,
                    m - i + 1,
                    n - i - ib + 1,
                    ib,
                    &v[at(i, i)..],
                    lda,
                    &t,
                    ldwork,
                    &mut a[at(i, i + ib)..],
                    lda,
                );
            }
            i += nb;
        }
    }
    if i <= k {
        dgeqr2(
            m - i + 1,
            n - i + 1,
            &mut a[at(i, i)..],
            lda,
            &mut tau[i - 1..],
        );
    }
}

/// `DGELSD`'s workspace query (`WORK(1)` with `LWORK = -1`): what NumPy's and
/// SciPy's `lstsq` allocate and pass back, which decides the block sizes inside.
pub fn dgelsd_lwork(m: usize, n: usize, nrhs: usize) -> i64 {
    let nb: i64 = 32; // ILAENV(1, …) for GEQRF/ORMQR/GEBRD/ORMBR/GELQF/ORMLQ
    let (m_, n_, nrhs_) = (m as i64, n as i64, nrhs as i64);
    let minmn0 = m.min(n);
    let mnthr = ((minmn0 as f32) * 1.6f32) as i64;
    let smlsiz = SMLSIZ as i64;
    let minmn = minmn0.max(1) as i64;
    let nlvl = (((minmn as f64 / (smlsiz + 1) as f64).ln() / 2f64.ln()) as i64 + 1).max(0);
    let mut maxwrk: i64 = 1;
    let mut minwrk: i64 = 1;
    if m >= n {
        let mut mm = m_;
        if m_ >= mnthr {
            mm = n_;
            maxwrk = maxwrk.max(n_ + n_ * nb);
            maxwrk = maxwrk.max(n_ + nrhs_ * nb);
        }
        maxwrk = maxwrk.max(3 * n_ + (mm + n_) * nb);
        maxwrk = maxwrk.max(3 * n_ + nrhs_ * nb);
        maxwrk = maxwrk.max(3 * n_ + (n_ - 1) * nb);
        let wlalsd =
            9 * n_ + 2 * n_ * smlsiz + 8 * n_ * nlvl + n_ * nrhs_ + (smlsiz + 1) * (smlsiz + 1);
        maxwrk = maxwrk.max(3 * n_ + wlalsd);
        minwrk = (3 * n_ + mm).max(3 * n_ + nrhs_).max(3 * n_ + wlalsd);
    }
    if n > m {
        let wlalsd =
            9 * m_ + 2 * m_ * smlsiz + 8 * m_ * nlvl + m_ * nrhs_ + (smlsiz + 1) * (smlsiz + 1);
        if n_ >= mnthr {
            maxwrk = m_ + m_ * nb;
            maxwrk = maxwrk.max(m_ * m_ + 4 * m_ + 2 * m_ * nb);
            maxwrk = maxwrk.max(m_ * m_ + 4 * m_ + nrhs_ * nb);
            maxwrk = maxwrk.max(m_ * m_ + 4 * m_ + (m_ - 1) * nb);
            if nrhs > 1 {
                maxwrk = maxwrk.max(m_ * m_ + m_ + m_ * nrhs_);
            } else {
                maxwrk = maxwrk.max(m_ * m_ + 2 * m_);
            }
            maxwrk = maxwrk.max(m_ + nrhs_ * nb);
            maxwrk = maxwrk.max(m_ * m_ + 4 * m_ + wlalsd);
            maxwrk = maxwrk.max(4 * m_ + m_ * m_ + m_.max(2 * m_ - 4).max(nrhs_).max(n_ - 3 * m_));
        } else {
            maxwrk = 3 * m_ + (n_ + m_) * nb;
            maxwrk = maxwrk.max(3 * m_ + nrhs_ * nb);
            maxwrk = maxwrk.max(3 * m_ + m_ * nb);
            maxwrk = maxwrk.max(3 * m_ + wlalsd);
        }
        minwrk = (3 * m_ + nrhs_).max(3 * m_ + m_).max(3 * m_ + wlalsd);
    }
    let _ = minwrk.min(maxwrk);
    maxwrk
}

// ── DGESDD (JOBZ = 'S'), the paths that bidiagonalise A directly ─────────────

/// `DBDSDC(uplo, 'I', n, d, e, U, ldu, VT, ldvt)` for `n ≤ SMLSIZ` (`DLASDQ` then a
/// selection sort); `U`, `VT` `n × n` column-major. Returns `info`.
#[allow(clippy::too_many_arguments)]
fn dbdsdc_small(
    upper: bool,
    n: usize,
    d: &mut [f64],
    e: &mut [f64],
    u: &mut [f64],
    ldu: usize,
    vt: &mut [f64],
    ldvt: usize,
) -> usize {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        u[0] = sign(1.0, d[0]);
        vt[0] = 1.0;
        d[0] = d[0].abs();
        return 0;
    }
    let nm1 = n - 1;
    let mut rot = vec![0.0f64; 2 * nm1];
    if !upper {
        for i in 0..nm1 {
            let (cs, sn, r) = dlartg(d[i], e[i]);
            d[i] = r;
            e[i] = sn * d[i + 1];
            d[i + 1] *= cs;
            rot[i] = cs;
            rot[nm1 + i] = -sn;
        }
    }
    for j in 0..n {
        for i in 0..n {
            u[i + j * ldu] = if i == j { 1.0 } else { 0.0 };
            vt[i + j * ldvt] = if i == j { 1.0 } else { 0.0 };
        }
    }
    let mut cdum = [0.0f64; 1];
    let info = dlasdq(0, n, n, n, 0, d, e, vt, ldvt, u, ldu, &mut cdum, 1);
    if info != 0 {
        return info;
    }
    for ii in 2..=n {
        let i = ii - 1;
        let mut kk = i;
        let mut p = d[i - 1];
        for j in ii..=n {
            if d[j - 1] > p {
                kk = j;
                p = d[j - 1];
            }
        }
        if kk != i {
            d[kk - 1] = d[i - 1];
            d[i - 1] = p;
            swap_strided(u, (i - 1) * ldu, (kk - 1) * ldu, n, 1);
            swap_strided(vt, i - 1, kk - 1, n, ldvt);
        }
    }
    if !upper {
        dlasr(true, false, n, n, &rot[..nm1], &rot[nm1..], u, ldu);
    }
    0
}

/// The largest `min(m, n)` [`dgesdd_s`] handles: `DBDSDC`'s `SMLSIZ` (beyond it the
/// divide and conquer `DLASD0` runs, not ported).
pub const DGESDD_MAX_MINMN: usize = 25;

/// `DGESDD('S', m, n, A, lda, …)` as reference LAPACK 3.12 (with OpenBLAS's
/// kernels) computes it, for the paths that bidiagonalise `A` directly — `m ≥ n`
/// with `m < ⌊11·n/6⌋`, or `m < n` with `n < ⌊11·m/6⌋` — and `min(m, n) ≤ 25`;
/// `None` elsewhere (not modelled). `a` column-major `m × n` (overwritten). Returns
/// `(U m×k, s, VT k×n)`, column-major, `k = min(m, n)`, or `Err(info)`.
#[allow(clippy::type_complexity)]
pub fn dgesdd_s(
    m: usize,
    n: usize,
    a: &mut [f64],
    lda: usize,
) -> Option<Result<(Vec<f64>, Vec<f64>, Vec<f64>), usize>> {
    let minmn = m.min(n);
    if minmn == 0 || minmn > DGESDD_MAX_MINMN {
        return None;
    }
    let mnthr = (minmn as f64 * 11.0 / 6.0) as usize;
    if (m >= n && m >= mnthr) || (m < n && n >= mnthr) {
        return None;
    }
    let eps = PREC;
    let smlnum = SFMIN.sqrt() / eps;
    let bignum = 1.0 / smlnum;
    let anrm = dlange_max(m, n, a, lda);
    if anrm.is_nan() {
        return Some(Err(usize::MAX));
    }
    let mut iscl = false;
    if anrm > 0.0 && anrm < smlnum {
        iscl = true;
        dlascl(anrm, smlnum, m, n, a, lda);
    } else if anrm > bignum {
        iscl = true;
        dlascl(anrm, bignum, m, n, a, lda);
    }
    let k = minmn;
    let mut s = vec![0.0f64; k];
    let mut e = vec![0.0f64; k];
    let (mut tauq, mut taup) = (vec![0.0f64; k], vec![0.0f64; k]);
    // the workspace only steers blocking, which these sizes never reach
    let lwork = 1i64 << 40;
    dgebrd(m, n, a, lda, &mut s, &mut e, &mut tauq, &mut taup, lwork);
    let mut u = vec![0.0f64; m * k];
    let mut vt = vec![0.0f64; k * n];
    if m >= n {
        // path 5: U (m×n) zero, DBDSDC('U') into its top n×n, then Q·U and VT·Pᵀ
        let info = {
            let mut ub = vec![0.0f64; n * n];
            let mut vb = vec![0.0f64; n * n];
            let info = dbdsdc_small(true, n, &mut s, &mut e, &mut ub, n, &mut vb, n);
            for j in 0..n {
                for i in 0..n {
                    u[i + j * m] = ub[i + j * n];
                    vt[i + j * k] = vb[i + j * n];
                }
            }
            info
        };
        if info != 0 {
            return Some(Err(info));
        }
        dormbr_left(true, true, m, n, n, a, lda, &tauq, &mut u, m, lwork);
        dormbr_p_right_t(n, n, n, a, lda, &taup, &mut vt, k);
    } else {
        // path 5t: VT (m×n) zero, DBDSDC('L') into its left m×m, then Q·U and VT·Pᵀ
        let info = {
            let mut ub = vec![0.0f64; m * m];
            let mut vb = vec![0.0f64; m * m];
            let info = dbdsdc_small(false, m, &mut s, &mut e, &mut ub, m, &mut vb, m);
            u.copy_from_slice(&ub);
            for j in 0..m {
                for i in 0..m {
                    vt[i + j * k] = vb[i + j * m];
                }
            }
            info
        };
        if info != 0 {
            return Some(Err(info));
        }
        dormbr_left(true, true, m, m, n, a, lda, &tauq, &mut u, m, lwork);
        dormbr_p_right_t(m, n, m, a, lda, &taup, &mut vt, k);
    }
    if iscl {
        if anrm > bignum {
            dlascl(bignum, anrm, k, 1, &mut s, k);
        }
        if anrm < smlnum {
            dlascl(smlnum, anrm, k, 1, &mut s, k);
        }
    }
    Some(Ok((u, s, vt)))
}

/// `DORMBR('P', 'R', 'T', m, n, k, A, tauP, C)`: `C·Pᵀ` → `DORMLQ('R', 'N', …)`
/// (unblocked for `k ≤ 32`).
#[allow(clippy::too_many_arguments)]
fn dormbr_p_right_t(
    m: usize,
    n: usize,
    k: usize,
    a: &mut [f64],
    lda: usize,
    tau: &[f64],
    c: &mut [f64],
    ldc: usize,
) {
    let nq = n;
    assert!(k <= 32, "blocked DORMLQ not modelled here");
    if nq > k {
        dorml2(false, true, m, n, k, a, lda, tau, c, ldc);
    } else if nq > 1 {
        dorml2(
            false,
            true,
            m,
            n - 1,
            nq - 1,
            &mut a[lda..],
            lda,
            tau,
            &mut c[ldc..],
            ldc,
        );
    }
}
