// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Direct BLAS FFI — zero abstraction overhead.
//!
//! Calls cblas_sgemm directly without going through ndarray, faer, or any
//! wrapper. This is the same approach that gave burnembed 2× speedup
//! over Burn's NdArray backend.
//!
//! The `blas` feature (default-on) links an external CBLAS. With
//! `--no-default-features` the extern is replaced by a portable scalar
//! gemm with the same calling convention so every consumer in this
//! file keeps working — slow, but correct on hosts without OpenBLAS /
//! Accelerate / MKL (e.g. CI containers, fresh Win dev boxes).

#[cfg(feature = "blas")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

#[cfg(not(feature = "blas"))]
#[allow(non_snake_case, clippy::too_many_arguments)]
#[inline]
unsafe fn cblas_sgemm(
    _order: i32,
    transa: i32,
    transb: i32,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    b: *const f32,
    ldb: i32,
    beta: f32,
    c: *mut f32,
    ldc: i32,
) {
    // Row-major scalar fallback. _order is ignored — every call site in
    // this file passes ROW_MAJOR (101). Supports both NoTrans/Trans on
    // either operand and arbitrary positive lda/ldb/ldc.
    let m = m as usize;
    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldc = ldc as usize;
    let trans_a = transa != NO_TRANS;
    let trans_b = transb != NO_TRANS;
    for i in 0..m {
        for j in 0..n {
            let mut acc: f32 = 0.0;
            for p in 0..k {
                let av = if trans_a {
                    unsafe { *a.add(p * lda + i) }
                } else {
                    unsafe { *a.add(i * lda + p) }
                };
                let bv = if trans_b {
                    unsafe { *b.add(j * ldb + p) }
                } else {
                    unsafe { *b.add(p * ldb + j) }
                };
                acc += av * bv;
            }
            let cp = unsafe { c.add(i * ldc + j) };
            unsafe {
                *cp = alpha * acc + beta * *cp;
            }
        }
    }
}

// ── f64 BLAS / LAPACK ─────────────────────────────────────────
//
// Accelerate's vecLib (macOS) and OpenBLAS (Linux/Win) both export
// these. They follow the exact same calling conventions as the f32
// variants — only the element type changes.
//
// `dgesv_` is the LAPACK Fortran ABI name (column-major, trailing
// underscore). It does an in-place LU factorization of `a` and then
// solves `a · x = b`, overwriting `b` with `x`. Row-major callers
// must transpose A in/out (or transpose b's leading-dim convention)
// — see the `dgesv` wrapper below for the row-major adapter.

#[cfg(feature = "blas")]
unsafe extern "C" {
    fn cblas_dgemm(
        order: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        b: *const f64,
        ldb: i32,
        beta: f64,
        c: *mut f64,
        ldc: i32,
    );

    /// LAPACK column-major dgesv:
    ///   A · X = B,  A: [n, n],  B: [n, nrhs],  pivot: [n] (i32).
    /// Overwrites A with its LU factors and B with the solution X.
    /// `info_out`: 0 = success, k>0 = U[k-1,k-1] is exactly zero.
    #[link_name = "dgesv_"]
    fn lapack_dgesv(
        n: *const i32,
        nrhs: *const i32,
        a: *mut f64,
        lda: *const i32,
        ipiv: *mut i32,
        b: *mut f64,
        ldb: *const i32,
        info_out: *mut i32,
    );

    /// Single-precision twin of dgesv. Same shape contract.
    #[link_name = "sgesv_"]
    fn lapack_sgesv(
        n: *const i32,
        nrhs: *const i32,
        a: *mut f32,
        lda: *const i32,
        ipiv: *mut i32,
        b: *mut f32,
        ldb: *const i32,
        info_out: *mut i32,
    );

    /// dpotrf — Cholesky factorization of an SPD matrix.
    /// `uplo` is a single byte: b'U' or b'L' (passed as i8).
    #[link_name = "dpotrf_"]
    fn lapack_dpotrf(
        uplo: *const i8,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        info_out: *mut i32,
    );

    /// dsyevd — symmetric eigendecomp via divide-and-conquer.
    /// `jobz`: b'N' (eigenvalues only) or b'V' (also eigenvectors).
    #[link_name = "dsyevd_"]
    fn lapack_dsyevd(
        jobz: *const i8,
        uplo: *const i8,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        w: *mut f64,
        work: *mut f64,
        lwork: *const i32,
        iwork: *mut i32,
        liwork: *const i32,
        info_out: *mut i32,
    );

    /// dgeqrf — QR factorization. On output, A's upper triangle holds R;
    /// elementary reflectors stored below the diagonal + scalar factors `tau`.
    #[link_name = "dgeqrf_"]
    fn lapack_dgeqrf(
        m: *const i32,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        tau: *mut f64,
        work: *mut f64,
        lwork: *const i32,
        info_out: *mut i32,
    );

    /// dorgqr — generate Q from the elementary reflectors produced by dgeqrf.
    #[link_name = "dorgqr_"]
    fn lapack_dorgqr(
        m: *const i32,
        n: *const i32,
        k: *const i32,
        a: *mut f64,
        lda: *const i32,
        tau: *const f64,
        work: *mut f64,
        lwork: *const i32,
        info_out: *mut i32,
    );

    /// dgesvd — singular value decomposition. `jobu`/`jobvt`: b'A' (all),
    /// b'S' (singular vectors only), b'O' (overwrite A), b'N' (none).
    #[link_name = "dgesvd_"]
    fn lapack_dgesvd(
        jobu: *const i8,
        jobvt: *const i8,
        m: *const i32,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        s: *mut f64,
        u: *mut f64,
        ldu: *const i32,
        vt: *mut f64,
        ldvt: *const i32,
        work: *mut f64,
        lwork: *const i32,
        info_out: *mut i32,
    );

    /// cblas_dtrsm — BLAS-3 row-major-friendly triangular solve.
    /// `op(A) · X = α·B`  (or  `X · op(A) = α·B` for `side=Right`).
    /// A is triangular, B overwritten with X.
    fn cblas_dtrsm(
        order: i32,
        side: i32,
        uplo: i32,
        transa: i32,
        diag: i32,
        m: i32,
        n: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        b: *mut f64,
        ldb: i32,
    );
}

#[cfg(not(feature = "blas"))]
#[allow(non_snake_case, clippy::too_many_arguments)]
#[inline]
unsafe fn cblas_dgemm(
    _order: i32,
    transa: i32,
    transb: i32,
    m: i32,
    n: i32,
    k: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    b: *const f64,
    ldb: i32,
    beta: f64,
    c: *mut f64,
    ldc: i32,
) {
    let m = m as usize;
    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldc = ldc as usize;
    let trans_a = transa != NO_TRANS;
    let trans_b = transb != NO_TRANS;
    for i in 0..m {
        for j in 0..n {
            let mut acc: f64 = 0.0;
            for p in 0..k {
                let av = if trans_a {
                    unsafe { *a.add(p * lda + i) }
                } else {
                    unsafe { *a.add(i * lda + p) }
                };
                let bv = if trans_b {
                    unsafe { *b.add(j * ldb + p) }
                } else {
                    unsafe { *b.add(p * ldb + j) }
                };
                acc += av * bv;
            }
            let cp = unsafe { c.add(i * ldc + j) };
            unsafe {
                *cp = alpha * acc + beta * *cp;
            }
        }
    }
}

/// Pure-Rust LU + solve fallback for builds without BLAS/LAPACK.
/// Partial pivoting; column-major in/out to match LAPACK's ABI.
/// Returns 0 on success, k+1 if U[k,k] is zero (singular).
#[cfg(not(feature = "blas"))]
#[allow(non_snake_case, clippy::too_many_arguments)]
unsafe fn lapack_dgesv(
    n: *const i32,
    nrhs: *const i32,
    a: *mut f64,
    lda: *const i32,
    ipiv: *mut i32,
    b: *mut f64,
    ldb: *const i32,
    info_out: *mut i32,
) {
    let nn = unsafe { *n } as usize;
    let nrhs = unsafe { *nrhs } as usize;
    let lda = unsafe { *lda } as usize;
    let ldb = unsafe { *ldb } as usize;
    // Column-major access helper: a[i, j] = a[j*lda + i]
    let aij = |a: *mut f64, i: usize, j: usize| unsafe { a.add(j * lda + i) };
    for k in 0..nn {
        // Pivot: row with max |a[k..,k]|
        let mut piv = k;
        let mut max_abs = unsafe { *aij(a, k, k) }.abs();
        for i in (k + 1)..nn {
            let v = unsafe { *aij(a, i, k) }.abs();
            if v > max_abs {
                max_abs = v;
                piv = i;
            }
        }
        unsafe {
            *ipiv.add(k) = (piv + 1) as i32;
        }
        if max_abs == 0.0 {
            unsafe {
                *info_out = (k + 1) as i32;
            }
            return;
        }
        // Swap rows piv and k in A
        if piv != k {
            for j in 0..nn {
                let p1 = aij(a, k, j);
                let p2 = aij(a, piv, j);
                unsafe {
                    std::ptr::swap(p1, p2);
                }
            }
            for j in 0..nrhs {
                let p1 = unsafe { b.add(j * ldb + k) };
                let p2 = unsafe { b.add(j * ldb + piv) };
                unsafe {
                    std::ptr::swap(p1, p2);
                }
            }
        }
        // Eliminate
        let akk = unsafe { *aij(a, k, k) };
        for i in (k + 1)..nn {
            let factor = unsafe { *aij(a, i, k) } / akk;
            unsafe {
                *aij(a, i, k) = factor;
            }
            for j in (k + 1)..nn {
                let v = unsafe { *aij(a, i, j) } - factor * unsafe { *aij(a, k, j) };
                unsafe {
                    *aij(a, i, j) = v;
                }
            }
            for j in 0..nrhs {
                let v = unsafe { *b.add(j * ldb + i) } - factor * unsafe { *b.add(j * ldb + k) };
                unsafe {
                    *b.add(j * ldb + i) = v;
                }
            }
        }
    }
    // Back-substitute U·x = y
    for j in 0..nrhs {
        for i in (0..nn).rev() {
            let mut sum = unsafe { *b.add(j * ldb + i) };
            for k in (i + 1)..nn {
                sum -= unsafe { *aij(a, i, k) } * unsafe { *b.add(j * ldb + k) };
            }
            unsafe {
                *b.add(j * ldb + i) = sum / *aij(a, i, i);
            }
        }
    }
    unsafe {
        *info_out = 0;
    }
}

/// f32 twin of `lapack_dgesv` for no-blas builds. Same algorithm,
/// f32 instead of f64.
#[cfg(not(feature = "blas"))]
#[allow(non_snake_case, clippy::too_many_arguments)]
unsafe fn lapack_sgesv(
    n: *const i32,
    nrhs: *const i32,
    a: *mut f32,
    lda: *const i32,
    ipiv: *mut i32,
    b: *mut f32,
    ldb: *const i32,
    info_out: *mut i32,
) {
    let nn = unsafe { *n } as usize;
    let nrhs = unsafe { *nrhs } as usize;
    let lda = unsafe { *lda } as usize;
    let ldb = unsafe { *ldb } as usize;
    let aij = |a: *mut f32, i: usize, j: usize| unsafe { a.add(j * lda + i) };
    for k in 0..nn {
        let mut piv = k;
        let mut max_abs = unsafe { *aij(a, k, k) }.abs();
        for i in (k + 1)..nn {
            let v = unsafe { *aij(a, i, k) }.abs();
            if v > max_abs {
                max_abs = v;
                piv = i;
            }
        }
        unsafe {
            *ipiv.add(k) = (piv + 1) as i32;
        }
        if max_abs == 0.0 {
            unsafe {
                *info_out = (k + 1) as i32;
            }
            return;
        }
        if piv != k {
            for j in 0..nn {
                let p1 = aij(a, k, j);
                let p2 = aij(a, piv, j);
                unsafe {
                    std::ptr::swap(p1, p2);
                }
            }
            for j in 0..nrhs {
                let p1 = unsafe { b.add(j * ldb + k) };
                let p2 = unsafe { b.add(j * ldb + piv) };
                unsafe {
                    std::ptr::swap(p1, p2);
                }
            }
        }
        let akk = unsafe { *aij(a, k, k) };
        for i in (k + 1)..nn {
            let factor = unsafe { *aij(a, i, k) } / akk;
            unsafe {
                *aij(a, i, k) = factor;
            }
            for j in (k + 1)..nn {
                let v = unsafe { *aij(a, i, j) } - factor * unsafe { *aij(a, k, j) };
                unsafe {
                    *aij(a, i, j) = v;
                }
            }
            for j in 0..nrhs {
                let v = unsafe { *b.add(j * ldb + i) } - factor * unsafe { *b.add(j * ldb + k) };
                unsafe {
                    *b.add(j * ldb + i) = v;
                }
            }
        }
    }
    for j in 0..nrhs {
        for i in (0..nn).rev() {
            let mut sum = unsafe { *b.add(j * ldb + i) };
            for k in (i + 1)..nn {
                sum -= unsafe { *aij(a, i, k) } * unsafe { *b.add(j * ldb + k) };
            }
            unsafe {
                *b.add(j * ldb + i) = sum / *aij(a, i, i);
            }
        }
    }
    unsafe {
        *info_out = 0;
    }
}

/// f64 GEMM. C = A @ B, all row-major, A: [m, k], B: [k, n], C: [m, n].
#[inline]
pub fn dgemm(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    unsafe {
        cblas_dgemm(
            ROW_MAJOR,
            NO_TRANS,
            NO_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
}

/// Solve `A · x = b` in-place. Row-major caller's API: `a` is `[n, n]`
/// row-major and `b` is `[n]` (single-RHS) or `[n, nrhs]` row-major.
/// On return, `b` holds the solution. `a` is overwritten with LU
/// factors (caller's copy is destroyed).
///
/// Implementation detail: LAPACK's `dgesv_` is column-major, so we
/// transpose `A` in place (square ⇒ cheap), call `dgesv_` with the
/// natural column-major `B` interpretation, and transpose `B` back if
/// `nrhs > 1`. For `nrhs = 1` the column-major and row-major layouts
/// of a vector are identical — no transpose on B needed.
///
/// Returns 0 on success; k > 0 means `U[k-1, k-1]` was exactly zero
/// (singular system).
pub fn dgesv(a: &mut [f64], b: &mut [f64], n: usize, nrhs: usize) -> i32 {
    assert_eq!(a.len(), n * n, "dgesv: A must be n×n");
    assert_eq!(b.len(), n * nrhs, "dgesv: B must be n×nrhs");
    // Row→col-major: in-place transpose of the square A.
    for i in 0..n {
        for j in (i + 1)..n {
            a.swap(i * n + j, j * n + i);
        }
    }
    // Same for B if nrhs > 1.
    if nrhs > 1 {
        let mut tmp = vec![0f64; n * nrhs];
        for i in 0..n {
            for j in 0..nrhs {
                tmp[j * n + i] = b[i * nrhs + j];
            }
        }
        b.copy_from_slice(&tmp);
    }
    let mut ipiv = vec![0i32; n];
    let mut info: i32 = 0;
    let nn = n as i32;
    let nrhs_i = nrhs as i32;
    unsafe {
        lapack_dgesv(
            &nn,
            &nrhs_i,
            a.as_mut_ptr(),
            &nn,
            ipiv.as_mut_ptr(),
            b.as_mut_ptr(),
            &nn,
            &mut info,
        );
    }
    // Col-major B back to row-major.
    if nrhs > 1 && info == 0 {
        let mut tmp = vec![0f64; n * nrhs];
        for j in 0..nrhs {
            for i in 0..n {
                tmp[i * nrhs + j] = b[j * n + i];
            }
        }
        b.copy_from_slice(&tmp);
    }
    info
}

/// f32 twin of `dgesv`. Same row-major caller's API + same return
/// code semantics. Uses LAPACK's `sgesv_` under the hood.
pub fn sgesv(a: &mut [f32], b: &mut [f32], n: usize, nrhs: usize) -> i32 {
    assert_eq!(a.len(), n * n, "sgesv: A must be n×n");
    assert_eq!(b.len(), n * nrhs, "sgesv: B must be n×nrhs");
    for i in 0..n {
        for j in (i + 1)..n {
            a.swap(i * n + j, j * n + i);
        }
    }
    if nrhs > 1 {
        let mut tmp = vec![0f32; n * nrhs];
        for i in 0..n {
            for j in 0..nrhs {
                tmp[j * n + i] = b[i * nrhs + j];
            }
        }
        b.copy_from_slice(&tmp);
    }
    let mut ipiv = vec![0i32; n];
    let mut info: i32 = 0;
    let nn = n as i32;
    let nrhs_i = nrhs as i32;
    unsafe {
        lapack_sgesv(
            &nn,
            &nrhs_i,
            a.as_mut_ptr(),
            &nn,
            ipiv.as_mut_ptr(),
            b.as_mut_ptr(),
            &nn,
            &mut info,
        );
    }
    if nrhs > 1 && info == 0 {
        let mut tmp = vec![0f32; n * nrhs];
        for j in 0..nrhs {
            for i in 0..n {
                tmp[i * nrhs + j] = b[j * n + i];
            }
        }
        b.copy_from_slice(&tmp);
    }
    info
}

const ROW_MAJOR: i32 = 101;
const NO_TRANS: i32 = 111;
const TRANS: i32 = 112;
const CBLAS_LEFT: i32 = 141;
#[allow(dead_code)]
const CBLAS_RIGHT: i32 = 142;
const CBLAS_UPPER: i32 = 121;
const CBLAS_LOWER: i32 = 122;
const CBLAS_NON_UNIT: i32 = 131;
#[allow(dead_code)]
const CBLAS_UNIT: i32 = 132;

// ── LAPACK fallback stubs (no-blas builds) ───────────────────────
//
// The new linalg ops (Cholesky / eigh / QR / SVD / dtrsm) require
// LAPACK and CBLAS Level 3 — implementing pure-Rust fallbacks is
// substantial work that's out of scope for a v1. Non-blas builds
// link the FFI symbol but never call it on an Op::Custom-routed
// path; if someone explicitly invokes one of these wrappers without
// the `blas` feature, the panic surfaces the missing dependency.

#[cfg(not(feature = "blas"))]
#[allow(non_snake_case, clippy::too_many_arguments)]
unsafe fn lapack_dpotrf(_: *const i8, _: *const i32, _: *mut f64, _: *const i32, info: *mut i32) {
    unsafe {
        *info = -1;
    }
    panic!("rlx-cpu: dpotrf requires the `blas` feature (LAPACK)");
}
#[cfg(not(feature = "blas"))]
#[allow(non_snake_case, clippy::too_many_arguments)]
unsafe fn lapack_dsyevd(
    _: *const i8,
    _: *const i8,
    _: *const i32,
    _: *mut f64,
    _: *const i32,
    _: *mut f64,
    _: *mut f64,
    _: *const i32,
    _: *mut i32,
    _: *const i32,
    info: *mut i32,
) {
    unsafe {
        *info = -1;
    }
    panic!("rlx-cpu: dsyevd requires the `blas` feature (LAPACK)");
}
#[cfg(not(feature = "blas"))]
#[allow(non_snake_case, clippy::too_many_arguments)]
unsafe fn lapack_dgeqrf(
    _: *const i32,
    _: *const i32,
    _: *mut f64,
    _: *const i32,
    _: *mut f64,
    _: *mut f64,
    _: *const i32,
    info: *mut i32,
) {
    unsafe {
        *info = -1;
    }
    panic!("rlx-cpu: dgeqrf requires the `blas` feature (LAPACK)");
}
#[cfg(not(feature = "blas"))]
#[allow(non_snake_case, clippy::too_many_arguments)]
unsafe fn lapack_dorgqr(
    _: *const i32,
    _: *const i32,
    _: *const i32,
    _: *mut f64,
    _: *const i32,
    _: *const f64,
    _: *mut f64,
    _: *const i32,
    info: *mut i32,
) {
    unsafe {
        *info = -1;
    }
    panic!("rlx-cpu: dorgqr requires the `blas` feature (LAPACK)");
}
#[cfg(not(feature = "blas"))]
#[allow(non_snake_case, clippy::too_many_arguments)]
unsafe fn lapack_dgesvd(
    _: *const i8,
    _: *const i8,
    _: *const i32,
    _: *const i32,
    _: *mut f64,
    _: *const i32,
    _: *mut f64,
    _: *mut f64,
    _: *const i32,
    _: *mut f64,
    _: *const i32,
    _: *mut f64,
    _: *const i32,
    info: *mut i32,
) {
    unsafe {
        *info = -1;
    }
    panic!("rlx-cpu: dgesvd requires the `blas` feature (LAPACK)");
}
#[cfg(not(feature = "blas"))]
#[allow(non_snake_case, clippy::too_many_arguments)]
unsafe fn cblas_dtrsm(
    _: i32,
    _: i32,
    _: i32,
    _: i32,
    _: i32,
    _: i32,
    _: i32,
    _: f64,
    _: *const f64,
    _: i32,
    _: *mut f64,
    _: i32,
) {
    panic!("rlx-cpu: cblas_dtrsm requires the `blas` feature");
}

// ── Linalg row-major wrappers ────────────────────────────────────

/// In-place Cholesky factorization of an SPD matrix `A`. The caller
/// passes `a` row-major, n×n. On output:
///   - if `lower`: lower triangle of `a` holds `L` such that `L·Lᵀ = A`,
///     upper triangle is zeroed.
///   - if `!lower`: upper triangle holds `U` such that `Uᵀ·U = A`,
///     lower triangle is zeroed.
/// Returns 0 on success; k>0 means the leading minor of order k is
/// not positive-definite (A is not SPD).
///
/// LAPACK's `dpotrf_` is column-major. For symmetric A, the col-major
/// bytes equal the row-major bytes (`A = Aᵀ`). After dpotrf with
/// UPLO='U' (col-major upper), the factor U lives in col-major's
/// upper triangle — which is row-major's lower triangle = L. Same
/// trick in reverse for the upper-row-major case.
pub fn dpotrf(a: &mut [f64], n: usize, lower: bool) -> i32 {
    assert_eq!(a.len(), n * n, "dpotrf: A must be n×n");
    let uplo: i8 = if lower { b'U' as i8 } else { b'L' as i8 };
    let nn = n as i32;
    let mut info: i32 = 0;
    unsafe {
        lapack_dpotrf(&uplo, &nn, a.as_mut_ptr(), &nn, &mut info);
    }
    if info != 0 {
        return info;
    }
    // Mask the unused triangle in the row-major view.
    if lower {
        for i in 0..n {
            for j in (i + 1)..n {
                a[i * n + j] = 0.0;
            }
        }
    } else {
        for i in 1..n {
            for j in 0..i {
                a[i * n + j] = 0.0;
            }
        }
    }
    info
}

/// In-place symmetric eigendecomposition. `a` is row-major n×n
/// symmetric. On output: `w` holds eigenvalues (length n, ascending);
/// `a` is overwritten with eigenvectors as columns (col-major) =
/// rows (row-major view, since the matrix's transpose interpretation
/// is the same set of orthonormal eigenvectors). Symmetric → no
/// transpose dance needed.
///
/// Returns 0 on success; k>0 means the algorithm failed to converge.
pub fn dsyevd(a: &mut [f64], w: &mut [f64], n: usize) -> i32 {
    assert_eq!(a.len(), n * n);
    assert_eq!(w.len(), n);
    let jobz: i8 = b'V' as i8;
    let uplo: i8 = b'U' as i8;
    let nn = n as i32;
    let mut info: i32 = 0;
    // dsyevd workspace (per LAPACK manual):
    //   lwork  ≥ 1 + 6n + 2n²       (jobz='V')
    //   liwork ≥ 3 + 5n             (jobz='V')
    let lwork = (1 + 6 * n + 2 * n * n) as i32;
    let liwork = (3 + 5 * n) as i32;
    let mut work = vec![0f64; lwork.max(1) as usize];
    let mut iwork = vec![0i32; liwork.max(1) as usize];
    unsafe {
        lapack_dsyevd(
            &jobz,
            &uplo,
            &nn,
            a.as_mut_ptr(),
            &nn,
            w.as_mut_ptr(),
            work.as_mut_ptr(),
            &lwork,
            iwork.as_mut_ptr(),
            &liwork,
            &mut info,
        );
    }
    info
}

/// QR factorization. Inputs:
///   - `a`: m×n row-major matrix (overwritten by the factorization)
///   - `q_out`: m×k row-major, k = min(m, n) — receives Q
///   - `r_out`: k×n row-major — receives R
/// Returns 0 on success; <0 means LAPACK got a bad arg.
pub fn dgeqrf_full(a: &mut [f64], m: usize, n: usize, q_out: &mut [f64], r_out: &mut [f64]) -> i32 {
    assert_eq!(a.len(), m * n, "dgeqrf: A must be m×n");
    let k = m.min(n);
    assert_eq!(q_out.len(), m * k, "Q must be m×min(m,n)");
    assert_eq!(r_out.len(), k * n, "R must be min(m,n)×n");

    // Row→col-major: transpose A in place.
    let mut a_col = transpose_to_col(a, m, n);
    let mut tau = vec![0f64; k];
    let mm = m as i32;
    let nn = n as i32;
    let kk = k as i32;
    let lwork = (n.max(1)) as i32;
    let mut work = vec![0f64; lwork.max(1) as usize];
    let mut info: i32 = 0;
    unsafe {
        lapack_dgeqrf(
            &mm,
            &nn,
            a_col.as_mut_ptr(),
            &mm,
            tau.as_mut_ptr(),
            work.as_mut_ptr(),
            &lwork,
            &mut info,
        );
    }
    if info != 0 {
        return info;
    }

    // Extract R from the upper triangle of the (k×n) leading block of
    // a_col (col-major). Row-major R has shape [k, n]; entry (i,j)
    // = a_col[j * m + i] for i ≤ j, else 0.
    for i in 0..k {
        for j in 0..n {
            let v = if i <= j { a_col[j * m + i] } else { 0.0 };
            r_out[i * n + j] = v;
        }
    }

    // Form Q in-place (col-major, m×k) using dorgqr.
    let mut work2 = vec![0f64; lwork.max(1) as usize];
    let mut info2: i32 = 0;
    unsafe {
        lapack_dorgqr(
            &mm,
            &kk,
            &kk,
            a_col.as_mut_ptr(),
            &mm,
            tau.as_ptr(),
            work2.as_mut_ptr(),
            &lwork,
            &mut info2,
        );
    }
    if info2 != 0 {
        return info2;
    }

    // Transpose Q (col-major m×k) → row-major m×k.
    for i in 0..m {
        for j in 0..k {
            q_out[i * k + j] = a_col[j * m + i];
        }
    }
    0
}

/// SVD: `a = U · diag(s) · V^T`. Inputs:
///   - `a`: m×n row-major (destroyed on return)
///   - `s`: length min(m, n) — singular values, descending
///   - `u`: m×min(m,n) row-major
///   - `vt`: min(m,n)×n row-major (V transposed)
/// Returns 0 on success; >0 means superdiagonals failed to converge.
pub fn dgesvd_thin(
    a: &mut [f64],
    m: usize,
    n: usize,
    s: &mut [f64],
    u: &mut [f64],
    vt: &mut [f64],
) -> i32 {
    assert_eq!(a.len(), m * n);
    let k = m.min(n);
    assert_eq!(s.len(), k);
    assert_eq!(u.len(), m * k);
    assert_eq!(vt.len(), k * n);

    let mut a_col = transpose_to_col(a, m, n);
    let mut u_col = vec![0f64; m * k];
    let mut vt_col = vec![0f64; k * n];
    let jobu = b'S' as i8;
    let jobvt = b'S' as i8;
    let mm = m as i32;
    let nn = n as i32;
    let ldu = m as i32;
    let ldvt = k as i32;
    // dgesvd workspace: lwork ≥ max(3*min(m,n) + max(m,n), 5*min(m,n))
    let lwork = (((3 * k + m.max(n)).max(5 * k)) as i32).max(1);
    let mut work = vec![0f64; lwork as usize];
    let mut info: i32 = 0;
    unsafe {
        lapack_dgesvd(
            &jobu,
            &jobvt,
            &mm,
            &nn,
            a_col.as_mut_ptr(),
            &mm,
            s.as_mut_ptr(),
            u_col.as_mut_ptr(),
            &ldu,
            vt_col.as_mut_ptr(),
            &ldvt,
            work.as_mut_ptr(),
            &lwork,
            &mut info,
        );
    }
    if info != 0 {
        return info;
    }
    // Transpose U (col-major m×k) → row-major m×k.
    for i in 0..m {
        for j in 0..k {
            u[i * k + j] = u_col[j * m + i];
        }
    }
    // Transpose V^T (col-major k×n) → row-major k×n.
    for i in 0..k {
        for j in 0..n {
            vt[i * n + j] = vt_col[j * k + i];
        }
    }
    0
}

/// Solve `op(A) · X = B` (or `X · op(A) = B` for `right`) where A is
/// triangular. Row-major callers throughout. Overwrites B with X.
///
///   - `a`: n×n row-major triangular
///   - `b`: m×n_rhs row-major (B = X · A) or n_rhs×n (A · X = B)?
///
/// Concretely: solve `A · X = B`:
///   - `a`: n×n row-major triangular (lower or upper per `lower`)
///   - `b`: n×nrhs row-major. Overwritten with X (n×nrhs).
pub fn dtrsm_lower_or_upper(
    a: &[f64],
    b: &mut [f64],
    n: usize,
    nrhs: usize,
    lower: bool,
    transpose_a: bool,
) {
    assert_eq!(a.len(), n * n);
    assert_eq!(b.len(), n * nrhs);
    unsafe {
        cblas_dtrsm(
            ROW_MAJOR,
            CBLAS_LEFT,
            if lower { CBLAS_LOWER } else { CBLAS_UPPER },
            if transpose_a { TRANS } else { NO_TRANS },
            CBLAS_NON_UNIT,
            n as i32,
            nrhs as i32,
            1.0,
            a.as_ptr(),
            n as i32,
            b.as_mut_ptr(),
            nrhs as i32,
        );
    }
}

fn transpose_to_col(a_row: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            out[j * m + i] = a_row[i * n + j];
        }
    }
    out
}

/// C = alpha * A @ B + beta * C
///
/// A: [m, k] row-major
/// B: [k, n] row-major
/// C: [m, n] row-major
#[inline]
pub fn sgemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    unsafe {
        cblas_sgemm(
            ROW_MAJOR,
            NO_TRANS,
            NO_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
}

/// C += alpha * A @ B (accumulate into existing C)
#[inline]
pub fn sgemm_accumulate(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    unsafe {
        cblas_sgemm(
            ROW_MAJOR,
            NO_TRANS,
            NO_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            1.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
}

/// C = A @ B^T (B transposed — useful for SDPA Q@K^T)
#[inline]
pub fn sgemm_bt(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, alpha: f32) {
    unsafe {
        cblas_sgemm(
            ROW_MAJOR,
            NO_TRANS,
            TRANS,
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            k as i32, // B is [n, k], transposed to [k, n]
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
}

/// C = A @ B with custom strides (for reading interleaved data).
/// lda = stride between rows of A, ldc = stride between rows of C.
#[inline]
pub fn sgemm_strided(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    lda: usize,
    ldc: usize,
) {
    unsafe {
        cblas_sgemm(
            ROW_MAJOR,
            NO_TRANS,
            NO_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            lda as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            c.as_mut_ptr(),
            ldc as i32,
        );
    }
}

/// NEON-vectorized bias addition in-place.
#[cfg(target_arch = "aarch64")]
pub fn bias_add(data: &mut [f32], bias: &[f32], m: usize, n: usize) {
    use std::arch::aarch64::*;
    let chunks = n / 4;
    unsafe {
        for row in 0..m {
            let base = row * n;
            for c in 0..chunks {
                let off = base + c * 4;
                let v = vld1q_f32(data.as_ptr().add(off));
                let b = vld1q_f32(bias.as_ptr().add(c * 4));
                vst1q_f32(data.as_mut_ptr().add(off), vaddq_f32(v, b));
            }
            for i in (chunks * 4)..n {
                data[base + i] += bias[i];
            }
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn bias_add(data: &mut [f32], bias: &[f32], m: usize, n: usize) {
    for row in 0..m {
        let base = row * n;
        for i in 0..n {
            data[base + i] += bias[i];
        }
    }
}

/// General sgemm with full control over transposition and strides.
/// C = alpha * op(A) @ op(B) + beta * C
/// op(X) = X if trans=false, X^T if trans=true
///
/// For row-major:
/// - NoTrans A: M×K with lda ≥ K
/// - Trans A:   stored K×M with lda ≥ M
/// - NoTrans B: K×N with ldb ≥ N
/// - Trans B:   stored N×K with ldb ≥ K
///
/// # Safety
/// Caller must ensure pointers are valid and output region is writable.
#[inline]
pub unsafe fn sgemm_general(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
    lda: usize,
    ldb: usize,
    ldc: usize,
    trans_a: bool,
    trans_b: bool,
) {
    unsafe {
        cblas_sgemm(
            ROW_MAJOR,
            if trans_a { TRANS } else { NO_TRANS },
            if trans_b { TRANS } else { NO_TRANS },
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a,
            lda as i32,
            b,
            ldb as i32,
            beta,
            c,
            ldc as i32,
        );
    }
}

/// sgemm + bias addition in one call. C = A @ B + bias (broadcast per row).
/// Auto-dispatches: NEON for tiny matrices, BLAS for everything else.
#[inline]
pub fn sgemm_bias(a: &[f32], b: &[f32], bias: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    // Cost model decides: NEON when BLAS overhead dominates the compute.
    if crate::cost::hw_model().prefer_neon_sgemm(m, k, n) {
        crate::kernels::neon_sgemm_bias_small(a, b, bias, c, m, k, n);
    } else {
        sgemm(a, b, c, m, k, n);
        bias_add(c, bias, m, n);
    }
}

/// sgemm with a generic epilogue closure (plan #1).
///
/// `C = epilogue(A @ B)`. The closure runs element-wise after the
/// matmul. Generics + `#[inline]` make the closure body monomorphize
/// per call site, so adding a new fusion type (matmul+scale, matmul+
/// clamp, matmul+gelu, etc.) is a new closure at the call site —
/// not a new `Thunk` variant + a new dispatch arm + a new boxed
/// function. Borrowed from MAX's `elementwise_lambda_fn` parameter
/// pattern on matmul; the Rust spelling is `impl Fn(f32) -> f32`.
///
/// For shapes where NEON beats BLAS, this routes through the NEON
/// path and applies the epilogue inline. Otherwise it does sgemm
/// then a single elementwise pass — still one less round-trip
/// through memory than sgemm + bias_add + activation.
#[inline]
pub fn sgemm_epilogue<E: Fn(f32) -> f32>(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    epilogue: E,
) {
    sgemm(a, b, c, m, k, n);
    for v in c.iter_mut() {
        *v = epilogue(*v);
    }
}

/// sgemm + per-row bias + arbitrary post-activation in one call.
/// `C[i, j] = activation(A[i,:] @ B[:,j] + bias[j])`.
///
/// Mirrors `sgemm_bias` but the activation can be any closure.
/// `FusedMatMulBiasAct` could be reimplemented on top of this
/// instead of carrying its own dispatch logic.
#[inline]
pub fn sgemm_bias_epilogue<E: Fn(f32) -> f32>(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    activation: E,
) {
    sgemm(a, b, c, m, k, n);
    // Fuse bias + activation in one pass over C.
    for i in 0..m {
        let row = &mut c[i * n..(i + 1) * n];
        for (j, v) in row.iter_mut().enumerate() {
            *v = activation(*v + bias[j]);
        }
    }
}

/// sgemm with auto-dispatch: NEON for tiny m, parallel BLAS for small-m, sequential BLAS otherwise.
#[inline]
pub fn sgemm_auto(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    // `parity-gemm` feature swaps in the same Rust `gemm` crate that
    // candle uses, yielding bit-exact reduction order. Useful for
    // parity tests and reproducibility-critical workloads; loses AMX.
    #[cfg(feature = "parity-gemm")]
    {
        sgemm_via_gemm_crate(a, b, c, m, k, n);
        return;
    }
    // `neon_sgemm_small` has a fixed `acc[8]` accumulator so m must be
    // ≤ 8. The cost model prefers NEON whenever its FLOP rate beats
    // BLAS-plus-overhead, which can fire for m up to thousands at
    // small k·n — we'd OOB on `acc[i]`. Cap explicitly.
    #[cfg(not(feature = "parity-gemm"))]
    if m <= 8 && crate::cost::hw_model().prefer_neon_sgemm(m, k, n) {
        crate::kernels::neon_sgemm_small(a, b, c, m, k, n);
    } else if m < 32 {
        // Multi-core AMX split for small m where AMX internal threading doesn't engage
        par_sgemm(a, b, c, m, k, n);
    } else {
        sgemm(a, b, c, m, k, n);
    }
}

/// Bit-exact CPU sgemm via the same `gemm` crate candle uses for its
/// CPU backend. Row-major `[m, k] @ [k, n] = [m, n]` overwrite (β = 0).
#[cfg(feature = "parity-gemm")]
fn sgemm_via_gemm_crate(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    use gemm::{Parallelism, gemm};
    let cfg = crate::config::RuntimeConfig::global();
    let workers = cfg.pool_workers + 1;
    let par = if workers > 1 {
        Parallelism::Rayon(workers)
    } else {
        Parallelism::None
    };
    // Row-major strides (cs=1, rs=#cols).
    unsafe {
        gemm(
            m,
            n,
            k,
            c.as_mut_ptr(),
            1,          // dst_cs
            n as isize, // dst_rs
            false,      // read_dst (β=0)
            a.as_ptr(),
            1,          // lhs_cs
            k as isize, // lhs_rs
            b.as_ptr(),
            1,          // rhs_cs
            n as isize, // rhs_rs
            0.0,        // alpha (zero out)
            1.0,        // beta
            false,      // conj_dst
            false,      // conj_lhs
            false,      // conj_rhs
            par,
        );
    }
}

/// Parallel sgemm without bias (split across n dimension for multi-core AMX).
pub fn par_sgemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    let cfg = crate::config::RuntimeConfig::global();
    let workers = cfg.pool_workers + 1;
    let total_flops = (m * k * n) as u64;
    if m >= 32 || total_flops < 2_000_000 || n < workers * 32 {
        sgemm(a, b, c, m, k, n);
        return;
    }
    let chunk = n / workers;
    let a_addr = a.as_ptr() as usize;
    let b_addr = b.as_ptr() as usize;
    let c_addr = c.as_mut_ptr() as usize;

    crate::pool::par_for(workers, 1, &|off, cnt| {
        for w in off..off + cnt {
            let n_start = w * chunk;
            let n_end = if w + 1 == workers { n } else { (w + 1) * chunk };
            let local_n = n_end - n_start;
            if local_n == 0 {
                continue;
            }
            unsafe {
                cblas_sgemm(
                    101,
                    111,
                    111,
                    m as i32,
                    local_n as i32,
                    k as i32,
                    1.0,
                    a_addr as *const f32,
                    k as i32,
                    (b_addr as *const f32).add(n_start),
                    n as i32,
                    0.0,
                    (c_addr as *mut f32).add(n_start),
                    n as i32,
                );
            }
        }
    });
}

/// Parallelized sgemm + bias: splits across the n dimension to use multiple
/// cores' AMX coprocessors on Apple Silicon. Critical for small-m workloads
/// where AMX internal threading doesn't engage.
///
/// For each worker thread: computes a column slice of C using its own AMX.
pub fn par_sgemm_bias(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    let cfg = crate::config::RuntimeConfig::global();
    let workers = cfg.pool_workers + 1; // total threads (workers + main)
    // Only split when AMX wastes capacity (m < 32 tile width).
    // For m ≥ 32, AMX's internal threading handles parallelism better.
    let total_flops = (m * k * n) as u64;
    if m >= 32 || total_flops < 2_000_000 || n < workers * 32 {
        sgemm_bias(a, b, bias, c, m, k, n);
        return;
    }

    // Split n into approximately equal chunks
    let chunk = n / workers;
    let a_addr = a.as_ptr() as usize;
    let b_addr = b.as_ptr() as usize;
    let bias_addr = bias.as_ptr() as usize;
    let c_addr = c.as_mut_ptr() as usize;

    crate::pool::par_for(workers, 1, &|off, cnt| {
        for w in off..off + cnt {
            let n_start = w * chunk;
            let n_end = if w + 1 == workers { n } else { (w + 1) * chunk };
            let local_n = n_end - n_start;
            if local_n == 0 {
                continue;
            }
            unsafe {
                // sgemm: A[m,k] @ B[:, n_start..n_end] → C[:, n_start..n_end]
                // ldb = n (B's full row stride), ldc = n (C's full row stride)

                cblas_sgemm(
                    101,
                    111,
                    111, // ROW_MAJOR, NO_TRANS, NO_TRANS
                    m as i32,
                    local_n as i32,
                    k as i32,
                    1.0,
                    a_addr as *const f32,
                    k as i32,
                    (b_addr as *const f32).add(n_start),
                    n as i32,
                    0.0,
                    (c_addr as *mut f32).add(n_start),
                    n as i32,
                );
                // bias_add for this column slice
                let local_bias =
                    std::slice::from_raw_parts((bias_addr as *const f32).add(n_start), local_n);
                let local_c =
                    std::slice::from_raw_parts_mut((c_addr as *mut f32).add(n_start), m * n);
                #[cfg(target_arch = "aarch64")]
                {
                    use std::arch::aarch64::*;
                    let chunks = local_n / 4;
                    for row in 0..m {
                        let base = row * n;
                        for c in 0..chunks {
                            let off = base + c * 4;
                            let v = vld1q_f32(local_c.as_ptr().add(off));
                            let bv = vld1q_f32(local_bias.as_ptr().add(c * 4));
                            vst1q_f32(local_c.as_mut_ptr().add(off), vaddq_f32(v, bv));
                        }
                        for i in (chunks * 4)..local_n {
                            local_c[base + i] += local_bias[i];
                        }
                    }
                }
                #[cfg(not(target_arch = "aarch64"))]
                for row in 0..m {
                    let base = row * n;
                    for i in 0..local_n {
                        local_c[base + i] += local_bias[i];
                    }
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgemm_identity() {
        // A = [[1,0],[0,1]], B = [[3,4],[5,6]]
        let a = [1.0, 0.0, 0.0, 1.0f32];
        let b = [3.0, 4.0, 5.0, 6.0f32];
        let mut c = [0.0f32; 4];
        sgemm(&a, &b, &mut c, 2, 2, 2);
        assert_eq!(c, [3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn sgemm_rectangular() {
        // [2,3] @ [3,2] = [2,2]
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32]; // [2,3]
        let b = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0f32]; // [3,2]
        let mut c = [0.0f32; 4];
        sgemm(&a, &b, &mut c, 2, 3, 2);
        // [1*1+2*0+3*1, 1*0+2*1+3*0] = [4, 2]
        // [4*1+5*0+6*1, 4*0+5*1+6*0] = [10, 5]
        assert_eq!(c, [4.0, 2.0, 10.0, 5.0]);
    }

    #[test]
    fn sgemm_bias_test() {
        let a = [1.0, 0.0, 0.0, 1.0f32]; // identity [2,2]
        let b = [3.0, 4.0, 5.0, 6.0f32]; // [2,2]
        let bias = [10.0, 20.0f32]; // [2]
        let mut c = [0.0f32; 4];
        sgemm_bias(&a, &b, &bias, &mut c, 2, 2, 2);
        assert_eq!(c, [13.0, 24.0, 15.0, 26.0]);
    }

    #[test]
    fn dgemm_identity() {
        let a = [1.0, 0.0, 0.0, 1.0f64];
        let b = [3.0, 4.0, 5.0, 6.0f64];
        let mut c = [0.0f64; 4];
        dgemm(&a, &b, &mut c, 2, 2, 2);
        assert_eq!(c, [3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn dgesv_2x2_known_solution() {
        // A = [[2, 1],
        //      [1, 3]],   b = [5, 10]
        // Solution: x = [1, 3]  (verified: 2·1 + 1·3 = 5; 1·1 + 3·3 = 10)
        let mut a = [2.0, 1.0, 1.0, 3.0_f64];
        let mut b = [5.0, 10.0_f64];
        let info = dgesv(&mut a, &mut b, 2, 1);
        assert_eq!(info, 0, "dgesv signaled singular: info={info}");
        let want = [1.0, 3.0_f64];
        for (i, (g, w)) in b.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-12, "x[{i}] = {g}, expected {w}");
        }
    }

    #[test]
    fn dgesv_3x3_general() {
        // A = [[ 4, -1,  0],
        //      [-1,  4, -1],
        //      [ 0, -1,  4]]    (1-D Laplacian-ish; symmetric pos-def)
        // b = [1, 0, -1]
        // Expected: solve via reference. We just check Ax ≈ b after.
        let a_orig = [4.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 4.0_f64];
        let mut a = a_orig;
        let mut b = [1.0, 0.0, -1.0_f64];
        let info = dgesv(&mut a, &mut b, 3, 1);
        assert_eq!(info, 0);
        // Verify: Ax ≈ b_original.
        let mut residual = [0.0_f64; 3];
        for i in 0..3 {
            for j in 0..3 {
                residual[i] += a_orig[i * 3 + j] * b[j];
            }
        }
        let want_b = [1.0, 0.0, -1.0_f64];
        for i in 0..3 {
            assert!(
                (residual[i] - want_b[i]).abs() < 1e-12,
                "residual[{i}] = {} vs {}",
                residual[i],
                want_b[i]
            );
        }
    }

    #[test]
    fn sgemm_bt_test() {
        // Q@K^T: Q=[2,3], K=[2,3], result=[2,2]
        let q = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0f32]; // [2,3]
        let k = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0f32]; // [2,3]
        let mut scores = [0.0f32; 4];
        sgemm_bt(&q, &k, &mut scores, 2, 3, 2, 1.0);
        // Q@K^T = [[1,0],[0,1]] (dot products of rows)
        assert_eq!(scores, [1.0, 0.0, 0.0, 1.0]);
    }

    /// Plan #1: epilogue closure runs after the matmul. Identity
    /// closure must match plain sgemm; relu closure must match the
    /// hand-fused reference.
    #[test]
    fn sgemm_epilogue_matches_post_pass() {
        let a = [1.0f32, -2.0, 3.0, -4.0]; // [2, 2]
        let b = [1.0f32, 0.0, 0.0, 1.0]; // [2, 2]  (identity)
        let mut c1 = [0f32; 4];
        let mut c2 = [0f32; 4];
        // identity epilogue == plain sgemm
        sgemm(&a, &b, &mut c1, 2, 2, 2);
        sgemm_epilogue(&a, &b, &mut c2, 2, 2, 2, |x| x);
        assert_eq!(c1, c2);

        // relu epilogue zeros negative outputs
        let mut c3 = [0f32; 4];
        sgemm_epilogue(&a, &b, &mut c3, 2, 2, 2, |x| x.max(0.0));
        assert_eq!(c3, [1.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn sgemm_bias_epilogue_matches_reference() {
        let a = [1.0f32, 2.0, 3.0, 4.0]; // [2, 2]
        let b = [1.0f32, 0.0, 0.0, 1.0]; // [2, 2]
        let bias = [10.0f32, 100.0];
        // Reference: A@B + bias, then activation
        // A@B = [[1, 2], [3, 4]]; +bias = [[11, 102], [13, 104]]; relu = same
        let mut c = [0f32; 4];
        sgemm_bias_epilogue(&a, &b, &bias, &mut c, 2, 2, 2, |x| x.max(0.0));
        assert_eq!(c, [11.0, 102.0, 13.0, 104.0]);
    }
}
