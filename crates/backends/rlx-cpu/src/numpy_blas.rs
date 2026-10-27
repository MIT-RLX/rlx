// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The exact BLAS/LAPACK entry points NumPy and SciPy call on macOS.
//!
//! NumPy ≥ 2.0's wheels link Accelerate's **new** interface with 64-bit integers
//! (`cblas_dgemm$NEWLAPACK$ILP64`, …); SciPy's link the new interface with 32-bit
//! integers (`dgelsd$NEWLAPACK`). Both differ from the legacy CLAPACK symbols the
//! rest of this crate binds (`cblas_dgemm`, `dgelsd_`) — the LAPACK drivers are
//! different code, and matching NumPy bit-for-bit means calling the same symbol
//! with the same arguments. (Measured: `np.polyfit` slopes on 7-sample windows
//! differ from legacy `dgelsd_` in the last bits on 16 % of solves.)
//!
//! These are thin, row-/column-major-agnostic wrappers: callers pass CBLAS
//! `order`/`trans` codes and leading dimensions exactly as NumPy would.
//! `rlx_linalg::numpy` builds NumPy's `dot` / `matmul` routing on top.
//!
//! On a non-Accelerate build every function falls back to a plain Rust loop of
//! the same mathematical operation (NumPy there uses its bundled OpenBLAS, which
//! is not bit-reproduced here).

/// CBLAS `CblasRowMajor` / `CblasColMajor`.
pub const ROW_MAJOR: i32 = 101;
pub const COL_MAJOR: i32 = 102;
/// CBLAS `CblasNoTrans` / `CblasTrans`.
pub const NO_TRANS: i32 = 111;
pub const TRANS: i32 = 112;
/// CBLAS `CblasUpper`.
pub const UPPER: i32 = 121;

#[cfg(rlx_cpu_blas_accelerate)]
unsafe extern "C" {
    #[link_name = "cblas_dgemm$NEWLAPACK$ILP64"]
    fn np_cblas_dgemm(
        order: i32,
        transa: i32,
        transb: i32,
        m: i64,
        n: i64,
        k: i64,
        alpha: f64,
        a: *const f64,
        lda: i64,
        b: *const f64,
        ldb: i64,
        beta: f64,
        c: *mut f64,
        ldc: i64,
    );
    #[link_name = "cblas_dgemv$NEWLAPACK$ILP64"]
    fn np_cblas_dgemv(
        order: i32,
        trans: i32,
        m: i64,
        n: i64,
        alpha: f64,
        a: *const f64,
        lda: i64,
        x: *const f64,
        incx: i64,
        beta: f64,
        y: *mut f64,
        incy: i64,
    );
    #[link_name = "cblas_dsyrk$NEWLAPACK$ILP64"]
    fn np_cblas_dsyrk(
        order: i32,
        uplo: i32,
        trans: i32,
        n: i64,
        k: i64,
        alpha: f64,
        a: *const f64,
        lda: i64,
        beta: f64,
        c: *mut f64,
        ldc: i64,
    );
    #[link_name = "cblas_ddot$NEWLAPACK$ILP64"]
    fn np_cblas_ddot(n: i64, x: *const f64, incx: i64, y: *const f64, incy: i64) -> f64;
    #[link_name = "dgelsd$NEWLAPACK"]
    fn sp_dgelsd(
        m: *const i32,
        n: *const i32,
        nrhs: *const i32,
        a: *mut f64,
        lda: *const i32,
        b: *mut f64,
        ldb: *const i32,
        s: *mut f64,
        rcond: *const f64,
        rank: *mut i32,
        work: *mut f64,
        lwork: *const i32,
        iwork: *mut i32,
        info: *mut i32,
    );
    #[link_name = "dgesv$NEWLAPACK$ILP64"]
    fn np_dgesv(
        n: *const i64,
        nrhs: *const i64,
        a: *mut f64,
        lda: *const i64,
        ipiv: *mut i64,
        b: *mut f64,
        ldb: *const i64,
        info: *mut i64,
    );
    #[link_name = "dsyevr$NEWLAPACK"]
    fn sp_dsyevr(
        jobz: *const u8,
        range: *const u8,
        uplo: *const u8,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        vl: *const f64,
        vu: *const f64,
        il: *const i32,
        iu: *const i32,
        abstol: *const f64,
        m: *mut i32,
        w: *mut f64,
        z: *mut f64,
        ldz: *const i32,
        isuppz: *mut i32,
        work: *mut f64,
        lwork: *const i32,
        iwork: *mut i32,
        liwork: *const i32,
        info: *mut i32,
    );
    #[link_name = "dgesdd$NEWLAPACK"]
    fn sp_dgesdd(
        jobz: *const u8,
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
        iwork: *mut i32,
        info: *mut i32,
    );
}

/// `RLX_NUMPY_BLAS_TRACE=1`: one stderr line per call (kind, shape, strides) — to see
/// which kernels a pipeline reaches.
pub(crate) fn trace(f: impl FnOnce() -> String) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ON.get_or_init(|| std::env::var("RLX_NUMPY_BLAS_TRACE").is_ok_and(|v| v == "1")) {
        eprintln!("[numpy_blas] {}", f());
    }
}

/// `cblas_dgemm` with `alpha = 1, beta = 0` as NumPy calls it (`C = op(A)·op(B)`).
/// `c` must hold at least the `m×n` result at leading dimension `ldc`.
#[allow(clippy::too_many_arguments)]
pub fn dgemm(
    order: i32,
    transa: i32,
    transb: i32,
    m: usize,
    n: usize,
    k: usize,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    c: &mut [f64],
    ldc: usize,
) {
    trace(|| {
        format!(
            "dgemm order={order} ta={transa} tb={transb} m={m} n={n} k={k} lda={lda} ldb={ldb} ldc={ldc}"
        )
    });
    #[cfg(rlx_cpu_blas_accelerate)]
    unsafe {
        np_cblas_dgemm(
            order,
            transa,
            transb,
            m as i64,
            n as i64,
            k as i64,
            1.0,
            a.as_ptr(),
            lda as i64,
            b.as_ptr(),
            ldb as i64,
            0.0,
            c.as_mut_ptr(),
            ldc as i64,
        );
    }
    #[cfg(not(rlx_cpu_blas_accelerate))]
    crate::numpy_openblas::dgemm(order, transa, transb, m, n, k, a, lda, b, ldb, c, ldc);
}

/// `cblas_dgemv` with `alpha = 1, beta = 0` (`y = op(A)·x`).
#[allow(clippy::too_many_arguments)]
pub fn dgemv(
    order: i32,
    trans: i32,
    m: usize,
    n: usize,
    a: &[f64],
    lda: usize,
    x: &[f64],
    incx: usize,
    y: &mut [f64],
    incy: usize,
) {
    trace(|| format!("dgemv order={order} t={trans} m={m} n={n} lda={lda} incx={incx}"));
    #[cfg(rlx_cpu_blas_accelerate)]
    unsafe {
        np_cblas_dgemv(
            order,
            trans,
            m as i64,
            n as i64,
            1.0,
            a.as_ptr(),
            lda as i64,
            x.as_ptr(),
            incx as i64,
            0.0,
            y.as_mut_ptr(),
            incy as i64,
        );
    }
    #[cfg(not(rlx_cpu_blas_accelerate))]
    if !crate::numpy_openblas::dgemv(order, trans, m, n, a, lda, x, incx, y, incy) {
        // above OpenBLAS's threading threshold — NOT at parity
        let (rows, cols) = if trans == NO_TRANS { (m, n) } else { (n, m) };
        for i in 0..rows {
            let mut s = 0.0;
            for j in 0..cols {
                s += idx2(order, trans, a, lda, i, j) * x[j * incx];
            }
            y[i * incy] = s;
        }
    }
}

/// `cblas_dsyrk(order, Upper, trans, n, k, 1, A, lda, 0, C, ldc)` — **only the upper
/// triangle** of `C` is written; NumPy mirrors it afterwards.
#[allow(clippy::too_many_arguments)]
pub fn dsyrk_upper(
    order: i32,
    trans: i32,
    n: usize,
    k: usize,
    a: &[f64],
    lda: usize,
    c: &mut [f64],
    ldc: usize,
) {
    trace(|| format!("dsyrk order={order} t={trans} n={n} k={k} lda={lda}"));
    #[cfg(rlx_cpu_blas_accelerate)]
    unsafe {
        np_cblas_dsyrk(
            order,
            UPPER,
            trans,
            n as i64,
            k as i64,
            1.0,
            a.as_ptr(),
            lda as i64,
            0.0,
            c.as_mut_ptr(),
            ldc as i64,
        );
    }
    #[cfg(not(rlx_cpu_blas_accelerate))]
    crate::numpy_openblas::dsyrk_upper(order, trans, n, k, a, lda, c, ldc);
}

/// `cblas_ddot`.
pub fn ddot(n: usize, x: &[f64], incx: usize, y: &[f64], incy: usize) -> f64 {
    trace(|| format!("ddot n={n} incx={incx} incy={incy}"));
    #[cfg(rlx_cpu_blas_accelerate)]
    unsafe {
        np_cblas_ddot(n as i64, x.as_ptr(), incx as i64, y.as_ptr(), incy as i64)
    }
    #[cfg(not(rlx_cpu_blas_accelerate))]
    crate::numpy_openblas::ddot(n, x, incx, y, incy)
}

/// `scipy.linalg.lstsq(a, b)` (driver `gelsd`, `cond = None` → `eps`) as SciPy
/// calls it: `dgelsd$NEWLAPACK` (32-bit ints) on Fortran-order copies, `ldb =
/// max(1, m, n)`, workspace sizes from the `gelsd_lwork` query (`int(work[0])`,
/// `iwork[0]`). `a`: m×n row-major, `b`: m×nrhs row-major → `x`: n×nrhs row-major.
/// Returns LAPACK's `info`.
pub fn dgelsd_scipy(
    a: &[f64],
    b: &[f64],
    m: usize,
    n: usize,
    nrhs: usize,
    cond: f64,
    x: &mut [f64],
) -> i32 {
    trace(|| format!("dgelsd m={m} n={n} nrhs={nrhs}"));
    #[cfg(rlx_cpu_blas_accelerate)]
    {
        let ldb = m.max(n).max(1);
        let lda = m.max(1);
        let mut a_col = vec![0.0f64; lda * n];
        for i in 0..m {
            for j in 0..n {
                a_col[i + j * lda] = a[i * n + j];
            }
        }
        let mut b_col = vec![0.0f64; ldb * nrhs];
        for i in 0..m {
            for j in 0..nrhs {
                b_col[i + j * ldb] = b[i * nrhs + j];
            }
        }
        let (mm, nn, nr, lda_i, ldb_i) = (m as i32, n as i32, nrhs as i32, lda as i32, ldb as i32);
        let mut s = vec![0.0f64; m.min(n).max(1)];
        let mut rank = 0i32;
        let mut info = 0i32;
        let mut wq = [0.0f64; 1];
        let mut iwq = [0i32; 1];
        let m1 = -1i32;
        unsafe {
            sp_dgelsd(
                &mm,
                &nn,
                &nr,
                a_col.as_mut_ptr(),
                &lda_i,
                b_col.as_mut_ptr(),
                &ldb_i,
                s.as_mut_ptr(),
                &cond,
                &mut rank,
                wq.as_mut_ptr(),
                &m1,
                iwq.as_mut_ptr(),
                &mut info,
            );
        }
        if info != 0 {
            return info;
        }
        let lwork = wq[0] as i32;
        let mut work = vec![0.0f64; lwork.max(1) as usize];
        let mut iwork = vec![0i32; iwq[0].max(1) as usize];
        unsafe {
            sp_dgelsd(
                &mm,
                &nn,
                &nr,
                a_col.as_mut_ptr(),
                &lda_i,
                b_col.as_mut_ptr(),
                &ldb_i,
                s.as_mut_ptr(),
                &cond,
                &mut rank,
                work.as_mut_ptr(),
                &lwork,
                iwork.as_mut_ptr(),
                &mut info,
            );
        }
        if info != 0 {
            return info;
        }
        for i in 0..n {
            for j in 0..nrhs {
                x[i * nrhs + j] = b_col[i + j * ldb];
            }
        }
        0
    }
    #[cfg(not(rlx_cpu_blas_accelerate))]
    {
        crate::blas::dgelsd_solve(a, b, m, n, nrhs, cond, x)
    }
}

/// `np.linalg.solve(a, b)` for one right-hand side as NumPy calls it: `a` (n×n,
/// row-major) copied to Fortran order, `dgesv$NEWLAPACK$ILP64`. Returns `x` or LAPACK's
/// `info` (> 0: singular).
pub fn dgesv_numpy(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>, i64> {
    trace(|| format!("dgesv n={n}"));
    let mut a_col = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            a_col[i + j * n] = a[i * n + j];
        }
    }
    let mut x = b[..n].to_vec();
    #[cfg(rlx_cpu_blas_accelerate)]
    {
        let (nn, one, ld) = (n as i64, 1i64, n.max(1) as i64);
        let mut ipiv = vec![0i64; n.max(1)];
        let mut info = 0i64;
        unsafe {
            np_dgesv(
                &nn,
                &one,
                a_col.as_mut_ptr(),
                &ld,
                ipiv.as_mut_ptr(),
                x.as_mut_ptr(),
                &ld,
                &mut info,
            )
        };
        if info != 0 {
            return Err(info);
        }
        Ok(x)
    }
    #[cfg(not(rlx_cpu_blas_accelerate))]
    if let Some(info) = crate::numpy_openblas::dgesv(n, &mut a_col, &mut x) {
        // OpenBLAS 0.3.30 Haswell, the Linux reference (n ≤ 33)
        return if info != 0 { Err(info as i64) } else { Ok(x) };
    }
    #[cfg(not(rlx_cpu_blas_accelerate))]
    {
        // n > 33: partial-pivoting LU, the algorithm (not the rounding) of OpenBLAS's
        // blocked getrf — NOT at parity
        let mut piv: Vec<usize> = (0..n).collect();
        for k in 0..n {
            let p = (k..n)
                .max_by(|&i, &j| {
                    a_col[i + k * n]
                        .abs()
                        .partial_cmp(&a_col[j + k * n].abs())
                        .unwrap()
                })
                .unwrap();
            if a_col[p + k * n] == 0.0 {
                return Err(k as i64 + 1);
            }
            if p != k {
                for j in 0..n {
                    a_col.swap(k + j * n, p + j * n);
                }
                x.swap(k, p);
                piv.swap(k, p);
            }
            for i in k + 1..n {
                let l = a_col[i + k * n] / a_col[k + k * n];
                for j in k + 1..n {
                    a_col[i + j * n] -= l * a_col[k + j * n];
                }
                x[i] -= l * x[k];
            }
        }
        for k in (0..n).rev() {
            let mut s = x[k];
            for j in k + 1..n {
                s -= a_col[k + j * n] * x[j];
            }
            x[k] = s / a_col[k + k * n];
        }
        Ok(x)
    }
}

/// `scipy.linalg.svd(a, full_matrices=False)` (driver `gesdd`) as SciPy calls it:
/// `dgesdd$NEWLAPACK` (32-bit ints), `jobz = 'S'`, Fortran-order copy of `a`, `lwork`
/// from the workspace query. `a`: m×n row-major. Returns `(u, s, vt)` in **Fortran
/// order** — `u` m×k, `vt` k×n with `k = min(m, n)` — as SciPy hands them back, or
/// LAPACK's `info`. Accelerate builds only.
#[cfg(rlx_cpu_blas_accelerate)]
pub fn dgesdd_scipy(a: &[f64], m: usize, n: usize) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), i32> {
    trace(|| format!("dgesdd m={m} n={n}"));
    let k = m.min(n);
    let lda = m.max(1);
    let mut a_col = vec![0.0f64; lda * n];
    for i in 0..m {
        for j in 0..n {
            a_col[i + j * lda] = a[i * n + j];
        }
    }
    let (mm, nn, lda_i) = (m as i32, n as i32, lda as i32);
    let (ldu, ldvt) = (m.max(1) as i32, k.max(1) as i32);
    let mut s = vec![0.0f64; k];
    let mut u = vec![0.0f64; m * k];
    let mut vt = vec![0.0f64; k * n];
    let mut iwork = vec![0i32; 8 * k.max(1)];
    let mut info = 0i32;
    let jobz = b'S';
    let mut wq = [0.0f64; 1];
    let m1 = -1i32;
    unsafe {
        sp_dgesdd(
            &jobz,
            &mm,
            &nn,
            a_col.as_mut_ptr(),
            &lda_i,
            s.as_mut_ptr(),
            u.as_mut_ptr(),
            &ldu,
            vt.as_mut_ptr(),
            &ldvt,
            wq.as_mut_ptr(),
            &m1,
            iwork.as_mut_ptr(),
            &mut info,
        );
    }
    if info != 0 {
        return Err(info);
    }
    // SciPy's `_compute_lwork` → `_check_work_float`: `int(value)` (float64 query)
    let lwork = (wq[0] as i32).max(1);
    let mut work = vec![0.0f64; lwork as usize];
    unsafe {
        sp_dgesdd(
            &jobz,
            &mm,
            &nn,
            a_col.as_mut_ptr(),
            &lda_i,
            s.as_mut_ptr(),
            u.as_mut_ptr(),
            &ldu,
            vt.as_mut_ptr(),
            &ldvt,
            work.as_mut_ptr(),
            &lwork,
            iwork.as_mut_ptr(),
            &mut info,
        );
    }
    if info != 0 {
        return Err(info);
    }
    Ok((u, s, vt))
}

/// `scipy.linalg.eigh(a)` (default driver `evr`, `lower=True`): `dsyevr$NEWLAPACK`
/// (32-bit ints) on a Fortran copy, `jobz='V', range='A', uplo='L'`, `vl=0, vu=1,
/// il=1, iu=n, abstol=0`. The workspace sizes come from SciPy's `syevr_lwork`
/// query, which asks with `jobz='N'` (and `iu=2`) — reproduced, since `lwork`
/// can change LAPACK's blocking. `a`: n×n row-major symmetric. Returns ascending
/// eigenvalues and the eigenvectors **Fortran-ordered** (column `j` = vector `j`),
/// or LAPACK's `info`. Accelerate builds only.
#[cfg(rlx_cpu_blas_accelerate)]
pub fn dsyevr_scipy(a: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>), i32> {
    trace(|| format!("dsyevr n={n}"));
    let ld = n.max(1);
    let mut a_col = vec![0.0f64; ld * n];
    for i in 0..n {
        for j in 0..n {
            a_col[i + j * ld] = a[i * n + j];
        }
    }
    let (nn, ld_i) = (n as i32, ld as i32);
    let (vl, vu, abstol) = (0.0f64, 1.0f64, 0.0f64);
    let mut m = 0i32;
    let mut info = 0i32;
    // workspace query, as syevr_lwork makes it
    let (mut wq, mut iwq) = ([0.0f64; 1], [0i32; 1]);
    let (mut w1, mut z1, mut isz1) = ([0.0f64; 1], [0.0f64; 1], [0i32; 1]);
    let (il_q, iu_q, m1) = (1i32, 2i32, -1i32);
    unsafe {
        sp_dsyevr(
            b"N".as_ptr(),
            b"A".as_ptr(),
            b"L".as_ptr(),
            &nn,
            a_col.as_mut_ptr(),
            &ld_i,
            &vl,
            &vu,
            &il_q,
            &iu_q,
            &abstol,
            &mut m,
            w1.as_mut_ptr(),
            z1.as_mut_ptr(),
            &ld_i,
            isz1.as_mut_ptr(),
            wq.as_mut_ptr(),
            &m1,
            iwq.as_mut_ptr(),
            &m1,
            &mut info,
        );
    }
    if info != 0 {
        return Err(info);
    }
    let (lwork, liwork) = ((wq[0] as i32).max(1), iwq[0].max(1));
    let mut w = vec![0.0f64; n];
    let mut z = vec![0.0f64; ld * n];
    let mut isuppz = vec![0i32; 2 * ld];
    let mut work = vec![0.0f64; lwork as usize];
    let mut iwork = vec![0i32; liwork as usize];
    let (il, iu) = (1i32, nn);
    unsafe {
        sp_dsyevr(
            b"V".as_ptr(),
            b"A".as_ptr(),
            b"L".as_ptr(),
            &nn,
            a_col.as_mut_ptr(),
            &ld_i,
            &vl,
            &vu,
            &il,
            &iu,
            &abstol,
            &mut m,
            w.as_mut_ptr(),
            z.as_mut_ptr(),
            &ld_i,
            isuppz.as_mut_ptr(),
            work.as_mut_ptr(),
            &lwork,
            iwork.as_mut_ptr(),
            &liwork,
            &mut info,
        );
    }
    if info != 0 {
        return Err(info);
    }
    Ok((w, z))
}

/// [`dgesdd_scipy`] without Accelerate (Linux, wasm32): the same `(u, s, vt)`
/// layout from [`crate::blas::dgesdd_thin`] — the platform LAPACK's `dgesdd` when
/// one is linked, else rlx's pure-Rust SVD. **Not** bit-identical to SciPy on
/// macOS (a different `dgesdd`, or a different algorithm); signs of singular
/// vector pairs may differ too.
#[cfg(not(rlx_cpu_blas_accelerate))]
pub fn dgesdd_scipy(a: &[f64], m: usize, n: usize) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), i32> {
    trace(|| format!("dgesdd m={m} n={n}"));
    // the Linux reference's own dgesdd, bit for bit, where modelled
    let mut a_col = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            a_col[i + j * m] = a[i * n + j];
        }
    }
    if let Some(r) = rlx_linux_ref::lapack::dgesdd_s(m, n, &mut a_col, m.max(1)) {
        return r.map_err(|i| i as i32);
    }
    let k = m.min(n);
    let mut a2 = a.to_vec();
    let (mut s, mut u, mut vt) = (vec![0.0f64; k], vec![0.0f64; m * k], vec![0.0f64; k * n]);
    let info = crate::blas::dgesdd_thin(&mut a2, m, n, &mut s, &mut u, &mut vt);
    if info != 0 {
        return Err(info);
    }
    // row-major → the Fortran order the Accelerate path returns
    let mut u_f = vec![0.0f64; m * k];
    for i in 0..m {
        for j in 0..k {
            u_f[i + j * m] = u[i * k + j];
        }
    }
    let mut vt_f = vec![0.0f64; k * n];
    for i in 0..k {
        for j in 0..n {
            vt_f[i + j * k] = vt[i * n + j];
        }
    }
    Ok((u_f, s, vt_f))
}

/// [`dsyevr_scipy`] without Accelerate: SciPy's `eigh` as the Linux x86-64
/// reference computes it — reference LAPACK's `dsyevr` on OpenBLAS Haswell,
/// `lwork = 33n` — bit for bit ([`rlx_linux_ref::dsyevr_lower`]) up to
/// [`rlx_linux_ref::DSYEVR_MAX_N`]; larger matrices fall back to
/// [`crate::blas::dsyevd_uplo`] (not bit-identical; eigenvector signs may differ).
#[cfg(not(rlx_cpu_blas_accelerate))]
pub fn dsyevr_scipy(a: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>), i32> {
    trace(|| format!("dsyevr n={n}"));
    if n <= rlx_linux_ref::DSYEVR_MAX_N {
        // the Fortran copy SciPy hands LAPACK (lower triangle read)
        let mut a_col = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a_col[i + j * n] = a[i * n + j];
            }
        }
        return rlx_linux_ref::dsyevr_lower(n, &mut a_col, n.max(1), 33 * n as i64);
    }
    let mut z = a[..n * n].to_vec();
    let mut w = vec![0.0f64; n];
    let info = crate::blas::dsyevd_uplo(&mut z, &mut w, n, true);
    if info != 0 {
        return Err(info);
    }
    Ok((w, z))
}

#[cfg(not(rlx_cpu_blas_accelerate))]
fn idx2(order: i32, trans: i32, a: &[f64], ld: usize, i: usize, j: usize) -> f64 {
    // element (i, j) of op(A)
    let (r, c) = if trans == NO_TRANS { (i, j) } else { (j, i) };
    if order == ROW_MAJOR {
        a[r * ld + c]
    } else {
        a[r + c * ld]
    }
}
