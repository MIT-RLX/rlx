// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The BLAS kernels NumPy and SciPy run on the Linux reference — OpenBLAS 0.3.30's
//! **Haswell** kernels (the core it selects on an AVX2+FMA Intel, e.g. msi's
//! i9-12900H), as `scipy_openblas` ships them in the numpy/scipy wheels — in pure
//! Rust, operation for operation: the inline-asm micro-kernels' lane layout and
//! reduction trees, the C tails as gcc builds them with `-mfma`
//! (`-ffp-contract=fast`: a product becomes an FMA when all its uses are adds in its
//! block), and the level-1 thread split for long vectors.
//!
//! Translated from OpenBLAS (BSD-3-Clause, `kernel/x86_64/ddot.c`,
//! `ddot_microk_haswell-2.c`, `driver/others/blas_l1_thread.c`).

/// OpenBLAS's thread count on the reference machine (`num_cpu_avail` = 20 cores on
/// msi): it decides how a level-1 call above its threshold is split, so the bits.
pub const REFERENCE_THREADS: usize = 20;

/// `ddot_kernel_8`: four 4-lane AVX2 accumulators over `n` (a multiple of 16)
/// elements, `vfmadd231pd`, then `(hi + lo)` per accumulator, `(a0+a1)+(a2+a3)` and
/// a horizontal add.
fn ddot_kernel_8(x: &[f64], y: &[f64], n: usize) -> f64 {
    let mut acc = [[0.0f64; 4]; 4];
    let mut i = 0;
    while i < n {
        for (a, lanes) in acc.iter_mut().enumerate() {
            for (l, v) in lanes.iter_mut().enumerate() {
                let k = i + 4 * a + l;
                *v = x[k].mul_add(y[k], *v);
            }
        }
        i += 16;
    }
    let h: Vec<[f64; 2]> = acc.iter().map(|a| [a[2] + a[0], a[3] + a[1]]).collect();
    let s0 = [h[0][0] + h[1][0], h[0][1] + h[1][1]];
    let s2 = [h[2][0] + h[3][0], h[2][1] + h[3][1]];
    let t = [s0[0] + s2[0], s0[1] + s2[1]];
    t[0] + t[1]
}

/// `dot_compute` (single thread).
fn dot_compute(n: usize, x: &[f64], incx: usize, y: &[f64], incy: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    if incx == 1 && incy == 1 {
        let n1 = n & !15;
        let mut dot = if n1 > 0 { ddot_kernel_8(x, y, n1) } else { 0.0 };
        for i in n1..n {
            dot += y[i] * x[i];
        }
        return dot;
    }
    let (mut t1, mut t2) = (0.0f64, 0.0f64);
    let n1 = n & !3;
    let (mut ix, mut iy, mut i) = (0, 0, 0);
    while i < n1 {
        let m1 = y[iy] * x[ix];
        let m2 = y[iy + incy] * x[ix + incx];
        let m3 = y[iy + 2 * incy] * x[ix + 2 * incx];
        let m4 = y[iy + 3 * incy] * x[ix + 3 * incx];
        t1 += m1 + m3;
        t2 += m2 + m4;
        ix += 4 * incx;
        iy += 4 * incy;
        i += 4;
    }
    while i < n {
        t1 += y[iy] * x[ix];
        ix += incx;
        iy += incy;
        i += 1;
    }
    t1 + t2
}

/// OpenBLAS `ddot` (Haswell) as called with `threads` worker threads: single
/// threaded up to n = 10000, else split like `blas_level1_thread_with_return_value`
/// (chunk `ceil(left / threads_left)`), the partial dots summed from `0.0` in order.
pub fn ddot_threads(
    n: usize,
    x: &[f64],
    incx: usize,
    y: &[f64],
    incy: usize,
    threads: usize,
) -> f64 {
    if incx == 0 || incy == 0 || n <= 10000 || threads <= 1 {
        return dot_compute(n, x, incx, y, incy);
    }
    let mut dot = 0.0f64;
    let (mut left, mut cpu, mut off) = (n, 0usize, 0usize);
    while left > 0 {
        let width = (left + threads - cpu - 1) / (threads - cpu);
        let width = width.min(left);
        dot += dot_compute(width, &x[off * incx..], incx, &y[off * incy..], incy);
        left -= width;
        off += width;
        cpu += 1;
    }
    dot
}

/// OpenBLAS `ddot` on the reference machine ([`REFERENCE_THREADS`]).
pub fn ddot(n: usize, x: &[f64], incx: usize, y: &[f64], incy: usize) -> f64 {
    ddot_threads(n, x, incx, y, incy, REFERENCE_THREADS)
}

/// `DGEMM_DEFAULT_Q` for Haswell: the level-3 drivers' `k` block.
const DGEMM_Q: usize = 256;

/// `DGEMM_DEFAULT_UNROLL_M` for Haswell.
const DGEMM_UNROLL_M: usize = 4;

/// The level-3 drivers' split of `k`: blocks of `GEMM_Q`, except that a remainder in
/// `(Q, 2Q)` is halved — `(min_l + 1) / 2` in `level3_syrk.c`, rounded up to
/// `GEMM_UNROLL_M` in `level3.c` (`gemm`).
fn k_blocks(k: usize, gemm: bool) -> Vec<(usize, usize)> {
    let mut out = vec![];
    let mut ls = 0;
    while ls < k {
        let mut min_l = k - ls;
        if min_l >= 2 * DGEMM_Q {
            min_l = DGEMM_Q;
        } else if min_l > DGEMM_Q {
            min_l = if gemm {
                (min_l / 2).div_ceil(DGEMM_UNROLL_M) * DGEMM_UNROLL_M
            } else {
                (min_l + 1) / 2
            };
        }
        out.push((ls, min_l));
        ls += min_l;
    }
    out
}

/// `cblas_dsyrk(order, Upper, trans, n, k, 1, A, lda, 0, C, ldc)` as OpenBLAS 0.3.30
/// Haswell computes it: `C` cleared by the beta kernel, then per `k` block the
/// `dgemm_kernel_4x8_haswell` micro-kernel's one-register-per-element
/// `vmulpd`/`vfmadd231pd` chain, scaled by `alpha = 1` and added to `C`
/// (`syrk_kernel.c` adds a diagonal block's triangle the same way). Only the upper
/// triangle is written, as NumPy expects.
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
    let row_major = order == crate::ROW_MAJOR;
    let no_trans = trans == crate::NO_TRANS;
    // element (i, l) of op(A), n×k
    let at = |i: usize, l: usize| {
        let (r, cc) = if no_trans { (i, l) } else { (l, i) };
        if row_major {
            a[r * lda + cc]
        } else {
            a[r + cc * lda]
        }
    };
    let blocks = k_blocks(k, false);
    for i in 0..n {
        for j in i..n {
            let mut s = 0.0f64;
            for &(ls, len) in &blocks {
                let mut acc = at(i, ls) * at(j, ls);
                for l in ls + 1..ls + len {
                    acc = at(i, l).mul_add(at(j, l), acc);
                }
                s += acc;
            }
            let o = if row_major { i * ldc + j } else { i + j * ldc };
            c[o] = s;
        }
    }
}

// ── level 1/2 kernels behind `dgesv` ─────────────────────────────────────────

/// `daxpy_k` (Haswell, unit strides): `y += da·x`, the first `n & !15` elements in
/// the `vfmadd231pd` micro-kernel, the tail as gcc builds the C loop (unfused).
fn daxpy(n: usize, da: f64, x: &[f64], y: &mut [f64]) {
    let n1 = n & !15;
    for i in 0..n1 {
        y[i] = da.mul_add(x[i], y[i]);
    }
    for i in n1..n {
        y[i] += da * x[i];
    }
}

/// `dgemv_n` (Haswell `dgemv_n_4.c`, unit strides): `y += alpha·A·x`, `A` `m×n`
/// column-major at `lda`. Rows `0..m & !3` go through the column-group kernels —
/// `4x4`: `fma(a2,x2,a0·x0) + fma(a3,x3,a1·x1)` then `fma(alpha, ·, y)`; `4x2`:
/// `fma(alpha, a0·x0 + a1·x1, y)`; `4x1` (SSE2): `y + a·(x·alpha)` — and the last
/// `m & 3` rows accumulate a plain dot over all columns, `y += alpha·temp`.
fn dgemv_n(
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    x: &[f64],
    y: &mut [f64],
    unit_x: bool,
) {
    if m == 0 || n == 0 {
        return;
    }
    let m1 = m & !3;
    let (n1, n2) = (n >> 2, n & 3);
    for r in 0..m1 {
        let mut yr = y[r];
        let mut c = 0;
        for _ in 0..n1 {
            let at = |k: usize| a[r + (c + k) * lda];
            let t4 = at(2).mul_add(x[c + 2], at(0) * x[c]);
            let t5 = at(3).mul_add(x[c + 3], at(1) * x[c + 1]);
            yr = alpha.mul_add(t4 + t5, yr);
            c += 4;
        }
        if !unit_x {
            // a strided `x` is copied four at a time; the rest go through `4x1`
            for cc in c..n {
                yr += a[r + cc * lda] * (x[cc] * alpha);
            }
            y[r] = yr;
            continue;
        }
        if n2 & 2 != 0 {
            let t = a[r + c * lda] * x[c] + a[r + (c + 1) * lda] * x[c + 1];
            yr = alpha.mul_add(t, yr);
            c += 2;
        }
        if n2 & 1 != 0 {
            yr += a[r + c * lda] * (x[c] * alpha);
        }
        y[r] = yr;
    }
    let m3 = m & 3;
    if m3 == 0 {
        return;
    }
    let mut temp = [0.0f64; 3];
    if lda == m3 && unit_x {
        // the `lda == m3 && inc_x == 1` loops: column pairs summed first
        let n4 = n & !3;
        let mut i = 0;
        while i < n4 {
            if m3 == 1 {
                temp[0] +=
                    a[i] * x[i] + a[i + 1] * x[i + 1] + a[i + 2] * x[i + 2] + a[i + 3] * x[i + 3];
            } else {
                for half in [0, 2] {
                    let (c0, c1) = (i + half, i + half + 1);
                    for (k, t) in temp.iter_mut().enumerate().take(m3) {
                        *t += a[k + c0 * lda] * x[c0] + a[k + c1 * lda] * x[c1];
                    }
                }
            }
            i += 4;
        }
        for c in n4..n {
            for (k, t) in temp.iter_mut().enumerate().take(m3) {
                *t += a[k + c * lda] * x[c];
            }
        }
    } else {
        for c in 0..n {
            for (k, t) in temp.iter_mut().enumerate().take(m3) {
                *t += a[m1 + k + c * lda] * x[c];
            }
        }
    }
    for (k, t) in temp.iter().enumerate().take(m3) {
        y[m1 + k] += alpha * t;
    }
}

/// `idamax_k`: the first index of the largest `|x|` (0-based).
fn idamax(x: &[f64]) -> usize {
    let mut best = 0;
    for (i, v) in x.iter().enumerate() {
        if v.abs() > x[best].abs() {
            best = i;
        }
    }
    best
}

/// `getf2_k` (`lapack/getf2/getf2_k.c`): OpenBLAS's left-looking LU of an `m×n`
/// column-major `a`, the one `getrf_single` runs when `min(m,n) ≤ 33` (its blocking
/// `≤ 2·GEMM_UNROLL_N`). `ipiv` is 1-based; returns LAPACK's `info`.
fn getf2(m: usize, n: usize, a: &mut [f64], lda: usize, ipiv: &mut [usize]) -> usize {
    let mut info = 0;
    for j in 0..n {
        let len = j.min(m);
        let col = j * lda;
        for i in 0..len {
            let ip = ipiv[i] - 1;
            if ip != i {
                a.swap(col + i, col + ip);
            }
        }
        for i in 1..len {
            // b[i] -= DOTU_K(i, a + i, lda, b, 1)
            let d = dot_compute(i, &a[i..], lda, &a[col..col + i], 1);
            a[col + i] -= d;
        }
        if j < m {
            let (left, right) = a.split_at_mut(col);
            // GEMV_N(m - j, j, 0, -1, a + j, lda, b, 1, b + j, 1)
            if j > 0 {
                let (bx, by) = right.split_at_mut(j);
                dgemv_n(m - j, j, -1.0, &left[j..], lda, bx, &mut by[..m - j], true);
            }
            let jp = j + idamax(&a[col + j..col + m]);
            ipiv[j] = jp + 1;
            let temp1 = a[col + jp];
            if temp1 != 0.0 {
                if temp1.abs() >= f64::MIN_POSITIVE {
                    let r = 1.0 / temp1;
                    if jp != j {
                        for c in 0..=j {
                            a.swap(j + c * lda, jp + c * lda);
                        }
                    }
                    for v in &mut a[col + j + 1..col + m] {
                        *v *= r;
                    }
                }
            } else if info == 0 {
                info = j + 1;
            }
        }
    }
    info
}

/// `dgesv_` (`interface/lapack/gesv.c`) for one right-hand side as OpenBLAS 0.3.30
/// Haswell computes it single-threaded (`n·nrhs < 10000`): `getf2`, `laswp`, then
/// `dtrsv_NLU` / `dtrsv_NUN` (`driver/level2/trsv_{L,U}.c`, one `DTB_ENTRIES = 64`
/// block: `daxpy` column sweeps, `B[i] /= A[i,i]`). `a` is `n×n` column-major (it is
/// overwritten by the factors), `b` the right-hand side → the solution. Returns
/// `Some(info)`, or `None` when `n > 33`, where OpenBLAS runs its blocked
/// `getrf_single` (not modelled).
pub fn dgesv(n: usize, a: &mut [f64], b: &mut [f64]) -> Option<usize> {
    if n > 33 {
        return None;
    }
    if n == 0 {
        return Some(0);
    }
    let lda = n;
    let mut ipiv = vec![0usize; n];
    let info = getf2(n, n, a, lda, &mut ipiv);
    if info != 0 {
        return Some(info);
    }
    for (i, &p) in ipiv.iter().enumerate() {
        if p - 1 != i {
            b.swap(i, p - 1);
        }
    }
    // dtrsv_NLU (unit lower)
    for i in 0..n {
        if i + 1 < n {
            let bi = b[i];
            let (_, rest) = b.split_at_mut(i + 1);
            daxpy(n - i - 1, -bi, &a[i + 1 + i * lda..(i + 1) * lda], rest);
        }
    }
    // dtrsv_NUN (upper)
    for i in 0..n {
        let k = n - 1 - i;
        b[k] /= a[k + k * lda];
        if i + 1 < n {
            let bk = b[k];
            daxpy(k, -bk, &a[k * lda..k * lda + k], &mut b[..k]);
        }
    }
    Some(0)
}

/// How OpenBLAS threads a column-major `m×n×k` product: the row ranges of the
/// `nthreads_m` thread rows (4-row kernel tiles start at each range's first row) and
/// the column sub-parts (`(start, width)`, one kernel call each), or `None` when it
/// runs single-threaded (`level3.c`). `interface/gemm.c` picks `⌊MNK/262144⌋` threads
/// (at most [`REFERENCE_THREADS`]); `level3_thread.c` splits them
/// `nthreads_m × nthreads_n` (`switch_ratio = 8`: `DYNAMIC_ARCH` reads it from
/// `gotoblas`, whose `setparam-ref.c` is compiled without `DOUBLE` — the single-precision
/// value, not param.h's double 4; measured on tall-thin products), minimising `n·tm + m·tn` and each
/// column part into `DIVIDE_RATE = 2` halves. The `DYNAMIC_ARCH` build compiles the
/// threaded driver without the Haswell `GEMM_PREFERED_SIZE`, so ranges are unrounded
/// (measured: a 100³ product splits its columns 34/33/33).
pub struct GemmThreads {
    pub rows: Vec<(usize, usize)>,
    pub cols: Vec<(usize, usize)>,
}

pub fn gemm_threads(m: usize, n: usize, k: usize) -> Option<GemmThreads> {
    const SWITCH: usize = 8;
    let mnk = m as f64 * n as f64 * k as f64;
    let thr = 65536.0 * 4.0;
    let nthreads = if mnk <= thr {
        1
    } else if mnk / (REFERENCE_THREADS as f64) < thr {
        (mnk / thr) as usize
    } else {
        REFERENCE_THREADS
    };
    if nthreads <= 1 {
        return None;
    }
    let mut tm = if m < 2 * SWITCH {
        1
    } else {
        let mut t = nthreads;
        while m < t * SWITCH {
            t /= 2;
        }
        t
    };
    let tn = if n < SWITCH * tm {
        1
    } else {
        let mut t = n.div_ceil(SWITCH * tm);
        if tm * t > nthreads {
            t = nthreads / tm;
        }
        let (mut cost, mut div) = (0usize, 0usize);
        let mut i = 1usize;
        while (i as f64) <= (tm as f64).sqrt() {
            if tm % i == 0 {
                let j = tm / i;
                let cost_i = n * j + m * t * i;
                let cost_j = n * i + m * t * j;
                if cost == 0 || cost_i < cost {
                    cost = cost_i;
                    div = i;
                }
                if cost_j < cost {
                    cost = cost_j;
                    div = j;
                }
            }
            i += 1;
        }
        if div > 1 {
            tm /= div;
            t *= div;
        }
        t
    };
    if tm * tn <= 1 {
        return None;
    }
    let mut rows = vec![];
    let (mut start, mut left, mut parts) = (0usize, m, 0usize);
    while left > 0 {
        let w = (left + tm - parts - 1) / (tm - parts);
        let w = w.min(left);
        rows.push((start, w));
        start += w;
        left -= w;
        parts += 1;
    }
    let mut cols = vec![];
    let (mut start, mut left) = (0usize, n);
    for j in 0..tn {
        let mut wn = (left + tn - j - 1) / (tn - j);
        left -= wn;
        for i in 0..tm {
            let w = ((wn + tm - i - 1) / (tm - i)).max(SWITCH).min(wn);
            wn -= w;
            if w > 0 {
                let div_n = w.div_ceil(2);
                let mut s = start;
                while s < start + w {
                    cols.push((s, div_n.min(start + w - s)));
                    s += div_n;
                }
            }
            start += w;
        }
    }
    Some(GemmThreads { rows, cols })
}

/// `cblas_dgemm(order, ta, tb, m, n, k, 1, A, lda, B, ldb, 0, C, ldc)` as OpenBLAS
/// 0.3.30 Haswell computes it on the reference's 20 threads (Haswell's small-matrix
/// permit is `0`, so every size takes `level3.c` / `level3_thread.c`): per `k` block
/// one `vmulpd`/`vfmadd231pd` chain per element — four rotating accumulators on a
/// kernel call's odd last column — added to the cleared `C`.
#[allow(clippy::too_many_arguments)]
pub fn dgemm(
    order: i32,
    ta: i32,
    tb: i32,
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
    dgemm_ab(order, ta, tb, m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc);
}

/// [`dgemm`] with general `alpha`, `beta`: `C` first scaled by the beta kernel
/// (`0` clears, `1` leaves it), then per `k` block `C + alpha·acc`.
#[allow(clippy::too_many_arguments)]
pub fn dgemm_ab(
    order: i32,
    ta: i32,
    tb: i32,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    beta: f64,
    c: &mut [f64],
    ldc: usize,
) {
    let row_major = order == crate::ROW_MAJOR;
    let el = |x: &[f64], ld: usize, trans: bool, i: usize, j: usize| {
        let (r, cc) = if trans { (j, i) } else { (i, j) };
        if row_major {
            x[r * ld + cc]
        } else {
            x[r + cc * ld]
        }
    };
    let (ta, tb) = (ta != crate::NO_TRANS, tb != crate::NO_TRANS);
    // the kernel works on column-major C: a row-major call computes Cᵀ = Bᵀ·Aᵀ
    let (mi, ni) = if row_major { (n, m) } else { (m, n) };
    let plan = gemm_threads(mi, ni, k);
    // threaded (`level3_thread.c`) blocks k as `level3_syrk.c` does
    let blocks = k_blocks(k, plan.is_none());
    // each kernel call's column range — one call per sub-part (threaded), or the whole
    // width (single thread: its 24/8-wide chunks are even) — and row ranges
    let (row_parts, col_parts) = match plan {
        Some(p) => (p.rows, p.cols),
        None => (vec![(0, mi)], vec![(0, ni)]),
    };
    let last_of_odd = |ci: usize| {
        col_parts
            .iter()
            .any(|&(s, w)| w % 2 == 1 && ci == s + w - 1)
    };
    let in_4row_tile = |ri: usize| row_parts.iter().any(|&(s, w)| ri >= s && ri - s < (w & !3));
    for i in 0..m {
        for j in 0..n {
            let (ri, ci) = if row_major { (j, i) } else { (i, j) };
            // the single-column tile (a call's last column when its width is odd) on
            // 4-row tiles: `KERNEL4x1` rotates k over four accumulators, `SAVE4x1` sums them
            let rotated = last_of_odd(ci) && in_4row_tile(ri);
            let prod = |l: usize, acc: f64| el(a, lda, ta, i, l).mul_add(el(b, ldb, tb, l, j), acc);
            let o = if row_major { i * ldc + j } else { i + j * ldc };
            let mut s = if beta == 0.0 {
                0.0
            } else if beta == 1.0 {
                c[o]
            } else {
                c[o] * beta
            };
            if alpha == 0.0 {
                c[o] = s;
                continue;
            }
            for &(ls, len) in &blocks {
                let acc = if rotated {
                    let mut r = [0.0f64; 4];
                    let full = len / 8 * 8;
                    for l in 0..full {
                        r[l % 4] = prod(ls + l, r[l % 4]);
                    }
                    for l in full..len {
                        r[0] = prod(ls + l, r[0]);
                    }
                    (r[0] + r[1]) + (r[2] + r[3])
                } else {
                    let mut acc = el(a, lda, ta, i, ls) * el(b, ldb, tb, ls, j);
                    for l in ls + 1..ls + len {
                        acc = prod(l, acc);
                    }
                    acc
                };
                s += alpha * acc;
            }
            c[o] = s;
        }
    }
}

/// `dgemv_t` (Haswell `dgemv_t_4.c`): `y[j] += alpha·Σᵢ A[i,j]·x[i]`, `A` `m×n`
/// column-major at `lda`, `x` unit stride. Rows `0..m & !3` in `NBMAX = 2048` blocks,
/// per block and column: the `4x4` micro-kernel (one 4-lane `vfmadd231pd` accumulator
/// by row mod 4, reduced `(l0+l2) + (l1+l3)`, added by SSE2 `add_y`: `y + t·alpha`),
/// the `4x2` SSE2 kernel (even/odd rows, plain), the `4x1` SSE2 kernel (rows mod 4 in
/// two 2-lane plain accumulators); the C statements as gcc contracts them
/// (`y = fma(t, alpha, y)`, the last `m & 3` rows `y + fma(a2,x2, fma(a0,x0, a1·x1))`).
fn dgemv_t(
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    x: &[f64],
    y: &mut [f64],
    incy: usize,
) {
    const NBMAX: usize = 2048;
    if m == 0 || n == 0 {
        return;
    }
    let m3 = m & 3;
    let rows = m - m3;
    let mut r0 = 0;
    while r0 < rows {
        let nb = (rows - r0).min(NBMAX);
        let col = |j: usize| &a[r0 + j * lda..r0 + j * lda + nb];
        let xs = &x[r0..r0 + nb];
        let n4 = n & !3;
        for j in 0..n4 {
            let c = col(j);
            let mut l = [0.0f64; 4];
            for i in 0..nb {
                l[i % 4] = c[i].mul_add(xs[i], l[i % 4]);
            }
            let t = (l[0] + l[2]) + (l[1] + l[3]);
            let yj = &mut y[j * incy];
            *yj = if incy == 1 {
                *yj + t * alpha
            } else {
                t.mul_add(alpha, *yj)
            };
        }
        let mut j = n4;
        if n & 2 != 0 {
            for jj in [j, j + 1] {
                let c = col(jj);
                let (mut e, mut o) = (0.0f64, 0.0f64);
                for i in (0..nb).step_by(2) {
                    e += c[i] * xs[i];
                    o += c[i + 1] * xs[i + 1];
                }
                let t = e + o;
                y[jj * incy] = t.mul_add(alpha, y[jj * incy]);
            }
            j += 2;
        }
        if n & 1 != 0 {
            let c = col(j);
            let (mut p, mut q) = ([0.0f64; 2], [0.0f64; 2]);
            for i in (0..nb).step_by(4) {
                p[0] += c[i] * xs[i];
                p[1] += c[i + 1] * xs[i + 1];
                q[0] += c[i + 2] * xs[i + 2];
                q[1] += c[i + 3] * xs[i + 3];
            }
            let t = (p[0] + q[0]) + (p[1] + q[1]);
            y[j * incy] = t.mul_add(alpha, y[j * incy]);
        }
        r0 += nb;
    }
    if m3 == 0 {
        return;
    }
    let xt: Vec<f64> = (0..m3).map(|k| x[rows + k] * alpha).collect();
    for j in 0..n {
        let aj = &a[rows + j * lda..];
        let yj = &mut y[j * incy];
        *yj = match m3 {
            3 => *yj + aj[2].mul_add(xt[2], aj[0].mul_add(xt[0], aj[1] * xt[1])),
            2 => *yj + aj[0].mul_add(xt[0], aj[1] * xt[1]),
            _ => aj[0].mul_add(xt[0], *yj),
        };
    }
}

/// `cblas_dgemv(order, trans, m, n, 1, A, lda, x, incx, 0, y, incy)` as OpenBLAS
/// 0.3.30 Haswell computes it: `y` cleared, then the column-major kernel (a row-major
/// call is the transposed column-major one), threaded as [`dgemv_ab`] describes.
/// Always `true` (kept for callers that fell back when threading was not modelled).
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
) -> bool {
    dgemv_ab(order, trans, m, n, 1.0, a, lda, x, incx, 0.0, y, incy)
}

/// `gemv_thread.c`'s split of `len` over `nthreads`: widths
/// `⌈left / threads left⌉`, at least 4.
fn gemv_ranges(len: usize, nthreads: usize) -> Vec<(usize, usize)> {
    let (mut out, mut i, mut cpu, mut start) = (vec![], len, 0usize, 0usize);
    while i > 0 {
        let mut width = (i + nthreads - cpu - 1) / (nthreads - cpu);
        if width < 4 {
            width = 4;
        }
        if i < width {
            width = i;
        }
        out.push((start, width));
        start += width;
        i -= width;
        cpu += 1;
    }
    out
}

/// [`dgemv`] with general `alpha`, `beta` (`y` scaled by `dscal` first unless
/// `beta == 1`). From `m·n ≥ 115200·4` OpenBLAS runs [`REFERENCE_THREADS`] threads
/// (`gemv_thread.c`): the transposed kernel splits the columns (each thread a
/// `dgemv_t` on its own column range); the plain one splits the rows — unless that
/// leaves threads idle and `m·n > 9216` with `m·threads ≤ 1024`, when it splits
/// `x` instead, each thread's `dgemv_n` into a zeroed buffer, added to `y` thread
/// by thread.
#[allow(clippy::too_many_arguments)]
pub fn dgemv_ab(
    order: i32,
    trans: i32,
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
) -> bool {
    let col_trans = (trans != crate::NO_TRANS) != (order == crate::ROW_MAJOR);
    // column-major dims
    let (cm, cn) = if order == crate::ROW_MAJOR {
        (n, m)
    } else {
        (m, n)
    };
    let (lenx, leny) = if col_trans { (cm, cn) } else { (cn, cm) };
    // `interface/gemv.c` returns on an empty matrix before scaling `y`
    if m == 0 || n == 0 {
        return true;
    }
    if beta != 1.0 {
        dscal(leny, beta, y, incy);
    }
    if alpha == 0.0 {
        return true;
    }
    let xs: Vec<f64> = (0..lenx).map(|i| x[i * incx]).collect();
    let threads = if m * n < 115_200 * 4 {
        1
    } else {
        REFERENCE_THREADS
    };
    if col_trans {
        for (s, w) in gemv_ranges(cn, threads) {
            dgemv_t(
                cm,
                w,
                alpha,
                &a[s * lda..],
                lda,
                &xs,
                &mut y[s * incy..],
                incy,
            );
        }
        return true;
    }
    let rows = gemv_ranges(cm, threads);
    if rows.len() < threads && (cm as f64) * (cn as f64) > 9216.0 && cm * threads <= 1024 {
        let mut parts = vec![];
        for (s, w) in gemv_ranges(cn, threads) {
            let mut yd = vec![0.0f64; cm];
            dgemv_n(
                cm,
                w,
                alpha,
                &a[s * lda..],
                lda,
                &xs[s..],
                &mut yd,
                incx == 1,
            );
            parts.push(yd);
        }
        for yd in parts {
            for j in 0..cm {
                y[j * incy] += yd[j];
            }
        }
        return true;
    }
    for (s, w) in rows {
        if incy == 1 {
            dgemv_n(w, cn, alpha, &a[s..], lda, &xs, &mut y[s..s + w], incx == 1);
        } else {
            let mut yb = vec![0.0; w];
            dgemv_n(w, cn, alpha, &a[s..], lda, &xs, &mut yb, incx == 1);
            for i in 0..w {
                y[(s + i) * incy] += yb[i];
            }
        }
    }
    true
}

// ── more level-1 kernels (the BLAS reference LAPACK calls) ───────────────────

/// `dnrm2` (OpenBLAS x86-64 `nrm2.S`, every core): the sum of squares in **x87
/// extended precision** — four accumulators by element index mod 4 over the first
/// `n & !7` elements, the rest into the first, combined `s3 + ((s2 + s0) + s1)` —
/// then `fsqrt` and one store to double. `n == 1` is `|x[0]|` (the interface).
pub fn dnrm2(n: usize, x: &[f64], incx: usize) -> f64 {
    use crate::x87::F80;
    if n == 0 || incx == 0 {
        return 0.0;
    }
    if n == 1 {
        return x[0].abs();
    }
    let mut acc = [F80::ZERO; 4];
    let n8 = n & !7;
    for i in 0..n {
        let v = F80::from_f64(x[i * incx]);
        let k = if i < n8 { i % 4 } else { 0 };
        acc[k] = acc[k].add(v.mul(v));
    }
    let r = acc[2].add(acc[0]).add(acc[1]);
    acc[3].add(r).sqrt().to_f64()
}

/// `drot` (Haswell `drot.c` + `drot_microk_haswell-2.c`, gcc-contracted tails):
/// `x' = fma(c, x, s·y)`, `y' = fma(c, y, -(s·x))` for every element.
pub fn drot(n: usize, x: &mut [f64], incx: usize, y: &mut [f64], incy: usize, c: f64, s: f64) {
    for i in 0..n {
        let (xi, yi) = (x[i * incx], y[i * incy]);
        x[i * incx] = c.mul_add(xi, s * yi);
        y[i * incy] = c.mul_add(yi, -(s * xi));
    }
}

/// `dscal` (`x *= alpha`, one rounding each; `alpha == 0` clears).
pub fn dscal(n: usize, alpha: f64, x: &mut [f64], incx: usize) {
    for i in 0..n {
        x[i * incx] = if alpha == 0.0 {
            0.0
        } else {
            alpha * x[i * incx]
        };
    }
}

/// `dger` (`../generic/ger.c` on Haswell): per column `j`, `daxpy(m, alpha·y[j], x,
/// A[:, j])` with the Haswell `daxpy`. `A` column-major at `lda`.
#[allow(clippy::too_many_arguments)]
pub fn dger(
    m: usize,
    n: usize,
    alpha: f64,
    x: &[f64],
    incx: usize,
    y: &[f64],
    incy: usize,
    a: &mut [f64],
    lda: usize,
) {
    if m == 0 || n == 0 || alpha == 0.0 {
        return;
    }
    let xs: Vec<f64> = (0..m).map(|i| x[i * incx]).collect();
    for j in 0..n {
        daxpy(m, alpha * y[j * incy], &xs, &mut a[j * lda..j * lda + m]);
    }
}

/// `dtrmm(side, uplo, transa, diag, m, n, alpha, A, B)` (column-major, `B` `m×n`
/// overwritten) as OpenBLAS 0.3.30 Haswell computes it for triangles up to `GEMM_Q`:
/// each element one chain over `k` ascending (zero outside the triangle; a unit
/// diagonal contributes `1·b`) over `alpha·B` (the driver pre-scales `B`).
/// `dtrmm_kernel_4x8_haswell.c` runs full
/// 4×8 tiles in its `vfmadd231pd` asm (fused) and the remainder tiles in C loops gcc
/// left unfused (measured: `mulsd`/`addsd`, no FMA): fused iff the element is in
/// a full 4×8 tile of its thread's range.
#[allow(clippy::too_many_arguments)]
pub fn dtrmm(
    left: bool,
    lower: bool,
    trans: bool,
    unit: bool,
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    b: &mut [f64],
    ldb: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    let k = if left { m } else { n };
    assert!(
        k <= DGEMM_Q,
        "dtrmm with a triangle above GEMM_Q is not modelled"
    );
    // op(A)[r, c]
    let op = |r: usize, c: usize| -> f64 {
        let (rr, cc) = if trans { (c, r) } else { (r, c) };
        let inside = if lower { rr >= cc } else { rr <= cc };
        if !inside {
            0.0
        } else if rr == cc && unit {
            1.0
        } else {
            a[rr + cc * lda]
        }
    };
    // the driver scales B by alpha first (`GEMM_BETA`), then runs the kernel with 1
    let src: Vec<f64> = if alpha == 1.0 {
        b.to_vec()
    } else {
        b.iter()
            .map(|v| if alpha == 0.0 { 0.0 } else { v * alpha })
            .collect()
    };
    let bb = |r: usize, c: usize| src[r + c * ldb];
    // `interface/trsm.c` (TRMM): 20 threads once m·n ≥ 256·4, splitting the non-triangular
    // dimension (`gemm_thread_n` on the left, `gemm_thread_m` on the right); each
    // thread's kernel tiles start at its own range
    let split = |len: usize| -> Vec<(usize, usize)> {
        if m * n < 256 * 4 {
            return vec![(0, len)];
        }
        let (mut out, mut left, mut cpu, mut start) = (vec![], len, 0usize, 0usize);
        while left > 0 {
            let w = (left + REFERENCE_THREADS - cpu - 1) / (REFERENCE_THREADS - cpu);
            let w = w.min(left);
            out.push((start, w));
            start += w;
            left -= w;
            cpu += 1;
        }
        out
    };
    let local = |idx: usize, parts: &[(usize, usize)], unroll: usize| {
        parts
            .iter()
            .any(|&(s0, w)| idx >= s0 && idx - s0 < (w / unroll) * unroll)
    };
    let (rows, cols) = if left {
        (vec![(0, m)], split(n))
    } else {
        (split(m), vec![(0, n)])
    };
    for j in 0..n {
        for i in 0..m {
            let fused = local(i, &rows, 4) && local(j, &cols, 8);
            let mut r = 0.0f64;
            for l in 0..k {
                let (x, y) = if left {
                    (op(i, l), bb(l, j))
                } else {
                    (bb(i, l), op(l, j))
                };
                r = if fused { x.mul_add(y, r) } else { r + x * y };
            }
            b[i + j * ldb] = r;
        }
    }
}

// ── the level-2/3 symmetric kernels `DSYTRD` calls ───────────────────────────

/// `y += alpha·x` with the Haswell `daxpy_k` (unit strides), for the crate's LAPACK.
pub(crate) fn daxpy_unit(n: usize, alpha: f64, x: &[f64], y: &mut [f64]) {
    daxpy(n, alpha, x, y);
}

/// `dsymv('L', n, alpha, A, lda, x, 1, beta, y, 1)` (OpenBLAS 0.3.30 Haswell,
/// single-threaded below `n = 200`): `y` scaled by `beta` (`dscal`), then
/// `kernel/x86_64/dsymv_L.c` — columns in groups of four; past a 4×4 diagonal
/// triangle (plain C), rows `j+4 .. n & !3` in `dsymv_L_microk_haswell-2.c` (`y =
/// fma(t_c, a_c, y)` over the four columns in turn; each column's dot in four
/// `vfmadd231pd` lanes by row mod 4, reduced `(l0+l2)+(l1+l3)` and added to the C
/// part), the rest in C gcc leaves unfused; the last `n & 3` columns one at a time.
/// `A` column-major, lower triangle read.
#[allow(clippy::too_many_arguments)]
pub fn dsymv_lower(
    n: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    x: &[f64],
    beta: f64,
    y: &mut [f64],
) {
    if beta != 1.0 {
        dscal(n, beta, y, 1);
    }
    if alpha == 0.0 || n == 0 {
        return;
    }
    if n < 200 {
        dsymv_l_kernel(n, n, alpha, a, lda, x, y);
        return;
    }
    // `symv_thread.c` (lower): columns split in 4-aligned ranges balancing the
    // triangle; thread `t` runs the kernel with alpha = 1 on rows `from..n`, its
    // columns, into a zeroed buffer; buffers summed in thread order (`daxpy` by 1),
    // then `y += alpha·buffer` (`daxpy`)
    let nthreads = REFERENCE_THREADS;
    let dnum = n as f64 * n as f64 / nthreads as f64;
    let (mut ranges, mut i, mut cpu) = (vec![], 0usize, 0usize);
    while i < n {
        let mut width = if nthreads - cpu > 1 {
            let di = (n - i) as f64;
            if di * di - dnum > 0.0 {
                (((-(di * di - dnum).sqrt() + di) as usize) + 3) & !3
            } else {
                n - i
            }
        } else {
            n - i
        };
        if nthreads - cpu > 1 && width < 4 {
            width = 4;
        }
        width = width.min(n - i);
        ranges.push((i, width));
        i += width;
        cpu += 1;
    }
    let mut bufs: Vec<Vec<f64>> = vec![];
    for &(from, w) in &ranges {
        let mut b = vec![0.0f64; n - from];
        dsymv_l_kernel(
            n - from,
            w,
            1.0,
            &a[from + from * lda..],
            lda,
            &x[from..],
            &mut b,
        );
        bufs.push(b);
    }
    let mut acc = bufs[0].clone();
    for (t, &(from, _)) in ranges.iter().enumerate().skip(1) {
        daxpy(n - from, 1.0, &bufs[t], &mut acc[from..]);
    }
    daxpy(n, alpha, &acc, y);
}

/// `kernel/x86_64/dsymv_L.c` itself: `m` rows, the first `offset` columns, unit
/// strides (see [`dsymv_lower`]).
fn dsymv_l_kernel(
    m: usize,
    offset: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    x: &[f64],
    y: &mut [f64],
) {
    let at = |r: usize, c: usize| a[r + c * lda];
    let offset1 = offset / 4 * 4;
    let mut j = 0;
    while j < offset1 {
        let tmp1 = [
            alpha * x[j],
            alpha * x[j + 1],
            alpha * x[j + 2],
            alpha * x[j + 3],
        ];
        let mut tmp2 = [0.0f64; 4];
        for c in 0..4 {
            y[j + c] += tmp1[c] * at(j + c, j + c);
        }
        // the 4×4 triangle below the diagonal
        for c in 0..3 {
            for i in j + c + 1..j + 4 {
                y[i] += tmp1[c] * at(i, j + c);
                tmp2[c] += at(i, j + c) * x[i];
            }
        }
        let c_rows = |from: usize, to: usize, y: &mut [f64], tmp2: &mut [f64; 4]| {
            for i in from..to {
                for c in 0..4 {
                    y[i] += tmp1[c] * at(i, j + c);
                    tmp2[c] += at(i, j + c) * x[i];
                }
            }
        };
        if m - (j + 1) >= 12 {
            let m2 = m / 4 * 4;
            if m2 > j + 4 {
                let mut lanes = [[0.0f64; 4]; 4];
                for i in j + 4..m2 {
                    let mut yi = y[i];
                    for c in 0..4 {
                        let av = at(i, j + c);
                        yi = tmp1[c].mul_add(av, yi);
                        lanes[c][i % 4] = av.mul_add(x[i], lanes[c][i % 4]);
                    }
                    y[i] = yi;
                }
                for c in 0..4 {
                    let l = lanes[c];
                    tmp2[c] += (l[0] + l[2]) + (l[1] + l[3]);
                }
            }
            c_rows(m2, m, y, &mut tmp2);
        } else {
            c_rows(j + 4, m, y, &mut tmp2);
        }
        for c in 0..4 {
            y[j + c] += alpha * tmp2[c];
        }
        j += 4;
    }
    for j in offset1..offset {
        let temp1 = alpha * x[j];
        let mut temp2 = 0.0f64;
        y[j] += temp1 * at(j, j);
        for i in j + 1..m {
            y[i] += temp1 * at(i, j);
            temp2 += at(i, j) * x[i];
        }
        y[j] += alpha * temp2;
    }
}

/// `dsyr2('L', n, alpha, x, 1, y, 1, A, lda)` (OpenBLAS 0.3.30, every size: the
/// threaded split is by column): per column `i`, `A[i.., i] += (alpha·x[i])·y[i..]`
/// then `+= (alpha·y[i])·x[i..]`, each a Haswell `daxpy`.
pub fn dsyr2_lower(n: usize, alpha: f64, x: &[f64], y: &[f64], a: &mut [f64], lda: usize) {
    if n == 0 || alpha == 0.0 {
        return;
    }
    for i in 0..n {
        let col = &mut a[i + i * lda..i + i * lda + (n - i)];
        daxpy(n - i, alpha * x[i], &y[i..n], col);
        daxpy(n - i, alpha * y[i], &x[i..n], col);
    }
}

/// One `dgemm_kernel_4x8_haswell` accumulation over `k` for the element at local
/// `(i, j)` of an `m × n` kernel call: a `vfmadd231pd` chain, except on the last
/// column of an odd-width call in a 4-row tile, where `KERNEL4x1` rotates `k` over
/// four accumulators.
fn kernel_acc(
    m: usize,
    n: usize,
    i: usize,
    j: usize,
    k: usize,
    prod: impl Fn(usize, f64) -> f64,
) -> f64 {
    let rotated = n % 2 == 1 && j == n - 1 && i < (m & !3);
    if rotated {
        let mut r = [0.0f64; 4];
        let full = k / 8 * 8;
        for l in 0..full {
            r[l % 4] = prod(l, r[l % 4]);
        }
        for l in full..k {
            r[0] = prod(l, r[0]);
        }
        (r[0] + r[1]) + (r[2] + r[3])
    } else {
        let mut acc = prod(0, 0.0);
        for l in 1..k {
            acc = prod(l, acc);
        }
        acc
    }
}

/// `syrk_thread.c`'s column split for a lower triangle: widths that balance the
/// triangle's area, rounded down to `GEMM_UNROLL_MN = 8`; the last thread takes the rest.
fn syrk_lower_ranges(n: usize, nthreads: usize) -> Vec<(usize, usize)> {
    const MASK: i64 = 7;
    let n_to = n as i64;
    let nf = n as f64;
    let dnum = (0.0 - nf * nf) / nthreads as f64;
    let (mut out, mut i, mut cpu) = (vec![], 0i64, 0usize);
    while i < n_to {
        let mut width = if nthreads - cpu > 1 {
            let di = (n_to - i) as f64;
            let dinum = di * di + dnum;
            let w = if dinum < 0.0 {
                ((di + MASK as f64) as i64) / (MASK + 1) * (MASK + 1)
            } else {
                ((-dinum.sqrt() + di + MASK as f64) as i64) / (MASK + 1) * (MASK + 1)
            };
            if w <= 0 || w > n_to - i { n_to - i } else { w }
        } else {
            n_to - i
        };
        if width <= 0 {
            width = n_to - i;
        }
        out.push((i as usize, width as usize));
        i += width;
        cpu += 1;
    }
    out
}

/// `dsyr2k('L', 'N', n, k, alpha, A, lda, B, ldb, 1, C, ldc)` as OpenBLAS 0.3.30
/// Haswell computes it (returns `false`, `C` untouched, past one `GEMM_Q`/`GEMM_P`
/// block): one thread below `n·k = 1000`, else [`REFERENCE_THREADS`] over the
/// column ranges of [`syrk_lower_ranges`]. Each thread runs `level3_syr2k.c` on
/// its columns `js..js+w` — `alpha·A·Bᵀ` (flag 1) then `alpha·B·Aᵀ` (flag 0)
/// through `syr2k_kernel.c`: the rows below its `w × w` block in one kernel call
/// `w` wide, then the block in 8-column strips — a strip's diagonal block computed
/// whole into a zeroed buffer `S` on the first pass and added as `C + (S[i,j] +
/// S[j,i])`, the rows below it `C + alpha·acc` on each pass. (The split is
/// arithmetically inert: every range starts on a multiple of 8, so the diagonal
/// strips fall where one thread puts them, and only the last range can be odd —
/// it has no rows under it. Measured: forcing one thread changes no bit.)
#[allow(clippy::too_many_arguments)]
pub fn dsyr2k_lower(
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    c: &mut [f64],
    ldc: usize,
) -> bool {
    if k > DGEMM_Q || n > 512 {
        return false;
    }
    if n == 0 || k == 0 || alpha == 0.0 {
        return true;
    }
    const MN: usize = 8;
    let threads = if n * k < 1000 { 1 } else { REFERENCE_THREADS };
    for (js, w) in syrk_lower_ranges(n, threads) {
        for pass in 0..2 {
            let (x, ldx, y, ldy) = if pass == 0 {
                (a, lda, b, ldb)
            } else {
                (b, ldb, a, lda)
            };
            // alpha · Σ_l x[r, l]·y[s, l], as a kernel call of shape (m, nn) sees (i, j)
            let acc = |r: usize, s: usize, m: usize, nn: usize, i: usize, j: usize| {
                alpha
                    * kernel_acc(m, nn, i, j, k, |l, t| {
                        x[r + l * ldx].mul_add(y[s + l * ldy], t)
                    })
            };
            // the rows under this thread's square block
            let under = n - js - w;
            for j in 0..w {
                for i in 0..under {
                    let r = js + w + i;
                    c[r + (js + j) * ldc] += acc(r, js + j, under, w, i, j);
                }
            }
            let mut lp = 0;
            while lp < w {
                let nn = MN.min(w - lp);
                let base = js + lp;
                if pass == 0 {
                    for j in 0..nn {
                        for i in j..nn {
                            let s_ij = acc(base + i, base + j, nn, nn, i, j);
                            // S[j, i] = alpha·A_j·B_iᵀ
                            let s_ji = acc(base + j, base + i, nn, nn, j, i);
                            c[base + i + (base + j) * ldc] += s_ij + s_ji;
                        }
                    }
                }
                let below = w - lp - nn;
                for j in 0..nn {
                    for i in 0..below {
                        let r = base + nn + i;
                        c[r + (base + j) * ldc] += acc(r, base + j, below, nn, i, j);
                    }
                }
                lp += MN;
            }
        }
    }
    true
}
