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
pub fn ddot_threads(n: usize, x: &[f64], incx: usize, y: &[f64], incy: usize, threads: usize) -> f64 {
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

/// The level-3 drivers' split of `k` (`driver/level3/level3_syrk.c`): blocks of
/// `GEMM_Q`, except that a remainder in `(Q, 2Q)` is halved (`(min_l + 1) / 2`).
fn k_blocks(k: usize) -> Vec<(usize, usize)> {
    let mut out = vec![];
    let mut ls = 0;
    while ls < k {
        let mut min_l = k - ls;
        if min_l >= 2 * DGEMM_Q {
            min_l = DGEMM_Q;
        } else if min_l > DGEMM_Q {
            min_l = (min_l + 1) / 2;
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
    let row_major = order == crate::numpy_blas::ROW_MAJOR;
    let no_trans = trans == crate::numpy_blas::NO_TRANS;
    // element (i, l) of op(A), n×k
    let at = |i: usize, l: usize| {
        let (r, cc) = if no_trans { (i, l) } else { (l, i) };
        if row_major { a[r * lda + cc] } else { a[r + cc * lda] }
    };
    let blocks = k_blocks(k);
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
