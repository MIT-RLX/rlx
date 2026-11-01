// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! simd128 GEMM for wasm32, where no BLAS is linked.
//!
//! Register-tiled: a 4-row × 8-column tile of C lives in eight `v128`
//! accumulators while `k` runs, B is read eight columns at a time (packed
//! first when it is transposed, so every load is contiguous), and A is
//! broadcast one element per row and step. Each element of C still takes one
//! multiply and one add per term, in `k` order, from the same starting value
//! as the scalar fallback in `blas.rs`, so the result is bit-identical to it;
//! only the speed changes (wasm SIMD arithmetic is IEEE per lane).
//!
//! Built with relaxed-simd on an engine whose `relaxed_madd` is a real fused
//! multiply-add ([`crate::avx2_mirror::simd::fused`]), each term is one
//! hardware FMA instead: every element of C is then the in-order FMA chain
//! (as on the GPU kernels), and the multiply-add pair becomes one
//! instruction.

use core::arch::wasm32::*;

const MR: usize = 4;
const NR: usize = 8;
/// From this many rows an untransposed B is packed into the tiles' layout.
const PACK_MIN_M: usize = 8;

/// How the fallback it replaces starts and finishes each element.
#[derive(Clone, Copy)]
enum Mode {
    /// `NoTrans · NoTrans`: C scaled by `beta` first, then `C += (alpha·a)·b`
    /// for every `k` whose `alpha·a` is nonzero.
    Accumulate,
    /// The transposed cases: `acc` from 0, `acc += a·b`, then
    /// `C = alpha·acc (+ beta·C)`.
    Scale,
}

/// Row-major `C = alpha·op(A)·op(B) + beta·C`; the contract of the
/// `cblas_sgemm` fallback in `blas.rs`.
///
/// # Safety
/// The pointers cover the strided extents.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn sgemm(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    trans_a: bool,
    b: *const f32,
    ldb: usize,
    trans_b: bool,
    beta: f32,
    c: *mut f32,
    ldc: usize,
) {
    #[cfg(target_feature = "relaxed-simd")]
    if crate::avx2_mirror::simd::fused() {
        unsafe {
            sgemm_with::<true>(
                m, n, k, alpha, a, lda, trans_a, b, ldb, trans_b, beta, c, ldc,
            )
        };
        return;
    }
    unsafe {
        sgemm_with::<false>(
            m, n, k, alpha, a, lda, trans_a, b, ldb, trans_b, beta, c, ldc,
        )
    };
}

/// One term of an element's sum: `s + a·b`, fused when `HW`.
#[inline(always)]
fn term<const HW: bool>(s: f32, a: f32, b: f32) -> f32 {
    if HW {
        crate::avx2_mirror::fmaf(a, b, s)
    } else {
        s + a * b
    }
}

/// [`term`] on four lanes.
#[inline(always)]
fn term4<const HW: bool>(s: v128, a: v128, b: v128) -> v128 {
    #[cfg(target_feature = "relaxed-simd")]
    if HW {
        return f32x4_relaxed_madd(a, b, s);
    }
    f32x4_add(s, f32x4_mul(a, b))
}

/// [`sgemm`]; `HW`: every term one fused multiply-add.
#[allow(clippy::too_many_arguments)]
unsafe fn sgemm_with<const HW: bool>(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    trans_a: bool,
    b: *const f32,
    ldb: usize,
    trans_b: bool,
    beta: f32,
    c: *mut f32,
    ldc: usize,
) {
    let mode = if !trans_a && !trans_b {
        Mode::Accumulate
    } else {
        Mode::Scale
    };
    // A few rows of an untransposed product: the fallback's i→p→j loop
    // streams whole rows of B and is faster than packing them (decode-shaped
    // m = 1 ran at 0.7x through the tiles).
    if matches!(mode, Mode::Accumulate) && m < PACK_MIN_M && !HW {
        unsafe {
            crate::blas::sgemm_loops(
                m, n, k, alpha, a, lda, trans_a, b, ldb, trans_b, beta, c, ldc,
            )
        };
        return;
    }
    if let Mode::Accumulate = mode {
        for i in 0..m {
            let row = unsafe { std::slice::from_raw_parts_mut(c.add(i * ldc), n) };
            if beta == 0.0 {
                row.fill(0.0);
            } else if beta != 1.0 {
                for v in row {
                    *v *= beta;
                }
            }
        }
    }
    let at = |i: usize, p: usize| unsafe {
        if trans_a {
            *a.add(p * lda + i)
        } else {
            *a.add(i * lda + p)
        }
    };
    // ... and its fused form: the same i→p→j stream, one FMA per term.
    if matches!(mode, Mode::Accumulate) && m < PACK_MIN_M {
        let n4 = n / 4 * 4;
        for i in 0..m {
            let crow = unsafe { c.add(i * ldc) };
            for p in 0..k {
                let av = alpha * unsafe { *a.add(i * lda + p) };
                if av == 0.0 {
                    continue;
                }
                let (sv, brow) = (f32x4_splat(av), unsafe { b.add(p * ldb) });
                let mut j = 0;
                while j < n4 {
                    unsafe {
                        let cj = crow.add(j) as *mut v128;
                        v128_store(
                            cj,
                            term4::<HW>(v128_load(cj), sv, v128_load(brow.add(j) as *const v128)),
                        );
                    }
                    j += 4;
                }
                for j in n4..n {
                    unsafe { *crow.add(j) = term::<HW>(*crow.add(j), av, *brow.add(j)) };
                }
            }
        }
        return;
    }
    let k_ = Tile::<_, HW> {
        k,
        alpha,
        beta,
        mode,
        at: &at,
        c,
        ldc,
    };
    let full = n / NR * NR;
    // Each 8-column panel of `op(B)`, `k` rows of 8 contiguous floats, reused
    // by every row tile.
    let mut packed = vec![0f32; k * NR];
    for j0 in (0..full).step_by(NR) {
        for (p, dst) in packed.chunks_exact_mut(NR).enumerate() {
            if trans_b {
                for (jj, d) in dst.iter_mut().enumerate() {
                    *d = unsafe { *b.add((j0 + jj) * ldb + p) };
                }
            } else {
                dst.copy_from_slice(unsafe { std::slice::from_raw_parts(b.add(p * ldb + j0), NR) });
            }
        }
        let (bp, bs) = (packed.as_ptr(), NR);
        let mut i = 0;
        while i + MR <= m {
            unsafe { k_.run::<MR>(i, j0, bp, bs) };
            i += MR;
        }
        match m - i {
            3 => unsafe { k_.run::<3>(i, j0, bp, bs) },
            2 => unsafe { k_.run::<2>(i, j0, bp, bs) },
            1 => unsafe { k_.run::<1>(i, j0, bp, bs) },
            _ => {}
        }
    }
    // The last `n % 8` columns, one element at a time.
    let bt = |p: usize, j: usize| unsafe {
        if trans_b {
            *b.add(j * ldb + p)
        } else {
            *b.add(p * ldb + j)
        }
    };
    for i in 0..m {
        for j in full..n {
            let cp = unsafe { c.add(i * ldc + j) };
            match mode {
                Mode::Accumulate => {
                    let mut s = unsafe { *cp };
                    for p in 0..k {
                        let av = alpha * at(i, p);
                        if av != 0.0 {
                            s = term::<HW>(s, av, bt(p, j));
                        }
                    }
                    unsafe { *cp = s };
                }
                Mode::Scale => {
                    let mut s = 0f32;
                    for p in 0..k {
                        s = term::<HW>(s, at(i, p), bt(p, j));
                    }
                    unsafe { *cp = finish(alpha, beta, s, *cp) };
                }
            }
        }
    }
}

/// `alpha·acc`, plus `beta·C` unless `beta == 0` (then C is not read, as in
/// CBLAS: stale NaN / Inf in a reused buffer must not leak through `0·NaN`).
#[inline(always)]
fn finish(alpha: f32, beta: f32, acc: f32, c: f32) -> f32 {
    if beta == 0.0 {
        alpha * acc
    } else {
        alpha * acc + beta * c
    }
}

struct Tile<'a, A: Fn(usize, usize) -> f32, const HW: bool> {
    k: usize,
    alpha: f32,
    beta: f32,
    mode: Mode,
    at: &'a A,
    c: *mut f32,
    ldc: usize,
}

impl<A: Fn(usize, usize) -> f32, const HW: bool> Tile<'_, A, HW> {
    /// Rows `i..i+R`, columns `j0..j0+8`, B rows at `bp + p·bs`.
    #[inline(always)]
    unsafe fn run<const R: usize>(&self, i: usize, j0: usize, bp: *const f32, bs: usize) {
        let cp: [*mut f32; R] =
            std::array::from_fn(|r| unsafe { self.c.add((i + r) * self.ldc + j0) });
        let mut acc = [[f32x4_splat(0.0); 2]; R];
        if let Mode::Accumulate = self.mode {
            for r in 0..R {
                acc[r] = unsafe {
                    [
                        v128_load(cp[r] as *const v128),
                        v128_load(cp[r].add(4) as *const v128),
                    ]
                };
            }
        }
        for p in 0..self.k {
            let (b0, b1) = unsafe {
                let row = bp.add(p * bs);
                (
                    v128_load(row as *const v128),
                    v128_load(row.add(4) as *const v128),
                )
            };
            for r in 0..R {
                let av = (self.at)(i + r, p);
                let av = match self.mode {
                    Mode::Accumulate => {
                        let av = self.alpha * av;
                        if av == 0.0 {
                            continue;
                        }
                        av
                    }
                    Mode::Scale => av,
                };
                let s = f32x4_splat(av);
                acc[r][0] = term4::<HW>(acc[r][0], s, b0);
                acc[r][1] = term4::<HW>(acc[r][1], s, b1);
            }
        }
        for r in 0..R {
            if let Mode::Scale = self.mode {
                let al = f32x4_splat(self.alpha);
                for h in 0..2 {
                    let scaled = f32x4_mul(al, acc[r][h]);
                    acc[r][h] = if self.beta == 0.0 {
                        scaled
                    } else {
                        let old = unsafe { v128_load(cp[r].add(4 * h) as *const v128) };
                        f32x4_add(scaled, f32x4_mul(f32x4_splat(self.beta), old))
                    };
                }
            }
            unsafe {
                v128_store(cp[r] as *mut v128, acc[r][0]);
                v128_store(cp[r].add(4) as *mut v128, acc[r][1]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    /// The kernel against the scalar fallback loops, bit for bit, over every
    /// transpose pair, edge sizes on each axis (row and column tails), alpha /
    /// beta in and out of `{0, 1}`, zeros in A (the fallback skips them) and
    /// a NaN-poisoned C under `beta == 0`.
    #[test]
    fn simd128_gemm_is_the_fallback_loops() {
        let mut s = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / 8_388_608.0) - 1.0
        };
        let mut cases = 0;
        for &(m, n, k) in &[
            (1, 8, 1),
            (1, 2304, 768),
            (3, 7, 5),
            (4, 8, 16),
            (5, 17, 33),
            (9, 24, 0),
            (40, 77, 64),
            (13, 1, 9),
        ] {
            for (ta, tb) in [(false, false), (true, false), (false, true), (true, true)] {
                for (alpha, beta) in [
                    (1.0f32, 0.0f32),
                    (1.0, 1.0),
                    (0.5, 0.0),
                    (-1.25, 0.75),
                    (2.0, 1.0),
                ] {
                    let (lda, ldb, ldc) = (
                        if ta { m + 3 } else { k + 2 },
                        if tb { k + 1 } else { n + 5 },
                        n + 4,
                    );
                    let a: Vec<f32> = (0..lda * if ta { k } else { m })
                        .map(|i| if i % 11 == 0 { 0.0 } else { rnd() })
                        .collect();
                    let b: Vec<f32> = (0..ldb * if tb { n } else { k })
                        .map(|_| rnd() * 3.0)
                        .collect();
                    let c0: Vec<f32> = (0..ldc * m)
                        .map(|i| {
                            if beta == 0.0 && i % 5 == 0 {
                                f32::NAN
                            } else {
                                rnd()
                            }
                        })
                        .collect();
                    let (mut want, mut got) = (c0.clone(), c0.clone());
                    unsafe {
                        crate::blas::sgemm_loops(
                            m,
                            n,
                            k,
                            alpha,
                            a.as_ptr(),
                            lda,
                            ta,
                            b.as_ptr(),
                            ldb,
                            tb,
                            beta,
                            want.as_mut_ptr(),
                            ldc,
                        );
                        super::sgemm_with::<false>(
                            m,
                            n,
                            k,
                            alpha,
                            a.as_ptr(),
                            lda,
                            ta,
                            b.as_ptr(),
                            ldb,
                            tb,
                            beta,
                            got.as_mut_ptr(),
                            ldc,
                        );
                    }
                    for i in 0..m {
                        for j in 0..n {
                            let (w, g) = (want[i * ldc + j], got[i * ldc + j]);
                            assert!(
                                w.to_bits() == g.to_bits() || (w.is_nan() && g.is_nan()),
                                "m{m} n{n} k{k} ta{ta} tb{tb} a{alpha} b{beta} at ({i},{j}): {w} vs {g}"
                            );
                        }
                    }
                    cases += 1;
                }
            }
        }
        assert_eq!(cases, 8 * 4 * 5);
    }

    /// With relaxed-simd on a fusing engine: every element is the in-order
    /// chain of fused multiply-adds (the scalar reference below, through the
    /// exact `fmaf`), from the fallback's start value and with its zero-skip.
    #[cfg(target_feature = "relaxed-simd")]
    #[test]
    fn fused_gemm_is_the_in_order_fma_chain() {
        assert!(
            crate::avx2_mirror::simd::fused(),
            "node's V8 fuses relaxed_madd on arm64"
        );
        let fma = crate::avx2_mirror::fmaf;
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / 8_388_608.0) - 1.0
        };
        let mut cases = 0;
        for &(m, n, k) in &[
            (1, 8, 1),
            (1, 2303, 768),
            (3, 7, 5),
            (5, 17, 33),
            (8, 24, 9),
            (9, 24, 0),
            (40, 77, 64),
            (13, 1, 9),
        ] {
            for (ta, tb) in [(false, false), (true, false), (false, true), (true, true)] {
                for (alpha, beta) in [(1.0f32, 0.0f32), (1.0, 1.0), (-1.25, 0.75)] {
                    let (lda, ldb, ldc) = (
                        if ta { m + 3 } else { k + 2 },
                        if tb { k + 1 } else { n + 5 },
                        n + 4,
                    );
                    let a: Vec<f32> = (0..lda * if ta { k } else { m })
                        .map(|i| if i % 11 == 0 { 0.0 } else { rnd() })
                        .collect();
                    let b: Vec<f32> = (0..ldb * if tb { n } else { k })
                        .map(|_| rnd() * 3.0)
                        .collect();
                    let c0: Vec<f32> = (0..ldc * m).map(|_| rnd()).collect();
                    let at = |i: usize, p: usize| if ta { a[p * lda + i] } else { a[i * lda + p] };
                    let bt = |p: usize, j: usize| if tb { b[j * ldb + p] } else { b[p * ldb + j] };
                    let mut got = c0.clone();
                    unsafe {
                        super::sgemm(
                            m,
                            n,
                            k,
                            alpha,
                            a.as_ptr(),
                            lda,
                            ta,
                            b.as_ptr(),
                            ldb,
                            tb,
                            beta,
                            got.as_mut_ptr(),
                            ldc,
                        )
                    };
                    for i in 0..m {
                        for j in 0..n {
                            let old = c0[i * ldc + j];
                            let want = if !ta && !tb {
                                let mut s = if beta == 0.0 {
                                    0.0
                                } else if beta != 1.0 {
                                    old * beta
                                } else {
                                    old
                                };
                                for p in 0..k {
                                    let av = alpha * at(i, p);
                                    if av != 0.0 {
                                        s = fma(av, bt(p, j), s);
                                    }
                                }
                                s
                            } else {
                                let mut s = 0f32;
                                for p in 0..k {
                                    s = fma(at(i, p), bt(p, j), s);
                                }
                                super::finish(alpha, beta, s, old)
                            };
                            let g = got[i * ldc + j];
                            assert_eq!(
                                g.to_bits(),
                                want.to_bits(),
                                "m{m} n{n} k{k} ta{ta} tb{tb} a{alpha} b{beta} at ({i},{j}): {g} vs {want}"
                            );
                        }
                    }
                    cases += 1;
                }
            }
        }
        assert_eq!(cases, 8 * 4 * 3);
    }
}
