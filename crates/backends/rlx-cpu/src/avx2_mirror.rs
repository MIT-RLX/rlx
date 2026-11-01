// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Mirrors of the x86-64 AVX2+FMA kernels, lane for lane.
//!
//! The targets with no SIMD arm here (wasm, riscv) used to run portable
//! loops of the same *functions* in a different *arithmetic*: no fused
//! multiply-adds, one accumulator instead of eight, so a browser run of a
//! LayerNorm or a GELU differed from the Linux x86-64 run in the last bits.
//! These compute what the AVX2 kernels compute: each `_mm256_fmadd_ps` is a
//! fused multiply-add, each reduction keeps the eight lanes and folds them in
//! the kernel's order, and the scalar tails are the kernels' own tails.
//!
//! Written once over [`Lanes`]: `f32` (one lane) everywhere, and on wasm with
//! simd128 a 4-lane `v128`. wasm has no FMA instruction, and its software
//! `fmaf` is both slow (these ran 13x slower than the inexact loops they
//! replace) and wrong when the result is a float subnormal (it rounds twice),
//! so on wasm the fused multiply-add is [`fmaf`] / `fma_f32x4` here.
//!
//! Compiled everywhere so that x86-64 can test each mirror against its kernel.

#![cfg_attr(any(target_arch = "aarch64", target_arch = "x86_64"), allow(dead_code))]

use crate::kernels::scalar_gelu;

/// TwoSum: `s = a + b` rounded, and its exact error.
#[inline(always)]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bv = s - a;
    (s, (a - (s - bv)) + (b - bv))
}

/// `a·b + c` rounded once to float, from double arithmetic alone.
///
/// The product of two floats is exact in a double, so `s = p + c` is the
/// exact value rounded once, with TwoSum error `e`. Rounding `s` to float is
/// the correct result unless `s` is exactly a midpoint between two floats and
/// `e != 0` (the exact value is off it): then `s` moves one double ulp towards
/// `e` first. A subnormal result sits on the `2^-149` grid, not 24 bits below
/// its leading one, so it is rounded on that grid directly.
#[inline]
pub(crate) fn fmaf(a: f32, b: f32, c: f32) -> f32 {
    let (s, e) = two_sum(a as f64 * b as f64, c as f64);
    if !s.is_finite() {
        return s as f32;
    }
    if s.abs() >= f32::MIN_POSITIVE as f64 || s == 0.0 {
        let bits = s.to_bits();
        let s = if bits & 0x1fff_ffff == 0x1000_0000 && e != 0.0 {
            f64::from_bits(if (s > 0.0) == (e > 0.0) {
                bits + 1
            } else {
                bits - 1
            })
        } else {
            s
        };
        return s as f32;
    }
    // Subnormal: units of 2^-149 (exact scalings by powers of two).
    let (us, ue) = (s * 2f64.powi(149), e * 2f64.powi(149));
    let mut q = us.round_ties_even();
    let d = us - q; // exact: `|us| < 2^23`
    if d.abs() == 0.5 && ue != 0.0 {
        // `us` is a tie that the exact value `us + ue` is not on: take the
        // neighbour on `ue`'s side (`us + d` is the one `q` is not).
        q = if (d > 0.0) == (ue > 0.0) {
            us + d
        } else {
            us - d
        };
    }
    (q * 2f64.powi(-149)) as f32
}

/// The vector operations the AVX2 kernels use, lane for lane.
pub(crate) trait Lanes: Copy {
    const W: usize;
    fn splat(x: f32) -> Self;
    /// # Safety
    /// `p` points at `W` floats.
    unsafe fn load(p: *const f32) -> Self;
    /// # Safety
    /// `p` points at `W` writable floats.
    unsafe fn store(self, p: *mut f32);
    fn add(self, o: Self) -> Self;
    fn sub(self, o: Self) -> Self;
    fn mul(self, o: Self) -> Self;
    fn div(self, o: Self) -> Self;
    /// `self·b + c`, rounded once (`_mm256_fmadd_ps`).
    fn fmadd(self, b: Self, c: Self) -> Self;
    /// `-(self·b) + c`, rounded once (`_mm256_fnmadd_ps`).
    fn fnmadd(self, b: Self, c: Self) -> Self;
    /// `_mm256_max_ps(self, o)`: `self > o ? self : o` (so `o` on NaN).
    fn max_ps(self, o: Self) -> Self;
    /// `_mm256_min_ps(self, o)`: `self < o ? self : o`.
    fn min_ps(self, o: Self) -> Self;
    /// Round to nearest, ties to even.
    fn round_even(self) -> Self;
    fn abs(self) -> Self;
    /// `self >= 0 ? 1 : -1` (`_CMP_GE_OQ` + blend: NaN gives -1).
    fn sign(self) -> Self;
    /// `2^n · self` by adding the integral `n` to the exponent field.
    fn add_exp(self, n: Self) -> Self;
}

impl Lanes for f32 {
    const W: usize = 1;
    #[inline(always)]
    fn splat(x: f32) -> Self {
        x
    }
    #[inline(always)]
    unsafe fn load(p: *const f32) -> Self {
        unsafe { *p }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f32) {
        unsafe { *p = self }
    }
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        self + o
    }
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        self - o
    }
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        self * o
    }
    #[inline(always)]
    fn div(self, o: Self) -> Self {
        self / o
    }
    #[inline(always)]
    fn fmadd(self, b: Self, c: Self) -> Self {
        #[cfg(target_arch = "wasm32")]
        return fmaf(self, b, c);
        #[cfg(not(target_arch = "wasm32"))]
        self.mul_add(b, c)
    }
    #[inline(always)]
    fn fnmadd(self, b: Self, c: Self) -> Self {
        (-self).fmadd(b, c)
    }
    #[inline(always)]
    fn max_ps(self, o: Self) -> Self {
        if self > o { self } else { o }
    }
    #[inline(always)]
    fn min_ps(self, o: Self) -> Self {
        if self < o { self } else { o }
    }
    #[inline(always)]
    fn round_even(self) -> Self {
        self.round_ties_even()
    }
    #[inline(always)]
    fn abs(self) -> Self {
        f32::abs(self)
    }
    #[inline(always)]
    fn sign(self) -> Self {
        if self >= 0.0 { 1.0 } else { -1.0 }
    }
    #[inline(always)]
    fn add_exp(self, n: Self) -> Self {
        f32::from_bits(self.to_bits().wrapping_add((n as i32 as u32) << 23))
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod simd {
    use core::arch::wasm32::*;

    /// Exact `a·b + c` per lane. The product of two floats is exact in a
    /// double, so the double sum `s` is the exact value rounded once, and
    /// rounding `s` to float is the correctly rounded result unless `s`
    /// landed exactly on a midpoint between two floats (the 29 bits a double
    /// has beyond a float are `100…0`) or is a float subnormal (fewer than 24
    /// bits, another grid). Both are rare; such a vector goes lane by lane
    /// through [`super::fmaf`], which settles them with the sum's error.
    #[inline(always)]
    fn sums(a: v128, b: v128, c: v128) -> (v128, v128) {
        let lo = |v: v128| f64x2_promote_low_f32x4(v);
        let hi = |v: v128| f64x2_promote_low_f32x4(i32x4_shuffle::<2, 3, 2, 3>(v, v));
        (
            f64x2_add(f64x2_mul(lo(a), lo(b)), lo(c)),
            f64x2_add(f64x2_mul(hi(a), hi(b)), hi(c)),
        )
    }

    /// Lanes whose double sum is a float midpoint, or `0 < |s| < 2^-126`
    /// (as bits: `0 <= |s|bits - 1 < 2^-126 bits - 1`).
    #[inline(always)]
    fn odd(s: v128) -> v128 {
        let mid = i64x2_eq(
            v128_and(s, i64x2_splat(0x1fff_ffff)),
            i64x2_splat(0x1000_0000),
        );
        let mag = i64x2_sub(v128_and(s, i64x2_splat(i64::MAX)), i64x2_splat(1));
        let tiny = i64x2_lt(
            mag,
            i64x2_splat((f32::MIN_POSITIVE as f64).to_bits() as i64 - 1),
        );
        // `|s| == 0` wraps `mag` to -1, which is "tiny": exclude it
        v128_or(mid, v128_and(tiny, i64x2_ge(mag, i64x2_splat(0))))
    }

    #[inline(always)]
    fn demote(sl: v128, sh: v128) -> v128 {
        i32x4_shuffle::<0, 1, 4, 5>(f32x4_demote_f64x2_zero(sl), f32x4_demote_f64x2_zero(sh))
    }

    /// [`fma_f32x4`] lane by lane through [`super::fmaf`].
    #[cold]
    #[inline(never)]
    fn fma_slow(a: v128, b: v128, c: v128) -> v128 {
        let f = super::fmaf;
        f32x4(
            f(
                f32x4_extract_lane::<0>(a),
                f32x4_extract_lane::<0>(b),
                f32x4_extract_lane::<0>(c),
            ),
            f(
                f32x4_extract_lane::<1>(a),
                f32x4_extract_lane::<1>(b),
                f32x4_extract_lane::<1>(c),
            ),
            f(
                f32x4_extract_lane::<2>(a),
                f32x4_extract_lane::<2>(b),
                f32x4_extract_lane::<2>(c),
            ),
            f(
                f32x4_extract_lane::<3>(a),
                f32x4_extract_lane::<3>(b),
                f32x4_extract_lane::<3>(c),
            ),
        )
    }

    /// Exact `a·b + c` per lane. The product of two floats is exact in a
    /// double, so the double sum `s` is the exact value rounded once, and
    /// rounding `s` to float is the correctly rounded result unless `s`
    /// landed exactly on a midpoint between two floats (the 29 bits a double
    /// has beyond a float are `100…0`) or is a float subnormal (fewer than 24
    /// bits, another grid). Both are rare; such a vector goes lane by lane
    /// through [`super::fmaf`], which settles them with the sum's error.
    #[inline(always)]
    pub(super) fn fma_f32x4(a: v128, b: v128, c: v128) -> v128 {
        let (sl, sh) = sums(a, b, c);
        if v128_any_true(v128_or(odd(sl), odd(sh))) {
            return fma_slow(a, b, c);
        }
        demote(sl, sh)
    }

    /// Eight lanes (one AVX register) as two `v128`, so each step of a
    /// kernel's dependency chain has two vectors in flight and one test.
    #[derive(Clone, Copy)]
    pub(crate) struct X2(v128, v128);

    impl super::Lanes for X2 {
        const W: usize = 8;
        #[inline(always)]
        fn splat(x: f32) -> Self {
            X2(f32x4_splat(x), f32x4_splat(x))
        }
        #[inline(always)]
        unsafe fn load(p: *const f32) -> Self {
            unsafe {
                X2(
                    v128_load(p as *const v128),
                    v128_load(p.add(4) as *const v128),
                )
            }
        }
        #[inline(always)]
        unsafe fn store(self, p: *mut f32) {
            unsafe {
                v128_store(p as *mut v128, self.0);
                v128_store(p.add(4) as *mut v128, self.1);
            }
        }
        #[inline(always)]
        fn add(self, o: Self) -> Self {
            X2(f32x4_add(self.0, o.0), f32x4_add(self.1, o.1))
        }
        #[inline(always)]
        fn sub(self, o: Self) -> Self {
            X2(f32x4_sub(self.0, o.0), f32x4_sub(self.1, o.1))
        }
        #[inline(always)]
        fn mul(self, o: Self) -> Self {
            X2(f32x4_mul(self.0, o.0), f32x4_mul(self.1, o.1))
        }
        #[inline(always)]
        fn div(self, o: Self) -> Self {
            X2(f32x4_div(self.0, o.0), f32x4_div(self.1, o.1))
        }
        #[inline(always)]
        fn fmadd(self, b: Self, c: Self) -> Self {
            let (l0, h0) = sums(self.0, b.0, c.0);
            let (l1, h1) = sums(self.1, b.1, c.1);
            if v128_any_true(v128_or(
                v128_or(odd(l0), odd(h0)),
                v128_or(odd(l1), odd(h1)),
            )) {
                return X2(fma_slow(self.0, b.0, c.0), fma_slow(self.1, b.1, c.1));
            }
            X2(demote(l0, h0), demote(l1, h1))
        }
        #[inline(always)]
        fn fnmadd(self, b: Self, c: Self) -> Self {
            X2(f32x4_neg(self.0), f32x4_neg(self.1)).fmadd(b, c)
        }
        #[inline(always)]
        fn max_ps(self, o: Self) -> Self {
            X2(self.0.max_ps(o.0), self.1.max_ps(o.1))
        }
        #[inline(always)]
        fn min_ps(self, o: Self) -> Self {
            X2(self.0.min_ps(o.0), self.1.min_ps(o.1))
        }
        #[inline(always)]
        fn round_even(self) -> Self {
            X2(f32x4_nearest(self.0), f32x4_nearest(self.1))
        }
        #[inline(always)]
        fn abs(self) -> Self {
            X2(f32x4_abs(self.0), f32x4_abs(self.1))
        }
        #[inline(always)]
        fn sign(self) -> Self {
            X2(self.0.sign(), self.1.sign())
        }
        #[inline(always)]
        fn add_exp(self, n: Self) -> Self {
            X2(self.0.add_exp(n.0), self.1.add_exp(n.1))
        }
    }

    impl super::Lanes for v128 {
        const W: usize = 4;
        #[inline(always)]
        fn splat(x: f32) -> Self {
            f32x4_splat(x)
        }
        #[inline(always)]
        unsafe fn load(p: *const f32) -> Self {
            unsafe { v128_load(p as *const v128) }
        }
        #[inline(always)]
        unsafe fn store(self, p: *mut f32) {
            unsafe { v128_store(p as *mut v128, self) }
        }
        #[inline(always)]
        fn add(self, o: Self) -> Self {
            f32x4_add(self, o)
        }
        #[inline(always)]
        fn sub(self, o: Self) -> Self {
            f32x4_sub(self, o)
        }
        #[inline(always)]
        fn mul(self, o: Self) -> Self {
            f32x4_mul(self, o)
        }
        #[inline(always)]
        fn div(self, o: Self) -> Self {
            f32x4_div(self, o)
        }
        #[inline(always)]
        fn fmadd(self, b: Self, c: Self) -> Self {
            fma_f32x4(self, b, c)
        }
        #[inline(always)]
        fn fnmadd(self, b: Self, c: Self) -> Self {
            fma_f32x4(f32x4_neg(self), b, c)
        }
        #[inline(always)]
        fn max_ps(self, o: Self) -> Self {
            v128_bitselect(self, o, f32x4_gt(self, o))
        }
        #[inline(always)]
        fn min_ps(self, o: Self) -> Self {
            v128_bitselect(self, o, f32x4_lt(self, o))
        }
        #[inline(always)]
        fn round_even(self) -> Self {
            f32x4_nearest(self)
        }
        #[inline(always)]
        fn abs(self) -> Self {
            f32x4_abs(self)
        }
        #[inline(always)]
        fn sign(self) -> Self {
            v128_bitselect(
                f32x4_splat(1.0),
                f32x4_splat(-1.0),
                f32x4_ge(self, f32x4_splat(0.0)),
            )
        }
        #[inline(always)]
        fn add_exp(self, n: Self) -> Self {
            i32x4_add(self, i32x4_shl(i32x4_trunc_sat_f32x4(n), 23))
        }
    }
}

/// The widest [`Lanes`] on this target.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
type Wide = simd::X2;
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
type Wide = f32;

/// The kernels' horizontal sum of eight lanes: `lo + hi`, then
/// `(s0 + s2) + (s1 + s3)`.
#[inline(always)]
fn hsum8(v: &[f32; 8]) -> f32 {
    let s: [f32; 4] = std::array::from_fn(|i| v[i] + v[i + 4]);
    (s[0] + s[2]) + (s[1] + s[3])
}

/// The kernels' horizontal max of eight lanes, folded like [`hsum8`] with
/// `max_ps`.
#[inline(always)]
fn hmax8(v: &[f32; 8]) -> f32 {
    let s: [f32; 4] = std::array::from_fn(|i| v[i].max_ps(v[i + 4]));
    s[0].max_ps(s[2]).max_ps(s[1].max_ps(s[3]))
}

/// [`crate::kernels::avx2_exp8`].
#[inline(always)]
#[allow(clippy::excessive_precision)] // the kernel's literals, verbatim
pub(crate) fn exp_lane<V: Lanes>(x: V) -> V {
    let x = x.max_ps(V::splat(-87.3)).min_ps(V::splat(88.7));
    let n = x.mul(V::splat(std::f32::consts::LOG2_E)).round_even();
    let r = n.fnmadd(
        V::splat(1.428606765330187e-6),
        n.fnmadd(V::splat(0.693145751953125), x),
    );
    let mut p = V::splat(0.001388888888888889);
    p = p.fmadd(r, V::splat(0.008333333333333333));
    p = p.fmadd(r, V::splat(0.041666666666666664));
    p = p.fmadd(r, V::splat(0.16666666666666666));
    p = p.fmadd(r, V::splat(0.5));
    p = p.fmadd(r, V::splat(1.0));
    p = p.fmadd(r, V::splat(1.0));
    p.add_exp(n)
}

/// The erf-GELU of `gelu_inplace_avx2` / `bias_gelu_avx2`.
#[inline(always)]
#[allow(clippy::excessive_precision)] // the kernel's literals, verbatim
fn gelu_lane<V: Lanes>(x: V) -> V {
    let one = V::splat(1.0);
    let erf_arg = x.mul(V::splat(std::f32::consts::FRAC_1_SQRT_2));
    let xa = erf_arg.abs();
    let sign = erf_arg.sign();
    let t = one.div(V::splat(0.3275911).fmadd(xa, one));
    let mut y = V::splat(1.061405429);
    y = y.fmadd(t, V::splat(-1.453152027));
    y = y.fmadd(t, V::splat(1.421413741));
    y = y.fmadd(t, V::splat(-0.284496736));
    y = y.fmadd(t, V::splat(0.254829592));
    y = y.mul(t);
    let e = exp_lane(V::splat(0.0).sub(xa.mul(xa)));
    let erf = sign.mul(y.fnmadd(e, one));
    x.mul(V::splat(0.5).mul(one.add(erf)))
}

#[inline(always)]
fn silu_lane<V: Lanes>(x: V) -> V {
    x.div(V::splat(1.0).add(exp_lane(V::splat(0.0).sub(x))))
}

#[inline(always)]
fn tanh_lane<V: Lanes>(x: V) -> V {
    let one = V::splat(1.0);
    let e = exp_lane(x.mul(V::splat(2.0)));
    e.sub(one).div(e.add(one))
}

/// `f` on the first `len / 8 · 8` elements (the kernel's vector body), `W`
/// at a time, then `tail` on the rest.
#[inline(always)]
fn by_lanes<V: Lanes>(data: &mut [f32], f: impl Fn(V) -> V, tail: impl Fn(f32) -> f32) {
    let body = data.len() / 8 * 8;
    let p = data.as_mut_ptr();
    let mut i = 0;
    // Two independent chains per iteration: the kernels are long dependency
    // chains (Horner steps), and one at a time leaves the units idle.
    while i + 2 * V::W <= body {
        // SAFETY: `i + 2W <= body <= len`.
        unsafe {
            let (x0, x1) = (V::load(p.add(i)), V::load(p.add(i + V::W)));
            let (y0, y1) = (f(x0), f(x1));
            y0.store(p.add(i));
            y1.store(p.add(i + V::W));
        }
        i += 2 * V::W;
    }
    while i < body {
        // SAFETY: `i + W <= body <= len` (`W` divides 8).
        unsafe { f(V::load(p.add(i))).store(p.add(i)) };
        i += V::W;
    }
    for v in &mut data[body..] {
        *v = tail(*v);
    }
}

/// `gelu_inplace_avx2`.
pub(crate) fn gelu_inplace(data: &mut [f32]) {
    by_lanes::<Wide>(data, gelu_lane, scalar_gelu);
}

/// `bias_gelu_avx2`: the bias added per row, the 8-lane body per row.
pub(crate) fn bias_gelu(data: &mut [f32], bias: &[f32], m: usize, n: usize) {
    let body = n / 8 * 8;
    for row in data[..m * n].chunks_exact_mut(n) {
        for (v, &b) in row[..body].iter_mut().zip(bias) {
            *v += b;
        }
        by_lanes::<Wide>(&mut row[..body], gelu_lane, scalar_gelu);
        for (v, &b) in row[body..].iter_mut().zip(&bias[body..n]) {
            *v = scalar_gelu(*v + b);
        }
    }
}

/// `silu_inplace_avx2`.
pub(crate) fn silu_inplace(data: &mut [f32]) {
    by_lanes::<Wide>(data, silu_lane, |x| x / (1.0 + (-x).exp()));
}

/// `vvexpf_avx2`, with the caller's tail function.
pub(crate) fn vvexpf(y: &mut [f32], x: &[f32], tail: impl Fn(f32) -> f32) {
    y.copy_from_slice(x);
    by_lanes::<Wide>(y, exp_lane, tail);
}

/// `vvtanhf_avx2`, with the caller's tail function.
pub(crate) fn vvtanhf(y: &mut [f32], x: &[f32], tail: impl Fn(f32) -> f32) {
    y.copy_from_slice(x);
    by_lanes::<Wide>(y, tanh_lane, tail);
}

/// Eight lanes as `8 / W` vectors.
struct Acc8<V: Lanes>([V; 8]);

impl<V: Lanes> Acc8<V> {
    fn splat(x: f32) -> Self {
        Acc8([V::splat(x); 8])
    }
    /// Apply `f(acc, x)` to the eight lanes and the eight floats at `p`.
    #[inline(always)]
    unsafe fn step(&mut self, p: *const f32, f: impl Fn(V, V) -> V) {
        let mut l = 0;
        while l < 8 {
            let v = &mut self.0[l / V::W];
            *v = f(*v, unsafe { V::load(p.add(l)) });
            l += V::W;
        }
    }
    fn lanes(&self) -> [f32; 8] {
        let mut out = [0f32; 8];
        let mut l = 0;
        while l < 8 {
            // SAFETY: `l + W <= 8`.
            unsafe { self.0[l / V::W].store(out.as_mut_ptr().add(l)) };
            l += V::W;
        }
        out
    }
}

/// `softmax_rows_avx2`.
pub(crate) fn softmax_rows(data: &mut [f32], rows: usize, cols: usize) {
    softmax_rows_with::<Wide>(data, rows, cols);
}

fn softmax_rows_with<V: Lanes>(data: &mut [f32], rows: usize, cols: usize) {
    let body = cols / 8 * 8;
    for row in data[..rows * cols].chunks_exact_mut(cols) {
        let p = row.as_mut_ptr();
        let mut vmax = Acc8::<V>::splat(f32::NEG_INFINITY);
        for c in (0..body).step_by(8) {
            unsafe { vmax.step(p.add(c), |a, x| a.max_ps(x)) };
        }
        let mut max_v = hmax8(&vmax.lanes());
        for &v in &row[body..] {
            if v > max_v {
                max_v = v;
            }
        }
        let mx = V::splat(max_v);
        by_lanes::<V>(&mut row[..body], |x| exp_lane(x.sub(mx)), |x| x);
        let mut vsum = Acc8::<V>::splat(0.0);
        for c in (0..body).step_by(8) {
            unsafe { vsum.step(p.add(c), |a, e| a.add(e)) };
        }
        let mut sum_v = hsum8(&vsum.lanes());
        for v in &mut row[body..] {
            let e = (*v - max_v).exp();
            *v = e;
            sum_v += e;
        }
        let inv = 1.0 / sum_v;
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

/// `layer_norm_row_avx2`.
pub(crate) fn layer_norm_row(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    output: &mut [f32],
    h: usize,
    eps: f32,
) {
    layer_norm_row_with::<Wide>(input, gamma, beta, output, h, eps);
}

fn layer_norm_row_with<V: Lanes>(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    output: &mut [f32],
    h: usize,
    eps: f32,
) {
    let inv_hf = 1.0 / h as f32;
    let body = h / 8 * 8;
    let ip = input.as_ptr();
    let mut vsum = Acc8::<V>::splat(0.0);
    for c in (0..body).step_by(8) {
        unsafe { vsum.step(ip.add(c), |a, x| a.add(x)) };
    }
    let mut sum = hsum8(&vsum.lanes());
    for &v in &input[body..h] {
        sum += v;
    }
    let mean = sum * inv_hf;
    let vm = V::splat(mean);
    let mut vdev = Acc8::<V>::splat(0.0);
    for c in (0..body).step_by(8) {
        unsafe {
            vdev.step(ip.add(c), |a, x| {
                let d = x.sub(vm);
                d.fmadd(d, a)
            })
        };
    }
    let mut dev = hsum8(&vdev.lanes());
    for &v in &input[body..h] {
        let d = v - mean;
        dev += d * d;
    }
    let var = (dev * inv_hf).max(0.0);
    let inv = 1.0 / (var + eps).sqrt();
    let vi = V::splat(inv);
    let mut i = 0;
    while i < body {
        // SAFETY: `i + W <= body <= h` for all four slices.
        unsafe {
            let norm = V::load(ip.add(i)).sub(vm).mul(vi);
            norm.fmadd(
                V::load(gamma.as_ptr().add(i)),
                V::load(beta.as_ptr().add(i)),
            )
            .store(output.as_mut_ptr().add(i));
        }
        i += V::W;
    }
    for i in body..h {
        output[i] = (input[i] - mean) * inv * gamma[i] + beta[i];
    }
}

#[cfg(all(test, any(target_arch = "aarch64", target_arch = "x86_64")))]
mod native_fma_tests {
    /// [`super::fmaf`] against the hardware fused multiply-add, on random
    /// bit patterns and on constructed double-rounding cases in the normal and
    /// the subnormal range (where wasm's software `fmaf` is wrong).
    #[test]
    fn fmaf_is_the_hardware_fma() {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut r = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 32) as u32
        };
        let mut bad = Vec::new();
        let mut check = |a: f32, b: f32, c: f32| {
            let (got, want) = (super::fmaf(a, b, c), a.mul_add(b, c));
            if got.to_bits() != want.to_bits() && !(got.is_nan() && want.is_nan()) {
                bad.push((a, b, c, got, want));
            }
        };
        let (up, dn) = (1.0 + f32::EPSILON, 1.0 - f32::EPSILON);
        for i in 0..2_000_000u32 {
            // normal: `c ± h(1 - 2^-46)`, `h` half an ulp of `c`
            let c = f32::from_bits((r() & 0x007f_ffff) | ((1 + r() % 253) << 23));
            let h = f32::from_bits(c.to_bits() & 0x7f80_0000) * f32::EPSILON * 0.5;
            let sg = if i & 1 == 0 { 1.0 } else { -1.0 };
            check(sg * up, h * dn, c);
            check(sg * up, h * dn, -c);
            // subnormal: `c ± 2^-150(1 - 2^-46)`
            let c = f32::from_bits(1 + r() % 0x0080_0000);
            check(sg * up * 2f32.powi(-75), dn * 2f32.powi(-75), c);
            check(sg * up * 2f32.powi(-75), dn * 2f32.powi(-75), -c);
            // random patterns, and products near the subnormal range
            check(
                f32::from_bits(r()),
                f32::from_bits(r()),
                f32::from_bits(r()),
            );
            let t = |x: u32| f32::from_bits((x & 0x807f_ffff) | ((40 + x % 60) << 23));
            check(t(r()), t(r()), f32::from_bits(r() & 0x80ff_ffff));
        }
        assert!(
            bad.is_empty(),
            "{} mismatches, first {:?}",
            bad.len(),
            bad[0]
        );
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    //! Each mirror against its AVX2 kernel, bit for bit, on lengths that leave
    //! every tail size and on inputs spanning the kernels' ranges (including
    //! the exp clamp, signed zeros and subnormals).

    fn avx2() -> bool {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }

    fn inputs(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        (0..n)
            .map(|i| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                match i % 61 {
                    0 => 0.0,
                    1 => -0.0,
                    2 => 1e-40,
                    3 => -200.0,
                    4 => 200.0,
                    _ => (((s >> 40) as f32 / 8_388_608.0) - 1.0) * scale,
                }
            })
            .collect()
    }

    fn same(what: &str, a: &[f32], b: &[f32]) {
        let bad: Vec<usize> = (0..a.len())
            .filter(|&i| a[i].to_bits() != b[i].to_bits())
            .collect();
        assert!(
            bad.is_empty(),
            "{what}: {} of {} differ, first at {} ({} vs {})",
            bad.len(),
            a.len(),
            bad[0],
            a[bad[0]],
            b[bad[0]]
        );
    }

    #[test]
    fn elementwise_mirrors_are_the_avx2_kernels() {
        if !avx2() {
            return;
        }
        for len in (0..40).chain([1000, 4099]) {
            for (seed, scale) in [(1u64, 6.0f32), (2, 40.0), (3, 0.01)] {
                let x = inputs(len, seed + len as u64, scale);

                let (mut a, mut b) = (x.clone(), x.clone());
                unsafe { crate::kernels::gelu_inplace_avx2(&mut a) };
                super::gelu_inplace(&mut b);
                same("gelu", &a, &b);

                let (mut a, mut b) = (x.clone(), x.clone());
                unsafe { crate::kernels::silu_inplace_avx2(&mut a) };
                super::silu_inplace(&mut b);
                same("silu", &a, &b);

                let (mut a, mut b) = (vec![0f32; len], vec![0f32; len]);
                unsafe { crate::vmath::vvexpf_avx2(&mut a, &x) };
                super::vvexpf(&mut b, &x, crate::vmath::acc_expf);
                same("exp", &a, &b);

                let (mut a, mut b) = (vec![0f32; len], vec![0f32; len]);
                unsafe { crate::vmath::vvtanhf_avx2(&mut a, &x) };
                super::vvtanhf(&mut b, &x, crate::vmath::acc_tanhf);
                same("tanh", &a, &b);
            }
        }
    }

    #[test]
    fn row_mirrors_are_the_avx2_kernels() {
        if !avx2() {
            return;
        }
        for n in (1..40).chain([768, 1001]) {
            let m = 3;
            for (seed, scale) in [(5u64, 6.0f32), (6, 300.0)] {
                let x = inputs(m * n, seed + n as u64, scale);
                let bias = inputs(n, seed + 99, 1.0);

                let (mut a, mut b) = (x.clone(), x.clone());
                unsafe { crate::kernels::bias_gelu_avx2(&mut a, &bias, m, n) };
                super::bias_gelu(&mut b, &bias, m, n);
                same("bias_gelu", &a, &b);

                let (mut a, mut b) = (x.clone(), x.clone());
                unsafe { crate::kernels::softmax_rows_avx2(&mut a, m, n) };
                super::softmax_rows(&mut b, m, n);
                same("softmax", &a, &b);

                let (g, be) = (inputs(n, seed + 7, 2.0), inputs(n, seed + 8, 2.0));
                let (mut a, mut b) = (vec![0f32; n], vec![0f32; n]);
                for r in 0..m {
                    let row = &x[r * n..(r + 1) * n];
                    unsafe { crate::kernels::layer_norm_row_avx2(row, &g, &be, &mut a, n, 1e-5) };
                    super::layer_norm_row(row, &g, &be, &mut b, n, 1e-5);
                    same("layer_norm", &a, &b);
                }
            }
        }
    }
}

#[cfg(all(test, target_arch = "wasm32", target_feature = "simd128"))]
mod simd_tests {
    //! The `v128` instantiation against the `f32` one, bit for bit, and its
    //! fused multiply-add against [`super::fmaf`] (which `native_fma_tests`
    //! checks against hardware FMA).
    use super::Lanes;
    use core::arch::wasm32::*;

    fn rng(seed: u64) -> impl FnMut() -> u32 {
        let mut s = seed | 1;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 32) as u32
        }
    }

    /// `fma_f32x4`, checked to agree with the `X2` fused multiply-add with
    /// the operands in either half (the other half benign, so each half's
    /// own test decides).
    fn fma4(a: [f32; 4], b: [f32; 4], c: [f32; 4]) -> [f32; 4] {
        let v = |x: [f32; 4]| f32x4(x[0], x[1], x[2], x[3]);
        let r = super::simd::fma_f32x4(v(a), v(b), v(c));
        let mut out = [0f32; 4];
        unsafe { r.store(out.as_mut_ptr()) };
        for half in 0..2 {
            let x2 = |x: [f32; 4], benign: f32| unsafe {
                let mut t = [benign; 8];
                v(x).store(t.as_mut_ptr().add(4 * half));
                super::simd::X2::load(t.as_ptr())
            };
            let mut wide = [0f32; 8];
            unsafe {
                x2(a, 1.0)
                    .fmadd(x2(b, 1.0), x2(c, 0.0))
                    .store(wide.as_mut_ptr())
            };
            for l in 0..4 {
                let w = wide[4 * half + l];
                assert!(
                    w.to_bits() == out[l].to_bits() || (w.is_nan() && out[l].is_nan()),
                    "X2 fma half {half} lane {l}: {w} vs {}",
                    out[l]
                );
            }
        }
        out
    }

    #[test]
    fn simd128_fma_is_correctly_rounded() {
        let mut r = rng(0x9e37_79b9_7f4a_7c15);
        let mut checked = 0u64;
        let mut check = |a: [f32; 4], b: [f32; 4], c: [f32; 4]| {
            let got = fma4(a, b, c);
            for l in 0..4 {
                let want = super::fmaf(a[l], b[l], c[l]);
                assert!(
                    got[l].to_bits() == want.to_bits() || (got[l].is_nan() && want.is_nan()),
                    "fma({:e}, {:e}, {:e}) = {:e}, want {:e}",
                    a[l],
                    b[l],
                    c[l],
                    got[l],
                    want
                );
            }
            checked += 4;
        };
        // Double rounding: `p = (1 + 2^-23)(1 - 2^-23)·h = h·(1 - 2^-46)` with
        // `h` half an ulp of `c`, so the double sum `c ± p` rounds onto the
        // float midpoint next to `c` while the exact value is just on `c`'s
        // side of it; ties-to-even then goes the wrong way for every odd `c`.
        let (up, dn) = (1.0 + f32::EPSILON, 1.0 - f32::EPSILON);
        let mut hard = 0u32;
        for i in 0..200_000u32 {
            let c = f32::from_bits((r() & 0x007f_ffff) | ((60 + r() % 140) << 23));
            let h = f32::from_bits(c.to_bits() & 0x7f80_0000) * f32::EPSILON * 0.5;
            let (a, b, cc) = ([up, -up, up, -up], [h * dn; 4], [c, c, -c, -c]);
            if i < 4 {
                // the construction is hard: the double sum is a float midpoint
                for l in 0..4 {
                    let s = (a[l] as f64 * b[l] as f64 + cc[l] as f64).to_bits();
                    assert_eq!(s & 0x1fff_ffff, 0x1000_0000);
                }
            }
            hard += (0..4)
                .filter(|&l| {
                    ((a[l] as f64 * b[l] as f64 + cc[l] as f64) as f32).to_bits()
                        != super::fmaf(a[l], b[l], cc[l]).to_bits()
                })
                .count() as u32;
            check(a, b, cc);
        }
        // and the same in the float subnormal range (`h = 2^-150`).
        for _ in 0..200_000u32 {
            let c = f32::from_bits(1 + r() % 0x007f_fffe);
            let (a, b) = (up * 2f32.powi(-75), dn * 2f32.powi(-75));
            let (a, b, cc) = ([a, -a, a, -a], [b; 4], [c, c, -c, -c]);
            hard += (0..4)
                .filter(|&l| {
                    ((a[l] as f64 * b[l] as f64 + cc[l] as f64) as f32).to_bits()
                        != super::fmaf(a[l], b[l], cc[l]).to_bits()
                })
                .count() as u32;
            check(a, b, cc);
        }
        // a plain double sum is wrong on a quarter of these (400488 of 1.6M)
        assert!(hard > 300_000, "only {hard} hard cases");
        // Random bit patterns (all classes), and ordinary magnitudes.
        for i in 0..2_000_000u32 {
            let f = |x: u32| {
                if i % 2 == 0 {
                    f32::from_bits(x)
                } else {
                    f32::from_bits((x & 0x807f_ffff) | ((100 + x % 56) << 23))
                }
            };
            check(
                [f(r()), f(r()), f(r()), f(r())],
                [f(r()), f(r()), f(r()), f(r())],
                [f(r()), f(r()), f(r()), f(r())],
            );
        }
        assert!(checked > 8_000_000);
    }

    fn same(what: &str, a: &[f32], b: &[f32]) {
        for i in 0..a.len() {
            assert!(
                a[i].to_bits() == b[i].to_bits() || (a[i].is_nan() && b[i].is_nan()),
                "{what} at {i}: {} vs {}",
                a[i],
                b[i]
            );
        }
    }

    /// Every mirror with lanes `V` against the scalar `f32` one.
    fn mirrors_match<V: Lanes>(tag: &str) {
        let mut r = rng(7);
        let mut x: Vec<f32> = (0..4096)
            .map(|_| ((r() >> 8) as f32 / 8_388_608.0 - 1.0) * 12.0)
            .collect();
        x[5] = 0.0;
        x[6] = -0.0;
        x[7] = 1e-40;
        x[8] = -200.0;
        x[9] = 200.0;
        for len in [0usize, 7, 8, 9, 16, 24, 64, 4095, 4096] {
            let x = &x[..len];
            let run = |f: fn(V) -> V, g: fn(f32) -> f32, name: &str| {
                let (mut a, mut b) = (x.to_vec(), x.to_vec());
                super::by_lanes::<V>(&mut a, f, |v| v);
                super::by_lanes::<f32>(&mut b, g, |v| v);
                same(&format!("{tag} {name}"), &a, &b);
            };
            run(super::gelu_lane::<V>, super::gelu_lane::<f32>, "gelu");
            run(super::silu_lane::<V>, super::silu_lane::<f32>, "silu");
            run(super::tanh_lane::<V>, super::tanh_lane::<f32>, "tanh");
            run(super::exp_lane::<V>, super::exp_lane::<f32>, "exp");
        }
        for (rows, cols) in [
            (3usize, 1usize),
            (3, 8),
            (2, 17),
            (2, 40),
            (4, 768),
            (2, 771),
        ] {
            let x = &x[..rows * cols];
            let (mut a, mut b) = (x.to_vec(), x.to_vec());
            super::softmax_rows_with::<V>(&mut a, rows, cols);
            super::softmax_rows_with::<f32>(&mut b, rows, cols);
            same(&format!("{tag} softmax"), &a, &b);
            let (g, be) = (&x[..cols], &x[x.len() - cols..]);
            for row in x.chunks_exact(cols) {
                let (mut a, mut b) = (vec![0f32; cols], vec![0f32; cols]);
                super::layer_norm_row_with::<V>(row, g, be, &mut a, cols, 1e-5);
                super::layer_norm_row_with::<f32>(row, g, be, &mut b, cols, 1e-5);
                same(&format!("{tag} layer_norm"), &a, &b);
            }
        }
    }

    #[test]
    fn simd128_mirrors_are_the_scalar_mirrors() {
        mirrors_match::<v128>("v128");
        mirrors_match::<super::simd::X2>("x2");
    }
}
