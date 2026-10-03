// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Vector transcendentals — RLX equivalents of Accelerate vForce
//! `vvexpf` / `vvtanhf` / `vvrecf` / `vvlogf` / `vvsqrtf` / `vvrsqrtf`.
//!
//! Host API (this module) is shared across GPU backends via
//! `rlx_<backend>::vmath` re-exports. Device kernels:
//! - CUDA / ROCm / wgpu: `unary` op ids 3=exp, 2=tanh, 17=recip
//! - Metal: `exp_inplace` / `tanh_inplace` / `rec_inplace`
//! - Vulkan / oneAPI: Activation-order unary ids + 17=recip
//!
//! | API | Behavior |
//! |-----|----------|
//! | [`vvexpf`] / [`vvtanhf`] / [`vvlogf`] | Accurate: public Accelerate vForce on Apple, libm elsewhere |
//! | [`vvrecf`] | NEON on aarch64, vectorized/libm fallback elsewhere |
//! | [`vvsqrtf`] / [`vvrsqrtf`] | Hardware SIMD by default; Apple vForce when `RLX_VMATH_ACCURATE=1` |
//! | [`vvexpf_fast`] / [`vvtanhf_fast`] / [`vvlogf_fast`] | Portable SIMD polynomial (~2e-7 rel) |
//! | [`vvexpf_hot`] / [`vvtanhf_hot`] / [`vvlogf_hot`] | SIMD `*_fast` by default; accurate path when `RLX_VMATH_ACCURATE=1` |
//!
//! Callers depend on this module — never link Accelerate vForce directly.

/// Select accurate CPU vector math instead of the default SIMD exp/tanh path.
#[inline]
pub fn vmath_accurate() -> bool {
    rlx_ir::env::flag("RLX_VMATH_ACCURATE")
}

/// `y[i] = exp(x[i])`. Lengths must match. Aliasing OK.
pub fn vvexpf(y: &mut [f32], x: &[f32]) {
    assert_eq!(y.len(), x.len());
    #[cfg(target_vendor = "apple")]
    {
        let n = y.len() as i32;
        unsafe {
            accelerate::vvexpf(y.as_mut_ptr(), x.as_ptr(), &n);
        }
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        for (yi, &xi) in y.iter_mut().zip(x.iter()) {
            *yi = xi.exp();
        }
    }
}

/// In-place `y[i] = exp(y[i])`.
#[inline]
pub fn vvexpf_inplace(y: &mut [f32]) {
    #[cfg(target_vendor = "apple")]
    {
        let n = y.len() as i32;
        unsafe {
            accelerate::vvexpf(y.as_mut_ptr(), y.as_ptr(), &n);
        }
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        for yi in y.iter_mut() {
            *yi = yi.exp();
        }
    }
}

/// Scalar mirror of [`crate::kernels::avx2_exp8`] — same Cody–Waite ln2 split,
/// same degree-6 Taylor series, same `2^n` exponent-bias trick, so the
/// vectorized and portable arms compute the *same* function.
///
/// Written in plain Rust on purpose. It has no libm call and no branches, so
/// LLVM auto-vectorizes a loop over it to packed SSE2 — which is in the
/// `x86_64` baseline, needs no `#[target_feature]` and no CPUID check, and so
/// lands on every x86 CPU including the pre-AVX Atoms. The same applies to
/// aarch64's NEON-less fallback, wasm and riscv.
#[inline(always)]
#[allow(clippy::excessive_precision)]
// see `avx2_exp8`: literals are deliberate
// Dead on aarch64 by design: NEON is baseline there, so `vvexpf_fast`
// always takes `vvexpf_neon` and never reaches this. Kept compiled (rather
// than cfg-d out) so the accuracy test below still runs on an Apple dev
// machine. Live on x86_64, wasm and riscv.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
pub(crate) fn exp_poly(x: f32) -> f32 {
    // Clamp first: the exponent-bias trick below has no overflow handling.
    //
    // `clamp`, not `x.max(-87.3).min(88.7)`. Those differ on NaN and it
    // matters: `f32::max` returns the non-NaN operand, so the max/min pair
    // turns NaN into -87.3 and this function returns ~1e-38 — a NaN silently
    // becoming zero, which is both wrong (libm gives NaN) and the exact thing
    // `RLX_DEBUG_NANS` exists to catch. `clamp` propagates NaN.
    //
    // KNOWN GAP: `avx2_exp8` builds its clamp from `_mm256_max_ps` /
    // `_mm256_min_ps`, which carry the same NaN-swallowing semantics, so the
    // AVX2 arm still returns ~0 for a NaN input where this one returns NaN.
    // Fixing that costs a compare + blend per call in the AVX2 kernel; left
    // as a deliberate, documented difference rather than copying the bug.
    let x = x.clamp(-87.3, 88.7);
    // n = round(x / ln2) to nearest-even, matching AVX2's
    // `_MM_FROUND_TO_NEAREST_INT`.
    //
    // Deliberately NOT `round_ties_even()`: that lowers to `roundps`, which is
    // SSE4.1 — absent on Bonnell and, more importantly, outside the x86_64
    // baseline, so LLVM emits a libm call instead and the loop stops
    // vectorizing. That would make this slower than the `expf` it replaces.
    // Adding and subtracting 1.5·2^23 instead forces the mantissa to drop its
    // fractional bits in the FP unit itself, under the default rounding mode,
    // using only SSE2 `addps`/`subps`. Safe from being folded to a no-op
    // because Rust never enables fast-math, so `(v + C) - C` is not a legal
    // rewrite. Exact for |v| < 2^22, and v = x·log2(e) is within ±128 here.
    const MAGIC: f32 = 12582912.0; // 1.5 * 2^23
    let v = x * std::f32::consts::LOG2_E;
    let n = (v + MAGIC) - MAGIC;
    // r = x − n·ln2_hi − n·ln2_lo
    let r = x - n * 0.693145751953125 - n * 1.428606765330187e-6;
    let mut p = 0.001388888888888889f32;
    p = p * r + 0.008333333333333333;
    p = p * r + 0.041666666666666664;
    p = p * r + 0.16666666666666666;
    p = p * r + 0.5;
    p = p * r + 1.0;
    p = p * r + 1.0;
    // 2^n by adding n to the f32 exponent field.
    f32::from_bits(p.to_bits().wrapping_add((n as i32 as u32) << 23))
}

/// Portable `ln(1 + u)` for `u >= 0`, accurate across the whole range.
///
/// `log_poly(1.0 + u)` alone is wrong twice over: once `u < 6e-8` the sum
/// rounds to exactly 1.0 and the answer collapses to 0, and just above that
/// the bits of `u` lost in the addition show up as relative error (a
/// threshold-and-series version of this measured 3.3e-6 at the crossover).
///
/// Both come from the same place — the rounding of `1 + u` — so correct for it
/// instead of working around it. `s - 1` is exact by Sterbenz for `s` in
/// `[0.5, 2]`, which makes `err` the exact rounding error of the sum, and
/// `ln(s - err) = ln(s) + ln(1 - err/s) ≈ ln(s) - err/s`. In the tail
/// `s == 1.0`, so `err == -u` and the result is exactly `u`. No branch, no
/// threshold, so it still vectorizes.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline(always)]
pub(crate) fn ln_1p_poly(u: f32) -> f32 {
    let s = 1.0 + u;
    let err = (s - 1.0) - u;
    log_poly(s) - err / s
}

/// Portable `softplus(x) = ln(1 + e^x)`, in the numerically stable form
/// `max(x, 0) + ln(1 + e^-|x|)`. Backs the Softplus / Mish / LogSigmoid
/// activations.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline(always)]
pub(crate) fn softplus_poly(x: f32) -> f32 {
    x.max(0.0) + ln_1p_poly(exp_poly(-x.abs()))
}

/// Portable fast `ln` (~1 ULP), the counterpart to [`exp_poly`].
///
/// Mantissa/exponent split plus the classic cephes `logf` minimax on
/// `[sqrt(0.5), sqrt(2))`. Branchless by construction — every special case is
/// a `select`, not an early return — so a loop over it auto-vectorizes to
/// baseline SSE2 / NEON, same as `exp_poly`. Denormals are scaled by 2^24
/// first and corrected afterwards, so the whole positive range is covered.
///
/// Returns `-inf` at 0 and NaN for negative input, matching `f32::ln`.
#[inline(always)]
#[allow(clippy::excessive_precision)] // cephes constants, verbatim
pub(crate) fn log_poly(x: f32) -> f32 {
    const LN2_HI: f32 = 0.693359375;
    const LN2_LO: f32 = -2.12194440e-4;
    // Denormals have no usable exponent field; scale them up and subtract the
    // scale back off at the end. `select`, so the loop still vectorizes.
    let denorm = x > 0.0 && x < f32::MIN_POSITIVE;
    let xs = if denorm { x * 16_777_216.0 } else { x };

    let bits = xs.to_bits();
    let mut e = ((bits >> 23) & 0xff) as i32 - 127;
    // Mantissa forced into [1, 2).
    let mut m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000);
    // Re-center to [sqrt(0.5), sqrt(2)) so the polynomial stays near zero.
    if m > std::f32::consts::SQRT_2 {
        m *= 0.5;
        e += 1;
    }
    let y = m - 1.0;
    let z = y * y;
    let mut p = 7.0376836292e-2f32;
    p = p * y - 1.1514610310e-1;
    p = p * y + 1.1676998740e-1;
    p = p * y - 1.2420140846e-1;
    p = p * y + 1.4249322787e-1;
    p = p * y - 1.6668057665e-1;
    p = p * y + 2.0000714765e-1;
    p = p * y - 2.4999993993e-1;
    p = p * y + 3.3333331174e-1;
    let ef = e as f32;
    // Cody-Waite ln2 split keeps the exponent term from dominating the error.
    let r = (p * y * z - 0.5 * z + y) + ef * LN2_LO + ef * LN2_HI;
    let r = if denorm {
        r - 24.0 * std::f32::consts::LN_2
    } else {
        r
    };
    if x == 0.0 {
        f32::NEG_INFINITY
    } else if x.is_nan() || x < 0.0 {
        // The NaN test is NOT redundant. `x < 0.0` is false for NaN, so
        // without it a NaN falls through to `r` — and `r` is FINITE there,
        // computed from the NaN payload's mantissa bits reinterpreted as a
        // number in [1, 2). A NaN turning into plausible garbage is worse than
        // one turning into zero, and this is the kind of thing
        // `RLX_DEBUG_NANS` can never see.
        f32::NAN
    } else {
        r
    }
}

/// Portable fast `tanh`, as `(e^2x − 1) / (e^2x + 1)` over [`exp_poly`] — the
/// same formulation as [`vvtanhf_avx2`] and the NEON arm, so all three agree.
///
/// Saturates correctly at both ends because `exp_poly` clamps: large positive
/// x gives (3.4e38 − 1)/(3.4e38 + 1) = 1, large negative gives −1.
///
/// Shares that formulation's one weakness: for |x| below ~1e-4, `e^2x`
/// rounds to 1.0 in f32 and the numerator cancels to zero, so the *relative*
/// error near the origin is poor (absolute error stays under 1e-7). That is
/// pre-existing behaviour for every `_fast`/`_hot` tanh here, not new.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline(always)]
pub(crate) fn tanh_poly(x: f32) -> f32 {
    let e = exp_poly(2.0 * x);
    (e - 1.0) / (e + 1.0)
}

/// Portable fast `exp` over a slice. Auto-vectorizes; see [`exp_poly`].
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn vvexpf_poly(y: &mut [f32], x: &[f32]) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi = exp_poly(xi);
    }
}

/// Fast SIMD `exp` (~2e-7 relative). Prefer [`vvexpf`] when ULPs matter.
pub fn vvexpf_fast(y: &mut [f32], x: &[f32]) {
    assert_eq!(y.len(), x.len());
    #[cfg(target_arch = "aarch64")]
    {
        vvexpf_neon(y, x);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                unsafe { vvexpf_avx2(y, x) };
                return;
            }
        }
        // Was `vvexpf`, i.e. a libm `expf` CALL PER ELEMENT — which is both
        // slow and unvectorizable, so a pre-AVX2 x86 (every Atom through
        // Tremont) got no SIMD at all on the hottest activation primitive.
        vvexpf_poly(y, x);
    }
}

/// In-place fast SIMD `exp`.
#[inline]
pub fn vvexpf_fast_inplace(y: &mut [f32]) {
    let x = unsafe { &*(y as *const [f32]) };
    vvexpf_fast(y, x);
}

/// `exp` selected for CPU activation hot paths.
#[inline]
pub fn vvexpf_hot(y: &mut [f32], x: &[f32]) {
    if vmath_accurate() {
        vvexpf(y, x);
    } else {
        vvexpf_fast(y, x);
    }
}

/// In-place `exp` selected for CPU activation hot paths.
#[inline]
pub fn vvexpf_hot_inplace(y: &mut [f32]) {
    if vmath_accurate() {
        vvexpf_inplace(y);
    } else {
        vvexpf_fast_inplace(y);
    }
}

/// Largest |x| for which [`sin_poly`] / [`cos_poly`] stay accurate.
///
/// A trig argument is unbounded, unlike `exp`'s (which saturation bounds), and
/// argument reduction is where f32 trig dies: the error in the reduced
/// argument is `ulp(x)`-scale, and it lands in the result as absolute error.
/// A two-part Cody-Waite split in f32 measured 2.4e-4 at |x| = 7216 for
/// exactly that reason — usable only to about |x| = 32.
///
/// So the reduction below runs in f64 while the polynomial stays f32. `x as
/// f64` is exact, and f64 has ~29 bits of headroom over f32, which pushes the
/// cancellation out past any argument a model will produce. This bound is
/// where f64's own ulp starts to matter; past it, libm's Payne-Hanek
/// reduction is the only correct answer, so the vector entries fall back.
pub(crate) const TRIG_FAST_MAX: f32 = 1.0e6;

/// `(sin(r), cos(r), quadrant)` for `x` reduced onto `[-π/4, π/4]`.
///
/// Branchless so a loop over it vectorizes: the quadrant comes out as an
/// integer the caller turns into selects, not into control flow. The f64
/// reduction halves the lane count for those few ops (2-wide instead of
/// 4-wide on SSE2) and is still far cheaper than a libm call per element.
#[inline(always)]
#[allow(clippy::excessive_precision)] // cephes constants, verbatim
fn sincos_reduce(x: f32) -> (f32, f32, i32) {
    // n = round(x · 2/π) via the magic-number round in f64 (see `exp_poly`:
    // `round_ties_even` would lower to SSE4.1 `roundsd` or a libm call and
    // stop the loop vectorizing). Exact while |n| < 2^51.
    const MAGIC_D: f64 = 6_755_399_441_055_744.0; // 1.5 * 2^52
    const TWO_OVER_PI_D: f64 = std::f64::consts::FRAC_2_PI;
    const PIO2_D: f64 = std::f64::consts::FRAC_PI_2;
    let xd = x as f64; // exact
    let nd = (xd * TWO_OVER_PI_D + MAGIC_D) - MAGIC_D;
    let r = (xd - nd * PIO2_D) as f32;

    let zz = r * r;
    // cephes sinf minimax on [-π/4, π/4]
    let mut sp = -1.9515295891e-4f32;
    sp = sp * zz + 8.3321608736e-3;
    sp = sp * zz - 1.6666654611e-1;
    let s = r + r * zz * sp;
    // cephes cosf minimax on [-π/4, π/4]
    let mut cp = 2.443315711809948e-5f32;
    cp = cp * zz - 1.388731625493765e-3;
    cp = cp * zz + 4.166664568298827e-2;
    let c = 1.0 - 0.5 * zz + zz * zz * cp;

    (s, c, nd as i32)
}

/// Portable fast `sin` for |x| <= [`TRIG_FAST_MAX`]. Branchless.
#[inline(always)]
pub(crate) fn sin_poly(x: f32) -> f32 {
    let (s, c, q) = sincos_reduce(x);
    let v = if q & 1 != 0 { c } else { s };
    if q & 2 != 0 { -v } else { v }
}

/// Portable fast `cos` for |x| <= [`TRIG_FAST_MAX`]. `cos x = sin(x + π/2)`,
/// i.e. the same reduction with the quadrant shifted by one.
#[inline(always)]
pub(crate) fn cos_poly(x: f32) -> f32 {
    let (s, c, q) = sincos_reduce(x);
    let q = q + 1;
    let v = if q & 1 != 0 { c } else { s };
    if q & 2 != 0 { -v } else { v }
}

/// Portable fast `tan` for |x| <= [`TRIG_FAST_MAX`].
///
/// I had dismissed `tan` on the grounds that a polynomial loses all relative
/// accuracy at the poles. That is true of computing it as `sin/cos` — and
/// false of the actual construction, which never forms that ratio. Reduce onto
/// `[-π/4, π/4]`, evaluate `tan(r)` there, and for an odd quadrant return
/// `-1/tan(r)`. Approaching a pole means `r → 0`, where `tan(r) ≈ r` has
/// EXCELLENT relative accuracy, and the reciprocal of an accurate small number
/// is an accurate large one. So the poles are the easy case, not the hard one.
///
/// Exact at the pole too, thanks to the f64 reduction: `f32(π/2)` is not π/2,
/// and the reduction recovers the ~-4.4e-8 difference rather than collapsing
/// `r` to zero and returning infinity.
#[inline(always)]
#[allow(clippy::excessive_precision)] // cephes constants, verbatim
pub(crate) fn tan_poly(x: f32) -> f32 {
    const MAGIC_D: f64 = 6_755_399_441_055_744.0; // 1.5 * 2^52
    let xd = x as f64; // exact
    let nd = (xd * std::f64::consts::FRAC_2_PI + MAGIC_D) - MAGIC_D;
    let r = (xd - nd * std::f64::consts::FRAC_PI_2) as f32;
    let zz = r * r;
    // cephes tanf minimax on [-π/4, π/4]
    let mut p = 9.38540185543e-3f32;
    p = p * zz + 3.11992232697e-3;
    p = p * zz + 2.44301354525e-2;
    p = p * zz + 5.34112807005e-2;
    p = p * zz + 1.33387994085e-1;
    p = p * zz + 3.33331568548e-1;
    let t = r + r * zz * p;
    // Odd quadrant: tan(r + π/2) = -cot(r).
    if (nd as i32) & 1 != 0 { -1.0 / t } else { t }
}

/// Fast `tan` over a slice, range-guarded exactly as [`vvsinf_fast`].
pub fn vvtanf_fast_inplace(y: &mut [f32]) {
    if max_abs4(y) <= TRIG_FAST_MAX {
        for yi in y.iter_mut() {
            *yi = tan_poly(*yi);
        }
    } else {
        for yi in y.iter_mut() {
            *yi = yi.tan();
        }
    }
}

/// Portable `atan`. Needs no range guard — every finite input folds onto
/// `[0, tan(π/8)]`, so there is no large-argument cliff like sin/cos have.
///
/// The fold has TWO stages, which is not optional: cephes' coefficients are a
/// minimax on `[0, tan(π/8)]` only. Applying them across all of `[0, 1]` with
/// a single `1/x` fold — the obvious simplification — measured **2.3e-2**.
#[inline(always)]
#[allow(clippy::excessive_precision)] // cephes constants, verbatim
pub(crate) fn atan_poly(x: f32) -> f32 {
    const TAN_3PI_8: f32 = 2.414_213_6;
    const TAN_PI_8: f32 = 0.414_213_57;
    let sign = if x < 0.0 { -1.0f32 } else { 1.0 };
    let a = x.abs();
    let big = a > TAN_3PI_8;
    let mid = !big && a > TAN_PI_8;
    // Selects, not branches, so the loop still vectorizes.
    let z = if big {
        -1.0 / a
    } else if mid {
        (a - 1.0) / (a + 1.0)
    } else {
        a
    };
    let offset = if big {
        std::f32::consts::FRAC_PI_2
    } else if mid {
        std::f32::consts::FRAC_PI_4
    } else {
        0.0
    };
    let zz = z * z;
    let mut p = 8.05374449538e-2f32;
    p = p * zz - 1.38776856032e-1;
    p = p * zz + 1.99777106478e-1;
    p = p * zz - 3.33329491539e-1;
    sign * (offset + z + z * zz * p)
}

/// Largest |x| in a slice, over four accumulators so it vectorizes (a float
/// reduction is not reassociable, so a plain `fold` stays serial).
#[inline]
fn max_abs4(x: &[f32]) -> f32 {
    let mut acc = [0f32; 4];
    let mut it = x.chunks_exact(4);
    for ch in &mut it {
        for i in 0..4 {
            acc[i] = acc[i].max(ch[i].abs());
        }
    }
    let mut m = acc[0].max(acc[1]).max(acc[2].max(acc[3]));
    for &v in it.remainder() {
        m = m.max(v.abs());
    }
    m
}

/// Fast `sin` over a slice, with a range guard.
///
/// Scans for the largest magnitude first and hands the whole slice to libm if
/// any element is outside `TRIG_FAST_MAX`. That scan is the price of making
/// the fast arm safe to use by default: a trig argument is unbounded, so
/// without it a large input would silently return noise instead of being
/// slower. One extra read pass over data already in cache, against a libm call
/// per element otherwise.
pub fn vvsinf_fast(y: &mut [f32], x: &[f32]) {
    assert_eq!(y.len(), x.len());
    if max_abs4(x) <= TRIG_FAST_MAX {
        for (yi, &xi) in y.iter_mut().zip(x.iter()) {
            *yi = sin_poly(xi);
        }
    } else {
        for (yi, &xi) in y.iter_mut().zip(x.iter()) {
            *yi = xi.sin();
        }
    }
}

/// In-place [`vvsinf_fast`].
pub fn vvsinf_fast_inplace(y: &mut [f32]) {
    if max_abs4(y) <= TRIG_FAST_MAX {
        for yi in y.iter_mut() {
            *yi = sin_poly(*yi);
        }
    } else {
        for yi in y.iter_mut() {
            *yi = yi.sin();
        }
    }
}

/// Fast `cos` over a slice, range-guarded exactly as [`vvsinf_fast`].
pub fn vvcosf_fast(y: &mut [f32], x: &[f32]) {
    assert_eq!(y.len(), x.len());
    if max_abs4(x) <= TRIG_FAST_MAX {
        for (yi, &xi) in y.iter_mut().zip(x.iter()) {
            *yi = cos_poly(xi);
        }
    } else {
        for (yi, &xi) in y.iter_mut().zip(x.iter()) {
            *yi = xi.cos();
        }
    }
}

/// In-place [`vvcosf_fast`].
pub fn vvcosf_fast_inplace(y: &mut [f32]) {
    if max_abs4(y) <= TRIG_FAST_MAX {
        for yi in y.iter_mut() {
            *yi = cos_poly(*yi);
        }
    } else {
        for yi in y.iter_mut() {
            *yi = yi.cos();
        }
    }
}

/// In-place fast `atan`. No range guard needed — `atan_poly` folds `|x| > 1`
/// onto `[0, 1]`, so every finite input is in range.
pub fn vvatanf_fast_inplace(y: &mut [f32]) {
    for yi in y.iter_mut() {
        *yi = atan_poly(*yi);
    }
}

/// Fast `ln` (~1 ULP). Prefer [`vvlogf`] when ULPs matter.
///
/// Completes the `_fast`/`_hot` pair that `exp` and `tanh` have had. Note that
/// `Activation::Log` and log-softmax deliberately stay on the accurate
/// [`vvlogf`]: they feed losses, where the extra ULP is worth more than the
/// throughput. This is for callers that have made the opposite trade.
pub fn vvlogf_fast(y: &mut [f32], x: &[f32]) {
    assert_eq!(y.len(), x.len());
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi = log_poly(xi);
    }
}

/// In-place fast `ln`.
#[inline]
pub fn vvlogf_fast_inplace(y: &mut [f32]) {
    for yi in y.iter_mut() {
        *yi = log_poly(*yi);
    }
}

/// `ln` selected for CPU activation hot paths: fast by default, accurate under
/// `RLX_VMATH_ACCURATE=1`.
#[inline]
pub fn vvlogf_hot(y: &mut [f32], x: &[f32]) {
    if vmath_accurate() {
        vvlogf(y, x);
    } else {
        vvlogf_fast(y, x);
    }
}

/// In-place `ln` for CPU activation hot paths.
#[inline]
pub fn vvlogf_hot_inplace(y: &mut [f32]) {
    if vmath_accurate() {
        vvlogf_inplace(y);
    } else {
        vvlogf_fast_inplace(y);
    }
}

/// `y[i] = tanh(x[i])`. Lengths must match. Aliasing OK.
pub fn vvtanhf(y: &mut [f32], x: &[f32]) {
    assert_eq!(y.len(), x.len());
    #[cfg(target_vendor = "apple")]
    {
        let n = y.len() as i32;
        unsafe {
            accelerate::vvtanhf(y.as_mut_ptr(), x.as_ptr(), &n);
        }
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        for (yi, &xi) in y.iter_mut().zip(x.iter()) {
            *yi = xi.tanh();
        }
    }
}

/// In-place `y[i] = tanh(y[i])`.
#[inline]
pub fn vvtanhf_inplace(y: &mut [f32]) {
    #[cfg(target_vendor = "apple")]
    {
        let n = y.len() as i32;
        unsafe {
            accelerate::vvtanhf(y.as_mut_ptr(), y.as_ptr(), &n);
        }
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        for yi in y.iter_mut() {
            *yi = yi.tanh();
        }
    }
}

/// Fast SIMD `tanh` via `exp(2x)` poly. Prefer [`vvtanhf`] when ULPs matter.
pub fn vvtanhf_fast(y: &mut [f32], x: &[f32]) {
    assert_eq!(y.len(), x.len());
    #[cfg(target_arch = "aarch64")]
    {
        vvtanhf_neon(y, x);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                unsafe { vvtanhf_avx2(y, x) };
                return;
            }
        }
        // Same `(e^2x − 1) / (e^2x + 1)` form as `vvtanhf_avx2`, over the
        // auto-vectorizing `exp_poly` instead of a libm `tanhf` per element.
        for (yi, &xi) in y.iter_mut().zip(x.iter()) {
            *yi = tanh_poly(xi);
        }
    }
}

/// In-place fast SIMD `tanh`.
#[inline]
pub fn vvtanhf_fast_inplace(y: &mut [f32]) {
    let x = unsafe { &*(y as *const [f32]) };
    vvtanhf_fast(y, x);
}

/// `tanh` selected for CPU activation hot paths.
#[inline]
pub fn vvtanhf_hot(y: &mut [f32], x: &[f32]) {
    if vmath_accurate() {
        vvtanhf(y, x);
    } else {
        vvtanhf_fast(y, x);
    }
}

/// In-place `tanh` selected for CPU activation hot paths.
#[inline]
pub fn vvtanhf_hot_inplace(y: &mut [f32]) {
    if vmath_accurate() {
        vvtanhf_inplace(y);
    } else {
        vvtanhf_fast_inplace(y);
    }
}

/// `y[i] = 1 / x[i]`. Lengths must match. Aliasing OK.
pub fn vvrecf(y: &mut [f32], x: &[f32]) {
    assert_eq!(y.len(), x.len());
    #[cfg(target_arch = "aarch64")]
    {
        vvrecf_neon(y, x);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                unsafe { vvrecf_avx2(y, x) };
                return;
            }
        }
        for (yi, &xi) in y.iter_mut().zip(x.iter()) {
            *yi = 1.0 / xi;
        }
    }
}

/// In-place `y[i] = 1 / y[i]`.
#[inline]
pub fn vvrecf_inplace(y: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        let x = unsafe { &*(y as *const [f32]) };
        vvrecf_neon(y, x);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let x = unsafe { &*(y as *const [f32]) };
        vvrecf(y, x);
    }
}

/// `y[i] = ln(x[i])`. Lengths must match. Aliasing OK.
pub fn vvlogf(y: &mut [f32], x: &[f32]) {
    assert_eq!(y.len(), x.len());
    #[cfg(target_vendor = "apple")]
    {
        let n = y.len() as i32;
        unsafe {
            accelerate::vvlogf(y.as_mut_ptr(), x.as_ptr(), &n);
        }
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        for (yi, &xi) in y.iter_mut().zip(x.iter()) {
            *yi = xi.ln();
        }
    }
}

/// In-place `y[i] = ln(y[i])`.
#[inline]
pub fn vvlogf_inplace(y: &mut [f32]) {
    #[cfg(target_vendor = "apple")]
    {
        let n = y.len() as i32;
        unsafe {
            accelerate::vvlogf(y.as_mut_ptr(), y.as_ptr(), &n);
        }
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        for yi in y.iter_mut() {
            *yi = yi.ln();
        }
    }
}

/// `y[i] = sqrt(x[i])`. Lengths must match. Aliasing OK.
pub fn vvsqrtf(y: &mut [f32], x: &[f32]) {
    assert_eq!(y.len(), x.len());
    #[cfg(target_vendor = "apple")]
    if vmath_accurate() {
        let n = y.len() as i32;
        unsafe {
            accelerate::vvsqrtf(y.as_mut_ptr(), x.as_ptr(), &n);
        }
        return;
    }
    vvsqrtf_simd(y, x);
}

/// In-place `y[i] = sqrt(y[i])`.
#[inline]
pub fn vvsqrtf_inplace(y: &mut [f32]) {
    #[cfg(target_vendor = "apple")]
    if vmath_accurate() {
        let n = y.len() as i32;
        unsafe {
            accelerate::vvsqrtf(y.as_mut_ptr(), y.as_ptr(), &n);
        }
        return;
    }
    let x = unsafe { &*(y as *const [f32]) };
    vvsqrtf_simd(y, x);
}

/// `y[i] = 1 / sqrt(x[i])`. Lengths must match. Aliasing OK.
pub fn vvrsqrtf(y: &mut [f32], x: &[f32]) {
    assert_eq!(y.len(), x.len());
    #[cfg(target_vendor = "apple")]
    if vmath_accurate() {
        let n = y.len() as i32;
        unsafe {
            accelerate::vvrsqrtf(y.as_mut_ptr(), x.as_ptr(), &n);
        }
        return;
    }
    vvrsqrtf_simd(y, x);
}

/// In-place `y[i] = 1 / sqrt(y[i])`.
#[inline]
pub fn vvrsqrtf_inplace(y: &mut [f32]) {
    #[cfg(target_vendor = "apple")]
    if vmath_accurate() {
        let n = y.len() as i32;
        unsafe {
            accelerate::vvrsqrtf(y.as_mut_ptr(), y.as_ptr(), &n);
        }
        return;
    }
    let x = unsafe { &*(y as *const [f32]) };
    vvrsqrtf_simd(y, x);
}

/// `y[i] = 1 / (1 + exp(-x[i]))` (logistic sigmoid).
pub fn vvsigmoidf(y: &mut [f32], x: &[f32]) {
    assert_eq!(y.len(), x.len());
    let mut tmp = vec![0.0f32; x.len()];
    for (t, &xi) in tmp.iter_mut().zip(x.iter()) {
        *t = -xi;
    }
    vvexpf_hot_inplace(&mut tmp);
    for t in tmp.iter_mut() {
        *t += 1.0;
    }
    vvrecf(y, &tmp);
}

#[cfg(target_vendor = "apple")]
mod accelerate {
    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        pub fn vvexpf(y: *mut f32, x: *const f32, n: *const i32);
        pub fn vvtanhf(y: *mut f32, x: *const f32, n: *const i32);
        pub fn vvlogf(y: *mut f32, x: *const f32, n: *const i32);
        pub fn vvsqrtf(y: *mut f32, x: *const f32, n: *const i32);
        pub fn vvrsqrtf(y: *mut f32, x: *const f32, n: *const i32);
    }
}

fn vvsqrtf_simd(y: &mut [f32], x: &[f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        vvsqrtf_neon(y, x);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx") {
            unsafe { vvsqrtf_avx(y, x) };
            return;
        }
        for (yi, &xi) in y.iter_mut().zip(x.iter()) {
            *yi = xi.sqrt();
        }
    }
}

fn vvrsqrtf_simd(y: &mut [f32], x: &[f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        vvrsqrtf_neon(y, x);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx") {
            unsafe { vvrsqrtf_avx(y, x) };
            return;
        }
        for (yi, &xi) in y.iter_mut().zip(x.iter()) {
            *yi = 1.0 / xi.sqrt();
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn vvexpf_neon(y: &mut [f32], x: &[f32]) {
    use crate::kernels::neon_exp4;
    use std::arch::aarch64::*;
    let n = x.len();
    let chunks = n / 4;
    unsafe {
        for c in 0..chunks {
            let off = c * 4;
            let v = vld1q_f32(x.as_ptr().add(off));
            vst1q_f32(y.as_mut_ptr().add(off), neon_exp4(v));
        }
    }
    for i in chunks * 4..n {
        y[i] = x[i].exp();
    }
}

#[cfg(target_arch = "aarch64")]
fn vvtanhf_neon(y: &mut [f32], x: &[f32]) {
    use crate::kernels::neon_exp4;
    use std::arch::aarch64::*;
    let n = x.len();
    let chunks = n / 4;
    unsafe {
        let two = vdupq_n_f32(2.0);
        let one = vdupq_n_f32(1.0);
        for c in 0..chunks {
            let off = c * 4;
            let v = vld1q_f32(x.as_ptr().add(off));
            // tanh(x) = (e^{2x} - 1) / (e^{2x} + 1)
            let e = neon_exp4(vmulq_f32(v, two));
            let num = vsubq_f32(e, one);
            let den = vaddq_f32(e, one);
            vst1q_f32(y.as_mut_ptr().add(off), vdivq_f32(num, den));
        }
    }
    for i in chunks * 4..n {
        y[i] = x[i].tanh();
    }
}

#[cfg(target_arch = "aarch64")]
fn vvrecf_neon(y: &mut [f32], x: &[f32]) {
    use std::arch::aarch64::*;
    let n = x.len();
    let chunks = n / 4;
    unsafe {
        let one = vdupq_n_f32(1.0);
        for c in 0..chunks {
            let off = c * 4;
            let v = vld1q_f32(x.as_ptr().add(off));
            vst1q_f32(y.as_mut_ptr().add(off), vdivq_f32(one, v));
        }
    }
    for i in chunks * 4..n {
        y[i] = 1.0 / x[i];
    }
}

#[cfg(target_arch = "aarch64")]
fn vvsqrtf_neon(y: &mut [f32], x: &[f32]) {
    use std::arch::aarch64::*;
    let n = x.len();
    let chunks = n / 4;
    unsafe {
        for c in 0..chunks {
            let off = c * 4;
            let v = vld1q_f32(x.as_ptr().add(off));
            vst1q_f32(y.as_mut_ptr().add(off), vsqrtq_f32(v));
        }
    }
    for i in chunks * 4..n {
        y[i] = x[i].sqrt();
    }
}

#[cfg(target_arch = "aarch64")]
fn vvrsqrtf_neon(y: &mut [f32], x: &[f32]) {
    use std::arch::aarch64::*;
    let n = x.len();
    let chunks = n / 4;
    unsafe {
        let one = vdupq_n_f32(1.0);
        for c in 0..chunks {
            let off = c * 4;
            let v = vld1q_f32(x.as_ptr().add(off));
            vst1q_f32(y.as_mut_ptr().add(off), vdivq_f32(one, vsqrtq_f32(v)));
        }
    }
    for i in chunks * 4..n {
        y[i] = 1.0 / x[i].sqrt();
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vvexpf_avx2(y: &mut [f32], x: &[f32]) {
    use crate::kernels::avx2_exp8;
    use std::arch::x86_64::*;
    let n = x.len();
    let chunks = n / 8;
    for c in 0..chunks {
        let off = c * 8;
        let v = _mm256_loadu_ps(x.as_ptr().add(off));
        _mm256_storeu_ps(y.as_mut_ptr().add(off), avx2_exp8(v));
    }
    for i in chunks * 8..n {
        y[i] = x[i].exp();
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vvtanhf_avx2(y: &mut [f32], x: &[f32]) {
    use crate::kernels::avx2_exp8;
    use std::arch::x86_64::*;
    let n = x.len();
    let chunks = n / 8;
    let two = _mm256_set1_ps(2.0);
    let one = _mm256_set1_ps(1.0);
    for c in 0..chunks {
        let off = c * 8;
        let v = _mm256_loadu_ps(x.as_ptr().add(off));
        let e = avx2_exp8(_mm256_mul_ps(v, two));
        let num = _mm256_sub_ps(e, one);
        let den = _mm256_add_ps(e, one);
        _mm256_storeu_ps(y.as_mut_ptr().add(off), _mm256_div_ps(num, den));
    }
    for i in chunks * 8..n {
        y[i] = x[i].tanh();
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vvrecf_avx2(y: &mut [f32], x: &[f32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let chunks = n / 8;
    let one = _mm256_set1_ps(1.0);
    for c in 0..chunks {
        let off = c * 8;
        let v = _mm256_loadu_ps(x.as_ptr().add(off));
        _mm256_storeu_ps(y.as_mut_ptr().add(off), _mm256_div_ps(one, v));
    }
    for i in chunks * 8..n {
        y[i] = 1.0 / x[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vvsqrtf_avx(y: &mut [f32], x: &[f32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let chunks = n / 8;
    for c in 0..chunks {
        let off = c * 8;
        let v = _mm256_loadu_ps(x.as_ptr().add(off));
        _mm256_storeu_ps(y.as_mut_ptr().add(off), _mm256_sqrt_ps(v));
    }
    for i in chunks * 8..n {
        y[i] = x[i].sqrt();
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vvrsqrtf_avx(y: &mut [f32], x: &[f32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let chunks = n / 8;
    let one = _mm256_set1_ps(1.0);
    for c in 0..chunks {
        let off = c * 8;
        let v = _mm256_loadu_ps(x.as_ptr().add(off));
        _mm256_storeu_ps(
            y.as_mut_ptr().add(off),
            _mm256_div_ps(one, _mm256_sqrt_ps(v)),
        );
    }
    for i in chunks * 8..n {
        y[i] = 1.0 / x[i].sqrt();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// `exp_poly` is the portable arm of `vvexpf_fast`, and on x86 it now runs
    /// wherever AVX2 is absent. Pin its accuracy against libm across the full
    /// clamped domain, and pin it against the AVX2 arm it mirrors, so the two
    /// cannot drift apart per-CPU.
    #[test]
    fn exp_poly_matches_libm_and_the_simd_arm() {
        let mut worst_rel = 0f32;
        let mut at = 0f32;
        let mut xi = -87.0f32;
        while xi <= 88.0 {
            let got = exp_poly(xi);
            let want = xi.exp();
            let rel = ((got - want) / want).abs();
            if rel > worst_rel {
                worst_rel = rel;
                at = xi;
            }
            xi += 0.013;
        }
        println!("exp_poly worst relative error {worst_rel:e} at x={at}");
        // Measured 2.53e-7 against glibc and 2.5e-7 against Apple libm,
        // i.e. the polynomial's own error, matching the ~2e-7 this module
        // documents for the `_fast` entry points. Bound keeps ~2x headroom
        // for libm differences across platforms; a broken polynomial would
        // miss by orders of magnitude, not by a factor of two.
        assert!(worst_rel < 5e-7, "exp_poly rel err {worst_rel:e} at x={at}");

        // Saturation ends, where the clamp decides the result.
        assert_eq!(exp_poly(-1000.0), exp_poly(-87.3));
        assert_eq!(exp_poly(1000.0), exp_poly(88.7));

        // NaN must PROPAGATE, not vanish. A `x.max(lo).min(hi)` clamp returns
        // the non-NaN operand and quietly yields ~1e-38 here, which would
        // erase a NaN mid-graph and defeat `RLX_DEBUG_NANS`.
        assert!(exp_poly(f32::NAN).is_nan(), "exp_poly swallowed a NaN");
        assert!(tanh_poly(f32::NAN).is_nan(), "tanh_poly swallowed a NaN");
        assert!(softplus_poly(f32::NAN).is_nan(), "softplus swallowed a NaN");
        assert!(log_poly(f32::NAN).is_nan(), "log_poly swallowed a NaN");
        assert!(atan_poly(f32::NAN).is_nan(), "atan_poly swallowed a NaN");
        assert!(sin_poly(f32::NAN).is_nan(), "sin_poly swallowed a NaN");

        // Both arms of `vvexpf_fast` must agree where both are reachable.
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            let x: Vec<f32> = (0..64).map(|i| i as f32 * 2.3 - 70.0).collect();
            let mut simd = vec![0f32; x.len()];
            let mut poly = vec![0f32; x.len()];
            // SAFETY: both feature bits checked above.
            unsafe { vvexpf_avx2(&mut simd, &x) };
            vvexpf_poly(&mut poly, &x);
            let worst = simd
                .iter()
                .zip(&poly)
                .map(|(a, b)| ((a - b) / b.max(f32::MIN_POSITIVE)).abs())
                .fold(0f32, f32::max);
            println!("avx2-vs-poly worst relative error {worst:e}");
            // Measured 1.11e-7 — about one f32 ULP. Same polynomial, so the
            // only difference is FMA contraction in the AVX2 arm.
            assert!(worst < 1e-6, "exp arms diverge by {worst:e} relative");
        }
    }

    /// `tanh_poly` backs the attention softcap, which applies it per score.
    /// Absolute error is the meaningful bound: the `(e^2x−1)/(e^2x+1)` form
    /// loses relative accuracy near the origin (documented on the function),
    /// and softcap only ever needs the saturating shape.
    #[test]
    fn tanh_poly_matches_libm_and_saturates() {
        let mut worst = 0f32;
        let mut at = 0f32;
        let mut x = -20.0f32;
        while x <= 20.0 {
            let d = (tanh_poly(x) - x.tanh()).abs();
            if d > worst {
                worst = d;
                at = x;
            }
            x += 0.0017;
        }
        println!("tanh_poly worst absolute error {worst:e} at x={at}");
        // Measured 1.19e-7 — one f32 ULP.
        assert!(worst < 1e-6, "tanh_poly abs err {worst:e} at x={at}");
        assert_eq!(tanh_poly(50.0), 1.0);
        assert_eq!(tanh_poly(-50.0), -1.0);
        assert_eq!(tanh_poly(0.0), 0.0);
    }

    /// `log_poly` is the `ln` counterpart to `exp_poly`, used by the softplus
    /// family. Sweep several decades, including denormals and the specials.
    #[test]
    fn log_poly_matches_libm() {
        let mut worst = 0f32;
        let mut at = 0f32;
        for k in -30i32..=30 {
            let scale = 2f32.powi(k);
            let mut t = 1.0f32;
            while t < 2.0 {
                let x = t * scale;
                if x > 0.0 && x.is_finite() {
                    let rel = ((log_poly(x) - x.ln()) / x.ln().abs().max(1.0)).abs();
                    if rel > worst {
                        worst = rel;
                        at = x;
                    }
                }
                t += 0.0007;
            }
        }
        println!("log_poly worst relative error {worst:e} at x={at}");
        assert!(worst < 1e-6, "log_poly rel err {worst:e} at x={at}");
        // Denormal input, scaled internally by 2^24.
        let d = f32::from_bits(1);
        assert!(
            (log_poly(d) - d.ln()).abs() / d.ln().abs() < 1e-6,
            "log_poly denormal: {} vs {}",
            log_poly(d),
            d.ln()
        );
        assert_eq!(log_poly(1.0), 0.0);
        assert_eq!(log_poly(0.0), f32::NEG_INFINITY);
        assert!(log_poly(-1.0).is_nan());
    }

    /// The softplus family backs Softplus / Mish / LogSigmoid. The tail is the
    /// interesting part: softplus(-20) is 2.06e-9, and a naive
    /// `log_poly(1.0 + u)` would return 0 there.
    #[test]
    fn softplus_family_matches_libm_including_the_tail() {
        let mut worst = 0f32;
        let mut at = 0f32;
        let mut x = -40.0f32;
        while x <= 40.0 {
            let want = (-(x.abs())).exp().ln_1p() + x.max(0.0);
            let got = softplus_poly(x);
            let rel = ((got - want) / want.abs().max(1e-30)).abs();
            if rel > worst {
                worst = rel;
                at = x;
            }
            x += 0.0031;
        }
        println!("softplus_poly worst relative error {worst:e} at x={at}");
        // Measured 3.37e-7, dominated by exp_poly's own 2.5e-7 (softplus of a
        // large negative x IS e^x). A threshold-based ln1p measured 3.3e-6.
        assert!(worst < 2e-6, "softplus rel err {worst:e} at x={at}");

        // The tail must not collapse to zero.
        for &x in &[-10.0f32, -20.0, -30.0] {
            let got = softplus_poly(x);
            let want = x.exp().ln_1p();
            assert!(got > 0.0, "softplus({x}) collapsed to {got}");
            assert!(
                (got - want).abs() / want < 1e-4,
                "softplus({x}) = {got}, want {want}"
            );
        }
        assert!((ln_1p_poly(0.0) - 0.0).abs() < 1e-9);
    }

    /// sin/cos/atan polynomial arms. The point of the sweep past
    /// `TRIG_FAST_MAX` is to confirm the *guard* is set correctly: the
    /// polynomial is expected to degrade out there, which is exactly why the
    /// vector entries fall back to libm rather than using it.
    #[test]
    fn sin_cos_poly_matches_libm_in_range() {
        let mut worst = 0f32;
        let mut at = 0f32;
        let mut x = -TRIG_FAST_MAX;
        while x <= TRIG_FAST_MAX {
            let d = (sin_poly(x) - x.sin())
                .abs()
                .max((cos_poly(x) - x.cos()).abs());
            if d > worst {
                worst = d;
                at = x;
            }
            x += 0.937; // irrational-ish stride: hits many quadrants
        }
        println!("sin/cos worst absolute error {worst:e} at x={at} (|x| <= {TRIG_FAST_MAX})");
        // Measured 1.19e-7 at |x| ~ 9.2e5 — one f32 ULP, thanks to the f64
        // reduction. The f32 two-part split this replaced: 2.4e-4.
        assert!(worst < 5e-7, "sin/cos abs err {worst:e} at x={at}");

        // Dense sweep near the origin, where most real arguments live.
        let mut worst_near = 0f32;
        let mut x = -6.3f32;
        while x <= 6.3 {
            worst_near = worst_near
                .max((sin_poly(x) - x.sin()).abs())
                .max((cos_poly(x) - x.cos()).abs());
            x += 0.0003;
        }
        println!("sin/cos worst absolute error near origin {worst_near:e}");
        assert!(worst_near < 2e-7, "sin/cos near origin {worst_near:e}");

        let mut worst_atan = 0f32;
        let mut x = -200.0f32;
        while x <= 200.0 {
            worst_atan = worst_atan.max((atan_poly(x) - x.atan()).abs());
            x += 0.0017;
        }
        println!("atan worst absolute error {worst_atan:e}");
        // Measured 1.19e-7 on aarch64 but 2.38e-7 on baseline x86 — exactly
        // 1 vs 2 ULP, because aarch64 contracts `z + z*zz*p` into an FMA and
        // baseline x86 has no FMA to contract into. Bound has to cover the
        // platform without it. (A single-fold atan measured 2.3e-2.)
        assert!(worst_atan < 1e-6, "atan abs err {worst_atan:e}");
        assert_eq!(atan_poly(0.0), 0.0);

        // The guard must actually engage and stay exact out there.
        let big = [TRIG_FAST_MAX * 4.0, -1e7, 3e8];
        let mut y = big;
        vvsinf_fast_inplace(&mut y);
        for (i, &b) in big.iter().enumerate() {
            assert_eq!(y[i], b.sin(), "guard did not fall back to libm at {b}");
        }
    }

    /// `tan` is the one I wrongly dismissed. Relative error is the metric
    /// here, since tan spans zero to ±huge, and the interesting region is
    /// exactly the poles.
    #[test]
    fn tan_poly_matches_libm_including_at_the_poles() {
        let mut worst = 0f32;
        let mut at = 0f32;
        let mut x = -30.0f32;
        while x <= 30.0 {
            let want = x.tan();
            if want.is_finite() {
                let rel = ((tan_poly(x) - want) / want).abs();
                if rel > worst {
                    worst = rel;
                    at = x;
                }
            }
            x += 0.00037;
        }
        println!("tan_poly worst RELATIVE error {worst:e} at x={at}");
        assert!(worst < 1e-5, "tan rel err {worst:e} at x={at}");

        // Walk right up to the poles, where the reciprocal branch takes over.
        for k in -3i32..=3 {
            let pole = std::f32::consts::FRAC_PI_2 + (k as f32) * std::f32::consts::PI;
            for &d in &[1e-3f32, 1e-4, 1e-5, 0.0, -1e-5, -1e-4, -1e-3] {
                let x = pole + d;
                let want = x.tan();
                let got = tan_poly(x);
                assert!(got.is_finite(), "tan_poly({x}) = {got}");
                let rel = ((got - want) / want).abs();
                assert!(
                    rel < 1e-5,
                    "near pole x={x} (k={k}, d={d}): got {got}, want {want}, rel {rel:e}"
                );
            }
        }
        // f32(π/2) is not π/2; the f64 reduction must recover the difference
        // rather than collapsing r to 0 and returning infinity.
        let at_pole = tan_poly(std::f32::consts::FRAC_PI_2);
        assert!(
            at_pole.is_finite() && at_pole.abs() > 1e6,
            "at pole: {at_pole}"
        );
        assert_eq!(tan_poly(0.0), 0.0);
    }

    #[test]
    fn vvexpf_close_to_libm() {
        let x: Vec<f32> = (-40..40).map(|i| i as f32 * 0.17).collect();
        let mut y = vec![0.0f32; x.len()];
        vvexpf(&mut y, &x);
        let ref_y: Vec<f32> = x.iter().map(|v| v.exp()).collect();
        // Apple vForce may differ from libm by a few ULPs on large |x|.
        assert!(max_abs_err(&y, &ref_y) < 1e-4, "vvexpf vs libm");

        let mut yf = vec![0.0f32; x.len()];
        vvexpf_fast(&mut yf, &x);
        assert!(max_abs_err(&yf, &ref_y) < 1e-4, "vvexpf_fast maxabs");
    }

    #[test]
    fn vvtanhf_close_to_libm() {
        let x: Vec<f32> = (-50..50).map(|i| i as f32 * 0.11).collect();
        let mut y = vec![0.0f32; x.len()];
        vvtanhf(&mut y, &x);
        let ref_y: Vec<f32> = x.iter().map(|v| v.tanh()).collect();
        assert!(max_abs_err(&y, &ref_y) < 1e-4, "vvtanhf vs libm");

        let mut yf = vec![0.0f32; x.len()];
        vvtanhf_fast(&mut yf, &x);
        assert!(max_abs_err(&yf, &ref_y) < 1e-4, "vvtanhf_fast maxabs");
    }

    #[test]
    fn vvrecf_and_inplace_exp() {
        let x = vec![0.5f32, 1.0, 2.0, 4.0, -1.0];
        let mut y = vec![0.0f32; x.len()];
        vvrecf(&mut y, &x);
        assert!((y[1] - 1.0).abs() < 1e-6);
        assert!((y[2] - 0.5).abs() < 1e-6);

        let mut recip_inplace = x.clone();
        vvrecf_inplace(&mut recip_inplace);
        assert_eq!(recip_inplace, y);

        let mut z = vec![0.0f32, 1.0, -0.5];
        vvexpf_inplace(&mut z);
        assert!((z[0] - 1.0).abs() < 1e-5);
        assert!((z[1] - 1.0f32.exp()).abs() < 1e-5);
    }

    #[test]
    fn vvlogf_sqrtf_and_rsqrtf_close_to_libm() {
        let x: Vec<f32> = (1..100).map(|i| i as f32 * 0.13).collect();
        let mut y = vec![0.0f32; x.len()];

        vvlogf(&mut y, &x);
        let log_ref: Vec<f32> = x.iter().map(|v| v.ln()).collect();
        assert!(max_abs_err(&y, &log_ref) < 1e-5, "vvlogf vs libm");
        let mut inplace = x.clone();
        vvlogf_inplace(&mut inplace);
        assert!(max_abs_err(&inplace, &log_ref) < 1e-5, "vvlogf inplace");

        vvsqrtf(&mut y, &x);
        let sqrt_ref: Vec<f32> = x.iter().map(|v| v.sqrt()).collect();
        assert!(max_abs_err(&y, &sqrt_ref) < 1e-5, "vvsqrtf vs libm");
        let mut inplace = x.clone();
        vvsqrtf_inplace(&mut inplace);
        assert!(max_abs_err(&inplace, &sqrt_ref) < 1e-5, "vvsqrtf inplace");

        vvrsqrtf(&mut y, &x);
        let rsqrt_ref: Vec<f32> = x.iter().map(|v| 1.0 / v.sqrt()).collect();
        assert!(max_abs_err(&y, &rsqrt_ref) < 1e-5, "vvrsqrtf vs libm");
        let mut inplace = x;
        vvrsqrtf_inplace(&mut inplace);
        assert!(max_abs_err(&inplace, &rsqrt_ref) < 1e-5, "vvrsqrtf inplace");
    }

    #[test]
    fn vvsigmoidf_basic() {
        let x = vec![0.0f32, 10.0, -10.0];
        let mut y = vec![0.0f32; 3];
        vvsigmoidf(&mut y, &x);
        assert!((y[0] - 0.5).abs() < 1e-5);
        assert!(y[1] > 0.999);
        assert!(y[2] < 0.001);
    }
}
