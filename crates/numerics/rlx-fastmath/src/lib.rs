// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The accurate f32 `exp` / `log` / `tanh` tier in Rust, vectorized: what
//! rlx-cpu's `vmath::vvexpf` & co. compute when neither a vendor library
//! (Accelerate's vForce, non-`pure-rust` Apple builds) nor the bit-exact
//! reference ports (`parity`, rlx-libm) are selected.
//!
//! On aarch64 the scalar functions run the NEON kernel on one lane, so a
//! slice's vector body and scalar tail give identical bits; elsewhere a
//! portable form of the same algorithm.
//! Accuracy against an f64 oracle over all 2^32 inputs is pinned by the tests
//! (`exhaustive_*`, `--ignored`).

#![allow(clippy::excessive_precision)]

const LOG2E: f32 = std::f32::consts::LOG2_E;
// ln 2 = LN2_HI + LN2_LO (f32 head, f32 tail)
const LN2_HI: f32 = f32::from_bits(0x3f31_7218);
const LN2_LO: f32 = f32::from_bits(0xb102_e308);
const EXP_HI: f32 = 88.722_84; // above: overflow → inf
const EXP_LO: f32 = -103.972_08; // below: underflow → 0
// Taylor degree 7 on |r| ≤ ln2/2: truncation ≤ 0.1 ulp
const E2: f32 = 0.5;
const E3: f32 = 1.0 / 6.0;
const E4: f32 = 1.0 / 24.0;
const E5: f32 = 1.0 / 120.0;
const E6: f32 = 1.0 / 720.0;
const E7: f32 = 1.0 / 5040.0;

/// `exp(x)`: Cody–Waite reduction, degree-7 polynomial, exact two-step
/// scaling by 2^n (so subnormal results round once).
#[inline(always)]
pub fn exp(x: f32) -> f32 {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64.
    unsafe {
        use std::arch::aarch64::*;
        vgetq_lane_f32::<0>(neon::exp4(vdupq_n_f32(x)))
    }
    #[cfg(not(target_arch = "aarch64"))]
    exp_portable(x)
}

/// The portable form of [`exp`] (other targets; the oracle-checked
/// reference for the NEON kernel's algorithm).
#[inline(always)]
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
pub fn exp_portable(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x > EXP_HI {
        return f32::INFINITY;
    }
    if x < EXP_LO {
        return 0.0;
    }
    let n = (x * LOG2E).round_ties_even();
    let r = (-n).mul_add(LN2_HI, x);
    let r = (-n).mul_add(LN2_LO, r);
    let mut p = E7;
    p = p.mul_add(r, E6);
    p = p.mul_add(r, E5);
    p = p.mul_add(r, E4);
    p = p.mul_add(r, E3);
    p = p.mul_add(r, E2);
    let p = (p * r).mul_add(r, r) + 1.0; // 1 + r + r²·P(r)
    let ni = n as i32;
    let n1 = ni >> 1;
    let n2 = ni - n1;
    let s1 = f32::from_bits(((n1 + 127) as u32) << 23);
    let s2 = f32::from_bits(((n2 + 127) as u32) << 23);
    (p * s1) * s2
}

const LG1: f32 = 0.666_666_626_93; // 0xaaaaaa.0p-24
const LG2: f32 = 0.400_009_721_52; // 0xccce13.0p-25
const LG3: f32 = 0.284_987_866_88; // 0x91e9ee.0p-25
const LG4: f32 = 0.242_790_788_41; // 0xf89e26.0p-26
const LOG_LN2_HI: f32 = 6.931_381_225_6e-1; // 0x3f317180
const LOG_LN2_LO: f32 = 9.058_000_614_5e-6; // 0x3717f7d1

/// `log(x)` (fdlibm `logf`'s algorithm): `x = 2^k·m`, `m ∈ [√½, √2)`,
/// `log(m) = 2s + s·R(s²)` with `s = f/(2+f)`.
#[inline(always)]
pub fn log(x: f32) -> f32 {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64.
    unsafe {
        use std::arch::aarch64::*;
        vgetq_lane_f32::<0>(neon::log4(vdupq_n_f32(x)))
    }
    #[cfg(not(target_arch = "aarch64"))]
    log_portable(x)
}

/// The portable form of [`log`] (other targets; the oracle-checked
/// reference for the NEON kernel's algorithm).
#[inline(always)]
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
pub fn log_portable(x: f32) -> f32 {
    let mut ix = x.to_bits();
    let mut k: i32 = 0;
    if ix < 0x0080_0000 || ix >> 31 != 0 {
        if ix << 1 == 0 {
            return f32::NEG_INFINITY;
        }
        if ix >> 31 != 0 {
            return f32::NAN;
        }
        // subnormal: scale up by 2^25
        k -= 25;
        ix = (x * 33_554_432.0).to_bits();
    } else if ix >= 0x7f80_0000 {
        return x; // inf, NaN
    }
    ix = ix.wrapping_add(0x3f80_0000 - 0x3f35_04f3);
    k += (ix >> 23) as i32 - 0x7f;
    ix = (ix & 0x007f_ffff) + 0x3f35_04f3;
    let f = f32::from_bits(ix) - 1.0;
    let s = f / (2.0 + f);
    let z = s * s;
    let w = z * z;
    let t1 = w * w.mul_add(LG4, LG2);
    let t2 = z * w.mul_add(LG3, LG1);
    let r = t2 + t1;
    let hfsq = 0.5 * f * f;
    let dk = k as f32;
    s.mul_add(hfsq + r, dk * LOG_LN2_LO) - hfsq + f + dk * LOG_LN2_HI
}

// tanh(x) = x + x³·P(x²) on |x| < TANH_POLY (least squares in relative
// error, 0.03 ulp model error); 1 − 2/(e^{2|x|} + 1) beyond.
const TANH_POLY: f32 = 0.625;
const T0: f32 = -0.333_333_330_691_167_16;
const T1: f32 = 0.133_333_023_954_394_5;
const T2: f32 = -0.053_959_634_198_039_39;
const T3: f32 = 0.021_774_065_148_300_246;
const T4: f32 = -0.008_364_255_392_178_164;
const T5: f32 = 0.002_317_965_965_614_568_3;

/// `tanh(x)`.
#[inline(always)]
pub fn tanh(x: f32) -> f32 {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64.
    unsafe {
        use std::arch::aarch64::*;
        vgetq_lane_f32::<0>(neon::tanh4(vdupq_n_f32(x)))
    }
    #[cfg(not(target_arch = "aarch64"))]
    tanh_portable(x)
}

/// The portable form of [`tanh`] (other targets; the oracle-checked
/// reference for the NEON kernel's algorithm).
#[inline(always)]
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
pub fn tanh_portable(x: f32) -> f32 {
    let a = x.abs();
    if a < TANH_POLY {
        let z = x * x;
        let mut p = T5;
        p = p.mul_add(z, T4);
        p = p.mul_add(z, T3);
        p = p.mul_add(z, T2);
        p = p.mul_add(z, T1);
        p = p.mul_add(z, T0);
        return (x * z).mul_add(p, x);
    }
    let e = exp(a + a);
    let r = 1.0 - 2.0 / (e + 1.0);
    r.copysign(x)
}

#[cfg(target_arch = "aarch64")]
pub mod neon {
    //! The same functions on four lanes.
    use super::*;
    use std::arch::aarch64::*;

    #[inline(always)]
    pub unsafe fn exp4(x: float32x4_t) -> float32x4_t {
        unsafe {
            let n = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(LOG2E)));
            let nn = vnegq_f32(n);
            let r = vfmaq_f32(x, nn, vdupq_n_f32(LN2_HI));
            let r = vfmaq_f32(r, nn, vdupq_n_f32(LN2_LO));
            let mut p = vdupq_n_f32(E7);
            p = vfmaq_f32(vdupq_n_f32(E6), p, r);
            p = vfmaq_f32(vdupq_n_f32(E5), p, r);
            p = vfmaq_f32(vdupq_n_f32(E4), p, r);
            p = vfmaq_f32(vdupq_n_f32(E3), p, r);
            p = vfmaq_f32(vdupq_n_f32(E2), p, r);
            let p = vaddq_f32(vfmaq_f32(r, vmulq_f32(p, r), r), vdupq_n_f32(1.0));
            // vcvt saturates (and maps NaN to 0): every lane gets a finite n
            let ni = vcvtq_s32_f32(n);
            // |x| ≤ 86.6 ⇒ n ∈ [-125, 125] and p·2^n is normal, where adding
            // n to p's exponent field is the exact product. One absolute
            // compare (false for NaN) decides whether any lane needs more.
            let y =
                vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(p), vshlq_n_s32::<23>(ni)));
            let ok = vcaleq_f32(x, vdupq_n_f32(86.6));
            if vminvq_u32(ok) != 0 {
                return y;
            }
            // rare lanes: the two-step scale (rounds once, like the scalar
            // form), overflow, underflow, NaN
            let nc = vminq_s32(vmaxq_s32(ni, vdupq_n_s32(-150)), vdupq_n_s32(128));
            let n1 = vshrq_n_s32::<1>(nc);
            let n2 = vsubq_s32(nc, n1);
            let bias = vdupq_n_s32(127);
            let s1 = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(n1, bias)));
            let s2 = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(n2, bias)));
            let mut z = vmulq_f32(vmulq_f32(p, s1), s2);
            z = vbslq_f32(
                vcgtq_f32(x, vdupq_n_f32(EXP_HI)),
                vdupq_n_f32(f32::INFINITY),
                z,
            );
            z = vbslq_f32(vcltq_f32(x, vdupq_n_f32(EXP_LO)), vdupq_n_f32(0.0), z);
            z = vbslq_f32(vceqq_f32(x, x), z, x);
            vbslq_f32(ok, y, z)
        }
    }

    #[inline(always)]
    pub unsafe fn log4(x: float32x4_t) -> float32x4_t {
        unsafe {
            let bits = vreinterpretq_u32_f32(x);
            // subnormal positive inputs: scale by 2^25, k -= 25
            let sub = vcltq_u32(bits, vdupq_n_u32(0x0080_0000));
            let xs = vbslq_f32(sub, vmulq_f32(x, vdupq_n_f32(33_554_432.0)), x);
            let k0 = vbslq_s32(sub, vdupq_n_s32(-25), vdupq_n_s32(0));
            let mut ix = vaddq_u32(
                vreinterpretq_u32_f32(xs),
                vdupq_n_u32(0x3f80_0000 - 0x3f35_04f3),
            );
            let k = vaddq_s32(
                k0,
                vsubq_s32(
                    vreinterpretq_s32_u32(vshrq_n_u32::<23>(ix)),
                    vdupq_n_s32(0x7f),
                ),
            );
            ix = vaddq_u32(
                vandq_u32(ix, vdupq_n_u32(0x007f_ffff)),
                vdupq_n_u32(0x3f35_04f3),
            );
            let f = vsubq_f32(vreinterpretq_f32_u32(ix), vdupq_n_f32(1.0));
            let s = vdivq_f32(f, vaddq_f32(vdupq_n_f32(2.0), f));
            let z = vmulq_f32(s, s);
            let w = vmulq_f32(z, z);
            let t1 = vmulq_f32(w, vfmaq_f32(vdupq_n_f32(LG2), w, vdupq_n_f32(LG4)));
            let t2 = vmulq_f32(z, vfmaq_f32(vdupq_n_f32(LG1), w, vdupq_n_f32(LG3)));
            let r = vaddq_f32(t2, t1);
            let hfsq = vmulq_f32(vmulq_f32(vdupq_n_f32(0.5), f), f);
            let dk = vcvtq_f32_s32(k);
            let a = vfmaq_f32(
                vmulq_f32(dk, vdupq_n_f32(LOG_LN2_LO)),
                s,
                vaddq_f32(hfsq, r),
            );
            let mut y = vaddq_f32(
                vaddq_f32(vsubq_f32(a, hfsq), f),
                vmulq_f32(dk, vdupq_n_f32(LOG_LN2_HI)),
            );
            // specials, in the scalar order: ±0 → -inf, negative → NaN, inf/NaN → x
            let is_zero = vceqq_u32(vshlq_n_u32::<1>(bits), vdupq_n_u32(0));
            let is_neg = vcgeq_u32(bits, vdupq_n_u32(0x8000_0000));
            let is_big = vcgeq_u32(bits, vdupq_n_u32(0x7f80_0000));
            y = vbslq_f32(is_big, x, y);
            y = vbslq_f32(is_neg, vdupq_n_f32(f32::NAN), y);
            vbslq_f32(is_zero, vdupq_n_f32(f32::NEG_INFINITY), y)
        }
    }

    #[inline(always)]
    pub unsafe fn tanh4(x: float32x4_t) -> float32x4_t {
        unsafe {
            let a = vabsq_f32(x);
            let z = vmulq_f32(x, x);
            let mut p = vdupq_n_f32(T5);
            p = vfmaq_f32(vdupq_n_f32(T4), p, z);
            p = vfmaq_f32(vdupq_n_f32(T3), p, z);
            p = vfmaq_f32(vdupq_n_f32(T2), p, z);
            p = vfmaq_f32(vdupq_n_f32(T1), p, z);
            p = vfmaq_f32(vdupq_n_f32(T0), p, z);
            let small = vfmaq_f32(x, vmulq_f32(x, z), p);
            let e = exp4(vaddq_f32(a, a));
            let one = vdupq_n_f32(1.0);
            let r = vsubq_f32(one, vdivq_f32(vdupq_n_f32(2.0), vaddq_f32(e, one)));
            // copysign(r, x)
            let sign = vdupq_n_u32(0x8000_0000);
            let big = vbslq_f32(sign, x, r);
            vbslq_f32(vcltq_f32(a, vdupq_n_f32(TANH_POLY)), small, big)
        }
    }
}

macro_rules! slice_fn {
    ($name:ident, $name_ip:ident, $scalar:ident, $vec:ident, $doc:literal) => {
        #[doc = $doc]
        pub fn $name(y: &mut [f32], x: &[f32]) {
            assert_eq!(y.len(), x.len());
            // Only the aarch64 block below advances `i` before the tail.
            #[cfg_attr(not(target_arch = "aarch64"), allow(unused_mut))]
            let mut i = 0;
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                // four independent vectors per iteration: the polynomial
                // chains are latency-bound, so this lets them overlap
                while i + 16 <= x.len() {
                    // SAFETY: i + 16 <= len for both slices.
                    unsafe {
                        let (xp, yp) = (x.as_ptr().add(i), y.as_mut_ptr().add(i));
                        let (a, b, c, d) = (
                            vld1q_f32(xp),
                            vld1q_f32(xp.add(4)),
                            vld1q_f32(xp.add(8)),
                            vld1q_f32(xp.add(12)),
                        );
                        let (a, b, c, d) =
                            (neon::$vec(a), neon::$vec(b), neon::$vec(c), neon::$vec(d));
                        vst1q_f32(yp, a);
                        vst1q_f32(yp.add(4), b);
                        vst1q_f32(yp.add(8), c);
                        vst1q_f32(yp.add(12), d);
                    }
                    i += 16;
                }
                while i + 4 <= x.len() {
                    // SAFETY: i + 4 <= len for both slices.
                    unsafe {
                        vst1q_f32(
                            y.as_mut_ptr().add(i),
                            neon::$vec(vld1q_f32(x.as_ptr().add(i))),
                        )
                    };
                    i += 4;
                }
            }
            for j in i..x.len() {
                y[j] = $scalar(x[j]);
            }
        }

        #[doc = $doc]
        pub fn $name_ip(y: &mut [f32]) {
            #[cfg_attr(not(target_arch = "aarch64"), allow(unused_mut))]
            let mut i = 0;
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                while i + 16 <= y.len() {
                    // SAFETY: i + 16 <= len.
                    unsafe {
                        let p = y.as_mut_ptr().add(i);
                        let (a, b, c, d) = (
                            vld1q_f32(p),
                            vld1q_f32(p.add(4)),
                            vld1q_f32(p.add(8)),
                            vld1q_f32(p.add(12)),
                        );
                        let (a, b, c, d) =
                            (neon::$vec(a), neon::$vec(b), neon::$vec(c), neon::$vec(d));
                        vst1q_f32(p, a);
                        vst1q_f32(p.add(4), b);
                        vst1q_f32(p.add(8), c);
                        vst1q_f32(p.add(12), d);
                    }
                    i += 16;
                }
                while i + 4 <= y.len() {
                    // SAFETY: i + 4 <= len.
                    unsafe {
                        let p = y.as_mut_ptr().add(i);
                        vst1q_f32(p, neon::$vec(vld1q_f32(p)));
                    }
                    i += 4;
                }
            }
            for v in &mut y[i..] {
                *v = $scalar(*v);
            }
        }
    };
}

slice_fn!(
    exp_slice,
    exp_inplace,
    exp,
    exp4,
    "`y = exp(x)` element-wise."
);
slice_fn!(
    log_slice,
    log_inplace,
    log,
    log4,
    "`y = log(x)` element-wise."
);
slice_fn!(
    tanh_slice,
    tanh_inplace,
    tanh,
    tanh4,
    "`y = tanh(x)` element-wise."
);

#[cfg(test)]
mod tests {
    use super::*;

    fn ulp_err(got: f32, want: f64) -> f64 {
        if got.is_nan() && want.is_nan() {
            return 0.0;
        }
        // correctly rounded (also when the rounding overflows to ±inf)
        if got.to_bits() == (want as f32).to_bits() {
            return 0.0;
        }
        let w = want as f32;
        let ulp = if w == 0.0 || !w.is_finite() {
            f32::from_bits(1) as f64
        } else {
            (f32::from_bits(w.abs().to_bits() + 1) - w.abs()) as f64
        };
        ((got as f64) - want).abs() / ulp
    }

    /// `(max ulp, fraction correctly rounded)` over every `step`-th bit
    /// pattern, and vector lanes == scalar on each of them.
    fn sweep(
        step: u64,
        f: fn(f32) -> f32,
        v: fn(&mut [f32], &[f32]),
        oracle: fn(f64) -> f64,
    ) -> (f64, f64) {
        let (mut worst, mut exact, mut n) = (0f64, 0u64, 0u64);
        let mut xs = Vec::with_capacity(4096);
        let mut i: u64 = 0;
        while i < 1 << 32 {
            xs.push(f32::from_bits(i as u32));
            if xs.len() == 4096 || i + step >= 1 << 32 {
                let mut ys = vec![0f32; xs.len()];
                v(&mut ys, &xs);
                for (&x, &y) in xs.iter().zip(&ys) {
                    let s = f(x);
                    assert!(
                        s.to_bits() == y.to_bits() || (s.is_nan() && y.is_nan()),
                        "lane != scalar at {x:e}: {y:e} vs {s:e}"
                    );
                    let want = oracle(x as f64);
                    let e = ulp_err(s, want);
                    if e > worst && std::env::var_os("FASTMATH_DEBUG").is_some() {
                        eprintln!(
                            "new worst {e} at {x:e} ({:#x}): got {s:e} want {want:e}",
                            x.to_bits()
                        );
                    }
                    worst = worst.max(e);
                    exact += (s == want as f32 || (s.is_nan() && want.is_nan())) as u64;
                    n += 1;
                }
                xs.clear();
            }
            i += step;
        }
        (worst, exact as f64 / n as f64)
    }

    #[test]
    fn sampled_accuracy() {
        let (e, _) = sweep(997, exp, exp_slice, f64::exp);
        assert!(e < 1.5, "exp {e} ulp");
        let (l, _) = sweep(997, log, log_slice, f64::ln);
        assert!(l < 1.5, "log {l} ulp");
        let (t, _) = sweep(997, tanh, tanh_slice, f64::tanh);
        assert!(t < 2.5, "tanh {t} ulp");
    }

    #[test]
    #[ignore = "all 2^32 inputs; run with --ignored --nocapture"]
    fn exhaustive_accuracy() {
        for (name, f, v, o) in [
            (
                "exp",
                exp as fn(f32) -> f32,
                exp_slice as fn(&mut [f32], &[f32]),
                f64::exp as fn(f64) -> f64,
            ),
            ("log", log, log_slice, f64::ln),
            ("tanh", tanh, tanh_slice, f64::tanh),
        ] {
            let (worst, cr) = sweep(1, f, v, o);
            eprintln!(
                "{name}: max {worst:.3} ulp, {:.4}% correctly rounded",
                100.0 * cr
            );
        }
    }
}
