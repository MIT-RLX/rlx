// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! glibc 2.43's double-precision `tanh` (`s_tanh.c`; its x86-64 FMA build
//! differs only in calling the FMA `expm1`, which [`crate::expm1`] is) and
//! `cbrt` (`s_cbrt.c`, built without FMA). Line-for-line ports.

/// glibc `tanh`.
pub fn tanh(x: f64) -> f64 {
    let bits = x.to_bits();
    let jx = (bits >> 32) as u32 as i32;
    let lx = bits as u32;
    let ix = jx & 0x7fff_ffff;
    if ix >= 0x7ff0_0000 {
        // ±inf → ±1, NaN → NaN
        return if jx >= 0 {
            1.0 / x + 1.0
        } else {
            1.0 / x - 1.0
        };
    }
    let z = if ix < 0x4036_0000 {
        // |x| < 22
        if (ix as u32 | lx) == 0 {
            return x;
        }
        if ix < 0x3c80_0000 {
            // |x| < 2^-55
            return x * (1.0 + x);
        }
        if ix >= 0x3ff0_0000 {
            let t = crate::expm1(2.0 * x.abs());
            1.0 - 2.0 / (t + 2.0)
        } else {
            let t = crate::expm1(-2.0 * x.abs());
            -t / (t + 2.0)
        }
    } else {
        1.0 - 1.0e-300
    };
    if jx >= 0 { z } else { -z }
}

/// `x = m·2^e`, `m` in `[0.5, 1)` (`e = 0` for 0, inf and NaN, as glibc's).
fn frexp(x: f64) -> (f64, i32) {
    let bits = x.to_bits();
    let e = ((bits >> 52) & 0x7ff) as i32;
    if e == 0x7ff || x == 0.0 {
        return (x, 0);
    }
    if e == 0 {
        // subnormal: scale into the normal range first
        let (m, e) = frexp(x * f64::from_bits(0x4350_0000_0000_0000)); // 2^54
        return (m, e - 54);
    }
    (
        f64::from_bits((bits & 0x800f_ffff_ffff_ffff) | 0x3fe0_0000_0000_0000),
        e - 1022,
    )
}

/// glibc `cbrt`.
pub fn cbrt(x: f64) -> f64 {
    cbrt_impl::<false>(x)
}

/// `cbrt`; `FMA`: as gcc builds it for aarch64, the Horner steps fused (the
/// other products, `2·xm` and `2·t2`, are exact either way).
pub(crate) fn cbrt_impl<const FMA: bool>(x: f64) -> f64 {
    const CBRT2: f64 = 1.259_921_049_894_873_164_8;
    const SQR_CBRT2: f64 = 1.587_401_051_968_199_474_8;
    const FACTOR: [f64; 5] = [1.0 / SQR_CBRT2, 1.0 / CBRT2, 1.0, CBRT2, SQR_CBRT2];
    let (xm, xe) = frexp(x.abs());
    if xe == 0 && (x == 0.0 || !x.is_finite()) {
        return x + x;
    }
    let u = if FMA {
        let f = crate::fma;
        let p = f(-0.145_263_899_385_486_377, xm, 0.784_932_344_976_639_262);
        let p = f(p, xm, -1.834_692_774_836_130_86);
        let p = f(p, xm, 2.446_931_225_635_344_30);
        let p = f(p, xm, -2.114_994_941_673_712_87);
        let p = f(p, xm, 1.508_191_937_815_848_96);
        f(p, xm, 0.354_895_765_043_919_860)
    } else {
        0.354_895_765_043_919_860
            + ((1.508_191_937_815_848_96
                + ((-2.114_994_941_673_712_87
                    + ((2.446_931_225_635_344_30
                        + ((-1.834_692_774_836_130_86
                            + (0.784_932_344_976_639_262 - 0.145_263_899_385_486_377 * xm)
                                * xm)
                            * xm))
                        * xm))
                    * xm))
                * xm)
    };
    let t2 = u * u * u;
    let ym = u * (t2 + 2.0 * xm) / (2.0 * t2 + xm) * FACTOR[(2 + xe % 3) as usize];
    // `ldexp`: |xe / 3| ≤ 358 and ym is near 1, so the scale is exact
    let scale = f64::from_bits(((1023 + xe / 3) as u64) << 52);
    (if x > 0.0 { ym } else { -ym }) * scale
}

/// glibc `cosh` (`e_cosh.c`; no FMA build, its `exp` / `expm1` the FMA ones).
pub fn cosh(x: f64) -> f64 {
    let ix = ((x.to_bits() >> 32) as u32) & 0x7fff_ffff;
    if ix < 0x4036_0000 {
        if ix < 0x3fd6_2e43 {
            if ix < 0x3c80_0000 {
                return 1.0;
            }
            let t = crate::expm1(x.abs());
            let w = 1.0 + t;
            return 1.0 + (t * t) / (w + w);
        }
        let t = crate::exp(x.abs());
        return 0.5 * t + 0.5 / t;
    }
    if ix < 0x4086_2e42 {
        return 0.5 * crate::exp(x.abs());
    }
    if x.to_bits() & 0x7fff_ffff_ffff_ffff <= 0x4086_33ce_8fb9_f87d {
        let w = crate::exp(0.5 * x.abs());
        let t = 0.5 * w;
        return t * w;
    }
    if ix >= 0x7ff0_0000 {
        return x * x;
    }
    1.0e300 * 1.0e300
}

/// glibc `sinh` (`e_sinh.c`; the FMA build fuses only `2·t − …`, where the
/// product is exact).
pub fn sinh(x: f64) -> f64 {
    let bits = x.to_bits();
    let jx = (bits >> 32) as u32 as i32;
    let ix = jx & 0x7fff_ffff;
    if ix >= 0x7ff0_0000 {
        return x + x;
    }
    let h = if jx < 0 { -0.5 } else { 0.5 };
    if ix < 0x4036_0000 {
        if ix < 0x3e30_0000 && 1.0e307 + x > 1.0 {
            return x;
        }
        let t = crate::expm1(x.abs());
        if ix < 0x3ff0_0000 {
            return h * (2.0 * t - t * t / (t + 1.0));
        }
        return h * (t + t / (t + 1.0));
    }
    if ix < 0x4086_2e42 {
        return h * crate::exp(x.abs());
    }
    let lx = bits as u32;
    if ix < 0x4086_33ce || (ix == 0x4086_33ce && lx <= 0x8fb9_f87d) {
        let w = crate::exp(0.5 * x.abs());
        let t = h * w;
        return t * w;
    }
    x * 1.0e307
}

/// glibc `log10` (`e_log10.c`; no FMA build, its `log` the FMA one).
pub fn log10(x: f64) -> f64 {
    const TWO54: f64 = f64::from_bits(0x4350_0000_0000_0000);
    const IVLN10: f64 = f64::from_bits(0x3fdb_cb7b_1526_e50e);
    const LOG10_2HI: f64 = f64::from_bits(0x3fd3_4413_509f_6000);
    const LOG10_2LO: f64 = f64::from_bits(0x3d59_fef3_11f1_2b36);
    let mut x = x;
    let mut hx = x.to_bits() as i64;
    let mut k: i32 = 0;
    if hx < 0x0010_0000_0000_0000 {
        if hx & 0x7fff_ffff_ffff_ffff == 0 {
            return -TWO54 / x.abs();
        }
        if hx < 0 {
            return (x - x) / (x - x);
        }
        k -= 54;
        x *= TWO54;
        hx = x.to_bits() as i64;
    }
    if hx >= 0x7ff0_0000_0000_0000 {
        return x + x;
    }
    k += ((hx >> 52) - 1023) as i32;
    // `(uint64_t) k`, sign-extended, has its top bit set iff k < 0
    let i = (k < 0) as i64;
    hx = (hx & 0x000f_ffff_ffff_ffff) | ((0x3ff - i) << 52);
    let y = (k as i64 + i) as f64;
    let x = f64::from_bits(hx as u64);
    let z = y * LOG10_2LO + IVLN10 * crate::log(x);
    z + y * LOG10_2HI
}

/// The shared subnormal / overflow tail of `exp2` and `exp10`; `FMA`: as gcc
/// fuses it (`scale·tmp` once in the `k > 0` arm; in the other it is shared
/// with the `if` block, so it stays a plain product).
fn exp_scale_special<const FMA: bool>(tmp: f64, sbits: u64, ki: u64) -> f64 {
    if ki & 0x8000_0000 == 0 {
        let scale = f64::from_bits(sbits.wrapping_sub(1 << 52));
        if FMA {
            return 2.0 * crate::fma(scale, tmp, scale);
        }
        return 2.0 * (scale + scale * tmp);
    }
    let scale = f64::from_bits(sbits.wrapping_add(1022 << 52));
    let st = scale * tmp;
    let mut y = scale + st;
    if y < 1.0 {
        let mut lo = scale - y + st;
        let hi = 1.0 + y;
        lo = 1.0 - hi + y + lo;
        y = (hi + lo) - 1.0;
        if y == 0.0 {
            y = 0.0;
        }
    }
    f64::from_bits(0x0010_0000_0000_0000) * y
}

/// glibc `exp2` (`e_exp2.c`; no FMA build on x86-64).
pub fn exp2(x: f64) -> f64 {
    exp2_impl::<false>(x)
}

/// `exp2`; `FMA`: as gcc builds it for aarch64 (every product whose uses are
/// all adds fused).
pub(crate) fn exp2_impl<const FMA: bool>(x: f64) -> f64 {
    use crate::tables::{EXP_DATA as D, EXP_TAB, EXP2_POLY, EXP2_SHIFT};
    const N: u64 = 128;
    let top12 = |v: f64| (v.to_bits() >> 52) as u32;
    let mut abstop = top12(x) & 0x7ff;
    let tiny = top12(f64::from_bits(0x3c90_0000_0000_0000)); // 0x1p-54
    if abstop.wrapping_sub(tiny) >= top12(512.0) - tiny {
        if abstop.wrapping_sub(tiny) >= 0x8000_0000 {
            return 1.0 + x;
        }
        if abstop >= top12(1024.0) {
            if x.to_bits() == f64::NEG_INFINITY.to_bits() {
                return 0.0;
            }
            if abstop >= top12(f64::INFINITY) {
                return 1.0 + x;
            }
            if x.to_bits() >> 63 == 0 {
                return crate::oflow(false);
            } else if x.to_bits() >= (-1075.0f64).to_bits() {
                return crate::uflow(false);
            }
        }
        if 2 * x.to_bits() > 2 * 928.0f64.to_bits() {
            abstop = 0;
        }
    }
    let shift = f64::from_bits(D[EXP2_SHIFT]);
    let c = |i: usize| f64::from_bits(D[EXP2_POLY + i]);
    let kd0 = x + shift;
    let ki = kd0.to_bits();
    let kd = kd0 - shift;
    let r = x - kd;
    let idx = (2 * (ki % N)) as usize;
    let top = ki << (52 - 7);
    let tail = f64::from_bits(D[EXP_TAB + idx]);
    let sbits = D[EXP_TAB + idx + 1].wrapping_add(top);
    let r2 = r * r;
    let tmp = if FMA {
        let f = crate::fma;
        let t = f(r, c(0), tail);
        let t = f(r2, f(r, c(2), c(1)), t);
        f(r2 * r2, f(r, c(4), c(3)), t)
    } else {
        tail + r * c(0) + r2 * (c(1) + r * c(2)) + r2 * r2 * (c(3) + r * c(4))
    };
    if abstop == 0 {
        return exp_scale_special::<FMA>(tmp, sbits, ki);
    }
    let scale = f64::from_bits(sbits);
    if FMA {
        return crate::fma(scale, tmp, scale);
    }
    scale + scale * tmp
}

/// glibc `exp10` (`e_exp10.c`; no FMA build on x86-64).
pub fn exp10(x: f64) -> f64 {
    exp10_impl::<false>(x)
}

/// `exp10`; `AARCH64`: as glibc builds it there — `k` by `round` / `lround`
/// (`TOINT_INTRINSICS`: ties away from zero, not the shift's ties to even)
/// and every product whose uses are all adds fused.
pub(crate) fn exp10_impl<const AARCH64: bool>(x: f64) -> f64 {
    use crate::tables::{EXP_DATA as D, EXP_SHIFT, EXP_TAB};
    const SMALL_TOP: u32 = 0x3c6;
    const THRESH: u32 = 0x41;
    let c = |i: usize| f64::from_bits(D[16 + i]);
    let ix = x.to_bits();
    let mut abstop = ((ix >> 52) & 0x7ff) as u32;
    if abstop.wrapping_sub(SMALL_TOP) >= THRESH {
        if abstop.wrapping_sub(SMALL_TOP) >= 0x8000_0000 {
            return x + 1.0;
        }
        if abstop == 0x7ff {
            return if ix == f64::NEG_INFINITY.to_bits() {
                0.0
            } else {
                x + 1.0
            };
        }
        if x >= f64::from_bits(0x4073_4413_509f_79ff) {
            return crate::oflow(false);
        }
        if x < -350.0 {
            return crate::uflow(false);
        }
        abstop = 0;
    }
    let shift = f64::from_bits(D[EXP_SHIFT]);
    let z = f64::from_bits(D[277]) * x;
    let (kd, ki) = if AARCH64 {
        let kd = crate::round(z);
        (kd, kd as i64 as u64)
    } else {
        let kd0 = z + shift;
        (kd0 - shift, kd0.to_bits())
    };
    let f = |a: f64, b: f64, c: f64| {
        if AARCH64 {
            crate::fma(a, b, c)
        } else {
            a * b + c
        }
    };
    let mut r = x;
    r = f(f64::from_bits(D[14]), kd, r);
    r = f(f64::from_bits(D[15]), kd, r);
    let e = ki << (52 - 7);
    let i = ((ki & 127) * 2) as usize;
    let sbits = D[EXP_TAB + i + 1].wrapping_add(e);
    let tail = f64::from_bits(D[EXP_TAB + i]);
    let r2 = r * r;
    let p = f(r, c(1), c(0));
    let mut y = f(r, c(3), c(2));
    y = f(r2, c(4), y);
    y = f(r2, y, p);
    y = f(y, r, tail);
    if abstop == 0 {
        return exp_scale_special::<AARCH64>(y, sbits, ki);
    }
    let s = f64::from_bits(sbits);
    f(s, y, s)
}
