// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! glibc 2.43's single-precision `powf`, `exp2f`, `log2f` (the x86-64 `-mfma`
//! variants: gcc fuses each multiply into the add that consumes it), and
//! `exp10f`, `hypotf` (built without FMA there). Not correctly rounded, so
//! ported line for line rather than computed in double and rounded.

#[cfg(not(any(test, feature = "std")))]
#[allow(unused_imports)]
use crate::fp::Fp as _;
use crate::{d, fma};

/// `__exp2f_data.tab`: `uint(2^(i/32)) - (i << 47)`.
const EXP2F_TAB: [u64; 32] = [
    0x3ff0000000000000,
    0x3fefd9b0d3158574,
    0x3fefb5586cf9890f,
    0x3fef9301d0125b51,
    0x3fef72b83c7d517b,
    0x3fef54873168b9aa,
    0x3fef387a6e756238,
    0x3fef1e9df51fdee1,
    0x3fef06fe0a31b715,
    0x3feef1a7373aa9cb,
    0x3feedea64c123422,
    0x3feece086061892d,
    0x3feebfdad5362a27,
    0x3feeb42b569d4f82,
    0x3feeab07dd485429,
    0x3feea47eb03a5585,
    0x3feea09e667f3bcd,
    0x3fee9f75e8ec5f74,
    0x3feea11473eb0187,
    0x3feea589994cce13,
    0x3feeace5422aa0db,
    0x3feeb737b0cdc5e5,
    0x3feec49182a3f090,
    0x3feed503b23e255d,
    0x3feee89f995ad3ad,
    0x3feeff76f2fb5e47,
    0x3fef199bdd85529c,
    0x3fef3720dcef9069,
    0x3fef5818dcfba487,
    0x3fef7c97337b9b5f,
    0x3fefa4afa2a490da,
    0x3fefd0765b6e4540,
];
const EXP2F_N: u64 = 32;
/// `0x1.8p+52`.
const SHIFT: f64 = 6755399441055744.0;
/// `__exp2f_data.poly`.
const EXP2F_C: [f64; 3] = [
    f64::from_bits(0x3fac6af84b912394),
    f64::from_bits(0x3fcebfce50fac4f3),
    f64::from_bits(0x3fe62e42ff0c52d6),
];

/// `(invc, logc)` of `__powf_log2_data` / `__log2f_data` (the same table).
const LOG2F_TAB: [(f64, f64); 16] = [
    (
        f64::from_bits(0x3ff661ec79f8f3be),
        f64::from_bits(0xbfdefec65b963019),
    ),
    (
        f64::from_bits(0x3ff571ed4aaf883d),
        f64::from_bits(0xbfdb0b6832d4fca4),
    ),
    (
        f64::from_bits(0x3ff49539f0f010b0),
        f64::from_bits(0xbfd7418b0a1fb77b),
    ),
    (
        f64::from_bits(0x3ff3c995b0b80385),
        f64::from_bits(0xbfd39de91a6dcf7b),
    ),
    (
        f64::from_bits(0x3ff30d190c8864a5),
        f64::from_bits(0xbfd01d9bf3f2b631),
    ),
    (
        f64::from_bits(0x3ff25e227b0b8ea0),
        f64::from_bits(0xbfc97c1d1b3b7af0),
    ),
    (
        f64::from_bits(0x3ff1bb4a4a1a343f),
        f64::from_bits(0xbfc2f9e393af3c9f),
    ),
    (
        f64::from_bits(0x3ff12358f08ae5ba),
        f64::from_bits(0xbfb960cbbf788d5c),
    ),
    (
        f64::from_bits(0x3ff0953f419900a7),
        f64::from_bits(0xbfaa6f9db6475fce),
    ),
    (1.0, 0.0),
    (
        f64::from_bits(0x3fee608cfd9a47ac),
        f64::from_bits(0x3fb338ca9f24f53d),
    ),
    (
        f64::from_bits(0x3feca4b31f026aa0),
        f64::from_bits(0x3fc476a9543891ba),
    ),
    (
        f64::from_bits(0x3feb2036576afce6),
        f64::from_bits(0x3fce840b4ac4e4d2),
    ),
    (
        f64::from_bits(0x3fe9c2d163a1aa2d),
        f64::from_bits(0x3fd40645f0c6651c),
    ),
    (
        f64::from_bits(0x3fe886e6037841ed),
        f64::from_bits(0x3fd88e9c2c1b9ff8),
    ),
    (
        f64::from_bits(0x3fe767dcf5534862),
        f64::from_bits(0x3fdce0a44eb17bcc),
    ),
];
const LOG2F_OFF: u32 = 0x3f33_0000;

/// `__math_oflowf(sign)`: `±0x1p97f * 0x1p97f`.
fn oflowf(sign: bool) -> f32 {
    let y = f32::from_bits(0x7000_0000);
    (if sign { -y } else { y }) * y
}
/// `__math_uflowf(sign)`: `±0x1p-95f * 0x1p-95f`.
fn uflowf(sign: bool) -> f32 {
    let y = f32::from_bits(0x1000_0000);
    (if sign { -y } else { y }) * y
}
/// `__math_may_uflowf(sign)`: `±0x1.4p-75f * 0x1.4p-75f`.
fn may_uflowf(sign: bool) -> f32 {
    let y = f32::from_bits(0x1a20_0000);
    (if sign { -y } else { y }) * y
}
/// `__math_invalidf(x)`: `(x - x) / (x - x)`.
fn invalidf(x: f32) -> f32 {
    (x - x) / (x - x)
}

/// `2^x·s`-style tail of `powf` / `exp2f`: `x = k/32 + r`, `2^(k/32)` from the
/// table with `sign_bias` added to the exponent bits (it flips the sign).
#[inline(always)]
fn exp2_inline(xd: f64, sign_bias: u64) -> f64 {
    let kd0 = xd + SHIFT / EXP2F_N as f64;
    let ki = kd0.to_bits();
    let kd = kd0 - SHIFT / EXP2F_N as f64;
    let r = xd - kd;
    let t = EXP2F_TAB[(ki % EXP2F_N) as usize].wrapping_add(ki.wrapping_add(sign_bias) << (52 - 5));
    let s = d(t);
    let z = fma(EXP2F_C[0], r, EXP2F_C[1]);
    let r2 = r * r;
    let y = fma(EXP2F_C[2], r, 1.0);
    let y = fma(z, r2, y);
    y * s
}

/// `log2(x)` in double for `powf`; `ix` normalized (a subnormal has a negative
/// biased exponent).
#[inline(always)]
fn powf_log2_inline(ix: u32) -> f64 {
    const A: [f64; 5] = [
        f64::from_bits(0x3fd27616c9496e0b),
        f64::from_bits(0xbfd71969a075c67a),
        f64::from_bits(0x3fdec70a6ca7badd),
        f64::from_bits(0xbfe7154748bef6c8),
        f64::from_bits(0x3ff71547652ab82b),
    ];
    let tmp = ix.wrapping_sub(LOG2F_OFF);
    let i = ((tmp >> (23 - 4)) % 16) as usize;
    let top = tmp & 0xff80_0000;
    let iz = ix.wrapping_sub(top);
    let k = (top as i32) >> 23;
    let (invc, logc) = LOG2F_TAB[i];
    let z = f32::from_bits(iz) as f64;
    let r = fma(z, invc, -1.0);
    let y0 = logc + k as f64;
    let r2 = r * r;
    let y = fma(A[0], r, A[1]);
    let p = fma(A[2], r, A[3]);
    let r4 = r2 * r2;
    let q = fma(A[4], r, y0);
    let q = fma(p, r2, q);
    fma(y, r4, q)
}

/// 0: not an integer, 1: odd, 2: even (`iy` non-zero finite).
fn checkint(iy: u32) -> u32 {
    let e = (iy >> 23) & 0xff;
    if e < 0x7f {
        return 0;
    }
    if e > 0x7f + 23 {
        return 2;
    }
    if iy & ((1 << (0x7f + 23 - e)) - 1) != 0 {
        return 0;
    }
    if iy & (1 << (0x7f + 23 - e)) != 0 {
        return 1;
    }
    2
}

fn zeroinfnan(ix: u32) -> bool {
    ix.wrapping_mul(2).wrapping_sub(1) >= 2 * 0x7f80_0000 - 1
}

fn issignaling(x: f32) -> bool {
    2 * (x.to_bits() ^ 0x0040_0000) > 2 * 0x7fc0_0000
}

/// glibc `powf` (`e_powf.c`, x86-64 `-mfma`).
pub fn powf(x: f32, y: f32) -> f32 {
    const SIGN_BIAS: u64 = 1 << (5 + 11);
    let mut sign_bias = 0u64;
    let mut ix = x.to_bits();
    let iy = y.to_bits();
    if ix.wrapping_sub(0x0080_0000) >= 0x7f80_0000 - 0x0080_0000 || zeroinfnan(iy) {
        if zeroinfnan(iy) {
            if iy.wrapping_mul(2) == 0 {
                return if issignaling(x) { x + y } else { 1.0 };
            }
            if ix == 0x3f80_0000 {
                return if issignaling(y) { x + y } else { 1.0 };
            }
            if ix.wrapping_mul(2) > 2 * 0x7f80_0000 || iy.wrapping_mul(2) > 2 * 0x7f80_0000 {
                return x + y;
            }
            if ix.wrapping_mul(2) == 2 * 0x3f80_0000 {
                return 1.0;
            }
            if (ix.wrapping_mul(2) < 2 * 0x3f80_0000) == (iy & 0x8000_0000 == 0) {
                return 0.0;
            }
            return y * y;
        }
        if zeroinfnan(ix) {
            let mut x2 = x * x;
            if ix & 0x8000_0000 != 0 && checkint(iy) == 1 {
                x2 = -x2;
                sign_bias = 1;
            }
            if ix.wrapping_mul(2) == 0 && iy & 0x8000_0000 != 0 {
                // __math_divzerof: (sign ? -1 : 1) / 0
                return if sign_bias != 0 {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                };
            }
            return if iy & 0x8000_0000 != 0 { 1.0 / x2 } else { x2 };
        }
        // x and y are non-zero finite
        if ix & 0x8000_0000 != 0 {
            let yint = checkint(iy);
            if yint == 0 {
                return invalidf(x);
            }
            if yint == 1 {
                sign_bias = SIGN_BIAS;
            }
            ix &= 0x7fff_ffff;
        }
        if ix < 0x0080_0000 {
            // normalize a subnormal x so its exponent goes negative
            ix = (f32::from_bits(ix) * f32::from_bits(0x4b00_0000)).to_bits();
            ix &= 0x7fff_ffff;
            ix = ix.wrapping_sub(23 << 23);
        }
    }
    let ylogx = y as f64 * powf_log2_inline(ix);
    if (ylogx.to_bits() >> 47 & 0xffff) >= 126.0f64.to_bits() >> 47 {
        let neg = sign_bias != 0;
        if ylogx <= -150.0 {
            return uflowf(neg);
        }
        if ylogx < -149.0 {
            return may_uflowf(neg);
        }
        if ylogx > f64::from_bits(0x405fffffffa3aae2) {
            if ylogx > f64::from_bits(0x405fffffffd1d571) {
                return oflowf(neg);
            }
            // |x^y| > 0x1.fffffep127: round-to-nearest rounds it away from 0
            if ylogx != f64::from_bits(0x405fffffffa3aae3) {
                return oflowf(neg);
            }
            return if neg { -f32::MAX } else { f32::MAX };
        }
    }
    exp2_inline(ylogx, sign_bias) as f32
}

/// glibc `exp2f` (`e_exp2f.c`, x86-64 `-mfma`).
pub fn exp2f(x: f32) -> f32 {
    let xd = x as f64;
    let abstop = (x.to_bits() >> 20) & 0x7ff;
    if abstop >= (128.0f32.to_bits() >> 20) {
        if x.to_bits() == f32::NEG_INFINITY.to_bits() {
            return 0.0;
        }
        if abstop >= (f32::INFINITY.to_bits() >> 20) {
            return x + x;
        }
        if x > 0.0 {
            return oflowf(false);
        }
        if x <= -150.0 {
            return uflowf(false);
        }
        if x < -149.0 {
            return may_uflowf(false);
        }
    }
    exp2_inline(xd, 0) as f32
}

/// glibc `log2f` (`e_log2f.c`, x86-64 `-mfma`).
pub fn log2f(x: f32) -> f32 {
    const A: [f64; 4] = [
        f64::from_bits(0xbfd712b6f70a7e4d),
        f64::from_bits(0x3fdecabf496832e0),
        f64::from_bits(0xbfe715479ffae3de),
        f64::from_bits(0x3ff715475f35c8b8),
    ];
    let mut ix = x.to_bits();
    if ix == 0x3f80_0000 {
        return 0.0;
    }
    if ix.wrapping_sub(0x0080_0000) >= 0x7f80_0000 - 0x0080_0000 {
        if ix.wrapping_mul(2) == 0 {
            // __math_divzerof(1)
            return f32::NEG_INFINITY;
        }
        if ix == 0x7f80_0000 {
            return x;
        }
        if ix & 0x8000_0000 != 0 || ix.wrapping_mul(2) >= 0xff00_0000 {
            return invalidf(x);
        }
        ix = (x * f32::from_bits(0x4b00_0000)).to_bits();
        ix = ix.wrapping_sub(23 << 23);
    }
    let tmp = ix.wrapping_sub(LOG2F_OFF);
    let i = ((tmp >> (23 - 4)) % 16) as usize;
    let top = tmp & 0xff80_0000;
    let iz = ix.wrapping_sub(top);
    let k = (tmp as i32) >> 23;
    let (invc, logc) = LOG2F_TAB[i];
    let z = f32::from_bits(iz) as f64;
    let r = fma(z, invc, -1.0);
    let y0 = logc + k as f64;
    let r2 = r * r;
    let y = fma(A[1], r, A[2]);
    let y = fma(A[0], r2, y);
    let p = fma(A[3], r, y0);
    fma(y, r2, p) as f32
}

/// glibc `exp10f` (`e_exp10f.c`; x86-64 has no FMA variant of it, so nothing
/// here is fused).
pub fn exp10f(x: f32) -> f32 {
    // log(10)/log(2)·32, to nearest
    const INVLN10N: f64 = f64::from_bits(0x400a934f0979a371) * 32.0;
    let c = [
        EXP2F_C[0] / 32.0 / 32.0 / 32.0,
        EXP2F_C[1] / 32.0 / 32.0,
        EXP2F_C[2] / 32.0,
    ];
    let xd = x as f64;
    let abstop = (x.to_bits() >> 19) & 0xfff;
    if abstop >= (38.0f32.to_bits() >> 19) {
        if x.to_bits() == f32::NEG_INFINITY.to_bits() {
            return 0.0;
        }
        if abstop >= (f32::INFINITY.to_bits() >> 19) {
            return x + x;
        }
        // 0x26.8826ap0: the largest x with 10^x < 2^128
        if x > f32::from_bits(0x421a_209a) {
            return oflowf(false);
        }
        // -0x2d.278d4p0: the smallest x with 10^x > 2^-150
        if x < f32::from_bits(0xc234_9e35) {
            return uflowf(false);
        }
        if x < f32::from_bits(0xc233_69f4) {
            return may_uflowf(false);
        }
    }
    let z = INVLN10N * xd;
    let kd0 = z + SHIFT;
    let ki = kd0.to_bits();
    let kd = kd0 - SHIFT;
    let r = z - kd;
    let t = EXP2F_TAB[(ki % EXP2F_N) as usize].wrapping_add(ki << (52 - 5));
    let s = d(t);
    let z = c[0] * r + c[1];
    let r2 = r * r;
    let y = c[2] * r + 1.0;
    let y = z * r2 + y;
    (y * s) as f32
}

/// glibc `hypotf` (`e_hypotf.c`): `sqrt(x² + y²)` in double, rounded once.
pub fn hypotf(x: f32, y: f32) -> f32 {
    if !x.is_finite() || !y.is_finite() {
        if (x.is_infinite() || y.is_infinite()) && !issignaling(x) && !issignaling(y) {
            return f32::INFINITY;
        }
        return x + y;
    }
    let (xd, yd) = (x as f64, y as f64);
    (xd * xd + yd * yd).sqrt() as f32
}

// ── atan2f: CORE-MATH's (glibc since 2.43), correctly rounded ───────────────

/// `(xh + xl)·(ch + cl)` as a double-double.
#[inline(always)]
fn muldd(xh: f64, xl: f64, ch: f64, cl: f64) -> (f64, f64) {
    let ahlh = ch * xl;
    let alhh = cl * xh;
    let ahhh = ch * xh;
    let mut ahhl = fma(ch, xh, -ahhh);
    ahhl += alhh + ahlh;
    let h = ahhh + ahhl;
    (h, (ahhh - h) + ahhl)
}

fn polydd(xh: f64, xl: f64, c: &[[f64; 2]]) -> (f64, f64) {
    let mut i = c.len() - 1;
    let (mut ch, mut cl) = (c[i][0], c[i][1]);
    while i > 0 {
        i -= 1;
        let (h, l) = muldd(xh, xl, ch, cl);
        ch = h;
        cl = l;
        let th = ch + c[i][0];
        let tl = (c[i][0] - th) + ch;
        ch = th;
        cl += tl + c[i][1];
    }
    (ch, cl)
}

/// `y/x` tiny: `z − z³/3`, `z = y/x`.
fn atan2f_tiny(y: f32, x: f32) -> f32 {
    let (dy, dx) = (y as f64, x as f64);
    let z = dy / dx;
    let mut e = fma(-z, dx, dy);
    const C: f64 = f64::from_bits(0xbfd5555555555555); // -1/3
    let zz = z * z;
    let cz = C * z;
    e = e / dx + cz * zz;
    let mut t = z.to_bits();
    if t & 0xfff_ffff == 0 {
        // a boundary case: nudge `t` off it so rounding to float is single
        if z * e > 0.0 {
            t = t.wrapping_add(1);
        } else {
            t = t.wrapping_sub(1);
        }
    }
    let rf = d(t) as f32;
    if rf == 0.0 {
        return uflowf(t >> 63 != 0);
    }
    rf
}

/// glibc `atan2f` (`e_atan2f.c`, CORE-MATH): correctly rounded.
pub fn atan2f(y: f32, x: f32) -> f32 {
    const CN: [f64; 7] = [
        1.0,
        f64::from_bits(0x40040e0698f94c35),
        f64::from_bits(0x400248c5da347f0d),
        f64::from_bits(0x3fed873386572976),
        f64::from_bits(0x3fc46fa40b20f1d0),
        f64::from_bits(0x3f833f5e041eed0f),
        f64::from_bits(0x3f1546bbf28667c5),
    ];
    const CD: [f64; 7] = [
        1.0,
        f64::from_bits(0x4006b8b143a3f6da),
        f64::from_bits(0x4008421201d18ed5),
        f64::from_bits(0x3ff8221d086914eb),
        f64::from_bits(0x3fd670657e3a07ba),
        f64::from_bits(0x3fa0f4951fd1e72d),
        f64::from_bits(0x3f4b3874b8798286),
    ];
    const PI: f64 = f64::from_bits(0x400921fb54442d18);
    const PI2: f64 = f64::from_bits(0x3ff921fb54442d18);
    const PI2L: f64 = f64::from_bits(0x3c91a62633145c07);
    const OFF: [f64; 8] = [0.0, PI2, PI, PI2, -0.0, -PI2, -PI, -PI2];
    const OFFL: [f64; 8] = [0.0, PI2L, 2.0 * PI2L, PI2L, -0.0, -PI2L, -2.0 * PI2L, -PI2L];
    const SGN: [f64; 2] = [1.0, -1.0];
    let (ux, uy) = (x.to_bits(), y.to_bits());
    let (ax, ay) = (ux & 0x7fff_ffff, uy & 0x7fff_ffff);
    let (sx, sy) = ((ux >> 31) as usize, (uy >> 31) as usize);
    if ay >= 0xff << 23 || ax >= 0xff << 23 {
        if ay > 0xff << 23 || ax > 0xff << 23 {
            return x + y;
        }
        let (yinf, xinf) = (ay == 0xff << 23, ax == 0xff << 23);
        if yinf && xinf {
            return if sx != 0 {
                (f64::from_bits(0x4002d97c7f3321d2) * SGN[sy]) as f32 // ±3π/4
            } else {
                (f64::from_bits(0x3fe921fb54442d18) * SGN[sy]) as f32 // ±π/4
            };
        }
        if xinf {
            return if sx != 0 {
                (PI * SGN[sy]) as f32
            } else {
                (0.0 * SGN[sy]) as f32
            };
        }
        if yinf {
            return (PI2 * SGN[sy]) as f32;
        }
    }
    if ay == 0 {
        if ax == 0 {
            let i = sy * 4 + sx * 2;
            return if sx != 0 {
                (OFF[i] + OFFL[i]) as f32
            } else {
                OFF[i] as f32
            };
        }
        if sx == 0 {
            return (0.0 * SGN[sy]) as f32;
        }
    }
    let gt = (ay > ax) as usize;
    let i = sy * 4 + sx * 2 + gt;
    let (zx, zy) = (x as f64, y as f64);
    let m = [0.0, 1.0];
    let mut z = (m[gt] * zx + m[1 - gt] * zy) / (m[gt] * zy + m[1 - gt] * zx);
    let dd = ax as i32 - ay as i32;
    let mut r = if dd < (27 << 23) && dd > -(27 << 23) {
        let z2 = z * z;
        let z4 = z2 * z2;
        let z8 = z4 * z4;
        let mut cn0 = CN[0] + z2 * CN[1];
        let cn2 = CN[2] + z2 * CN[3];
        let mut cn4 = CN[4] + z2 * CN[5];
        let cn6 = CN[6];
        cn0 += z4 * cn2;
        cn4 += z4 * cn6;
        cn0 += z8 * cn4;
        let mut cd0 = CD[0] + z2 * CD[1];
        let cd2 = CD[2] + z2 * CD[3];
        let mut cd4 = CD[4] + z2 * CD[5];
        let cd6 = CD[6];
        cd0 += z4 * cd2;
        cd4 += z4 * cd6;
        cd0 += z8 * cd4;
        cn0 / cd0
    } else {
        1.0
    };
    z *= SGN[gt];
    r = z * r + OFF[i];
    if (r.to_bits().wrapping_add(8) & 0xfff_ffff) <= 16 {
        if ay < ax && (ax - ay) >> 23 >= 25 {
            return atan2f_tiny(y, x);
        }
        let (mut zh, mut zl) = if gt == 0 {
            let zh = zy / zx;
            (zh, fma(zh, -zx, zy) / zx)
        } else {
            let zh = zx / zy;
            (zh, fma(zh, -zy, zx) / zy)
        };
        let (z2h, z2l) = muldd(zh, zl, zh, zl);
        let (ph, pl) = polydd(z2h, z2l, &ATAN2F_C);
        zh *= SGN[gt];
        zl *= SGN[gt];
        let (ph, pl) = muldd(zh, zl, ph, pl);
        let sh = ph + OFF[i];
        let sl = ((OFF[i] - sh) + ph) + pl + OFFL[i];
        let th = (sh as f32) as f64;
        let dh = sh - th;
        let mut tm = dh + sl;
        let mut tth = th.to_bits();
        if th + th * f64::from_bits(0x3c30000000000000)
            == th - th * f64::from_bits(0x3c30000000000000)
        {
            tth &= 0x7ff << 52;
            tth = tth.wrapping_sub(24 << 52);
            if tm.abs() > d(tth) {
                tm *= 1.25;
            } else {
                tm *= 0.75;
            }
        }
        r = th + tm;
    }
    let rf = r as f32;
    if rf == 0.0 {
        return uflowf(rf.to_bits() >> 1 != 0);
    }
    rf
}

/// CORE-MATH `atan2f`'s accurate-path polynomial (double-double).
const ATAN2F_C: [[f64; 2]; 32] = [
    [
        f64::from_bits(0x3ff0000000000000),
        f64::from_bits(0xba88c1dac5492248),
    ],
    [
        f64::from_bits(0xbfd5555555555555),
        f64::from_bits(0xbc755553bf3a2abe),
    ],
    [
        f64::from_bits(0x3fc999999999999a),
        f64::from_bits(0xbc699deed1ec9071),
    ],
    [
        f64::from_bits(0xbfc2492492492492),
        f64::from_bits(0xbc5fd99c8d18269a),
    ],
    [
        f64::from_bits(0x3fbc71c71c71c717),
        f64::from_bits(0xbc2651eee4c4d9d0),
    ],
    [
        f64::from_bits(0xbfb745d1745d1649),
        f64::from_bits(0xbc5632683d6c44a6),
    ],
    [
        f64::from_bits(0x3fb3b13b13b11c63),
        f64::from_bits(0x3c5bf69c1f8af41d),
    ],
    [
        f64::from_bits(0xbfb11111110e6338),
        f64::from_bits(0x3c23c3e431e8bb68),
    ],
    [
        f64::from_bits(0x3fae1e1e1dc45c4a),
        f64::from_bits(0xbc4be2db05c77bbf),
    ],
    [
        f64::from_bits(0xbfaaf286b8164b4f),
        f64::from_bits(0x3c2a4673491f0942),
    ],
    [
        f64::from_bits(0x3fa86185e9ad4846),
        f64::from_bits(0x3c4e12e32d79fcee),
    ],
    [
        f64::from_bits(0xbfa642c6d5161fae),
        f64::from_bits(0x3c43ce76c1ca03f0),
    ],
    [
        f64::from_bits(0x3fa47ad6f277e5bf),
        f64::from_bits(0xbc3abd8d85bdb714),
    ],
    [
        f64::from_bits(0xbfa2f64a2ee8896d),
        f64::from_bits(0x3c2ef87d4b615323),
    ],
    [
        f64::from_bits(0x3fa1a6a2b31741b5),
        f64::from_bits(0x3c1a5d9d973547ee),
    ],
    [
        f64::from_bits(0xbfa07fbdad65e0a6),
        f64::from_bits(0xbc265ac07f5d35f4),
    ],
    [
        f64::from_bits(0x3f9ee9932a9a5f8b),
        f64::from_bits(0x3c2f8b9623f6f55a),
    ],
    [
        f64::from_bits(0xbf9ce8b5b9584dc6),
        f64::from_bits(0x3c2fe5af96e8ea2d),
    ],
    [
        f64::from_bits(0x3f9ac9cb288087b7),
        f64::from_bits(0xbc3450cdfceaf5ca),
    ],
    [
        f64::from_bits(0xbf984b025351f3e6),
        f64::from_bits(0x3c2579561b0d73da),
    ],
    [
        f64::from_bits(0x3f952f5b8ecdd52b),
        f64::from_bits(0x3c3036bd2c6fba47),
    ],
    [
        f64::from_bits(0xbf9163a8c44909dc),
        f64::from_bits(0x3c318f735ffb9f16),
    ],
    [
        f64::from_bits(0x3f8a400dce3eea6f),
        f64::from_bits(0xbc2c90569c0c1b5c),
    ],
    [
        f64::from_bits(0xbf81caa78ae6db3a),
        f64::from_bits(0xbc24c60f8161ea09),
    ],
    [
        f64::from_bits(0x3f752672453c0731),
        f64::from_bits(0x3c1834efb598c338),
    ],
    [
        f64::from_bits(0xbf65850c5be137cf),
        f64::from_bits(0xbc0445fc150ca7f5),
    ],
    [
        f64::from_bits(0x3f523eb98d22e1ca),
        f64::from_bits(0xbbf388fbaf1d7830),
    ],
    [
        f64::from_bits(0xbf38f4e974a40741),
        f64::from_bits(0x3bd271198a97da34),
    ],
    [
        f64::from_bits(0x3f1a5cf2e9cf76e5),
        f64::from_bits(0xbbb887eb4a63b665),
    ],
    [
        f64::from_bits(0xbef420c270719e32),
        f64::from_bits(0x3b8efd595b27888b),
    ],
    [
        f64::from_bits(0x3ec3ba2d69b51677),
        f64::from_bits(0xbb64fb06829cdfc7),
    ],
    [
        f64::from_bits(0xbe829b7e6f676385),
        f64::from_bits(0xbb2a783b6de718fb),
    ],
];
