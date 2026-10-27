// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # rlx-libm — glibc's `libm`, bit for bit, in pure Rust
//!
//! NumPy, SciPy and torch on Linux call glibc's `exp`/`log`/`log2`/`pow` (and torch's
//! sampler `logf`/`sincosf`, checked against aarch64 glibc). On
//! x86_64 glibc selects the variant compiled with `-mfma -mavx2`
//! (`__exp_fma`, …) on any CPU with FMA, and gcc (`-ffp-contract=fast`) turns a
//! multiply into FMAs when every use of the product is an add or subtract. These
//! are those functions — the algorithms are ARM's optimized-routines (which glibc
//! carries verbatim; tables in [`tables`] are its data, as compiled) with each
//! FMA where gcc's build has one (gcc fuses a product only when every use of it is
//! an add/subtract in the product's own basic block) — so a wasm or non-glibc build reproduces the
//! Linux reference exactly. Verified against glibc 2.43 on x86_64 (Alder Lake).
//!
//! Exceptions and `errno` are not modelled; the returned values are.

#![allow(clippy::excessive_precision, clippy::unreadable_literal)]

mod complex;
mod erfc;
mod expm1_log1p;
mod sin;
mod tables;
pub use complex::{cexp, csqrt, hypot};
pub use erfc::erfc;
#[doc(hidden)]
pub use erfc::hard_cases as erfc_hard_cases;
pub use expm1_log1p::{expm1, log1p};
pub use sin::{cos, sin, tan};
use tables::*;

#[inline(always)]
fn d(bits: u64) -> f64 {
    f64::from_bits(bits)
}
#[inline(always)]
fn top12(x: f64) -> u32 {
    (x.to_bits() >> 52) as u32
}
#[inline(always)]
fn fma(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

/// `__math_oflow(sign)`: `±0x1p769 * 0x1p769`.
fn oflow(sign: bool) -> f64 {
    let y = if sign {
        -f64::from_bits(0x7000000000000000)
    } else {
        f64::from_bits(0x7000000000000000)
    };
    y * f64::from_bits(0x7000000000000000)
}
/// `__math_uflow(sign)`: `±0x1p-767 * 0x1p-767`.
fn uflow(sign: bool) -> f64 {
    let y = if sign {
        -f64::from_bits(0x1000000000000000)
    } else {
        f64::from_bits(0x1000000000000000)
    };
    y * f64::from_bits(0x1000000000000000)
}

// ── exp ───────────────────────────────────────────────────────────────────────

const EXP_TABLE_BITS: u32 = 7;
const EXP_N: u64 = 1 << EXP_TABLE_BITS;

fn exp_specialcase(tmp: f64, sbits: u64, ki: u64) -> f64 {
    if ki & 0x80000000 == 0 {
        // k > 0: the exponent of scale might have overflowed by <= 460
        let scale = d(sbits.wrapping_sub(1009u64 << 52));
        let y = f64::from_bits(0x7f00000000000000) * fma(scale, tmp, scale); // 0x1p1009
        return y;
    }
    // k < 0: subnormal range
    let scale = d(sbits.wrapping_add(1022u64 << 52));
    // `scale * tmp` is also used inside the `if` below — another basic block — and
    // gcc only fuses a product whose uses are all adds in its own block: unfused.
    let st = scale * tmp;
    let mut y = scale + st;
    if y < 1.0 {
        let lo = scale - y + st;
        let hi = 1.0 + y;
        let lo = 1.0 - hi + y + lo;
        y = (hi + lo) - 1.0;
        if y == 0.0 {
            y = 0.0;
        }
    }
    f64::from_bits(0x0010000000000000) * y // 0x1p-1022
}

/// glibc `exp` (`__exp_fma`).
pub fn exp(x: f64) -> f64 {
    let mut abstop = top12(x) & 0x7ff;
    if abstop.wrapping_sub(top12(f64::from_bits(0x3c90000000000000)))
        >= top12(512.0) - top12(f64::from_bits(0x3c90000000000000))
    {
        if abstop.wrapping_sub(top12(f64::from_bits(0x3c90000000000000))) >= 0x80000000 {
            return 1.0 + x; // tiny x
        }
        if abstop >= top12(1024.0) {
            if x.to_bits() == f64::NEG_INFINITY.to_bits() {
                return 0.0;
            }
            if abstop >= top12(f64::INFINITY) {
                return 1.0 + x;
            }
            return if x.to_bits() >> 63 != 0 {
                uflow(false)
            } else {
                oflow(false)
            };
        }
        abstop = 0; // large |x|: special-cased below
    }
    // z = InvLn2N*x has one use (+ Shift): gcc fuses it
    let kd0 = fma(d(EXP_DATA[EXP_INVLN2N]), x, d(EXP_DATA[EXP_SHIFT]));
    let ki = kd0.to_bits();
    let kd = kd0 - d(EXP_DATA[EXP_SHIFT]);
    let r = fma(
        kd,
        d(EXP_DATA[EXP_NEGLN2LON]),
        fma(kd, d(EXP_DATA[EXP_NEGLN2HIN]), x),
    );
    let idx = (2 * (ki % EXP_N)) as usize;
    let top = ki << (52 - EXP_TABLE_BITS);
    let tail = d(EXP_DATA[EXP_TAB + idx]);
    let sbits = EXP_DATA[EXP_TAB + idx + 1].wrapping_add(top);
    let r2 = r * r;
    let (c2, c3, c4, c5) = (
        d(EXP_DATA[EXP_POLY]),
        d(EXP_DATA[EXP_POLY + 1]),
        d(EXP_DATA[EXP_POLY + 2]),
        d(EXP_DATA[EXP_POLY + 3]),
    );
    // tmp = tail + r + r2*(C2 + r*C3) + r2*r2*(C4 + r*C5)
    let tmp = fma(r2 * r2, fma(r, c5, c4), fma(r2, fma(r, c3, c2), tail + r));
    if abstop == 0 {
        return exp_specialcase(tmp, sbits, ki);
    }
    let scale = d(sbits);
    fma(scale, tmp, scale)
}

// ── log ───────────────────────────────────────────────────────────────────────

const LOG_TABLE_BITS: u32 = 7;
const LOG_OFF: u64 = 0x3fe6000000000000;

/// glibc `log` (`__log_fma`).
pub fn log(x: f64) -> f64 {
    let mut ix = x.to_bits();
    let top = (ix >> 48) as u32;
    const LO: u64 = 0x3fee000000000000; // 1.0 - 0x1p-4
    const HI: u64 = 0x3ff1090000000000; // 1.0 + 0x1.09p-4
    let b = |i: usize| d(LOG_DATA[LOG_POLY1 + i]);
    if ix.wrapping_sub(LO) < HI - LO {
        if ix == 1.0f64.to_bits() {
            return 0.0;
        }
        let r = x - 1.0;
        let r2 = r * r;
        let r3 = r * r2;
        // y = r3*(B1 + r*B2 + r2*B3 + r3*(B4 + r*B5 + r2*B6 + r3*(B7 + r*B8 + r2*B9 + r3*B10)))
        let inner3 = fma(r3, b(10), fma(r2, b(9), fma(r, b(8), b(7))));
        let inner2 = fma(r3, inner3, fma(r2, b(6), fma(r, b(5), b(4))));
        let p = fma(r3, inner2, fma(r2, b(3), fma(r, b(2), b(1))));
        // `y = r3*(...)` is only used by `y += lo`: gcc fuses it
        // N > 64: Dekker split of r (exact power-of-two scaling, so FMA-neutral)
        let w = r * f64::from_bits(0x41a0000000000000); // 0x1p27
        let rhi = r + w - w;
        let rlo = r - rhi;
        let w = rhi * rhi * b(0);
        let hi = r + w;
        let mut lo = r - hi + w;
        lo = fma(b(0) * rlo, rhi + r, lo);
        return fma(r3, p, lo) + hi;
    }
    if top.wrapping_sub(0x0010) >= 0x7ff0 - 0x0010 {
        if ix << 1 == 0 {
            return f64::NEG_INFINITY; // __math_divzero(1)
        }
        if ix == f64::INFINITY.to_bits() {
            return x;
        }
        if top & 0x8000 != 0 || top & 0x7ff0 == 0x7ff0 {
            return (x - x) / (x - x); // __math_invalid
        }
        ix = (x * f64::from_bits(0x4330000000000000)).to_bits(); // 0x1p52
        ix = ix.wrapping_sub(52u64 << 52);
    }
    let tmp = ix.wrapping_sub(LOG_OFF);
    let i = ((tmp >> (52 - LOG_TABLE_BITS)) % (1 << LOG_TABLE_BITS)) as usize;
    let k = (tmp as i64) >> 52;
    let iz = ix.wrapping_sub(tmp & (0xfffu64 << 52));
    let invc = d(LOG_DATA[LOG_TAB + 2 * i]);
    let logc = d(LOG_DATA[LOG_TAB + 2 * i + 1]);
    let z = d(iz);
    let r = fma(z, invc, -1.0);
    let kd = k as f64;
    let (ln2hi, ln2lo) = (d(LOG_DATA[LOG_LN2HI]), d(LOG_DATA[LOG_LN2LO]));
    let w = fma(kd, ln2hi, logc);
    let hi = w + r;
    let lo = fma(kd, ln2lo, w - hi + r);
    let r2 = r * r;
    let a = |i: usize| d(LOG_DATA[LOG_POLY + i]);
    // y = lo + r2*A0 + r*r2*(A1 + r*A2 + r2*(A3 + r*A4)) + hi
    let p = fma(r2, fma(r, a(4), a(3)), fma(r, a(2), a(1)));
    let y = fma(r * r2, p, fma(r2, a(0), lo));
    y + hi
}

// ── log2 ──────────────────────────────────────────────────────────────────────

const LOG2_TABLE_BITS: u32 = 6;

/// glibc `log2` (`__log2_fma`).
pub fn log2(x: f64) -> f64 {
    let mut ix = x.to_bits();
    let top = (ix >> 48) as u32;
    const LO: u64 = 0x3feea4af00000000; // asuint64(1.0 - 0x1.5b51p-5)
    const HI: u64 = 0x3ff0b55900000000; // asuint64(1.0 + 0x1.6ab2p-5)
    let (inv_hi, inv_lo) = (d(LOG2_DATA[LOG2_INVLN2HI]), d(LOG2_DATA[LOG2_INVLN2LO]));
    if ix.wrapping_sub(LO) < HI - LO {
        if ix == 1.0f64.to_bits() {
            return 0.0;
        }
        let b = |i: usize| d(LOG2_DATA[LOG2_POLY1 + i]);
        let r = x - 1.0;
        // `hi` also feeds an explicit fma: a plain multiply
        let hi = r * inv_hi;
        let mut lo = fma(r, inv_lo, fma(r, inv_hi, -hi));
        let r2 = r * r;
        let r4 = r2 * r2;
        // p = r2*(B0 + r*B1) feeds `hi + p` and `... + p` in this block: fused into both
        let q = fma(r, b(1), b(0));
        let y = fma(r2, q, hi);
        lo += fma(r2, q, hi - y);
        let p1 = fma(r2, fma(r, b(5), b(4)), fma(r, b(3), b(2)));
        let p2 = fma(r2, fma(r, b(9), b(8)), fma(r, b(7), b(6)));
        lo = fma(r4, fma(r4, p2, p1), lo);
        return y + lo;
    }
    if top.wrapping_sub(0x0010) >= 0x7ff0 - 0x0010 {
        if ix << 1 == 0 {
            return f64::NEG_INFINITY;
        }
        if ix == f64::INFINITY.to_bits() {
            return x;
        }
        if top & 0x8000 != 0 || top & 0x7ff0 == 0x7ff0 {
            return (x - x) / (x - x);
        }
        ix = (x * f64::from_bits(0x4330000000000000)).to_bits();
        ix = ix.wrapping_sub(52u64 << 52);
    }
    let tmp = ix.wrapping_sub(LOG_OFF);
    let i = ((tmp >> (52 - LOG2_TABLE_BITS)) % (1 << LOG2_TABLE_BITS)) as usize;
    let k = (tmp as i64) >> 52;
    let iz = ix.wrapping_sub(tmp & (0xfffu64 << 52));
    let invc = d(LOG2_DATA[LOG2_TAB + 2 * i]);
    let logc = d(LOG2_DATA[LOG2_TAB + 2 * i + 1]);
    let z = d(iz);
    let kd = k as f64;
    let r = fma(z, invc, -1.0);
    let t1 = r * inv_hi;
    let t2 = fma(r, inv_lo, fma(r, inv_hi, -t1));
    let t3 = kd + logc;
    let hi = t3 + t1;
    let lo = t3 - hi + t1 + t2;
    let r2 = r * r;
    let r4 = r2 * r2;
    let a = |i: usize| d(LOG2_DATA[LOG2_POLY + i]);
    // p = A0 + r*A1 + r2*(A2 + r*A3) + r4*(A4 + r*A5); y = lo + r2*p + hi
    let p = fma(
        r4,
        fma(r, a(5), a(4)),
        fma(r2, fma(r, a(3), a(2)), fma(r, a(1), a(0))),
    );
    fma(r2, p, lo) + hi
}

// ── pow ───────────────────────────────────────────────────────────────────────

const POW_LOG_TABLE_BITS: u32 = 7;
const POW_OFF: u64 = 0x3fe6955500000000;
const SIGN_BIAS: u64 = 0x800 << EXP_TABLE_BITS;

/// `log_inline(ix, &tail)`: `(y, tail)` with `y + tail = log(x)`.
fn pow_log_inline(ix: u64) -> (f64, f64) {
    let tmp = ix.wrapping_sub(POW_OFF);
    let i = ((tmp >> (52 - POW_LOG_TABLE_BITS)) % (1 << POW_LOG_TABLE_BITS)) as usize;
    let k = (tmp as i64) >> 52;
    let iz = ix.wrapping_sub(tmp & (0xfffu64 << 52));
    let z = d(iz);
    let kd = k as f64;
    let invc = d(POW_LOG_DATA[POW_TAB + 4 * i]);
    let logc = d(POW_LOG_DATA[POW_TAB + 4 * i + 2]);
    let logctail = d(POW_LOG_DATA[POW_TAB + 4 * i + 3]);
    let (ln2hi, ln2lo) = (d(POW_LOG_DATA[POW_LN2HI]), d(POW_LOG_DATA[POW_LN2LO]));
    let a = |i: usize| d(POW_LOG_DATA[POW_POLY + i]);
    let r = fma(z, invc, -1.0);
    let t1 = fma(kd, ln2hi, logc);
    let t2 = t1 + r;
    let lo1 = fma(kd, ln2lo, logctail);
    let lo2 = t1 - t2 + r;
    // ar / ar2 / ar3 have multiply or explicit-fma uses: plain multiplies
    let ar = a(0) * r;
    let ar2 = r * ar;
    let ar3 = r * ar2;
    let hi = t2 + ar2;
    let lo3 = fma(ar, r, -ar2);
    let lo4 = t2 - hi + ar2;
    // p = ar3*(A1 + r*A2 + ar2*(A3 + r*A4 + ar2*(A5 + r*A6))), only used by the sum below
    let q = fma(
        ar2,
        fma(ar2, fma(r, a(6), a(5)), fma(r, a(4), a(3))),
        fma(r, a(2), a(1)),
    );
    let lo = fma(ar3, q, lo1 + lo2 + lo3 + lo4);
    let y = hi + lo;
    (y, hi - y + lo)
}

fn pow_specialcase(tmp: f64, sbits: u64, ki: u64) -> f64 {
    if ki & 0x80000000 == 0 {
        let scale = d(sbits.wrapping_sub(1009u64 << 52));
        let y = fma(scale, tmp, scale);
        return y * f64::from_bits(0x7f00000000000000); // 0x1p1009
    }
    let sbits = sbits.wrapping_add(1022u64 << 52);
    let scale = d(sbits);
    // `scale * tmp` is also used inside the `if`: unfused (as in exp)
    let st = scale * tmp;
    let mut y = scale + st;
    if y.abs() < 1.0 {
        let one = if y < 0.0 { -1.0 } else { 1.0 };
        let lo = scale - y + st;
        let hi = one + y;
        let lo = one - hi + y + lo;
        y = (hi + lo) - one;
        if y == 0.0 {
            y = d(sbits & 0x8000000000000000);
        }
    }
    f64::from_bits(0x0010000000000000) * y
}

fn pow_exp_inline(x: f64, xtail: f64, sign_bias: u64) -> f64 {
    let mut abstop = top12(x) & 0x7ff;
    let tiny = top12(f64::from_bits(0x3c90000000000000));
    if abstop.wrapping_sub(tiny) >= top12(512.0) - tiny {
        if abstop.wrapping_sub(tiny) >= 0x80000000 {
            let one = 1.0 + x;
            return if sign_bias != 0 { -one } else { one };
        }
        if abstop >= top12(1024.0) {
            return if x.to_bits() >> 63 != 0 {
                uflow(sign_bias != 0)
            } else {
                oflow(sign_bias != 0)
            };
        }
        abstop = 0;
    }
    let kd0 = fma(d(EXP_DATA[EXP_INVLN2N]), x, d(EXP_DATA[EXP_SHIFT]));
    let ki = kd0.to_bits();
    let kd = kd0 - d(EXP_DATA[EXP_SHIFT]);
    let mut r = fma(
        kd,
        d(EXP_DATA[EXP_NEGLN2LON]),
        fma(kd, d(EXP_DATA[EXP_NEGLN2HIN]), x),
    );
    r += xtail;
    let idx = (2 * (ki % EXP_N)) as usize;
    let top = ki.wrapping_add(sign_bias) << (52 - EXP_TABLE_BITS);
    let tail = d(EXP_DATA[EXP_TAB + idx]);
    let sbits = EXP_DATA[EXP_TAB + idx + 1].wrapping_add(top);
    let r2 = r * r;
    let (c2, c3, c4, c5) = (
        d(EXP_DATA[EXP_POLY]),
        d(EXP_DATA[EXP_POLY + 1]),
        d(EXP_DATA[EXP_POLY + 2]),
        d(EXP_DATA[EXP_POLY + 3]),
    );
    let tmp = fma(r2 * r2, fma(r, c5, c4), fma(r2, fma(r, c3, c2), tail + r));
    if abstop == 0 {
        return pow_specialcase(tmp, sbits, ki);
    }
    let scale = d(sbits);
    fma(scale, tmp, scale)
}

/// 0: not an integer, 1: odd, 2: even (for finite non-zero `iy`).
fn checkint(iy: u64) -> u32 {
    let e = (iy >> 52 & 0x7ff) as i32;
    if e < 0x3ff {
        return 0;
    }
    if e > 0x3ff + 52 {
        return 2;
    }
    let sh = (0x3ff + 52 - e) as u32;
    if iy & ((1u64 << sh) - 1) != 0 {
        return 0;
    }
    if iy & (1u64 << sh) != 0 {
        return 1;
    }
    2
}

fn zeroinfnan(i: u64) -> bool {
    (2u64.wrapping_mul(i)).wrapping_sub(1) >= 2 * f64::INFINITY.to_bits() - 1
}

fn issignaling(x: f64) -> bool {
    2u64.wrapping_mul(x.to_bits() ^ 0x0008000000000000) > 2 * 0x7ff8000000000000u64
}

/// glibc `pow` (`__pow_fma`).
pub fn pow(x: f64, y: f64) -> f64 {
    let mut sign_bias = 0u64;
    let mut ix = x.to_bits();
    let iy = y.to_bits();
    let mut topx = top12(x);
    let topy = top12(y);
    if topx.wrapping_sub(0x001) >= 0x7ff - 0x001
        || (topy & 0x7ff).wrapping_sub(0x3be) >= 0x43e - 0x3be
    {
        if zeroinfnan(iy) {
            if iy << 1 == 0 {
                return if issignaling(x) { x + y } else { 1.0 };
            }
            if ix == 1.0f64.to_bits() {
                return if issignaling(y) { x + y } else { 1.0 };
            }
            if ix << 1 > f64::INFINITY.to_bits() << 1 || iy << 1 > f64::INFINITY.to_bits() << 1 {
                return x + y;
            }
            if ix << 1 == 1.0f64.to_bits() << 1 {
                return 1.0;
            }
            if (ix << 1 < 1.0f64.to_bits() << 1) == (iy >> 63 == 0) {
                return 0.0;
            }
            return y * y;
        }
        if zeroinfnan(ix) {
            let mut x2 = x * x;
            if ix >> 63 != 0 && checkint(iy) == 1 {
                x2 = -x2;
            }
            return if iy >> 63 != 0 { 1.0 / x2 } else { x2 };
        }
        if ix >> 63 != 0 {
            let yint = checkint(iy);
            if yint == 0 {
                return (x - x) / (x - x);
            }
            if yint == 1 {
                sign_bias = SIGN_BIAS;
            }
            ix &= 0x7fffffffffffffff;
            topx &= 0x7ff;
        }
        if (topy & 0x7ff).wrapping_sub(0x3be) >= 0x43e - 0x3be {
            if ix == 1.0f64.to_bits() {
                return 1.0;
            }
            if (topy & 0x7ff) < 0x3be {
                return if ix > 1.0f64.to_bits() {
                    1.0 + y
                } else {
                    1.0 - y
                };
            }
            return if (ix > 1.0f64.to_bits()) == (topy < 0x800) {
                oflow(false)
            } else {
                uflow(false)
            };
        }
        if topx == 0 {
            ix = (f64::from_bits(ix) * f64::from_bits(0x4330000000000000)).to_bits();
            ix &= 0x7fffffffffffffff;
            ix = ix.wrapping_sub(52u64 << 52);
        }
    }
    let (hi, lo) = pow_log_inline(ix);
    // `ehi` also feeds an explicit fma: a plain multiply
    let ehi = y * hi;
    let elo = fma(y, lo, fma(y, hi, -ehi));
    pow_exp_inline(ehi, elo, sign_bias)
}

// ── single precision: `logf`, `sincosf` (sysdeps/ieee754/flt-32) ─────────────

/// glibc `logf` (`e_logf.c`; the `-mfma` x86-64 variant and aarch64 contract alike —
/// for a float result the placement of the FMAs is inert: un-fusing them changes no
/// output on 15M sampled inputs).
pub fn logf(x: f32) -> f32 {
    const OFF: u32 = 0x3f33_0000;
    const LN2: f64 = f64::from_bits(0x3fe62e42fefa39ef);
    // (invc, logc)
    const T: [(u64, u64); 16] = [
        (0x3ff661ec79f8f3be, 0xbfd57bf7808caade),
        (0x3ff571ed4aaf883d, 0xbfd2bef0a7c06ddb),
        (0x3ff49539f0f010b0, 0xbfd01eae7f513a67),
        (0x3ff3c995b0b80385, 0xbfcb31d8a68224e9),
        (0x3ff30d190c8864a5, 0xbfc6574f0ac07758),
        (0x3ff25e227b0b8ea0, 0xbfc1aa2bc79c8100),
        (0x3ff1bb4a4a1a343f, 0xbfba4e76ce8c0e5e),
        (0x3ff12358f08ae5ba, 0xbfb1973c5a611ccc),
        (0x3ff0953f419900a7, 0xbfa252f438e10c1e),
        (0x3ff0000000000000, 0x0000000000000000),
        (0x3fee608cfd9a47ac, 0x3faaa5aa5df25984),
        (0x3feca4b31f026aa0, 0x3fbc5e53aa362eb4),
        (0x3feb2036576afce6, 0x3fc526e57720db08),
        (0x3fe9c2d163a1aa2d, 0x3fcbc2860d224770),
        (0x3fe886e6037841ed, 0x3fd1058bc8a07ee1),
        (0x3fe767dcf5534862, 0x3fd4043057b6ee09),
    ];
    const A: [u64; 3] = [0xbfd00ea348b88334, 0x3fd5575b0be00b6a, 0xbfdffffef20a4123];
    let mut ix = x.to_bits();
    if ix == 0x3f80_0000 {
        return 0.0;
    }
    if ix.wrapping_sub(0x0080_0000) >= 0x7f80_0000 - 0x0080_0000 {
        if ix.wrapping_mul(2) == 0 {
            return f32::NEG_INFINITY;
        }
        if ix == 0x7f80_0000 {
            return x;
        }
        if (ix & 0x8000_0000) != 0 || ix.wrapping_mul(2) >= 0xff00_0000 {
            return f32::NAN;
        }
        ix = (x * f32::from_bits(0x4b00_0000)).to_bits();
        ix = ix.wrapping_sub(23 << 23);
    }
    let tmp = ix.wrapping_sub(OFF);
    let i = ((tmp >> (23 - 4)) % 16) as usize;
    let k = (tmp as i32) >> 23;
    let iz = ix.wrapping_sub(tmp & 0xff80_0000);
    let (invc, logc) = (f64::from_bits(T[i].0), f64::from_bits(T[i].1));
    let z = f32::from_bits(iz) as f64;
    let a = |j: usize| f64::from_bits(A[j]);
    // gcc contracts every single-use product into its add
    let r = fma(z, invc, -1.0);
    let y0 = fma(k as f64, LN2, logc);
    let r2 = r * r;
    let mut y = fma(a(1), r, a(2));
    y = fma(a(0), r2, y);
    y = fma(y, r2, y0 + r);
    y as f32
}

/// `sincos_t` (`sincosf_poly.h`): the aarch64 (`TOINT_INTRINSICS`) table.
struct SinCosT {
    sign: [f64; 4],
    hpi_inv: f64,
    hpi: f64,
    c: [f64; 5],
    s: [f64; 3],
}

const SINCOSF_TABLE: [SinCosT; 2] = [
    SinCosT {
        sign: [1.0, -1.0, -1.0, 1.0],
        hpi_inv: f64::from_bits(0x3fe45f306dc9c883),
        hpi: f64::from_bits(0x3ff921fb54442d18),
        c: [
            f64::from_bits(0x3ff0000000000000),
            f64::from_bits(0xbfdffffffd0c621c),
            f64::from_bits(0x3fa55553e1068f19),
            f64::from_bits(0xbf56c087e89a359d),
            f64::from_bits(0x3ef99343027bf8c3),
        ],
        s: [
            f64::from_bits(0xbfc555545995a603),
            f64::from_bits(0x3f81107605230bc4),
            f64::from_bits(0xbf2994eb3774cf24),
        ],
    },
    SinCosT {
        sign: [1.0, -1.0, -1.0, 1.0],
        hpi_inv: f64::from_bits(0x3fe45f306dc9c883),
        hpi: f64::from_bits(0x3ff921fb54442d18),
        c: [
            f64::from_bits(0xbff0000000000000),
            f64::from_bits(0x3fdffffffd0c621c),
            f64::from_bits(0xbfa55553e1068f19),
            f64::from_bits(0x3f56c087e89a359d),
            f64::from_bits(0xbef99343027bf8c3),
        ],
        s: [
            f64::from_bits(0xbfc555545995a603),
            f64::from_bits(0x3f81107605230bc4),
            f64::from_bits(0xbf2994eb3774cf24),
        ],
    },
];

const INV_PIO4: [u32; 24] = [
    0xa2, 0xa2f9, 0xa2f983, 0xa2f9836e, 0xf9836e4e, 0x836e4e44, 0x6e4e4415, 0x4e441529, 0x441529fc,
    0x1529fc27, 0x29fc2757, 0xfc2757d1, 0x2757d1f5, 0x57d1f534, 0xd1f534dd, 0xf534ddc0, 0x34ddc0db,
    0xddc0db62, 0xc0db6295, 0xdb629599, 0x6295993c, 0x95993c43, 0x993c4390, 0x3c439041,
];

fn abstop12f(x: f32) -> u32 {
    (x.to_bits() >> 20) & 0x7ff
}

fn sincosf_poly(x: f64, x2: f64, p: &SinCosT, n: i64) -> (f32, f32) {
    let x4 = x2 * x2;
    let x3 = x2 * x;
    let c2 = fma(x2, p.c[4], p.c[3]);
    let s1 = fma(x2, p.s[2], p.s[1]);
    let c1 = fma(x2, p.c[1], p.c[0]);
    let x5 = x3 * x2;
    let x6 = x4 * x2;
    let s = fma(x3, p.s[0], x);
    let c = fma(x4, p.c[2], c1);
    let sv = fma(x5, s1, s) as f32;
    let cv = fma(x6, c2, c) as f32;
    if n & 1 != 0 { (cv, sv) } else { (sv, cv) }
}

/// glibc `sincosf(x)` → `(sin x, cos x)` (`s_sincosf.c`) as aarch64 builds it
/// (`TOINT_INTRINSICS`: the quadrant by `round`/`lround`, `x − n·π/2` fused).
pub fn sincosf(y: f32) -> (f32, f32) {
    let pio4 = f32::from_bits(0x3f490fdb);
    let mut x = y as f64;
    let at = abstop12f(y);
    if at < abstop12f(pio4) {
        let x2 = x * x;
        if at < abstop12f(f32::from_bits(0x3980_0000)) {
            return (y, 1.0);
        }
        return sincosf_poly(x, x2, &SINCOSF_TABLE[0], 0);
    }
    if at < abstop12f(120.0) {
        let p0 = &SINCOSF_TABLE[0];
        let r = x * p0.hpi_inv;
        let n = r.round() as i64;
        x = fma(-r.round(), p0.hpi, x);
        let s = p0.sign[(n & 3) as usize];
        let p = if n & 2 != 0 { &SINCOSF_TABLE[1] } else { p0 };
        return sincosf_poly(x * s, x * x, p, n);
    }
    if at < abstop12f(f32::INFINITY) {
        let xi = y.to_bits();
        let sign = (xi >> 31) as i64;
        let arr = &INV_PIO4[((xi >> 26) & 15) as usize..];
        let shift = (xi >> 23) & 7;
        let xm = ((xi & 0xff_ffff) | 0x80_0000) << shift;
        let mut res0 = xm.wrapping_mul(arr[0]) as u64;
        let res1 = xm as u64 * arr[4] as u64;
        let res2 = xm as u64 * arr[8] as u64;
        res0 = (res2 >> 32) | (res0 << 32);
        res0 = res0.wrapping_add(res1);
        let n = res0.wrapping_add(1 << 61) >> 62;
        res0 = res0.wrapping_sub(n << 62);
        x = (res0 as i64) as f64 * f64::from_bits(0x3c1921fb54442d18);
        let n = n as i64;
        let p0 = &SINCOSF_TABLE[0];
        let s = p0.sign[((n + sign) & 3) as usize];
        let p = if (n + sign) & 2 != 0 {
            &SINCOSF_TABLE[1]
        } else {
            p0
        };
        return sincosf_poly(x * s, x * x, p, n);
    }
    let nan = y - y;
    (nan, nan)
}

/// The reference platform's libm for a port that must agree with NumPy / SciPy /
/// torch: Apple's system libm on Apple targets (the macOS reference), glibc's —
/// these pure-Rust ports — everywhere else (the Linux reference; wasm included,
/// where `f64::ln` & co. would otherwise be wasi-libc's or Rust's own).
pub mod reference {
    const APPLE: bool = cfg!(target_vendor = "apple");

    /// `log(x)`.
    pub fn log(x: f64) -> f64 {
        if APPLE { x.ln() } else { super::log(x) }
    }
    /// `exp(x)`.
    pub fn exp(x: f64) -> f64 {
        if APPLE { x.exp() } else { super::exp(x) }
    }
    /// `log2(x)`.
    pub fn log2(x: f64) -> f64 {
        if APPLE { x.log2() } else { super::log2(x) }
    }
    /// `pow(x, y)`.
    pub fn pow(x: f64, y: f64) -> f64 {
        if APPLE { x.powf(y) } else { super::pow(x, y) }
    }
    /// `sin(x)`.
    pub fn sin(x: f64) -> f64 {
        if APPLE { x.sin() } else { super::sin(x) }
    }
    /// `cos(x)`.
    pub fn cos(x: f64) -> f64 {
        if APPLE { x.cos() } else { super::cos(x) }
    }
    /// `erfc(x)` (glibc's is correctly rounded since 2.42).
    pub fn erfc(x: f64) -> f64 {
        #[cfg(target_vendor = "apple")]
        {
            unsafe extern "C" {
                #[link_name = "erfc"]
                fn c_erfc(x: f64) -> f64;
            }
            // SAFETY: a pure libm function of one double.
            unsafe { c_erfc(x) }
        }
        #[cfg(not(target_vendor = "apple"))]
        super::erfc(x)
    }
    /// `tan(x)`.
    pub fn tan(x: f64) -> f64 {
        if APPLE { x.tan() } else { super::tan(x) }
    }
    /// `hypot(x, y)`.
    pub fn hypot(x: f64, y: f64) -> f64 {
        if APPLE {
            x.hypot(y)
        } else {
            super::hypot(x, y)
        }
    }
    /// `expm1(x)`.
    pub fn expm1(x: f64) -> f64 {
        if APPLE { x.exp_m1() } else { super::expm1(x) }
    }
    /// `log1p(x)`.
    pub fn log1p(x: f64) -> f64 {
        if APPLE { x.ln_1p() } else { super::log1p(x) }
    }
    /// `logf(x)`.
    pub fn logf(x: f32) -> f32 {
        if APPLE { x.ln() } else { super::logf(x) }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn sane() {
        assert_eq!(super::exp(0.0), 1.0);
        assert!((super::exp(1.0) - std::f64::consts::E).abs() < 1e-15);
        assert_eq!(super::log(1.0), 0.0);
        assert!((super::log(std::f64::consts::E) - 1.0).abs() < 1e-15);
    }
}
