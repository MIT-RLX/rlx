// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The float operations `core` lacks, in Rust: `sqrt`, `floor`, `ceil`,
//! `trunc`, `round`, `rint` (`round_ties_even`) and their f32 forms, each exact
//! or correctly rounded as IEEE 754 defines them, for `no_std` code (rlx-ir's
//! and rlx-mcu's subsets). The ports use them through [`Fp`] without the
//! `std` feature; with it, the inherent methods (single instructions) win.

#[cfg(not(any(test, feature = "std")))]
pub(crate) trait Fp: Sized {
    fn sqrt(self) -> Self;
    fn floor(self) -> Self;
    fn round(self) -> Self;
    fn round_ties_even(self) -> Self;
    fn mul_add(self, b: Self, c: Self) -> Self;
}

/// Towards zero: drop the fraction bits.
pub fn trunc(x: f64) -> f64 {
    let b = x.to_bits();
    let e = ((b >> 52) & 0x7ff) as i32 - 1023;
    if e >= 52 {
        return x; // integral, inf or NaN
    }
    if e < 0 {
        return f64::from_bits(b & (1 << 63)); // ±0
    }
    f64::from_bits(b & !((1u64 << (52 - e)) - 1))
}

/// `⌊√n⌋` (Newton from above).
fn isqrt(n: u128) -> u128 {
    if n < 2 {
        return n;
    }
    let mut x = 1u128 << (128 - n.leading_zeros()).div_ceil(2);
    loop {
        let y = (x + n / x) >> 1;
        if y >= x {
            return x;
        }
        x = y;
    }
}

/// Correctly rounded square root, on integers.
pub fn sqrt(x: f64) -> f64 {
    {
        let this = x;
        let b = this.to_bits();
        if this == 0.0 || this.is_nan() || this == f64::INFINITY {
            return this + this; // ±0, NaN (quieted) and +inf are their own roots
        }
        if b >> 63 != 0 {
            return (this - this) / (this - this); // negative: NaN
        }
        // self = m·2^e, m with its leading bit at 52 (subnormals normalized)
        let mut ef = ((b >> 52) & 0x7ff) as i32;
        let mut m = b & ((1 << 52) - 1);
        if ef == 0 {
            let sh = m.leading_zeros() as i32 - 11;
            m <<= sh;
            ef = 1 - sh;
        } else {
            m |= 1 << 52;
        }
        let mut e = ef - 1075;
        if e & 1 != 0 {
            m <<= 1;
            e -= 1;
        }
        // √(m·2^54) has 54–55 bits: one more than kept, plus the remainder
        let n = (m as u128) << 54;
        let r = isqrt(n);
        let exact = r * r == n;
        let shift = (128 - r.leading_zeros()) - 53;
        let mut q = r >> shift;
        let rem = r & ((1 << shift) - 1);
        let half = 1u128 << (shift - 1);
        if rem > half || (rem == half && (!exact || q & 1 == 1)) {
            q += 1;
        }
        let mut e2 = (e - 54) / 2 + shift as i32;
        if q >> 53 != 0 {
            q >>= 1;
            e2 += 1;
        }
        let biased = (e2 + 52 + 1023) as u64;
        f64::from_bits(biased << 52 | (q as u64 & ((1 << 52) - 1)))
    }
}

/// Towards −∞.
pub fn floor(x: f64) -> f64 {
    let t = trunc(x);
    if t != x && x < 0.0 { t - 1.0 } else { t }
}

/// Towards +∞.
pub fn ceil(x: f64) -> f64 {
    let t = trunc(x);
    if t != x && x > 0.0 { t + 1.0 } else { t }
}

/// To nearest, half away from zero.
pub fn round(x: f64) -> f64 {
    let t = trunc(x);
    let d = x - t; // exact
    if d.abs() >= 0.5 {
        t + 1.0f64.copysign(x)
    } else {
        t
    }
}

/// To nearest, half to even (C's `rint` in the default rounding mode).
pub fn rint(x: f64) -> f64 {
    let t = trunc(x);
    let d = (x - t).abs(); // exact
    let odd = t.abs() < 9007199254740992.0 && (t as i64) & 1 != 0;
    let r = if d > 0.5 || (d == 0.5 && odd) {
        t + 1.0f64.copysign(x)
    } else {
        t
    };
    r.copysign(x)
}

/// `x mod y` with the sign of `x` (C's `fmod`, Rust's `%`): always exact,
/// by long division on the significands.
pub fn fmod(x: f64, y: f64) -> f64 {
    let (mut ux, mut uy) = (x.to_bits(), y.to_bits());
    let mut ex = ((ux >> 52) & 0x7ff) as i64;
    let mut ey = ((uy >> 52) & 0x7ff) as i64;
    let sx = ux & (1 << 63);
    if uy << 1 == 0 || y.is_nan() || ex == 0x7ff {
        return (x * y) / (x * y); // y = 0, NaN operand or infinite x: NaN
    }
    if ux << 1 <= uy << 1 {
        return if ux << 1 == uy << 1 { 0.0 * x } else { x };
    }
    // normalize to integer significands with their exponents
    if ex == 0 {
        let sh = (ux << 12).leading_zeros() as i64;
        ex -= sh;
        ux <<= sh + 1;
    } else {
        ux &= u64::MAX >> 12;
        ux |= 1 << 52;
    }
    if ey == 0 {
        let sh = (uy << 12).leading_zeros() as i64;
        ey -= sh;
        uy <<= sh + 1;
    } else {
        uy &= u64::MAX >> 12;
        uy |= 1 << 52;
    }
    while ex > ey {
        if ux >= uy {
            ux -= uy;
            if ux == 0 {
                return 0.0 * x;
            }
        }
        ux <<= 1;
        ex -= 1;
    }
    if ux >= uy {
        ux -= uy;
        if ux == 0 {
            return 0.0 * x;
        }
    }
    let sh = ux.leading_zeros() as i64 - 11;
    ux <<= sh;
    ex -= sh;
    if ex > 0 {
        ux -= 1 << 52;
        ux |= (ex as u64) << 52;
    } else {
        ux >>= 1 - ex;
    }
    f64::from_bits(ux | sx)
}

/// `x − y` when `x > y`, else `+0` (C's `fdim`; NaN in, NaN out).
pub fn fdim(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        x + y
    } else if x > y {
        x - y
    } else {
        0.0
    }
}

/// [`fmod`] for `f32` (exact, so exact in `f64` too).
pub fn fmodf(x: f32, y: f32) -> f32 {
    fmod(x as f64, y as f64) as f32
}
/// [`fdim`] for `f32` (the `f64` difference of two `f32` rounds once).
pub fn fdimf(x: f32, y: f32) -> f32 {
    if x.is_nan() || y.is_nan() {
        x + y
    } else if x > y {
        x - y
    } else {
        0.0
    }
}

/// Correctly rounded `f32` square root: the `f64` root rounded once (`f64`
/// carries more than `2·24 + 2` bits, so the double rounding is exact).
pub fn sqrtf(x: f32) -> f32 {
    sqrt(x as f64) as f32
}
/// [`floor`] for `f32` (exact in `f64`).
pub fn floorf(x: f32) -> f32 {
    floor(x as f64) as f32
}
/// [`ceil`] for `f32`.
pub fn ceilf(x: f32) -> f32 {
    ceil(x as f64) as f32
}
/// [`trunc`] for `f32`.
pub fn truncf(x: f32) -> f32 {
    trunc(x as f64) as f32
}
/// [`round`] for `f32`.
pub fn roundf(x: f32) -> f32 {
    round(x as f64) as f32
}
/// [`rint`] for `f32`.
pub fn rintf(x: f32) -> f32 {
    rint(x as f64) as f32
}

#[cfg(not(any(test, feature = "std")))]
impl Fp for f64 {
    fn sqrt(self) -> f64 {
        sqrt(self)
    }
    fn floor(self) -> f64 {
        floor(self)
    }
    fn round(self) -> f64 {
        round(self)
    }
    fn round_ties_even(self) -> f64 {
        rint(self)
    }
    fn mul_add(self, b: f64, c: f64) -> f64 {
        crate::fma::fma_emulated(self, b, c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each against `std`'s instruction, bit for bit.
    #[test]
    fn matches_std() {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut r = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let same = |a: f64, b: f64| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
        for i in 0..2_000_000u32 {
            let x = if i % 2 == 0 {
                f64::from_bits(r())
            } else {
                (r() >> 11) as f64 / 1024.0 * if i % 4 == 1 { 1.0 } else { -1.0 } + 0.5
            };
            assert!(same(sqrt(x), f64::sqrt(x)), "sqrt {x:e}");
            assert!(same(ceil(x), f64::ceil(x)), "ceil {x:e}");
            assert!(same(trunc(x), f64::trunc(x)), "trunc {x:e}");
            assert!(same(floor(x), f64::floor(x)), "floor {x:e}");
            assert!(same(round(x), f64::round(x)), "round {x:e}");
            assert!(same(rint(x), f64::round_ties_even(x)), "rte {x:e}");
            let y = if i % 3 == 0 {
                f64::from_bits(r())
            } else {
                (r() >> 40) as f64 * 0.37 - 1e6
            };
            assert!(same(fmod(x, y), x % y), "fmod {x:e} {y:e}");
            let (a, b) = (x as f32, y as f32);
            let same32 = |p: f32, q: f32| p.to_bits() == q.to_bits() || (p.is_nan() && q.is_nan());
            assert!(same32(fmodf(a, b), a % b), "fmodf {a:e} {b:e}");
            assert!(same32(sqrtf(a), a.sqrt()), "sqrtf {a:e}");
        }
        for (x, y) in [
            (5.5, 0.0),
            (f64::INFINITY, 2.0),
            (3.0, f64::INFINITY),
            (-0.0, 1.0),
            (-7.0, 7.0),
            (5e-324, 3e-324),
            (1e308, 3e-324),
            (-1.0, f64::NAN),
            (f64::MAX, f64::MIN_POSITIVE),
        ] {
            assert!(same(fmod(x, y), x % y), "fmod {x:e} {y:e}");
        }
        for x in [
            0.5,
            -0.5,
            1.5,
            -1.5,
            2.5,
            -2.5,
            -0.0,
            0.0,
            4503599627370495.5,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ] {
            assert!(same(rint(x), f64::round_ties_even(x)), "rte {x}");
            assert!(same(sqrt(x), f64::sqrt(x)), "sqrt {x}");
            assert!(same(round(x), f64::round(x)), "round {x}");
        }
    }
}
