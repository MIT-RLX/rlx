// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! x87 extended precision in software: the 64-bit-significand `long double`
//! arithmetic x86-64 code runs with the FPU's default precision control
//! (round-to-nearest-even). Each operation rounds exactly once to 64 bits, as one
//! `fadd`/`fmul`/`fdiv`/`fsqrt` does; [`F80::to_f64`] is the store to a `double`
//! (the second rounding).
//!
//! Finite values only (zero and normal numbers; the exponent range is wide enough
//! that the sums of squares and quotients modelled here never leave it).

use std::cmp::Ordering;

/// An x87 extended value: `(-1)^neg · m · 2^(e-63)` with `m`'s top bit set, or zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct F80 {
    neg: bool,
    e: i32,
    m: u64,
}

impl F80 {
    pub const ZERO: F80 = F80 {
        neg: false,
        e: 0,
        m: 0,
    };

    pub fn is_zero(self) -> bool {
        self.m == 0
    }

    /// Exact conversion of a finite `f64` (`fld` of a double).
    pub fn from_f64(x: f64) -> F80 {
        assert!(x.is_finite(), "F80 models finite values");
        if x == 0.0 {
            return F80 {
                neg: x.is_sign_negative(),
                ..F80::ZERO
            };
        }
        let bits = x.to_bits();
        let neg = bits >> 63 != 0;
        let be = ((bits >> 52) & 0x7ff) as i32;
        let frac = bits & ((1 << 52) - 1);
        let (m53, e) = if be == 0 {
            (frac, -1074 + 52)
        } else {
            (frac | (1 << 52), be - 1023)
        };
        // m53 · 2^(e-52) → normalise so the top bit is bit 63
        let lz = m53.leading_zeros() as i32;
        F80 {
            neg,
            e: e - (lz - 11),
            m: m53 << lz,
        }
    }

    /// Exact conversion of an integer.
    pub fn from_u64(n: u64) -> F80 {
        if n == 0 {
            return F80::ZERO;
        }
        let lz = n.leading_zeros();
        F80 {
            neg: false,
            e: 63 - lz as i32,
            m: n << lz,
        }
    }

    pub fn neg(self) -> F80 {
        F80 {
            neg: !self.neg,
            ..self
        }
    }

    pub fn abs(self) -> F80 {
        F80 { neg: false, ..self }
    }

    /// `v · 2^(e127 - 127)` (plus `sticky`: nonzero bits below `v`'s last), rounded
    /// to 64 bits, nearest-even.
    fn round(neg: bool, e127: i32, v: u128, sticky: bool) -> F80 {
        if v == 0 {
            return F80 { neg, ..F80::ZERO };
        }
        let lz = v.leading_zeros() as i32;
        let v = v << lz;
        let e = e127 - lz;
        let m = (v >> 64) as u64;
        let rest = v as u64;
        let half = 1u64 << 63;
        let up = rest > half || (rest == half && (sticky || m & 1 != 0));
        if !up {
            return F80 { neg, e, m };
        }
        match m.checked_add(1) {
            Some(m) => F80 { neg, e, m },
            None => F80 {
                neg,
                e: e + 1,
                m: 1 << 63,
            },
        }
    }

    pub fn mul(self, o: F80) -> F80 {
        let neg = self.neg != o.neg;
        if self.is_zero() || o.is_zero() {
            return F80 { neg, ..F80::ZERO };
        }
        // m1·m2 ∈ [2^126, 2^128) · 2^(e1+e2-126)
        let p = self.m as u128 * o.m as u128;
        F80::round(neg, self.e + o.e + 1, p, false)
    }

    pub fn div(self, o: F80) -> F80 {
        assert!(!o.is_zero(), "F80 division by zero");
        let neg = self.neg != o.neg;
        if self.is_zero() {
            return F80 { neg, ..F80::ZERO };
        }
        // (m1 << 64) / m2 ∈ [2^63, 2^65), then 63 more quotient bits from the
        // remainder (a 64-bit quotient alone has no guard bit); the rest is sticky
        let num = (self.m as u128) << 64;
        let d = o.m as u128;
        let (q, r) = (num / d, num % d);
        let (q2, r2) = ((r << 63) / d, (r << 63) % d);
        let v = (q << 63) | q2;
        // v · 2^(e1 - e2 - 64 - 63)
        F80::round(neg, self.e - o.e, v, r2 != 0)
    }

    pub fn add(self, o: F80) -> F80 {
        if self.is_zero() {
            return if o.is_zero() {
                F80 {
                    neg: self.neg && o.neg,
                    ..F80::ZERO
                }
            } else {
                o
            };
        }
        if o.is_zero() {
            return self;
        }
        let (a, b) = if (self.e, self.m) >= (o.e, o.m) {
            (self, o)
        } else {
            (o, self)
        };
        let d = (a.e - b.e) as u32;
        // a at bits 126..63 (one headroom bit for the carry)
        let va = (a.m as u128) << 63;
        let (vb, sticky) = if d >= 127 {
            (0u128, true)
        } else {
            let full = (b.m as u128) << 63;
            let s = full >> d;
            (s, (s << d) != full)
        };
        if a.neg == b.neg {
            F80::round(a.neg, a.e + 1, va + vb, sticky)
        } else {
            // |a| ≥ |b|; a lost sticky bit borrows one unit from the truncated difference
            let diff = va - vb - u128::from(sticky);
            if diff == 0 && !sticky {
                return F80::ZERO;
            }
            F80::round(a.neg, a.e + 1, diff, sticky)
        }
    }

    pub fn sub(self, o: F80) -> F80 {
        self.add(o.neg())
    }

    /// `fsqrt` (non-negative operands).
    pub fn sqrt(self) -> F80 {
        assert!(!self.neg || self.is_zero(), "F80 sqrt of a negative");
        if self.is_zero() {
            return self;
        }
        // R = m·2^sh ∈ [2^126, 2^128) with e - 63 - sh even
        let mut sh = 63;
        if (self.e - 63 - sh).rem_euclid(2) != 0 {
            sh += 1;
        }
        let r = (self.m as u128) << sh;
        let s = r.isqrt();
        // a tie is impossible; round up iff R - s² > s
        let up = r - s * s > s;
        let mut m = s as u64;
        let mut e = (self.e - 63 - sh) / 2 + 63;
        if up {
            let (v, carry) = m.overflowing_add(1);
            m = if carry {
                e += 1;
                1 << 63
            } else {
                v
            };
        }
        F80 { neg: false, e, m }
    }

    /// The store to `double` (round-to-nearest-even; normal results).
    pub fn to_f64(self) -> f64 {
        if self.is_zero() {
            return if self.neg { -0.0 } else { 0.0 };
        }
        let lo = self.m & 0x7ff;
        let mut m = self.m >> 11;
        let mut e = self.e;
        if lo > 0x400 || (lo == 0x400 && m & 1 != 0) {
            m += 1;
            if m == 1 << 53 {
                m >>= 1;
                e += 1;
            }
        }
        assert!(
            (-1022..=1023).contains(&e),
            "F80 → f64 outside the normal range"
        );
        let bits =
            (u64::from(self.neg) << 63) | (((e + 1023) as u64) << 52) | (m & ((1 << 52) - 1));
        f64::from_bits(bits)
    }

    /// From the x87 register image `(sign | biased exponent, significand)` (normal
    /// values and zero).
    pub fn from_bits80(se: u16, m: u64) -> F80 {
        let neg = se >> 15 != 0;
        if m == 0 {
            return F80 { neg, ..F80::ZERO };
        }
        assert!(m >> 63 == 1, "F80 models normal values");
        F80 {
            neg,
            e: (se & 0x7fff) as i32 - 16383,
            m,
        }
    }

    /// The x87 register image: `(sign | biased exponent, significand)` as `fstpt`
    /// stores it (explicit integer bit; zero is `(sign << 15, 0)`).
    pub fn to_bits80(self) -> (u16, u64) {
        let sign = u16::from(self.neg) << 15;
        if self.is_zero() {
            return (sign, 0);
        }
        (sign | (self.e + 16383) as u16, self.m)
    }

    pub fn cmp_abs(self, o: F80) -> Ordering {
        (self.m != 0, self.e, self.m).cmp(&(o.m != 0, o.e, o.m))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn f(x: f64) -> F80 {
        F80::from_f64(x)
    }

    #[test]
    fn exact_round_trips_and_ops() {
        for x in [1.0, -2.5, 3.0e-300, 1.0e300, PI, 2.3e-308] {
            assert_eq!(f(x).to_f64(), x);
        }
        assert_eq!(f(1.5).mul(f(2.0)).to_f64(), 3.0);
        assert_eq!(f(1.0).add(f(2.0)).to_f64(), 3.0);
        assert_eq!(f(1.0).sub(f(3.0)).to_f64(), -2.0);
        assert_eq!(f(9.0).sqrt().to_f64(), 3.0);
        assert_eq!(f(1.0).div(f(4.0)).to_f64(), 0.25);
        assert!(f(1.0).sub(f(1.0)).is_zero());
    }

    /// Against a 128-bit-exact reference: the extended results round to f64 within
    /// one ulp of the f64 operation (they differ only by the double rounding).
    #[test]
    fn close_to_f64_ops() {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 1e3
        };
        for _ in 0..20000 {
            let (a, b) = (next(), next());
            let ulp = |x: f64, y: f64| (x.to_bits() as i64 - y.to_bits() as i64).abs();
            assert!(ulp(f(a).add(f(b)).to_f64(), a + b) <= 1);
            assert!(ulp(f(a).mul(f(b)).to_f64(), a * b) <= 1);
            assert!(ulp(f(a).div(f(b)).to_f64(), a / b) <= 1);
            assert!(ulp(f(a.abs()).sqrt().to_f64(), a.abs().sqrt()) <= 1);
        }
    }
}
