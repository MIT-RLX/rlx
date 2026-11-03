// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A correctly rounded `f64` fused multiply-add from plain `f64` arithmetic.
//!
//! wasm has no FMA instruction, so every `f64::mul_add` in a bit-exact port
//! becomes a call to a software `fma` that works on integers — about 9 ns
//! extra per call under V8, where a multiply-add costs 2.5 ns. [`fma_emulated`]
//! gets the same correctly rounded result from about 30 `f64` operations
//! (Boldo & Melquiond, "Emulation of FMA and correctly rounded sums: proved
//! algorithms using rounding to odd", IEEE TC 2008, Algorithm 5):
//!
//! 1. `a·b = uh + ul` exactly (Dekker's product, Veltkamp split),
//! 2. `c + uh = th + tl` exactly (Knuth's TwoSum),
//! 3. `v = RO(tl + ul)`, the sum rounded to odd,
//! 4. `fma = RN(th + v)`.
//!
//! The proof needs every intermediate to stay normal and finite. Inputs whose
//! exponents could break that — and zeros, infinities, NaNs — take the exact
//! integer path ([`fma_exact`]) instead, so the result is the correctly
//! rounded `a·b + c` for every input.
//!
//! With the `wasm-fma` feature on wasm32 this also defines the `fma` symbol,
//! which replaces compiler-builtins' weak software `fma` for the whole
//! program: every `f64::mul_add` in the binary uses it, with no change at the
//! call sites.

/// Exponents (unbiased) inside which every intermediate of the emulation is
/// a normal, finite double. `|a|, |b|, |c|` in `[2^-450, 2^450)`: the product
/// and sums stay below `2^902` (the Veltkamp split multiplies by `2^27 + 1`),
/// and any nonzero exact result — at worst the sum's error term, `2^-106`
/// below the smallest input scale — stays above `2^-1022`.
const SAFE_EXP: i32 = 450;

#[inline(always)]
fn exponent(x: f64) -> i32 {
    ((x.to_bits() >> 52) & 0x7ff) as i32 - 1023
}

/// Veltkamp split: `a = hi + lo`, each with at most 26 significant bits.
#[inline(always)]
fn split(a: f64) -> (f64, f64) {
    let c = 134_217_729.0 * a; // 2^27 + 1
    let hi = c - (c - a);
    (hi, a - hi)
}

/// `a·b = p + e` exactly (Dekker), for products that neither overflow nor
/// underflow.
#[inline(always)]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let (ah, al) = split(a);
    let (bh, bl) = split(b);
    let e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
    (p, e)
}

/// `a + b = s + e` exactly (Knuth).
#[inline(always)]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let e = (a - (s - bb)) + (b - bb);
    (s, e)
}

/// `x + y` rounded to odd: exact sums are returned as they are; otherwise the
/// neighbour of the bracketing pair whose last significand bit is 1.
#[inline(always)]
fn add_round_to_odd(x: f64, y: f64) -> f64 {
    let (s, e) = two_sum(x, y);
    let bits = s.to_bits();
    if e == 0.0 || bits & 1 == 1 {
        return s;
    }
    // `s` is the even neighbour; the true sum lies beyond it toward `e`.
    // Stepping the bit pattern by one moves one ulp away from (+1) or toward
    // (−1) zero, whatever the sign of `s`.
    let away = (e > 0.0) == (s > 0.0);
    f64::from_bits(if away { bits + 1 } else { bits - 1 })
}

/// `x` as `(negative, significand, exponent)` with `x = ±m·2^e`; finite `x`.
#[inline(always)]
fn decode(x: f64) -> (bool, u64, i32) {
    let b = x.to_bits();
    let ef = ((b >> 52) & 0x7ff) as i32;
    let frac = b & ((1 << 52) - 1);
    if ef == 0 {
        (b >> 63 != 0, frac, -1074)
    } else {
        (b >> 63 != 0, frac | 1 << 52, ef - 1075)
    }
}

/// Correctly rounded `a·b + c` on integers: the exact 106-bit product, the
/// addend aligned to it with a sticky bit, one rounding to nearest-even
/// (subnormal results included). For every input; [`fma_emulated`] takes it
/// where its `f64` emulation could leave the normal range.
pub fn fma_exact(a: f64, b: f64, c: f64) -> f64 {
    if a.is_finite() && b.is_finite() && c.is_infinite() {
        // a finite product never overflows the exact sum: it is `c`
        return c;
    }
    if !a.is_finite() || !b.is_finite() || !c.is_finite() {
        // an infinite or NaN factor, or a NaN addend: IEEE's own answer
        return a * b + c;
    }
    if a == 0.0 || b == 0.0 {
        // `a·b` is exactly ±0, and the sum rounds once
        return a * b + c;
    }
    let (sa, ma, ea) = decode(a);
    let (sb, mb, eb) = decode(b);
    let (sc, mc, ec) = decode(c);
    // Both terms as `m·2^e` with the leading bit at 125 (headroom for a carry).
    const TOP: u32 = 125;
    let norm = |m: u128, e: i32| -> (u128, i32) {
        let sh = TOP as i32 - (127 - m.leading_zeros() as i32);
        (m << sh, e - sh)
    };
    let (mp, ep) = norm(ma as u128 * mb as u128, ea + eb);
    let sp = sa != sb;
    let (r, er, neg) = if mc == 0 {
        (mp, ep, sp)
    } else {
        let (mq, eq) = norm(mc as u128, ec);
        // the larger magnitude first
        let ((xm, xe, xs), (ym, ye, ys)) = if (ep, mp) >= (eq, mq) {
            ((mp, ep, sp), (mq, eq, sc))
        } else {
            ((mq, eq, sc), (mp, ep, sp))
        };
        let d = (xe - ye) as u32;
        let y = if d >= 127 {
            (ym != 0) as u128
        } else {
            (ym >> d) | ((ym & ((1u128 << d) - 1)) != 0) as u128
        };
        if xs == ys {
            (xm + y, xe, xs)
        } else {
            (xm - y, xe, xs)
        }
    };
    if r == 0 {
        return 0.0; // exact cancellation: +0 to nearest
    }
    let p = 127 - r.leading_zeros() as i32; // leading bit of r
    // keep 53 bits, fewer where the result is subnormal
    let mut shift = p - 52;
    if er + shift < -1074 {
        shift = -1074 - er;
    }
    let (mut q, e) = if shift <= 0 {
        (r << (-shift) as u32, er + shift)
    } else if shift >= 127 {
        // below half the least subnormal (r < 2^127): rounds to zero
        (0, er + shift)
    } else {
        let sh = shift as u32;
        let q = r >> sh;
        let rem = r & ((1u128 << sh) - 1);
        let half = 1u128 << (sh - 1);
        let up = rem > half || (rem == half && q & 1 == 1);
        (q + up as u128, er + shift)
    };
    let mut e = e;
    if q >> 53 != 0 {
        q >>= 1;
        e += 1;
    }
    let sign = (neg as u64) << 63;
    if q == 0 {
        return f64::from_bits(sign);
    }
    let q = q as u64;
    if q >> 52 == 0 {
        // subnormal (e == -1074)
        return f64::from_bits(sign | q);
    }
    let biased = e + 52 + 1023;
    if biased >= 0x7ff {
        return f64::from_bits(sign | 0x7ff0_0000_0000_0000);
    }
    f64::from_bits(sign | (biased as u64) << 52 | (q & ((1 << 52) - 1)))
}

/// Correctly rounded `a·b + c` (round to nearest, ties to even), from `f64`
/// arithmetic where that is provably exact and [`fma_exact`] elsewhere.
#[inline]
pub fn fma_emulated(a: f64, b: f64, c: f64) -> f64 {
    let (ea, eb, ec) = (exponent(a), exponent(b), exponent(c));
    let in_range = |e: i32| (-SAFE_EXP..SAFE_EXP).contains(&e);
    // Zeros are common in real data (padding, ReLU outputs) and cheap: with a
    // zero factor and a finite other one, `a·b` is exactly ±0, so `a*b + c`
    // rounds once — the fused result, signed zeros included. A zero `c` with an
    // in-range product leaves the single rounding of `a·b`.
    if (a == 0.0 && b.is_finite()) || (b == 0.0 && a.is_finite()) {
        return a * b + c;
    }
    if c == 0.0 && in_range(ea) && in_range(eb) {
        return a * b + c;
    }
    if !(in_range(ea) && in_range(eb) && in_range(ec)) {
        // Zeros, subnormals, huge values, infinities and NaNs.
        return fma_exact(a, b, c);
    }
    let (uh, ul) = two_prod(a, b);
    let (th, tl) = two_sum(c, uh);
    th + add_round_to_odd(tl, ul)
}

/// Correctly rounded `f32` fused multiply-add. `a·b` is exact in `f64`
/// (48 bits, never out of range); the sum is formed exactly as `s + e` and
/// rounded to odd in `f64`, which then rounds to `f32` once, correctly.
pub fn fmaf(a: f32, b: f32, c: f32) -> f32 {
    let p = a as f64 * b as f64;
    let (s, e) = two_sum(p, c as f64);
    if e == 0.0 || !s.is_finite() || s.to_bits() & 1 == 1 {
        return s as f32;
    }
    // Inexact with an even last bit: the other bracket, towards `e`, is odd.
    let up = (e > 0.0) == (s > 0.0);
    let bits = if up { s.to_bits() + 1 } else { s.to_bits() - 1 };
    f64::from_bits(bits) as f32
}

/// The program-wide `fma` (see the module docs). Not `#[inline]`: it is the
/// target of the `fma` libcall that `f64::mul_add` lowers to on wasm.
#[cfg(all(feature = "wasm-fma", target_family = "wasm"))]
#[unsafe(no_mangle)]
pub extern "C" fn fma(a: f64, b: f64, c: f64) -> f64 {
    fma_emulated(a, b, c)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`fma_exact`] against the hardware fused multiply-add (`mul_add` is
    /// one instruction on aarch64 and on x86-64 with FMA): random bit patterns,
    /// and constructed cancellation, subnormal, overflow and midpoint cases.
    #[test]
    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "fma")
    ))]
    fn fmaf_is_the_hardware_fma() {
        let mut s = 0x2545_f491_4f6c_dd1du64;
        let mut r = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for i in 0..4_000_000u32 {
            let v = r();
            let (a, b) = (f32::from_bits(v as u32), f32::from_bits((v >> 32) as u32));
            // half the cases: `c` near `-a·b` (cancellation, ties)
            let c = if i % 2 == 0 {
                f32::from_bits(r() as u32)
            } else {
                -(a * b) * f32::from_bits(0x3f80_0000 ^ (r() as u32 & 0xff))
            };
            let (x, y) = (fmaf(a, b, c), a.mul_add(b, c));
            assert!(
                x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()),
                "fmaf({a:e}, {b:e}, {c:e}) = {x:e}, hardware {y:e}"
            );
        }
    }

    #[test]
    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "fma")
    ))]
    fn fma_exact_is_the_hardware_fma() {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut r = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let same = |x: f64, y: f64| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan());
        let mut n = 0;
        for i in 0..4_000_000u32 {
            let (a, b, c) = match i % 4 {
                0 => (
                    f64::from_bits(r()),
                    f64::from_bits(r()),
                    f64::from_bits(r()),
                ),
                // cancellation: c ≈ −a·b
                1 => {
                    let a = f64::from_bits(r() >> 2 | 0x3000_0000_0000_0000);
                    let b = f64::from_bits(r() >> 2 | 0x3000_0000_0000_0000);
                    let t = (r() % 9) as i64 - 4;
                    (a, b, -f64::from_bits(((a * b).to_bits() as i64 + t) as u64))
                }
                // tiny products and addends: subnormal results
                2 => {
                    let e = |r: &mut dyn FnMut() -> u64| {
                        f64::from_bits(r() & 0x800f_ffff_ffff_ffff | ((r() % 40 + 480) << 52))
                    };
                    (
                        e(&mut r),
                        e(&mut r),
                        f64::from_bits(r() & 0x801f_ffff_ffff_ffff),
                    )
                }
                // huge: overflow, and near-overflow cancellation
                _ => {
                    let h = |r: &mut dyn FnMut() -> u64| {
                        f64::from_bits(r() & 0x800f_ffff_ffff_ffff | ((r() % 30 + 1500) << 52))
                    };
                    (h(&mut r), h(&mut r), -f64::MAX * (r() % 3) as f64)
                }
            };
            let (got, want) = (fma_exact(a, b, c), a.mul_add(b, c));
            assert!(
                same(got, want),
                "fma({a:e}, {b:e}, {c:e}) = {got:e}, hardware {want:e}"
            );
            n += 1;
        }
        assert_eq!(n, 4_000_000);
    }

    /// A deterministic stream of doubles spread over `[2^-e, 2^e)` in
    /// magnitude, both signs, with random significands.
    struct Gen(u64);
    impl Gen {
        fn next(&mut self) -> u64 {
            // splitmix64
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
        fn double(&mut self, max_exp: i64) -> f64 {
            let r = self.next();
            let exp = (r % (2 * max_exp as u64 + 1)) as i64 - max_exp;
            let mant = self.next() & ((1u64 << 52) - 1);
            let sign = (r >> 63) << 63;
            f64::from_bits(sign | (((exp + 1023) as u64) << 52) | mant)
        }
    }

    /// The emulation against the hardware FMA (aarch64 `fmadd`, x86 `vfmadd`,
    /// both correctly rounded), bit for bit, over inputs chosen to stress it:
    /// independent random operands, `c ≈ −a·b` (massive cancellation, where the
    /// round-to-odd step decides the result), and products whose low half
    /// lands exactly on a rounding tie.
    #[test]
    #[cfg(not(target_family = "wasm"))]
    fn matches_hardware_fma_bit_for_bit() {
        let mut g = Gen(0x5eed);
        let mut checked = 0u64;
        let mut check = |a: f64, b: f64, c: f64| {
            let want = a.mul_add(b, c);
            let got = fma_emulated(a, b, c);
            assert!(
                want.to_bits() == got.to_bits() || (want.is_nan() && got.is_nan()),
                "fma({a:e}, {b:e}, {c:e}) = {got:e} ({:#x}), want {want:e} ({:#x})",
                got.to_bits(),
                want.to_bits()
            );
            checked += 1;
        };
        for _ in 0..2_000_000 {
            // independent operands across the whole range, both paths
            check(g.double(1100), g.double(1100), g.double(1100));
            // fast-path operands of similar scale
            let (a, b) = (g.double(40), g.double(40));
            check(a, b, g.double(80));
            // cancellation: c is the rounded product, negated, nudged
            let p = a * b;
            let nudge = f64::from_bits(p.to_bits().wrapping_add(g.next() % 5).wrapping_sub(2));
            check(a, b, -nudge);
            // ties: c chosen so the exact sum's tail is about half an ulp
            check(a, b, -(p + p * f64::EPSILON * 0.5));
        }
        // zero factors and zero addends against in-range values
        for _ in 0..200_000 {
            let (x, y) = (g.double(40), g.double(400));
            for z in [0.0, -0.0] {
                check(z, x, y);
                check(x, z, y);
                check(x, y, z);
                check(x, -y, z);
                check(z, z, y);
                check(z, -z, z);
            }
        }
        // the special values and range edges, every combination
        let specials = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
            5e-324,
            -5e-324,
            f64::MAX,
            -f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            f64::from_bits((1023 + 449) << 52),
            f64::from_bits((1023 + 450) << 52),
            f64::from_bits((1023 - 450) << 52),
            f64::from_bits((1023 - 451) << 52),
            1.0 + f64::EPSILON,
            1.0 - f64::EPSILON / 2.0,
            3.0,
            1.0 / 3.0,
        ];
        for &a in &specials {
            for &b in &specials {
                for &c in &specials {
                    check(a, b, c);
                }
            }
        }
        assert!(checked > 8_000_000);
    }
}
