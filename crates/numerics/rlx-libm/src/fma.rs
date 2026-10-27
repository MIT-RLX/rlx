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
//! integer path ([`libm_crate::fma`]) instead, so the result is the correctly
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

/// Correctly rounded `a·b + c` (round to nearest, ties to even), from `f64`
/// arithmetic where that is provably exact and [`libm_crate::fma`] elsewhere.
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
        return libm_crate::fma(a, b, c);
    }
    let (uh, ul) = two_prod(a, b);
    let (th, tl) = two_sum(c, uh);
    th + add_round_to_odd(tl, ul)
}

/// The program-wide `fma` (see the module docs). Not `#[inline]`: it is the
/// target of the `fma` libcall that `f64::mul_add` lowers to on wasm.
#[cfg(all(feature = "wasm-fma", target_arch = "wasm32"))]
#[unsafe(no_mangle)]
pub extern "C" fn fma(a: f64, b: f64, c: f64) -> f64 {
    fma_emulated(a, b, c)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    #[cfg(not(target_arch = "wasm32"))]
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
