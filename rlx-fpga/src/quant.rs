// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Integer-only Q0.31 requantize, gemmlowp / TFLite-Micro / CMSIS-NN style.
//!
//! `rlx-cortexm`'s requant is `sat_i8(round(acc * f32_mult) + out_zp)`,
//! which leans on the M4F's FPU. FPGA fabric has no FPU — a soft one is
//! hundreds of LUTs per requant, which kills throughput — so we split
//! every real multiplier `M ∈ (0, 1)` into:
//!
//! * `M0 : i32` — the significand quantized to Q0.31, i.e. an integer in
//!   `[2^30, 2^31)` representing `significand ∈ [0.5, 1)`.
//! * `shift : i32` — a non-negative right shift applied after the
//!   multiply.
//!
//! With `M ≈ significand · 2^exp` (`exp ≤ 0`):
//!   `acc · M ≈ srdhm(acc, M0) >> shift`,
//! where `srdhm` is *saturating-rounding-doubling-high-multiply* (the
//! "round to the high 32 bits of the doubled product" gemmlowp uses),
//! and `>>` is *rounding* divide by power of two.
//!
//! Both primitives are pure integer; the Verilog version is a direct
//! transliteration of the Rust here.

#![allow(clippy::cast_possible_truncation)]

/// Saturating cast i32 → i8.
#[inline(always)]
pub fn sat_i8(v: i32) -> i8 {
    if v > i8::MAX as i32 {
        i8::MAX
    } else if v < i8::MIN as i32 {
        i8::MIN
    } else {
        v as i8
    }
}

/// Split a positive real multiplier into a Q0.31 significand `M0` and a
/// right shift. Required: `0 < m_real < 1` (NN epilogues are
/// always-shrinking; if you hit `≥ 1` the trainer should renormalize).
///
/// Returns `(M0, shift)` such that
///   `m_real ≈ M0 / 2^31 · 2^(-shift)`,
/// with `M0 ∈ [2^30, 2^31)` and `shift ∈ [0, 31]`.
pub fn quantize_multiplier(m_real: f32) -> (i32, i32) {
    assert!(
        m_real.is_finite() && m_real > 0.0,
        "quantize_multiplier: m_real must be finite and positive (got {m_real})"
    );
    assert!(
        m_real < 1.0,
        "quantize_multiplier: m_real ≥ 1 unsupported (got {m_real}); \
             FPGA epilogue assumes shrinking requant"
    );

    // Decompose the f32 directly via its bit layout: value = (1 + frac/2^23) * 2^(exp - 127).
    // Renormalize to [0.5, 1):
    //   significand = (1 + frac/2^23) / 2 = 1/2 + frac/2^24,
    //   exp_norm    = (exp - 127) + 1     = exp - 126.
    // Then M0 = round(significand · 2^31) = 2^30 + (frac << 7) (exact, no rounding).
    let bits = m_real.to_bits();
    let frac = (bits & 0x007F_FFFF) as i32;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let exp_norm = exp - 126;

    let m0 = (1i32 << 30) + (frac << 7);
    let shift = -exp_norm;

    // shift > 31 means m_real < 2^-31; underflow is rare but valid — clamp
    // at 31 (the result will round to 0, which is the right answer).
    let shift = shift.clamp(0, 31);
    (m0, shift)
}

/// Saturating Rounding Doubling High Mul — bit-exact match for
/// gemmlowp's `SaturatingRoundingDoublingHighMul`. Computes
/// `trunc_toward_zero((2·a·b) / 2^32)` with the asymmetric nudge that
/// gemmlowp / CMSIS-NN / TFLite-Micro all use, plus the lone saturation
/// point `(i32::MIN, i32::MIN) → i32::MAX`.
///
/// Note: this is *truncate-toward-zero* division by `2^31`, **not** an
/// arithmetic right shift. The two differ for negative half-values
/// (`>>` rounds toward -∞, `/` toward 0). The Verilog implementation
/// mirrors the `/` semantics explicitly.
#[inline]
pub fn srdhm(a: i32, b: i32) -> i32 {
    if a == i32::MIN && b == i32::MIN {
        return i32::MAX;
    }
    let ab = (a as i64) * (b as i64);
    let nudge: i64 = if ab >= 0 { 1 << 30 } else { 1 - (1 << 30) };
    ((ab + nudge) / (1i64 << 31)) as i32
}

/// Rounding Divide by Power of Two: arithmetic right shift with
/// round-half-away-from-zero. `shift ∈ [0, 31]`.
#[inline]
pub fn rdpot(x: i32, shift: i32) -> i32 {
    debug_assert!((0..=31).contains(&shift));
    if shift == 0 {
        return x;
    }
    let mask = (1i32 << shift) - 1;
    let remainder = x & mask;
    let threshold = (mask >> 1) + i32::from(x < 0);
    (x >> shift) + i32::from(remainder > threshold)
}

/// Full requantize epilogue: `i32` accumulator → `i8` output code.
#[inline]
pub fn requantize_q31(acc: i32, m0: i32, shift: i32, out_zp: i32) -> i8 {
    let prod = srdhm(acc, m0);
    let shifted = rdpot(prod, shift);
    sat_i8(shifted + out_zp)
}

// ─── Q0.15 variant: half-width epilogue for the `Tune` size/energy knob.
//
// Same algorithm as Q0.31 but with a 16-bit M0. The product `acc · M0`
// fits in i48 (vs i64 for Q0.31), so the multiplier is roughly half the
// area / power on real silicon. The price is ≤1 extra ulp at the
// requantize boundary — argmax is robust to that, but downstream layers
// that care about exact logits should stick with Q0.31.

/// Convert a Q0.31 `(M0, shift)` pair to its Q0.15 equivalent.
///
/// Right-shifts `M0_q31` by 16 with round-half-up; `shift` is unchanged
/// because of the algebra:
///
/// ```text
///   acc · M0_q31 / 2^31 / 2^shift  ≈  acc · (M0_q31>>16) / 2^15 / 2^shift
/// ```
///
/// The lost 16 low bits of `M0` are the source of the ≤1 ulp drift.
#[inline]
pub fn q31_to_q15(m0_q31: i32, shift: i32) -> (i16, i32) {
    // Round-half-up shift by 16. (m0_q31 is always positive in our use,
    // so no sign-asymmetry concerns.)
    let m0_q15 = ((m0_q31 as i64 + (1 << 15)) >> 16) as i32;
    // Edge case: if M0_q31 = 2^31 - 1, rounding can push M0_q15 to 2^15.
    // Re-normalize as `quantize_multiplier` would: halve and bump shift.
    let (m0_q15, shift) = if m0_q15 >= (1 << 15) {
        (m0_q15 / 2, shift - 1)
    } else {
        (m0_q15, shift)
    };
    let m0_q15 = m0_q15.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    (m0_q15, shift)
}

/// Saturating Rounding Doubling High Mul, Q0.15 variant. Computes
/// `trunc_toward_zero(2·a·b / 2^16) = trunc(a·b / 2^15)` with the same
/// asymmetric nudge gemmlowp uses. No saturation case in the i32×i16 →
/// i48 path (the `(i32::MIN, i32::MIN)` overflow only matters at the
/// full-width Q0.31 boundary).
#[inline]
pub fn srdhm_q15(a: i32, b: i16) -> i32 {
    let ab = (a as i64) * (b as i64);
    let nudge: i64 = if ab >= 0 { 1 << 14 } else { 1 - (1 << 14) };
    ((ab + nudge) / (1i64 << 15)) as i32
}

/// Q0.15 requantize epilogue.
#[inline]
pub fn requantize_q15(acc: i32, m0: i16, shift: i32, out_zp: i32) -> i8 {
    let prod = srdhm_q15(acc, m0);
    let shifted = rdpot(prod, shift);
    sat_i8(shifted + out_zp)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `quantize_multiplier(0.5)` is `(2^30, 0)`: M0=2^30 represents 0.5
    /// exactly in Q0.31, no extra shift.
    #[test]
    fn quantize_half() {
        let (m0, sh) = quantize_multiplier(0.5);
        assert_eq!(m0, 1i32 << 30);
        assert_eq!(sh, 0);
    }

    /// `quantize_multiplier(0.25)` is `(2^30, 1)`: same M0, one extra shift.
    #[test]
    fn quantize_quarter() {
        let (m0, sh) = quantize_multiplier(0.25);
        assert_eq!(m0, 1i32 << 30);
        assert_eq!(sh, 1);
    }

    /// `M0 ∈ [2^30, 2^31)` always.
    #[test]
    fn m0_in_q031_range() {
        for m in [1e-3, 1e-2, 1e-1, 0.3, 0.49, 0.99, 1e-6, 1e-9] {
            let (m0, _) = quantize_multiplier(m);
            assert!(
                m0 >= 1 << 30 && m0 < i32::MAX,
                "m={m}: M0={m0} out of [2^30, 2^31)"
            );
        }
    }

    /// `requantize_q31(acc, M0, shift, 0) ≈ round(acc · m_real)` to ≤ 1 ulp.
    #[test]
    fn requant_close_to_real_multiplier() {
        let mults = [1e-3_f32, 7.87e-3, 3e-2, 0.1, 0.5, 0.99];
        for &m in &mults {
            let (m0, shift) = quantize_multiplier(m);
            for acc in [-100_000_i32, -1, 0, 1, 100, 12_345, 100_000] {
                let got = requantize_q31(acc, m0, shift, 0) as i32;
                let want = (acc as f64 * m as f64).round() as i32;
                let want_clamped = want.clamp(i8::MIN as i32, i8::MAX as i32);
                let diff = (got - want_clamped).abs();
                assert!(
                    diff <= 1,
                    "m={m} acc={acc}: got {got}, want {want_clamped} (diff {diff})"
                );
            }
        }
    }

    /// SRDHM's lone saturation case.
    #[test]
    fn srdhm_saturates_at_min_min() {
        assert_eq!(srdhm(i32::MIN, i32::MIN), i32::MAX);
    }

    /// `srdhm(a, 2^30) ≈ a / 2` for the easy non-tie inputs. (Tie cases
    /// follow gemmlowp's truncate-toward-zero rounding — covered by the
    /// `requant_close_to_real_multiplier` test below.)
    #[test]
    fn srdhm_half_is_divide_by_two_no_ties() {
        let half = 1i32 << 30;
        assert_eq!(srdhm(0, half), 0);
        assert_eq!(srdhm(4, half), 2);
        assert_eq!(srdhm(-4, half), -2);
        assert_eq!(srdhm(100, half), 50);
        assert_eq!(srdhm(-100, half), -50);
    }

    /// `rdpot` rounds half away from zero.
    #[test]
    fn rdpot_rounds_half_away_from_zero() {
        // 5 / 2 = 2.5 → 3
        assert_eq!(rdpot(5, 1), 3);
        // -5 / 2 = -2.5 → -3
        assert_eq!(rdpot(-5, 1), -3);
        // 4 / 2 = 2
        assert_eq!(rdpot(4, 1), 2);
        // shift=0 is a no-op
        assert_eq!(rdpot(12345, 0), 12345);
    }

    /// Q0.15 stays within ≤1 ulp of Q0.31 for typical NN multipliers.
    #[test]
    fn q15_within_one_ulp_of_q31() {
        let mults = [1e-3_f32, 7.87e-3, 3e-2, 0.1, 0.5, 0.9];
        for &m in &mults {
            let (m0_q31, sh_q31) = quantize_multiplier(m);
            let (m0_q15, sh_q15) = q31_to_q15(m0_q31, sh_q31);
            for acc in [-10_000_i32, -1, 0, 1, 100, 9_999, 50_000] {
                let want = requantize_q31(acc, m0_q31, sh_q31, 0) as i32;
                let got = requantize_q15(acc, m0_q15, sh_q15, 0) as i32;
                assert!(
                    (want - got).abs() <= 1,
                    "m={m} acc={acc}: q31={want}, q15={got}"
                );
            }
        }
    }
}
