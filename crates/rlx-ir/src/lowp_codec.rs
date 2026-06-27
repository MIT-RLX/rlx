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

//! Canonical scalar f32 ↔ low-precision codec for [`ScaledFormat`].
//!
//! One place owns the bit-exact decode/encode for every native-compute
//! low-precision element format (FP8 e4m3/e5m2 OCP + AMD FNUZ, FP6 e2m3/e3m2,
//! FP4 e2m1). Hardware tensor cores do this in silicon; this module is the
//! CPU **parity oracle** every GPU backend is checked against, plus the
//! reference quantizer that produces operands for [`crate::op::Op::ScaledMatMul`].
//!
//! Decode is exact (closed-form from the field bits). Encode is
//! nearest-representable by exhaustive search of the (≤256-entry) code space —
//! simple and obviously correct for an oracle, with round-half-to-even ties and
//! saturation of overflow/inf to the largest finite magnitude (the saturating
//! quantization mode hardware uses). NaN inputs map to `0`.

use crate::quant::ScaledFormat;

/// Decode one packed low-precision code (in the low [`ScaledFormat::bit_width`]
/// bits of `code`) to f32. Returns `f32::NAN` / `±inf` for the formats that
/// encode them.
#[inline]
pub fn decode(fmt: ScaledFormat, code: u8) -> f32 {
    // FP4 has a tiny fixed table shared with the existing NVFP4 path.
    if matches!(fmt, ScaledFormat::F4E2M1) {
        return crate::nvfp4::fp4_e2m1_to_f32(code);
    }

    let (e_bits, m_bits, bias) = fmt.fields();
    let width = e_bits + m_bits; // bits excluding sign
    let sign_bit = (code >> width) & 1;
    let exp = (u32::from(code) >> m_bits) & ((1u32 << e_bits) - 1);
    let mant = u32::from(code) & ((1u32 << m_bits) - 1);
    let sign = if sign_bit == 1 { -1.0f32 } else { 1.0f32 };
    let max_exp = (1u32 << e_bits) - 1;

    // Special encodings, per family.
    if fmt.is_fnuz() {
        // AMD FNUZ: the would-be −0 slot (sign=1, exp=0, mant=0) is the
        // sole NaN; there is no infinity and no negative zero.
        if sign_bit == 1 && exp == 0 && mant == 0 {
            return f32::NAN;
        }
    } else if fmt.has_inf() {
        // OCP E5M2 (IEEE-like): max exponent → inf (mant 0) or NaN.
        if exp == max_exp {
            return if mant == 0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            };
        }
    } else if matches!(fmt, ScaledFormat::F8E4M3) {
        // OCP E4M3 (finite): only S.1111.111 is NaN; everything else finite.
        if exp == max_exp && mant == (1u32 << m_bits) - 1 {
            return f32::NAN;
        }
    }
    // FP6 e2m3 / e3m2 are all-finite — no special codes.

    let m_div = (1u32 << m_bits) as f32;
    let val = if exp == 0 {
        // Subnormal (and zero): no implicit leading 1.
        (mant as f32 / m_div) * 2f32.powi(1 - bias)
    } else {
        (1.0 + mant as f32 / m_div) * 2f32.powi(exp as i32 - bias)
    };
    sign * val
}

/// Encode an f32 to the nearest representable code (round-half-to-even,
/// saturating overflow/inf to the max finite magnitude, NaN → `0`).
#[inline]
pub fn encode(fmt: ScaledFormat, x: f32) -> u8 {
    if x.is_nan() {
        return 0;
    }
    let n_codes: u16 = 1 << fmt.bit_width();
    let mut best: u8 = 0;
    let mut best_err = f64::INFINITY;
    let mut best_mant_lsb: u8 = 1; // prefer even mantissa LSB on ties
    let xd = x as f64;
    for c in 0..n_codes {
        let code = c as u8;
        let v = decode(fmt, code);
        if !v.is_finite() {
            continue; // never round *to* inf/NaN; overflow saturates instead
        }
        let err = (v as f64 - xd).abs();
        let mant_lsb = code & 1;
        if err < best_err || (err == best_err && mant_lsb < best_mant_lsb) {
            best_err = err;
            best = code;
            best_mant_lsb = mant_lsb;
        }
    }
    best
}

/// Largest finite magnitude representable in `fmt`.
pub fn max_finite(fmt: ScaledFormat) -> f32 {
    let n_codes: u16 = 1 << fmt.bit_width();
    (0..n_codes)
        .map(|c| decode(fmt, c as u8).abs())
        .filter(|v| v.is_finite())
        .fold(0.0f32, f32::max)
}

/// Decode a whole buffer of codes (1 code per byte) into f32.
pub fn decode_slice(fmt: ScaledFormat, codes: &[u8], out: &mut [f32]) {
    debug_assert_eq!(codes.len(), out.len());
    for (o, &c) in out.iter_mut().zip(codes) {
        *o = decode(fmt, c);
    }
}

/// Round a positive f32 scale **up** to the nearest power of two and return its
/// OCP **E8M0** byte (biased exponent, bias 127; `0xFF` is NaN, `0x00` is
/// 2^−127). Used for `ScaleLayout::BlockMxE8M0`.
#[inline]
pub fn f32_to_e8m0(scale: f32) -> u8 {
    if !scale.is_finite() || scale <= 0.0 {
        return 0; // degenerate block → smallest scale
    }
    // ceil(log2(scale)) as a biased E8M0 exponent.
    let mut exp = scale.log2().ceil() as i32 + 127;
    exp = exp.clamp(0, 254);
    exp as u8
}

/// Decode an OCP E8M0 scale byte to f32 (`2^(byte-127)`).
#[inline]
pub fn e8m0_to_f32(byte: u8) -> f32 {
    if byte == 0xFF {
        return f32::NAN;
    }
    2f32.powi(byte as i32 - 127)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::ScaledFormat::*;

    const ALL: [ScaledFormat; 7] = [
        F8E4M3, F8E5M2, F8E4M3Fnuz, F8E5M2Fnuz, F6E2M3, F6E3M2, F4E2M1,
    ];

    #[test]
    fn round_trip_every_representable_code() {
        // decode → encode → decode must be a fixed point for every finite code.
        for fmt in ALL {
            let n = 1u16 << fmt.bit_width();
            for c in 0..n {
                let v = decode(fmt, c as u8);
                if !v.is_finite() {
                    continue;
                }
                let re = decode(fmt, encode(fmt, v));
                assert_eq!(
                    re, v,
                    "{fmt}: code {c:#04x} decoded {v}, re-encoded to {re}"
                );
            }
        }
    }

    #[test]
    fn known_max_finite() {
        assert_eq!(max_finite(F8E4M3), 448.0);
        assert_eq!(max_finite(F8E5M2), 57344.0);
        assert_eq!(max_finite(F8E4M3Fnuz), 240.0);
        assert_eq!(max_finite(F8E5M2Fnuz), 57344.0);
        assert_eq!(max_finite(F6E2M3), 7.5);
        assert_eq!(max_finite(F6E3M2), 28.0);
        assert_eq!(max_finite(F4E2M1), 6.0);
    }

    #[test]
    fn unity_decodes_to_one() {
        // 1.0 = (1 + 0) * 2^0 → exp == bias, mant == 0.
        for fmt in ALL {
            let (e, m, bias) = fmt.fields();
            let code = ((bias as u32) << m) as u8;
            let _ = e;
            assert_eq!(decode(fmt, code), 1.0, "{fmt}: 1.0 code {code:#04x}");
            // and encode(1.0) round-trips
            assert_eq!(decode(fmt, encode(fmt, 1.0)), 1.0, "{fmt}");
        }
    }

    #[test]
    fn e4m3_has_one_nan_no_inf() {
        assert!(decode(F8E4M3, 0x7F).is_nan());
        assert!(decode(F8E4M3, 0xFF).is_nan());
        // every other code is finite
        let finite = (0..256u16)
            .filter(|&c| decode(F8E4M3, c as u8).is_finite())
            .count();
        assert_eq!(finite, 254);
    }

    #[test]
    fn e5m2_has_inf_and_nan() {
        assert_eq!(decode(F8E5M2, 0x7C), f32::INFINITY); // 0.11111.00
        assert_eq!(decode(F8E5M2, 0xFC), f32::NEG_INFINITY);
        assert!(decode(F8E5M2, 0x7D).is_nan());
    }

    #[test]
    fn fnuz_single_nan_no_neg_zero() {
        assert!(decode(F8E4M3Fnuz, 0x80).is_nan());
        assert!(decode(F8E5M2Fnuz, 0x80).is_nan());
        // +0 only
        assert_eq!(decode(F8E4M3Fnuz, 0x00), 0.0);
        // no infinities anywhere
        assert!(!(0..256u16).any(|c| decode(F8E4M3Fnuz, c as u8).is_infinite()));
    }

    #[test]
    fn saturates_on_overflow() {
        // Values past the max finite saturate (never become inf for finite fmts).
        for fmt in ALL {
            let big = 1.0e9;
            let v = decode(fmt, encode(fmt, big));
            assert!(v.is_finite(), "{fmt} overflow not finite");
            assert_eq!(v, max_finite(fmt), "{fmt} did not saturate to max");
        }
    }

    #[test]
    fn e8m0_round_trips_powers_of_two() {
        for p in -10..=10i32 {
            let s = 2f32.powi(p);
            assert_eq!(e8m0_to_f32(f32_to_e8m0(s)), s, "2^{p}");
        }
    }
}
