// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Generic low-precision numerics for **emulated mixed-precision training** —
//! quantize an `f32` to the grid of an arbitrary binary float format
//! `fXmYeZ` (`X` total bits, `Y` mantissa bits, `Z` exponent bits, 1 sign),
//! then keep computing in `f32`. This makes training work at *any* precision on
//! *any* backend (including exotic formats with no hardware kernel — `nvf4`,
//! `f8`, `bf8`) by round-tripping values through the format's representable set
//! instead of relying on native half/quarter-precision matmul.
//!
//! Pair with a straight-through estimator (quantize on the forward, identity on
//! the backward) for QAT-style low-precision training — see
//! [`Func`](crate::Func)'s QAT helpers / the `rlx-tinystories` `--fake-quant`.
//!
//! The [`float_format!`] macro generates a marker type per format with a
//! `const`-parameterized [`FloatFormat`] impl; [`parse_format`] resolves a
//! runtime spec string (`"nvf4"`, `"f8e4m3"`, `"bf8"`, or generic `"f8m3e4"`).

/// Round `x` to the nearest value representable by a binary float format with a
/// sign bit, `exp_bits` exponent bits (IEEE bias `2^(exp_bits-1) - 1`) and
/// `man_bits` explicit mantissa bits, saturating at `±max_normal` (no Inf
/// produced). Round-to-nearest-**even**; subnormals supported; `0`/NaN pass
/// through; `±Inf` saturates to `±max_normal`.
///
/// The whole grid is emulated in `f32`, so the result is an ordinary `f32` that
/// happens to be one of the format's representable values.
#[inline]
pub fn quantize(x: f32, exp_bits: u32, man_bits: u32, max_normal: f32) -> f32 {
    if x == 0.0 || x.is_nan() {
        return x;
    }
    let neg = x.is_sign_negative();
    let ax = x.abs();
    if ax >= max_normal {
        return if neg { -max_normal } else { max_normal };
    }
    let bias = (1i32 << (exp_bits - 1)) - 1;
    let emin = 1 - bias; // smallest normal exponent
    // Exponent of `ax`, floored, clamped up into the subnormal region so the
    // ULP never shrinks below the format's smallest step.
    let e = ax.log2().floor() as i32;
    let e_eff = e.max(emin);
    // Unit-in-the-last-place at this magnitude.
    let ulp = exp2i(e_eff - man_bits as i32);
    let q = round_half_even(ax / ulp) * ulp;
    let q = q.min(max_normal);
    if neg { -q } else { q }
}

/// `2^n` for a (possibly negative) integer exponent, without `powi` overflow
/// pitfalls at the extremes.
#[inline]
fn exp2i(n: i32) -> f32 {
    // f32 exponent range is [-149, 127]; clamp keeps subnormal ULPs finite.
    2f32.powi(n.clamp(-149, 127))
}

/// Round to nearest integer, ties to even (banker's rounding).
#[inline]
fn round_half_even(v: f32) -> f32 {
    let f = v.floor();
    let diff = v - f;
    if diff < 0.5 {
        f
    } else if diff > 0.5 {
        f + 1.0
    } else if (f as i64) & 1 == 0 {
        f // already even
    } else {
        f + 1.0
    }
}

/// Quantize every element of `xs` in place to `(exp_bits, man_bits, max_normal)`.
#[inline]
pub fn quantize_slice(xs: &mut [f32], exp_bits: u32, man_bits: u32, max_normal: f32) {
    for x in xs {
        *x = quantize(*x, exp_bits, man_bits, max_normal);
    }
}

/// Quantize `xs` in place with **per-tensor (absmax) scaling** — the standard
/// microscaling trick (MXFP4 / NVFP4): the largest-magnitude element is mapped
/// to the format's max, so narrow formats use their full dynamic range instead
/// of flushing everything below the smallest step to zero. `q = Q(x/s)·s` with
/// `s = absmax / max_normal`. Essential for ≤6-bit formats (e.g. `nvf4`, whose
/// smallest nonzero magnitude is 0.5 — without scaling, ~0.02 weights vanish).
#[inline]
pub fn quantize_slice_scaled(xs: &mut [f32], exp_bits: u32, man_bits: u32, max_normal: f32) {
    let absmax = xs.iter().fold(0f32, |a, &x| a.max(x.abs()));
    if absmax == 0.0 {
        return;
    }
    let scale = absmax / max_normal;
    for x in xs {
        *x = quantize(*x / scale, exp_bits, man_bits, max_normal) * scale;
    }
}

/// A compile-time-known binary float format.
pub trait FloatFormat: Copy {
    /// Exponent bits.
    const EXP_BITS: u32;
    /// Explicit mantissa bits.
    const MAN_BITS: u32;
    /// Largest finite magnitude (format-specific: e.g. 448 for E4M3, 6 for E2M1).
    const MAX_NORMAL: f32;
    /// Canonical spec string (`"f8e4m3"`, `"nvf4"`, …).
    const NAME: &'static str;
    /// Total bits (`1 + EXP_BITS + MAN_BITS`).
    const BITS: u32 = 1 + Self::EXP_BITS + Self::MAN_BITS;

    /// Round `x` to this format's grid.
    #[inline]
    fn quantize(x: f32) -> f32 {
        quantize(x, Self::EXP_BITS, Self::MAN_BITS, Self::MAX_NORMAL)
    }
}

/// Generate a marker type + [`FloatFormat`] impl per format, plus a `BUILTIN`
/// lookup table. `$name = (exp_bits, man_bits, max_normal, "spec")`.
#[macro_export]
macro_rules! float_format {
    ($($(#[$m:meta])* $name:ident = ($exp:expr, $man:expr, $max:expr, $spec:literal);)*) => {
        $(
            $(#[$m])*
            #[derive(Clone, Copy, Debug, Default)]
            pub struct $name;
            impl $crate::lowp::FloatFormat for $name {
                const EXP_BITS: u32 = $exp;
                const MAN_BITS: u32 = $man;
                const MAX_NORMAL: f32 = $max;
                const NAME: &'static str = $spec;
            }
        )*
        /// All built-in formats: `(spec, exp_bits, man_bits, max_normal)`.
        pub const BUILTIN: &[(&str, u32, u32, f32)] = &[ $(($spec, $exp, $man, $max)),* ];
    };
}

float_format! {
    /// IEEE binary16 (half).
    F16   = (5, 10, 65504.0,        "f16");
    /// bfloat16 (truncated f32; 8 exp / 7 man).
    Bf16  = (8, 7,  3.3895314e38,   "bf16");
    /// OCP FP8 **E4M3** (max 448) — the common fp8 weight/activation format.
    F8E4M3 = (4, 3, 448.0,          "f8e4m3");
    /// OCP FP8 **E5M2** — a.k.a. **bf8** (more range, less mantissa).
    F8E5M2 = (5, 2, 57344.0,        "bf8");
    /// **NVFP4** element format (**E2M1**, grid {0,±.5,±1,±1.5,±2,±3,±4,±6}).
    Nvf4  = (2, 1, 6.0,             "nvf4");
    /// FP6 E3M2 (a middle ground).
    F6E3M2 = (3, 2, 28.0,           "f6e3m2");
}

/// Resolve a runtime precision spec to `(exp_bits, man_bits, max_normal)`.
///
/// Accepts a built-in [`BUILTIN`] name / common alias (`"f8"`, `"e4m3"`,
/// `"e5m2"`, `"fp8"`, …), or a **generic** `fXmYeZ` string — `X` total bits,
/// `Y` mantissa bits, `Z` exponent bits — e.g. `"f8m3e4"` (== E4M3),
/// `"f4m1e2"` (== NVFP4). For generic specs `max_normal` uses the IEEE
/// Inf-reserving convention `(2 − 2^-Y)·2^bias`.
pub fn parse_format(spec: &str) -> Option<(u32, u32, f32)> {
    let s = spec.trim().to_ascii_lowercase();
    for (name, e, m, max) in BUILTIN {
        if s == *name {
            return Some((*e, *m, *max));
        }
    }
    match s.as_str() {
        "fp8" | "f8" | "e4m3" => return Some((4, 3, 448.0)),
        "e5m2" => return Some((5, 2, 57344.0)),
        "fp4" | "f4" | "e2m1" => return Some((2, 1, 6.0)),
        "half" => return Some((5, 10, 65504.0)),
        _ => {}
    }
    // Generic `f{X}m{Y}e{Z}`.
    let rest = s.strip_prefix('f')?;
    let (x_str, rest) = rest.split_once('m')?;
    let (y_str, z_str) = rest.split_once('e')?;
    let total: u32 = x_str.parse().ok()?;
    let man: u32 = y_str.parse().ok()?;
    let exp: u32 = z_str.parse().ok()?;
    if exp == 0 || 1 + man + exp != total {
        return None; // sign + exp + man must account for every bit
    }
    let bias = (1i32 << (exp - 1)) - 1;
    let max = (2.0 - exp2i(-(man as i32))) * exp2i(bias);
    Some((exp, man, max))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_values_pass_through() {
        // Powers of two and simple fractions are representable everywhere.
        for &(e, m, max) in &[(5u32, 10u32, 65504.0f32), (4, 3, 448.0), (2, 1, 6.0)] {
            for &v in &[0.0, 1.0, -1.0, 0.5, 2.0, -4.0] {
                assert_eq!(quantize(v, e, m, max), v, "fmt e{e}m{m} v{v}");
            }
        }
    }

    #[test]
    fn saturates_no_inf() {
        assert_eq!(quantize(1e30, 4, 3, 448.0), 448.0);
        assert_eq!(quantize(-1e30, 4, 3, 448.0), -448.0);
        assert_eq!(quantize(f32::INFINITY, 5, 10, 65504.0), 65504.0);
        assert_eq!(quantize(100.0, 2, 1, 6.0), 6.0); // nvf4 saturates at 6
    }

    #[test]
    fn nvf4_grid_is_exact() {
        // E2M1 representable magnitudes.
        for &v in &[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] {
            assert_eq!(quantize(v, 2, 1, 6.0), v, "nvf4 {v}");
            assert_eq!(quantize(-v, 2, 1, 6.0), -v);
        }
        // 5.0 rounds to the nearest of {4,6} → tie → even (4 = 2²·1.0).
        assert_eq!(quantize(5.0, 2, 1, 6.0), 4.0);
        // 2.5 → nearest of {2,3}; ulp at e=1 is 1 → RNE(2.5)=2 (even).
        assert_eq!(quantize(2.5, 2, 1, 6.0), 2.0);
    }

    #[test]
    fn rounds_toward_nearest() {
        // f16 has 10 mantissa bits: 1 + 2^-11 rounds back to 1.0.
        assert_eq!(quantize(1.0 + 2f32.powi(-11), 5, 10, 65504.0), 1.0);
        // 1 + 2^-9 is representable (a multiple of 2^-10).
        assert_eq!(
            quantize(1.0 + 2f32.powi(-9), 5, 10, 65504.0),
            1.0 + 2f32.powi(-9)
        );
        // f8e4m3 has 3 mantissa bits → step of 1/8 at [1,2): 1.1 → 1.125.
        assert_eq!(quantize(1.1, 4, 3, 448.0), 1.125);
    }

    #[test]
    fn parse_named_and_generic() {
        assert_eq!(parse_format("nvf4"), Some((2, 1, 6.0)));
        assert_eq!(parse_format("bf8"), Some((5, 2, 57344.0)));
        assert_eq!(parse_format("f8e4m3"), Some((4, 3, 448.0)));
        // Generic fXmYeZ resolves the (exp, man) split (max via IEEE convention).
        assert_eq!(parse_format("f8m3e4").map(|(e, m, _)| (e, m)), Some((4, 3)));
        assert_eq!(parse_format("f4m1e2").map(|(e, m, _)| (e, m)), Some((2, 1)));
        assert_eq!(
            parse_format("f16m10e5").map(|(e, m, _)| (e, m)),
            Some((5, 10))
        );
        // Bad specs.
        assert_eq!(parse_format("f8m4e4"), None); // 1+4+4 != 8
        assert_eq!(parse_format("nonsense"), None);
    }

    #[test]
    fn scaled_uses_full_range() {
        // Small weights (~0.02) under nvf4 *without* scaling all vanish to 0.
        let mut unscaled = vec![0.02f32, -0.05, 0.1, -0.01];
        quantize_slice(&mut unscaled, 2, 1, 6.0);
        assert!(unscaled.iter().all(|&x| x == 0.0), "{unscaled:?}");
        // *With* per-tensor scaling they survive, and the absmax maps exactly.
        let mut scaled = vec![0.02f32, -0.05, 0.1, -0.01];
        quantize_slice_scaled(&mut scaled, 2, 1, 6.0);
        assert!(scaled.iter().any(|&x| x != 0.0), "{scaled:?}");
        assert!(
            (scaled[2] - 0.1).abs() < 1e-6,
            "absmax preserved: {}",
            scaled[2]
        );
    }

    #[test]
    fn format_trait_consts() {
        assert_eq!(Nvf4::BITS, 4);
        assert_eq!(F8E4M3::BITS, 8);
        assert_eq!(F16::BITS, 16);
        assert_eq!(F8E4M3::quantize(1.1), 1.125);
    }
}
