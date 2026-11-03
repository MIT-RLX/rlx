// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! glibc's single-precision functions that are CORE-MATH's since 2.41:
//! correctly rounded, so any correctly rounded implementation is glibc's.
//! Each is rlx-libm's own glibc port of the function in double (`acosh` from
//! the `log` / `log1p` ports) rounded once, except on the few inputs where
//! the double's error straddles a float rounding boundary; those take glibc
//! 2.43's (correct) value from a table, as CORE-MATH settles its own hard
//! cases. Every table was found, and every function checked, against glibc
//! 2.43 on x86-64 over all 2^32 inputs: 0 mismatches.
//!
//! The tables hold for this crate's double functions: change one, and its
//! table must be found again (`tools/`: exhaustive against glibc).

#[cfg(not(any(test, feature = "std")))]
#[allow(unused_imports)]
use crate::fp::Fp as _;

macro_rules! correctly_rounded {
    ($($(#[$doc:meta])* $name:ident = $f:expr, [$(($x:literal, $y:literal)),* $(,)?];)*) => {$(
        $(#[$doc])*
        pub fn $name(x: f32) -> f32 {
            const HARD: &[(u32, u32)] = &[$(($x, $y)),*];
            let bits = x.to_bits();
            if let Some(&(_, y)) = HARD.iter().find(|&&(h, _)| h == bits) {
                return f32::from_bits(y);
            }
            let f: fn(f64) -> f64 = $f;
            f(x as f64) as f32
        }
    )*};
}

/// `acosh` in double from glibc's `log` / `log1p`: `log1p(t + √(t(x+1)))`,
/// `t = x − 1`, below 2, where both are exact enough near 1.
fn acosh_d(x: f64) -> f64 {
    if x.is_nan() || x < 1.0 {
        return (x - x) / (x - x);
    }
    if x < 2.0 {
        let t = x - 1.0;
        crate::log1p(t + (t * (x + 1.0)).sqrt())
    } else {
        crate::log(x + (x * x - 1.0).sqrt())
    }
}

correctly_rounded! {
    /// glibc `acosf`.
    acosf = crate::acos, [(0x3288_85a3, 0x3fc9_0fdb), (0x3982_6222, 0x3fc9_07b5)];
    /// glibc `asinf`.
    asinf = crate::asin, [];
    /// glibc `atanf`.
    atanf = crate::atan, [(0x3d8d_6b23, 0x3d8d_31c3), (0xbd8d_6b23, 0xbd8d_31c3)];
    /// glibc `acoshf`.
    acoshf = acosh_d, [(0x6558_90d3, 0x4254_d1f9), (0x6eb1_a8ec, 0x4284_5a89)];
    /// glibc `asinhf`.
    asinhf = crate::asinh, [
        (0x4bdd_65a5, 0x418f_034b),
        (0x6558_90d3, 0x4254_d1f9),
        (0x6eb1_a8ec, 0x4284_5a89),
        (0xcbdd_65a5, 0xc18f_034b),
        (0xe558_90d3, 0xc254_d1f9),
        (0xeeb1_a8ec, 0xc284_5a89),
    ];
    /// glibc `atanhf`.
    atanhf = crate::atanh, [];
    /// glibc `cbrtf`.
    cbrtf = crate::cbrt, [];
    /// glibc `coshf`.
    coshf = crate::cosh, [];
    /// glibc `sinhf`.
    sinhf = crate::sinh, [(0x3a12_85ff, 0x3a12_85ff), (0xba12_85ff, 0xba12_85ff)];
    /// glibc `tanf`.
    tanf = crate::tan, [];
    /// glibc `expm1f`.
    expm1f = crate::expm1, [];
    /// glibc `log10f`.
    log10f = crate::log10, [(0x0efe_ee7a, 0xc1e9_9d23)];
    /// glibc `erff`.
    erff = crate::erf, [];
    /// glibc `erfcf`.
    erfcf = crate::erfc, [(0xb76c_9f62, 0x3f80_0085)];
    /// glibc `lgammaf`.
    lgammaf = crate::lgamma, [(0x3b7c_53aa, 0x40b1_d661), (0xb0d6_f2ca, 0x41a2_3559)];
    /// glibc `tanhf`.
    tanhf = crate::tanh, [];
    /// glibc `tgammaf`.
    tgammaf = crate::tgamma, [(0x27de_86a9, 0x5713_4133), (0x27e0_5475, 0x5712_1211)];
}
