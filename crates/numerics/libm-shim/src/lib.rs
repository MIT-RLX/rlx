// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `libm` crate's API, as far as rlx's dependency tree calls it (naga's
//! constant evaluator, num-traits' `no_std` `Float`), forwarded to
//! rlx-libm's ports (glibc 2.43's algorithms). The workspace patches
//! crates.io's `libm` with this crate, so the only libm in an rlx build is
//! rlx-libm; the real crate stays a validation dev-dependency.
#![no_std]

pub use rlx_libm::{
    acos, acosf, acosh, acoshf, asin, asinf, asinh, asinhf, atan, atan2, atan2f, atanf, atanh,
    atanhf, cbrt, cbrtf, ceil, ceilf, cos, cosh, coshf, erf, erfc, erfcf, erff, exp, exp2, exp2f,
    exp10, exp10f, expf, expm1, expm1f, fdim, fdimf, floor, floorf, fmaf, fmod, fmodf, hypot,
    hypotf, lgamma, lgamma_r, lgammaf, log, log1p, log1pf, log2, log2f, log10, log10f, logf, pow,
    powf, rint, rintf, round, roundf, sin, sinh, sinhf, sqrt, sqrtf, tan, tanf, tanh, tanhf,
    tgamma, tgammaf, trunc, truncf,
};

/// `fma(a, b, c)`, correctly rounded.
pub fn fma(a: f64, b: f64, c: f64) -> f64 {
    rlx_libm::fma_emulated(a, b, c)
}
/// `sinf(x)`.
pub fn sinf(x: f32) -> f32 {
    rlx_libm::sincosf(x).0
}
/// `cosf(x)`.
pub fn cosf(x: f32) -> f32 {
    rlx_libm::sincosf(x).1
}
/// `(sinf(x), cosf(x))`.
pub fn sincosf(x: f32) -> (f32, f32) {
    rlx_libm::sincosf(x)
}
/// `(sin(x), cos(x))`.
pub fn sincos(x: f64) -> (f64, f64) {
    (rlx_libm::sin(x), rlx_libm::cos(x))
}
/// `|x|`.
pub fn fabs(x: f64) -> f64 {
    x.abs()
}
/// `|x|`.
pub fn fabsf(x: f32) -> f32 {
    x.abs()
}
/// The magnitude of `x` with the sign of `y`.
pub fn copysign(x: f64, y: f64) -> f64 {
    x.copysign(y)
}
/// The magnitude of `x` with the sign of `y`.
pub fn copysignf(x: f32, y: f32) -> f32 {
    x.copysign(y)
}
/// C's `fmax` (a NaN operand yields the other).
pub fn fmax(x: f64, y: f64) -> f64 {
    x.max(y)
}
/// C's `fmaxf`.
pub fn fmaxf(x: f32, y: f32) -> f32 {
    x.max(y)
}
/// C's `fmin`.
pub fn fmin(x: f64, y: f64) -> f64 {
    x.min(y)
}
/// C's `fminf`.
pub fn fminf(x: f32, y: f32) -> f32 {
    x.min(y)
}

#[cfg(test)]
mod tests {
    /// The shim is rlx-libm (spot-check the two families naga calls).
    #[test]
    fn forwards_to_rlx_libm() {
        for x in [-3.7f64, -0.5, 0.0, 0.25, 1.5, 2.5, 7.9, 1e10] {
            assert_eq!(super::rint(x).to_bits(), x.round_ties_even().to_bits());
            assert_eq!(super::asinh(x).to_bits(), rlx_libm::asinh(x).to_bits());
            assert_eq!(
                super::sinf(x as f32).to_bits(),
                rlx_libm::sincosf(x as f32).0.to_bits()
            );
        }
    }
}
