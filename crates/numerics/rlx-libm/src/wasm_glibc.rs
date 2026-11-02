// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! glibc's libm for the whole wasm program (feature `wasm-glibc`).
//!
//! On wasm32, `f32::exp`, `f64::sin`, … lower to libcalls that
//! compiler-builtins answers with weak definitions of musl's algorithms. Those
//! round differently from glibc (musl `cosf` differs from glibc's on about 1%
//! of Hann-window arguments), so a browser build cannot reproduce a Linux run
//! that calls them. Defining the symbols here replaces the weak ones for every
//! call in the binary, in every crate, with the ports above. Each export was
//! checked against glibc 2.43 on x86-64: the f32 ones on all 2^32 inputs, the
//! f64 ones on 3.2·10^8 samples each, with 0 mismatches.
//!
//! Functions with no port here (`powf`, `atan2`, …) stay musl's, unless
//! `export-glibc` is on: then they are the `libm` crate's (pure Rust, the
//! same code on every target), and the whole set is exported natively too, so
//! a macOS or Linux binary rounds as the browser does.
#![allow(unsafe_code)]

macro_rules! export {
    ($($sym:literal => fn $name:ident($($a:ident: $t:ty),*) -> $r:ty = $f:path;)*) => {$(
        #[unsafe(export_name = $sym)]
        pub extern "C" fn $name($($a: $t),*) -> $r {
            $f($($a),*)
        }
    )*};
}

fn sinf_(x: f32) -> f32 {
    crate::sincosf(x).0
}
fn cosf_(x: f32) -> f32 {
    crate::sincosf(x).1
}

export! {
    "expf" => fn glibc_expf(x: f32) -> f32 = crate::expf;
    "logf" => fn glibc_logf(x: f32) -> f32 = crate::logf;
    "log1pf" => fn glibc_log1pf(x: f32) -> f32 = crate::log1pf;
    "tanhf" => fn glibc_tanhf(x: f32) -> f32 = crate::tanhf;
    "sinf" => fn glibc_sinf(x: f32) -> f32 = sinf_;
    "cosf" => fn glibc_cosf(x: f32) -> f32 = cosf_;
    "exp" => fn glibc_exp(x: f64) -> f64 = crate::exp;
    "log" => fn glibc_log(x: f64) -> f64 = crate::log;
    "log2" => fn glibc_log2(x: f64) -> f64 = crate::log2;
    "pow" => fn glibc_pow(x: f64, y: f64) -> f64 = crate::pow;
    "sin" => fn glibc_sin(x: f64) -> f64 = crate::sin;
    "cos" => fn glibc_cos(x: f64) -> f64 = crate::cos;
    "tan" => fn glibc_tan(x: f64) -> f64 = crate::tan;
    "expm1" => fn glibc_expm1(x: f64) -> f64 = crate::expm1;
    "log1p" => fn glibc_log1p(x: f64) -> f64 = crate::log1p;
    "hypot" => fn glibc_hypot(x: f64, y: f64) -> f64 = crate::hypot;
    "erfc" => fn glibc_erfc(x: f64) -> f64 = crate::erfc;
}

/// The functions with no glibc port: the `libm` crate's, on every target.
#[cfg(feature = "export-glibc")]
mod rest {
    use libm_crate as m;
    export! {
        "acos" => fn x_acos(x: f64) -> f64 = m::acos;
        "acosf" => fn x_acosf(x: f32) -> f32 = m::acosf;
        "asin" => fn x_asin(x: f64) -> f64 = m::asin;
        "asinf" => fn x_asinf(x: f32) -> f32 = m::asinf;
        "atan" => fn x_atan(x: f64) -> f64 = m::atan;
        "atanf" => fn x_atanf(x: f32) -> f32 = m::atanf;
        "atan2" => fn x_atan2(y: f64, x: f64) -> f64 = m::atan2;
        "atan2f" => fn x_atan2f(y: f32, x: f32) -> f32 = m::atan2f;
        "acosh" => fn x_acosh(x: f64) -> f64 = m::acosh;
        "acoshf" => fn x_acoshf(x: f32) -> f32 = m::acoshf;
        "asinh" => fn x_asinh(x: f64) -> f64 = m::asinh;
        "asinhf" => fn x_asinhf(x: f32) -> f32 = m::asinhf;
        "atanh" => fn x_atanh(x: f64) -> f64 = m::atanh;
        "atanhf" => fn x_atanhf(x: f32) -> f32 = m::atanhf;
        "cbrt" => fn x_cbrt(x: f64) -> f64 = m::cbrt;
        "cbrtf" => fn x_cbrtf(x: f32) -> f32 = m::cbrtf;
        "cosh" => fn x_cosh(x: f64) -> f64 = m::cosh;
        "coshf" => fn x_coshf(x: f32) -> f32 = m::coshf;
        "sinh" => fn x_sinh(x: f64) -> f64 = m::sinh;
        "sinhf" => fn x_sinhf(x: f32) -> f32 = m::sinhf;
        "tanh" => fn x_tanh(x: f64) -> f64 = m::tanh;
        "tanf" => fn x_tanf(x: f32) -> f32 = m::tanf;
        "exp2" => fn x_exp2(x: f64) -> f64 = m::exp2;
        "exp2f" => fn x_exp2f(x: f32) -> f32 = m::exp2f;
        "exp10" => fn x_exp10(x: f64) -> f64 = m::exp10;
        "exp10f" => fn x_exp10f(x: f32) -> f32 = m::exp10f;
        "__exp10" => fn x_apple_exp10(x: f64) -> f64 = m::exp10;
        "__exp10f" => fn x_apple_exp10f(x: f32) -> f32 = m::exp10f;
        "expm1f" => fn x_expm1f(x: f32) -> f32 = m::expm1f;
        "log10" => fn x_log10(x: f64) -> f64 = m::log10;
        "log10f" => fn x_log10f(x: f32) -> f32 = m::log10f;
        "log2f" => fn x_log2f(x: f32) -> f32 = m::log2f;
        "powf" => fn x_powf(x: f32, y: f32) -> f32 = m::powf;
        "hypotf" => fn x_hypotf(x: f32, y: f32) -> f32 = m::hypotf;
        "erf" => fn x_erf(x: f64) -> f64 = m::erf;
        "erff" => fn x_erff(x: f32) -> f32 = m::erff;
        "erfcf" => fn x_erfcf(x: f32) -> f32 = m::erfcf;
        "lgamma" => fn x_lgamma(x: f64) -> f64 = m::lgamma;
        "lgammaf" => fn x_lgammaf(x: f32) -> f32 = m::lgammaf;
        "tgamma" => fn x_tgamma(x: f64) -> f64 = m::tgamma;
        "tgammaf" => fn x_tgammaf(x: f32) -> f32 = m::tgammaf;
    }

    // A sin and a cos of one argument: LLVM merges them into one call.
    #[cfg(target_vendor = "apple")]
    #[repr(C)]
    pub struct Pair64(f64, f64);
    #[cfg(target_vendor = "apple")]
    #[repr(C)]
    pub struct Pair32(f32, f32);
    #[cfg(target_vendor = "apple")]
    #[unsafe(export_name = "__sincos_stret")]
    pub extern "C" fn x_sincos_stret(x: f64) -> Pair64 {
        Pair64(crate::sin(x), crate::cos(x))
    }
    #[cfg(target_vendor = "apple")]
    #[unsafe(export_name = "__sincosf_stret")]
    pub extern "C" fn x_sincosf_stret(x: f32) -> Pair32 {
        let (s, c) = crate::sincosf(x);
        Pair32(s, c)
    }
    #[cfg(not(target_vendor = "apple"))]
    #[unsafe(export_name = "sincos")]
    pub unsafe extern "C" fn x_sincos(x: f64, s: *mut f64, c: *mut f64) {
        unsafe {
            *s = crate::sin(x);
            *c = crate::cos(x);
        }
    }
    #[cfg(not(target_vendor = "apple"))]
    #[unsafe(export_name = "sincosf")]
    pub unsafe extern "C" fn x_sincosf(x: f32, s: *mut f32, c: *mut f32) {
        let (sv, cv) = crate::sincosf(x);
        unsafe {
            *s = sv;
            *c = cv;
        }
    }
}
