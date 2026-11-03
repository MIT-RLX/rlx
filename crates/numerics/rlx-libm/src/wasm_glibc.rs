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
//! With `export-glibc` the same set is exported natively too, so a macOS or
//! Linux binary rounds as the browser does.
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
    "tanh" => fn glibc_tanh(x: f64) -> f64 = crate::tanh;
    "cbrt" => fn glibc_cbrt(x: f64) -> f64 = crate::cbrt;
    "cosh" => fn glibc_cosh(x: f64) -> f64 = crate::cosh;
    "sinh" => fn glibc_sinh(x: f64) -> f64 = crate::sinh;
    "log10" => fn glibc_log10(x: f64) -> f64 = crate::log10;
    "exp2" => fn glibc_exp2(x: f64) -> f64 = crate::exp2;
    "exp10" => fn glibc_exp10(x: f64) -> f64 = crate::exp10;
    "__exp10" => fn glibc_apple_exp10(x: f64) -> f64 = crate::exp10;
    "erf" => fn glibc_erf(x: f64) -> f64 = crate::erf;
    "atan" => fn glibc_atan(x: f64) -> f64 = crate::atan;
    "asin" => fn glibc_asin(x: f64) -> f64 = crate::asin;
    "acos" => fn glibc_acos(x: f64) -> f64 = crate::acos;
    "atanh" => fn glibc_atanh(x: f64) -> f64 = crate::atanh;
    "asinh" => fn glibc_asinh(x: f64) -> f64 = crate::asinh;
    "acosh" => fn glibc_acosh(x: f64) -> f64 = crate::acosh;
    "tgamma" => fn glibc_tgamma(x: f64) -> f64 = crate::tgamma;
    "lgamma" => fn glibc_lgamma(x: f64) -> f64 = crate::lgamma;
    "atan2" => fn glibc_atan2(y: f64, x: f64) -> f64 = crate::atan2;
    "powf" => fn glibc_powf(x: f32, y: f32) -> f32 = crate::powf;
    "exp2f" => fn glibc_exp2f(x: f32) -> f32 = crate::exp2f;
    "exp10f" => fn glibc_exp10f(x: f32) -> f32 = crate::exp10f;
    "__exp10f" => fn glibc_apple_exp10f(x: f32) -> f32 = crate::exp10f;
    "log2f" => fn glibc_log2f(x: f32) -> f32 = crate::log2f;
    "hypotf" => fn glibc_hypotf(x: f32, y: f32) -> f32 = crate::hypotf;
    "atan2f" => fn glibc_atan2f(y: f32, x: f32) -> f32 = crate::atan2f;
    "acosf" => fn glibc_acosf(x: f32) -> f32 = crate::acosf;
    "asinf" => fn glibc_asinf(x: f32) -> f32 = crate::asinf;
    "atanf" => fn glibc_atanf(x: f32) -> f32 = crate::atanf;
    "acoshf" => fn glibc_acoshf(x: f32) -> f32 = crate::acoshf;
    "asinhf" => fn glibc_asinhf(x: f32) -> f32 = crate::asinhf;
    "atanhf" => fn glibc_atanhf(x: f32) -> f32 = crate::atanhf;
    "cbrtf" => fn glibc_cbrtf(x: f32) -> f32 = crate::cbrtf;
    "coshf" => fn glibc_coshf(x: f32) -> f32 = crate::coshf;
    "sinhf" => fn glibc_sinhf(x: f32) -> f32 = crate::sinhf;
    "tanf" => fn glibc_tanf(x: f32) -> f32 = crate::tanf;
    "expm1f" => fn glibc_expm1f(x: f32) -> f32 = crate::expm1f;
    "log10f" => fn glibc_log10f(x: f32) -> f32 = crate::log10f;
    "erff" => fn glibc_erff(x: f32) -> f32 = crate::erff;
    "erfcf" => fn glibc_erfcf(x: f32) -> f32 = crate::erfcf;
    "lgammaf" => fn glibc_lgammaf(x: f32) -> f32 = crate::lgammaf;
    "tgammaf" => fn glibc_tgammaf(x: f32) -> f32 = crate::tgammaf;
}

/// `sincos` / `__sincos_stret`, which LLVM merges a sin and cos of one
/// argument into, and the libcalls std's other float methods lower to off
/// wasm (`%` → `fmod`, `abs_sub` → `fdim`, and on x86-64 without SSE4.1 / FMA
/// `floor` … `rint` and `fma`): exported natively under `export-glibc`, so
/// no float operation of the binary reaches the platform's C libm.
#[cfg(feature = "export-glibc")]
mod rest {
    export! {
        "fmod" => fn x_fmod(x: f64, y: f64) -> f64 = crate::fmod;
        "fmodf" => fn x_fmodf(x: f32, y: f32) -> f32 = crate::fmodf;
        "fdim" => fn x_fdim(x: f64, y: f64) -> f64 = crate::fdim;
        "fdimf" => fn x_fdimf(x: f32, y: f32) -> f32 = crate::fdimf;
        "fmaf" => fn x_fmaf(a: f32, b: f32, c: f32) -> f32 = crate::fmaf;
        "floor" => fn x_floor(x: f64) -> f64 = crate::floor;
        "floorf" => fn x_floorf(x: f32) -> f32 = crate::floorf;
        "ceil" => fn x_ceil(x: f64) -> f64 = crate::ceil;
        "ceilf" => fn x_ceilf(x: f32) -> f32 = crate::ceilf;
        "trunc" => fn x_trunc(x: f64) -> f64 = crate::trunc;
        "truncf" => fn x_truncf(x: f32) -> f32 = crate::truncf;
        "round" => fn x_round(x: f64) -> f64 = crate::round;
        "roundf" => fn x_roundf(x: f32) -> f32 = crate::roundf;
        "rint" => fn x_rint(x: f64) -> f64 = crate::rint;
        "rintf" => fn x_rintf(x: f32) -> f32 = crate::rintf;
        "roundeven" => fn x_roundeven(x: f64) -> f64 = crate::rint;
        "roundevenf" => fn x_roundevenf(x: f32) -> f32 = crate::rintf;
    }
    // wasm's `fma` is `wasm-fma`'s (the same function).
    #[cfg(not(all(feature = "wasm-fma", target_family = "wasm")))]
    export! {
        "fma" => fn x_fma(a: f64, b: f64, c: f64) -> f64 = crate::fma_emulated;
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
