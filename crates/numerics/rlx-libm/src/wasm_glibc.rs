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
//! Functions with no port here (`powf`, `atan2`, …) stay musl's.
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
