// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Every one- and two-argument `f64` function rlx-libm exports, by glibc
//! name, for `tools/glibc-exhaustive --f64` (sampled: f64 has no exhaustive).

use rlx_libm as l;

/// One one-argument f64 port, by glibc name.
pub type F64Fn = (&'static str, fn(f64) -> f64);
/// One two-argument f64 port, by glibc name.
pub type F64Fn2 = (&'static str, fn(f64, f64) -> f64);

const F64_UNARY: &[F64Fn] = &[
    ("exp", l::exp),
    ("exp2", l::exp2),
    ("exp10", l::exp10),
    ("expm1", l::expm1),
    ("log", l::log),
    ("log2", l::log2),
    ("log10", l::log10),
    ("log1p", l::log1p),
    ("sin", l::sin),
    ("cos", l::cos),
    ("tan", l::tan),
    ("asin", l::asin),
    ("acos", l::acos),
    ("atan", l::atan),
    ("sinh", l::sinh),
    ("cosh", l::cosh),
    ("tanh", l::tanh),
    ("asinh", l::asinh),
    ("acosh", l::acosh),
    ("atanh", l::atanh),
    ("cbrt", l::cbrt),
    ("erf", l::erf),
    ("erfc", l::erfc),
    ("tgamma", l::tgamma),
    ("lgamma", l::lgamma),
    ("floor", l::floor),
    ("ceil", l::ceil),
    ("trunc", l::trunc),
    ("round", l::round),
    ("rint", l::rint),
];

const F64_BINARY: &[F64Fn2] = &[
    ("pow", l::pow),
    ("atan2", l::atan2),
    ("hypot", l::hypot),
    ("fmod", l::fmod),
    ("fdim", l::fdim),
];

/// The one-argument functions of the glibc build named (`aarch64`: the
/// `rlx_libm::aarch64` ports where that build differs).
pub fn f64_unary(aarch64: bool) -> Vec<F64Fn> {
    let mut v = F64_UNARY.to_vec();
    if aarch64 {
        for e in &mut v {
            match e.0 {
                "exp2" => e.1 = l::aarch64::exp2,
                "exp10" => e.1 = l::aarch64::exp10,
                "cbrt" => e.1 = l::aarch64::cbrt,
                _ => {}
            }
        }
    }
    v
}

/// The two-argument functions of the glibc build named.
pub fn f64_binary(aarch64: bool) -> Vec<F64Fn2> {
    let mut v = F64_BINARY.to_vec();
    if aarch64 {
        for e in &mut v {
            if e.0 == "hypot" {
                e.1 = l::aarch64::hypot;
            }
        }
    }
    v
}
