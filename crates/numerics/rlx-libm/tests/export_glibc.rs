// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! With `export-glibc`, std's float methods in the binary are rlx-libm's Rust
//! ports, not the platform's C libm: every method that lowers to a libcall
//! (`f64::sin`, `%`, …) gives the port's bits. Run with
//! `cargo test -p rlx-libm --features export-glibc --test export_glibc`.
#![cfg(feature = "export-glibc")]

use std::hint::black_box as bb;

fn inputs() -> Vec<f64> {
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    (0..200_000)
        .map(|i| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            // a spread of magnitudes, half in the range math libs differ in
            if i % 2 == 0 {
                f64::from_bits(s)
            } else {
                ((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 40.0
            }
        })
        .collect()
}

fn same64(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}
fn same32(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

#[test]
fn std_float_methods_are_the_rust_ports() {
    type F64 = (&'static str, fn(f64) -> f64, fn(f64) -> f64);
    let f64s: &[F64] = &[
        ("sin", |x| bb(x).sin(), rlx_libm::sin),
        ("cos", |x| bb(x).cos(), rlx_libm::cos),
        ("tan", |x| bb(x).tan(), rlx_libm::tan),
        ("exp", |x| bb(x).exp(), rlx_libm::exp),
        ("exp2", |x| bb(x).exp2(), rlx_libm::exp2),
        ("exp_m1", |x| bb(x).exp_m1(), rlx_libm::expm1),
        ("ln", |x| bb(x).ln(), rlx_libm::log),
        ("log2", |x| bb(x).log2(), rlx_libm::log2),
        ("log10", |x| bb(x).log10(), rlx_libm::log10),
        ("ln_1p", |x| bb(x).ln_1p(), rlx_libm::log1p),
        ("cbrt", |x| bb(x).cbrt(), rlx_libm::cbrt),
        ("sinh", |x| bb(x).sinh(), rlx_libm::sinh),
        ("cosh", |x| bb(x).cosh(), rlx_libm::cosh),
        ("tanh", |x| bb(x).tanh(), rlx_libm::tanh),
        ("asin", |x| bb(x).asin(), rlx_libm::asin),
        ("acos", |x| bb(x).acos(), rlx_libm::acos),
        ("atan", |x| bb(x).atan(), rlx_libm::atan),
        ("floor", |x| bb(x).floor(), rlx_libm::floor),
        ("round", |x| bb(x).round(), rlx_libm::round),
        ("sin_cos", |x| bb(x).sin_cos().1, rlx_libm::cos),
    ];
    type F2 = (&'static str, fn(f64, f64) -> f64, fn(f64, f64) -> f64);
    let f2s: &[F2] = &[
        ("powf", |x, y| bb(x).powf(y), rlx_libm::pow),
        ("atan2", |x, y| bb(x).atan2(y), rlx_libm::atan2),
        ("hypot", |x, y| bb(x).hypot(y), rlx_libm::hypot),
        ("%", |x, y| bb(x) % y, rlx_libm::fmod),
    ];
    type F32 = (&'static str, fn(f32) -> f32, fn(f32) -> f32);
    let f32s: &[F32] = &[
        ("sinf", |x| bb(x).sin(), |x| rlx_libm::sincosf(x).0),
        ("cosf", |x| bb(x).cos(), |x| rlx_libm::sincosf(x).1),
        ("expf", |x| bb(x).exp(), rlx_libm::expf),
        ("lnf", |x| bb(x).ln(), rlx_libm::logf),
        ("log2f", |x| bb(x).log2(), rlx_libm::log2f),
        ("log10f", |x| bb(x).log10(), rlx_libm::log10f),
        ("tanhf", |x| bb(x).tanh(), rlx_libm::tanhf),
        ("tanf", |x| bb(x).tan(), rlx_libm::tanf),
        ("atanf", |x| bb(x).atan(), rlx_libm::atanf),
        ("cbrtf", |x| bb(x).cbrt(), rlx_libm::cbrtf),
    ];
    let xs = inputs();
    let mut bad = Vec::new();
    for &(name, s, p) in f64s {
        if let Some(x) = xs.iter().find(|&&x| !same64(s(x), p(x))) {
            bad.push(format!("{name}({x:e})"));
        }
    }
    for &(name, s, p) in f2s {
        if let Some(w) = xs
            .windows(2)
            .find(|w| !same64(s(w[0], w[1] * 0.01), p(w[0], w[1] * 0.01)))
        {
            bad.push(format!("{name}({:e}, {:e})", w[0], w[1] * 0.01));
        }
    }
    for &(name, s, p) in f32s {
        if let Some(&x) = xs.iter().find(|&&x| !same32(s(x as f32), p(x as f32))) {
            bad.push(format!("{name}({:e})", x as f32));
        }
    }
    assert!(bad.is_empty(), "std reached another libm: {bad:?}");
}
