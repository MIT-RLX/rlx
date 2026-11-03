// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A second opinion: the glibc ports against musl's independent algorithms
//! (the real `libm` crate, in this standalone tool so rlx's `libm` patch
//! does not replace it), to a few ULPs. Neither is the
//! reference — glibc itself is, through `glibc_2_43_vectors` — but a port that
//! drifts from both by more than a handful of ULPs is broken, whatever the
//! vectors say about the inputs they sample.

fn ulps64(a: f64, b: f64) -> u64 {
    if a == b || (a.is_nan() && b.is_nan()) {
        return 0;
    }
    if a.signum() != b.signum() {
        return u64::MAX;
    }
    (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs()
}

fn ulps32(a: f32, b: f32) -> u64 {
    ulps64(a as f64, b as f64) >> 29
}

struct R(u64);
impl R {
    fn u(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[test]
fn f64_ports_agree_with_musl() {
    use rlx_libm as l;
    type F = fn(f64) -> f64;
    let cases: [(&str, F, F, f64, f64); 23] = [
        ("exp", l::exp, libm::exp, -700.0, 700.0),
        ("log", l::log, libm::log, 1e-300, 1e300),
        ("log2", l::log2, libm::log2, 1e-300, 1e300),
        ("log10", l::log10, libm::log10, 1e-300, 1e300),
        ("exp2", l::exp2, libm::exp2, -1000.0, 1000.0),
        ("exp10", l::exp10, libm::exp10, -300.0, 300.0),
        ("expm1", l::expm1, libm::expm1, -40.0, 700.0),
        ("log1p", l::log1p, libm::log1p, -0.999, 1e300),
        ("sin", l::sin, libm::sin, -1e6, 1e6),
        ("cos", l::cos, libm::cos, -1e6, 1e6),
        ("tan", l::tan, libm::tan, -1e3, 1e3),
        ("asin", l::asin, libm::asin, -1.0, 1.0),
        ("acos", l::acos, libm::acos, -1.0, 1.0),
        ("atan", l::atan, libm::atan, -1e6, 1e6),
        ("sinh", l::sinh, libm::sinh, -700.0, 700.0),
        ("cosh", l::cosh, libm::cosh, -700.0, 700.0),
        ("tanh", l::tanh, libm::tanh, -30.0, 30.0),
        ("asinh", l::asinh, libm::asinh, -1e300, 1e300),
        ("acosh", l::acosh, libm::acosh, 1.0, 1e300),
        ("atanh", l::atanh, libm::atanh, -0.999_999, 0.999_999),
        ("cbrt", l::cbrt, libm::cbrt, -1e300, 1e300),
        ("erf", l::erf, libm::erf, -6.0, 6.0),
        ("tgamma", l::tgamma, libm::tgamma, 0.01, 170.0),
    ];
    let mut bad = Vec::new();
    for (name, ours, musl, lo, hi) in cases {
        let mut r = R(0x9e37_79b9_7f4a_7c15 ^ name.len() as u64);
        // log-spaced when the range spans many decades
        let wide = lo > 0.0 && hi / lo > 1e6;
        let worst = (0..200_000)
            .map(|_| {
                let t = r.u();
                let x = if wide {
                    lo * (hi / lo).powf(t)
                } else {
                    lo + (hi - lo) * t
                };
                ulps64(ours(x), musl(x))
            })
            .max()
            .unwrap();
        // musl's tgamma is a Lanczos sum several ULPs from the true value
        // (ours is glibc's correctly rounded one); every other musl function
        // is within an ULP or two
        let tol = if name == "tgamma" { 10 } else { 4 };
        if worst > tol {
            bad.push(format!("{name}: {worst} ulps"));
        }
    }
    assert!(bad.is_empty(), "{bad:?}");
}

#[test]
fn f32_ports_agree_with_musl() {
    use rlx_libm as l;
    type F = fn(f32) -> f32;
    let cases: [(&str, F, F, f32, f32); 12] = [
        ("expf", l::expf, libm::expf, -80.0, 80.0),
        ("logf", l::logf, libm::logf, 1e-30, 1e30),
        ("exp2f", l::exp2f, libm::exp2f, -120.0, 120.0),
        ("log2f", l::log2f, libm::log2f, 1e-30, 1e30),
        ("tanhf", l::tanhf, libm::tanhf, -10.0, 10.0),
        ("atanf", l::atanf, libm::atanf, -1e6, 1e6),
        ("asinf", l::asinf, libm::asinf, -1.0, 1.0),
        ("cbrtf", l::cbrtf, libm::cbrtf, -1e30, 1e30),
        ("erff", l::erff, libm::erff, -4.0, 4.0),
        ("expm1f", l::expm1f, libm::expm1f, -10.0, 80.0),
        ("log10f", l::log10f, libm::log10f, 1e-30, 1e30),
        ("sinhf", l::sinhf, libm::sinhf, -80.0, 80.0),
    ];
    let mut bad = Vec::new();
    for (name, ours, musl, lo, hi) in cases {
        let mut r = R(0x2545_f491_4f6c_dd1d ^ name.len() as u64);
        let wide = lo > 0.0 && hi / lo > 1e6;
        let worst = (0..200_000)
            .map(|_| {
                let t = r.u() as f32;
                let x = if wide {
                    lo * (hi / lo).powf(t)
                } else {
                    lo + (hi - lo) * t
                };
                ulps32(ours(x), musl(x))
            })
            .max()
            .unwrap();
        if worst > 2 {
            bad.push(format!("{name}: {worst} ulps"));
        }
    }
    let p = (0..100_000)
        .map(|i| {
            let x = 0.01 + i as f32 * 4e-5;
            let y = -20.0 + (i % 400) as f32 * 0.1;
            ulps32(l::powf(x, y), libm::powf(x, y))
        })
        .max()
        .unwrap();
    if p > 2 {
        bad.push(format!("powf: {p} ulps"));
    }
    assert!(bad.is_empty(), "{bad:?}");
}
