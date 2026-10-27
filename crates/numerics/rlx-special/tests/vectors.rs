// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bit-exact parity against Boost.Math (`vectors/boost.txt`, from
//! `tools/boost_oracle.cpp` built as SciPy builds Boost), against SciPy's cephes
//! (`vectors/cephes.txt`, from `tools/xsf_oracle.cpp`), and against SciPy itself
//! (`vectors/scipy.txt`) for every SciPy entry point the two ports cover.
//!
//! Tolerance is **zero**: a result is compared as its IEEE-754 bit pattern.

use std::collections::BTreeMap;

fn f(bits: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(bits, 16).expect("hex bit pattern"))
}

fn eval(fname: &str, a: &[f64]) -> f64 {
    use rlx_special::*;
    match fname {
        "ibeta" | "betainc" => ibeta(a[0], a[1], a[2]),
        "ibetac" | "betaincc" => ibetac(a[0], a[1], a[2]),
        "beta" => beta(a[0], a[1]),
        "gamma_p" => gamma_p(a[0], a[1]),
        "gamma_q" => gamma_q(a[0], a[1]),
        "tgamma" => tgamma(a[0]),
        "lgamma" => lgamma(a[0]),
        "tgamma1pm1" => tgamma1pm1(a[0]),
        "tgamma_delta_ratio" => tgamma_delta_ratio(a[0], a[1]),
        "erf" => erf(a[0]),
        "erfc" => erfc(a[0]),
        "log1pmx" => log1pmx(a[0]),
        "powm1" => powm1(a[0], a[1]),
        "lanczos_sum" => lanczos::lanczos_sum(a[0]),
        "lanczos_sum_exp_g_scaled" => lanczos::lanczos_sum_exp_g_scaled(a[0]),
        "lanczos_sum_near_1" => lanczos::lanczos_sum_near_1(a[0]),
        "lanczos_sum_near_2" => lanczos::lanczos_sum_near_2(a[0]),
        "students_t_cdf" | "stdtr" => students_t_cdf(a[0], a[1]),
        "t_sf" => students_t_sf(a[0], a[1]),
        s if s.starts_with("scipy_") => eval_cephes(&s["scipy_".len()..], a),
        other => panic!("vector file names an unknown function {other:?}"),
    }
}

fn eval_cephes(fname: &str, a: &[f64]) -> f64 {
    use rlx_special::cephes::*;
    match fname {
        "erf" => erf(a[0]),
        "erfc" => erfc(a[0]),
        "ndtr" => ndtr(a[0]),
        "ndtri" => ndtri(a[0]),
        "igam" => igam(a[0], a[1]),
        "igamc" => igamc(a[0], a[1]),
        "chdtrc" => chdtrc(a[0], a[1]),
        "lgam" => lgam(a[0]),
        "Gamma" => gamma(a[0]),
        "zeta" => zeta(a[0], a[1]),
        "c_log1p" => log1p(a[0]),
        "c_expm1" => expm1(a[0]),
        "c_log1pmx" => log1pmx(a[0]),
        "lgam1p" => lgam1p(a[0]),
        "sr_cdf" => rlx_special::quadpack::studentized_range_cdf_inf(a[0], a[1]),
        "sr_sf" => rlx_special::quadpack::studentized_range_sf_inf(a[0], a[1]),
        "c_erfc" => rlx_special::quadpack::libc_erfc(a[0]),
        other => panic!("cephes vector names an unknown function {other:?}"),
    }
}

/// Same NaN payload is not required: any NaN equals any NaN.
fn same(x: f64, y: f64) -> bool {
    x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan())
}

fn check(file: &str, text: &str, eval: fn(&str, &[f64]) -> f64) {
    let mut per_fn: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut examples = Vec::new();
    for line in text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let mut tok = line.split_whitespace();
        let fname = tok.next().unwrap();
        let vals: Vec<f64> = tok.map(f).collect();
        let (args, want) = vals.split_at(vals.len() - 1);
        let got = eval(fname, args);
        let e = per_fn.entry(fname.to_string()).or_default();
        e.0 += 1;
        if !same(got, want[0]) {
            e.1 += 1;
            if examples.len() < 12 {
                examples.push(format!(
                    "{fname}{args:?}: got {got:e} ({:016x}) want {:e} ({:016x})",
                    got.to_bits(),
                    want[0],
                    want[0].to_bits()
                ));
            }
        }
    }
    let total: usize = per_fn.values().map(|v| v.0).sum();
    let bad: usize = per_fn.values().map(|v| v.1).sum();
    for (k, (n, b)) in &per_fn {
        eprintln!("{file}: {k:28} {b:6} / {n:6} mismatched");
    }
    assert!(
        total > 1000,
        "{file}: only {total} vectors — the fixture is not being read"
    );
    assert_eq!(
        bad,
        0,
        "{file}: {bad} of {total} values differ, e.g.\n{}",
        examples.join("\n")
    );
}

/// The vectors of this target's reference: macOS arm64 on Apple targets, the Linux
/// x86-64 SciPy wheel (msi) elsewhere.
#[cfg(target_vendor = "apple")]
macro_rules! vectors {
    ($f:literal) => {
        include_str!(concat!("vectors/", $f))
    };
}
#[cfg(not(target_vendor = "apple"))]
macro_rules! vectors {
    ($f:literal) => {
        include_str!(concat!("vectors/linux-x86_64/", $f))
    };
}

#[test]
fn bit_exact_against_boost_oracle() {
    check("boost.txt", vectors!("boost.txt"), eval);
}

#[test]
fn bit_exact_against_scipy() {
    check("scipy.txt", vectors!("scipy.txt"), eval);
}

#[test]
fn bit_exact_against_xsf_oracle() {
    check("cephes.txt", vectors!("cephes.txt"), eval_cephes);
}

#[test]
fn quadpack_studentized_range_against_scipy() {
    check("quadpack.txt", vectors!("quadpack.txt"), eval_cephes);
}
