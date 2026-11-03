// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The glibc 2.43 ports against glibc itself, on every platform: each file in
//! [`VECTOR_FILES`] holds FNV-1a hashes of one glibc build's output bits over
//! input sets that `tools/glibc_ref_2_43.py` and this test generate
//! identically from integer operations. Zero tolerance.
//!
//! The two builds differ for five functions ([`AARCH64_DIFFERS`]); there the
//! aarch64 vectors check `rlx_libm::aarch64`, everywhere else the root ports.

/// The vector files, by the glibc build they record.
const VECTOR_FILES: &[(&str, &str)] = &[
    // x86-64 with FMA (the `*_fma` ifunc variants): Ubuntu 26.04, the msi rig
    (
        "glibc_x86_64_2_43.txt",
        include_str!("vectors/glibc_x86_64_2_43.txt"),
    ),
    // aarch64: the same Ubuntu 26.04 glibc package, in an arm64 container
    (
        "glibc_aarch64_2_43.txt",
        include_str!("vectors/glibc_aarch64_2_43.txt"),
    ),
];

/// Where glibc's aarch64 build rounds differently from its x86-64 one: the
/// ports of the aarch64 build are `rlx_libm::aarch64::*`.
const AARCH64_DIFFERS: &[&str] = &["expf", "exp2", "exp10", "cbrt", "hypot"];

const M: u64 = u64::MAX;

struct Xs(u64);
impl Iterator for Xs {
    type Item = u64;
    fn next(&mut self) -> Option<u64> {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        Some(self.0)
    }
}

fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h = (*h ^ u64::from(b)).wrapping_mul(0x100000001b3);
    }
}

/// `(width, mantissa bits)` of the type a function takes.
fn shape(f32: bool) -> (u32, u32) {
    if f32 { (32, 23) } else { (64, 52) }
}

fn draw(g: &mut Xs, kind: &str, a: u64, b: u64, f32: bool) -> u64 {
    let (w, mb) = shape(f32);
    let mut raw = g.next().unwrap();
    if f32 {
        raw &= 0xffff_ffff;
    }
    if kind == "bits" {
        return raw;
    }
    let sign = raw >> (w - 1);
    let mant = raw & ((1 << mb) - 1);
    let e = a + ((raw >> mb) & 0xffff) % (b - a + 1);
    (sign << (w - 1)) | (e << mb) | mant
}

type F1 = fn(f64) -> f64;
type F2 = fn(f64, f64) -> f64;
type F3 = fn(f64, f64, f64) -> f64;
type Fsc = fn(f64) -> (f64, f64);

/// A ported function, by arity, with whether it takes `f32`.
#[derive(Clone, Copy)]
enum Fun {
    U(bool, F1),
    B(bool, F2),
    T(bool, F3),
    /// `sincos` / `sincosf`: both outputs hashed, sin first.
    Sc(bool, Fsc),
}

macro_rules! f32_1 {
    ($f:expr) => {
        Fun::U(true, |x: f64| $f(x as f32) as f64)
    };
}
macro_rules! f32_2 {
    ($f:expr) => {
        Fun::B(true, |x: f64, y: f64| $f(x as f32, y as f32) as f64)
    };
}

fn port(name: &str, aarch64: bool) -> Option<Fun> {
    use Fun::*;
    use rlx_libm as l;
    if aarch64 && AARCH64_DIFFERS.contains(&name) {
        use l::aarch64 as a;
        return Some(match name {
            "expf" => f32_1!(a::expf),
            "exp2" => U(false, a::exp2),
            "exp10" => U(false, a::exp10),
            "cbrt" => U(false, a::cbrt),
            "hypot" => B(false, a::hypot),
            _ => unreachable!(),
        });
    }
    Some(match name {
        "acosf" => f32_1!(l::acosf),
        "asinf" => f32_1!(l::asinf),
        "atanf" => f32_1!(l::atanf),
        "acoshf" => f32_1!(l::acoshf),
        "asinhf" => f32_1!(l::asinhf),
        "atanhf" => f32_1!(l::atanhf),
        "cbrtf" => f32_1!(l::cbrtf),
        "coshf" => f32_1!(l::coshf),
        "sinhf" => f32_1!(l::sinhf),
        "tanf" => f32_1!(l::tanf),
        "expm1f" => f32_1!(l::expm1f),
        "log10f" => f32_1!(l::log10f),
        "erff" => f32_1!(l::erff),
        "erfcf" => f32_1!(l::erfcf),
        "lgammaf" => f32_1!(l::lgammaf),
        "tgammaf" => f32_1!(l::tgammaf),
        "exp2f" => f32_1!(l::exp2f),
        "exp10f" => f32_1!(l::exp10f),
        "log2f" => f32_1!(l::log2f),
        "expf" => f32_1!(l::expf),
        "logf" => f32_1!(l::logf),
        "log1pf" => f32_1!(l::log1pf),
        "tanhf" => f32_1!(l::tanhf),
        "sinf" => f32_1!(|x| l::sincosf(x).0),
        "cosf" => f32_1!(|x| l::sincosf(x).1),
        "floorf" => f32_1!(l::floorf),
        "ceilf" => f32_1!(l::ceilf),
        "truncf" => f32_1!(l::truncf),
        "roundf" => f32_1!(l::roundf),
        "rintf" | "roundevenf" => f32_1!(l::rintf),
        "sincosf" => Sc(true, |x| {
            let (s, c) = l::sincosf(x as f32);
            (s as f64, c as f64)
        }),
        "powf" => f32_2!(l::powf),
        "atan2f" => f32_2!(l::atan2f),
        "hypotf" => f32_2!(l::hypotf),
        "fmodf" => f32_2!(l::fmodf),
        "fdimf" => f32_2!(l::fdimf),
        "fmaf" => T(true, |a, b, c| l::fmaf(a as f32, b as f32, c as f32) as f64),
        "tanh" => U(false, l::tanh),
        "cbrt" => U(false, l::cbrt),
        "cosh" => U(false, l::cosh),
        "sinh" => U(false, l::sinh),
        "log10" => U(false, l::log10),
        "exp2" => U(false, l::exp2),
        "exp10" => U(false, l::exp10),
        "erf" => U(false, l::erf),
        "erfc" => U(false, l::erfc),
        "atan" => U(false, l::atan),
        "asin" => U(false, l::asin),
        "acos" => U(false, l::acos),
        "atanh" => U(false, l::atanh),
        "asinh" => U(false, l::asinh),
        "acosh" => U(false, l::acosh),
        "tgamma" => U(false, l::tgamma),
        "lgamma" => U(false, l::lgamma),
        "exp" => U(false, l::exp),
        "log" => U(false, l::log),
        "log2" => U(false, l::log2),
        "expm1" => U(false, l::expm1),
        "log1p" => U(false, l::log1p),
        "sin" => U(false, l::sin),
        "cos" => U(false, l::cos),
        "tan" => U(false, l::tan),
        "floor" => U(false, l::floor),
        "ceil" => U(false, l::ceil),
        "trunc" => U(false, l::trunc),
        "round" => U(false, l::round),
        "rint" | "roundeven" => U(false, l::rint),
        "sincos" => Sc(false, |x| (l::sin(x), l::cos(x))),
        "atan2" => B(false, l::atan2),
        "pow" => B(false, l::pow),
        "hypot" => B(false, l::hypot),
        "fmod" => B(false, l::fmod),
        "fdim" => B(false, l::fdim),
        "fma" => T(false, l::fma_emulated),
        _ => return None,
    })
}

/// A value's bits for hashing; any NaN as the canonical quiet NaN.
fn out_bits(v: f64, f32: bool, h: &mut u64) {
    if f32 {
        let v = v as f32;
        let b = if v.is_nan() { 0x7fc0_0000 } else { v.to_bits() };
        fnv(h, &b.to_le_bytes());
    } else {
        let b = if v.is_nan() {
            0x7ff8_0000_0000_0000
        } else {
            v.to_bits()
        };
        fnv(h, &b.to_le_bytes());
    }
}

fn value(bits: u64, f32: bool) -> f64 {
    if f32 {
        f32::from_bits(bits as u32) as f64
    } else {
        f64::from_bits(bits)
    }
}

#[test]
fn ports_match_glibc_2_43_x86_64() {
    check_file(VECTOR_FILES[0].0, VECTOR_FILES[0].1, false);
}

#[test]
fn aarch64_ports_match_glibc_2_43_aarch64() {
    check_file(VECTOR_FILES[1].0, VECTOR_FILES[1].1, true);
}

fn check_file(file: &str, text: &str, aarch64: bool) {
    let (mut checked, mut bad) = (0, Vec::new());
    for line in text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let t: Vec<&str> = line.split(' ').collect();
        let (name, kind) = (t[0], t[1]);
        let p = |i: usize| t[i].parse::<u64>().unwrap();
        let (a, b, count) = (p(2), p(3), p(4) as usize);
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut g = Xs(if kind == "bits" {
            a
        } else {
            match a.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(b) & M {
                0 => 1,
                s => s,
            }
        });
        let f = port(name, aarch64).unwrap_or_else(|| panic!("{file}: unknown function {name}"));
        let f32 = match f {
            Fun::U(w, _) | Fun::B(w, _) | Fun::T(w, _) | Fun::Sc(w, _) => w,
        };
        let mut arg = |i: usize| {
            if kind == "stride" {
                value(a + i as u64 * b, f32)
            } else {
                value(draw(&mut g, kind, a, b, f32), f32)
            }
        };
        for i in 0..count {
            match f {
                Fun::U(_, f) => out_bits(f(arg(i)), f32, &mut h),
                Fun::B(_, f) => {
                    let (x, y) = (arg(i), arg(i));
                    out_bits(f(x, y), f32, &mut h)
                }
                Fun::T(_, f) => {
                    let (x, y, z) = (arg(i), arg(i), arg(i));
                    out_bits(f(x, y, z), f32, &mut h)
                }
                Fun::Sc(_, f) => {
                    let (s, c) = f(arg(i));
                    out_bits(s, f32, &mut h);
                    out_bits(c, f32, &mut h);
                }
            }
        }
        checked += 1;
        if format!("{h:016x}") != t[5] {
            bad.push(line.to_string());
        }
    }
    assert!(checked > 150, "{file}: only {checked} vector lines");
    assert!(
        bad.is_empty(),
        "{file}: differ from glibc:\n{}",
        bad.join("\n")
    );
}

/// Every C symbol rlx-libm exports (`src/wasm_glibc.rs`), by its glibc name:
/// Apple's aliases fold onto the function they alias.
fn exported_symbols() -> Vec<String> {
    let src = include_str!("../src/wasm_glibc.rs");
    let mut out: Vec<String> = Vec::new();
    for l in src.lines() {
        let l = l.trim();
        let name = if l.starts_with('"') && l.contains("\" => fn ") {
            l[1..].split('"').next()
        } else if let Some(r) = l.strip_prefix("#[unsafe(export_name = \"") {
            r.split('"').next()
        } else {
            None
        };
        if let Some(n) = name {
            let n = match n {
                "__sincos_stret" => "sincos",
                "__sincosf_stret" => "sincosf",
                n => n.trim_start_matches("__"),
            };
            if !out.iter().any(|o| o == n) {
                out.push(n.to_string());
            }
        }
    }
    out
}

/// No exported function without glibc vectors: each symbol has lines in the
/// vector file of every glibc build its port is checked against.
#[test]
fn every_export_has_glibc_vectors() {
    let syms = exported_symbols();
    assert!(syms.len() > 70, "parsed only {} exports", syms.len());
    let mut missing = Vec::new();
    for (file, text) in VECTOR_FILES {
        let names: std::collections::HashSet<&str> = text
            .lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
            .map(|l| l.split(' ').next().unwrap())
            .collect();
        for s in &syms {
            if !names.contains(s.as_str()) {
                missing.push(format!("{s} (in {file})"));
            }
        }
    }
    assert!(
        missing.is_empty(),
        "exports without glibc vectors: {missing:?}"
    );
}
