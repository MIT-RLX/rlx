// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Every one-argument f32 port against glibc on all 2^32 inputs: prints each
//! function's mismatch count (and the first few), and writes the hashes of
//! glibc's outputs to the vectors file given as the argument. Exits 1 on any
//! mismatch, without writing.
#![allow(unsafe_code)]

#[path = "../../../tests/common/f32_unary.rs"]
mod common;
#[path = "../../../tests/common/f64_fns.rs"]
mod f64_fns;
#[path = "../../../src/sys.rs"]
mod sys;

use common::*;
use std::sync::atomic::{AtomicU64, Ordering};

fn glibc(name: &str) -> impl Fn(f32) -> f32 + Sync {
    let c = std::ffi::CString::new(name).unwrap();
    let p = sys::sym(&c);
    // SAFETY: the libm function of this name is `float f(float)`.
    let f: unsafe extern "C" fn(f32) -> f32 = unsafe { std::mem::transmute(p) };
    // SAFETY: a pure function of one float.
    move |x| unsafe { f(x) }
}

/// `--f64 N`: every f64 port against glibc on `N` draws each (half raw bit
/// patterns, half with the exponent in [2^-63, 2^57]), mismatches listed.
fn sample_f64(n: u64) {
    let mut failed = false;
    let draw = |i: u64, s: &mut u64| {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        if i.is_multiple_of(2) {
            f64::from_bits(*s)
        } else {
            let e = 960 + (*s >> 52) % 121;
            f64::from_bits((*s & (1 << 63)) | e << 52 | (*s & ((1 << 52) - 1)))
        }
    };
    let same = |a: f64, b: f64| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
    for (name, port) in f64_fns::f64_unary(AARCH64) {
        let c = std::ffi::CString::new(name).unwrap();
        // SAFETY: `double f(double)`.
        let g: unsafe extern "C" fn(f64) -> f64 = unsafe { std::mem::transmute(sys::sym(&c)) };
        let (mut s, mut bad, mut first) = (0x9e37_79b9_7f4a_7c15u64, 0u64, Vec::new());
        for i in 0..n {
            let x = draw(i, &mut s);
            // SAFETY: a pure function.
            let r = unsafe { g(x) };
            if !same(r, port(x)) {
                bad += 1;
                if first.len() < 3 {
                    first.push((x, r, port(x)));
                }
            }
        }
        println!("{name:8} {bad:>9} / {n}  {first:?}");
        failed |= bad != 0;
    }
    for (name, port) in f64_fns::f64_binary(AARCH64) {
        let c = std::ffi::CString::new(name).unwrap();
        // SAFETY: `double f(double, double)`.
        let g: unsafe extern "C" fn(f64, f64) -> f64 = unsafe { std::mem::transmute(sys::sym(&c)) };
        let (mut s, mut bad, mut first) = (0x2545_f491_4f6c_dd1du64, 0u64, Vec::new());
        for i in 0..n {
            let (x, y) = (draw(i, &mut s), draw(i, &mut s));
            // SAFETY: a pure function.
            let r = unsafe { g(x, y) };
            if !same(r, port(x, y)) {
                bad += 1;
                if first.len() < 3 {
                    first.push((x, y, r, port(x, y)));
                }
            }
        }
        println!("{name:8} {bad:>9} / {n}  {first:?}");
        failed |= bad != 0;
    }
    std::process::exit(failed as i32);
}

/// The glibc build this binary runs against.
const AARCH64: bool = cfg!(target_arch = "aarch64");

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--f64") {
        return sample_f64(args.get(2).map_or(10_000_000, |n| n.parse().unwrap()));
    }
    let out = args
        .get(1)
        .expect("usage: glibc-exhaustive <vectors file> | --f64 [N]")
        .clone();
    let (machine, release) = (std::env::consts::ARCH, glibc_version());
    let mut lines = vec![format!(
        "# {machine} glibc {release}: all 2^32 inputs per function"
    )];
    let mut failed = false;
    for (name, port) in f32_unary(AARCH64) {
        let g = glibc(name);
        let bad = AtomicU64::new(0);
        let first = std::sync::Mutex::new(Vec::new());
        // The glibc hash, checking the port on the way.
        let h = exhaustive_hash(&|x| {
            let r = g(x);
            if canon(r) != canon(port(x)) && bad.fetch_add(1, Ordering::Relaxed) < 4 {
                first.lock().unwrap().push((x, r, port(x)));
            }
            r
        });
        let n = bad.into_inner();
        println!(
            "{name:12} {n:>10} mismatches  {:?}",
            first.into_inner().unwrap()
        );
        failed |= n != 0;
        lines.push(format!("{name} {h:016x}"));
    }
    if failed {
        eprintln!("ports differ from glibc; {out} not written");
        std::process::exit(1);
    }
    std::fs::write(&out, lines.join("\n") + "\n").unwrap();
    println!("wrote {out}");
}

fn glibc_version() -> String {
    unsafe extern "C" {
        fn gnu_get_libc_version() -> *const std::ffi::c_char;
    }
    // SAFETY: glibc returns a static NUL-terminated string.
    unsafe { std::ffi::CStr::from_ptr(gnu_get_libc_version()) }
        .to_string_lossy()
        .into_owned()
}
