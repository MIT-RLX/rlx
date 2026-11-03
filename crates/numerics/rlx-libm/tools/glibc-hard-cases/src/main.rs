// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Every f32 input: rlx-libm's own double function rounded once vs glibc's
//! correctly rounded f32 function, on all 2^32 inputs. Prints each function's
//! count and its `core_math_f32` table rows. glibc is reached through `dlsym`
//! (a plain `extern "C"` declaration can bind to compiler-builtins' musl copy
//! instead). C is reached only for validation: here, `tools/glibc-exhaustive`
//! and the `dlsym` shim `src/sys.rs` (tests and the opt-in `libm` feature).
// `(x - x) / (x - x)`: the IEEE NaN idiom, as in rlx-libm's ports.
#![allow(clippy::eq_op)]
use std::sync::atomic::{AtomicU64, Ordering};
unsafe extern "C" {
    fn dlopen(f: *const core::ffi::c_char, flag: i32) -> *mut core::ffi::c_void;
    fn dlsym(h: *mut core::ffi::c_void, s: *const core::ffi::c_char) -> *mut core::ffi::c_void;
}
fn g(name: &core::ffi::CStr) -> extern "C" fn(f32) -> f32 {
    unsafe { std::mem::transmute(dlsym(dlopen(c"libm.so.6".as_ptr(), 2), name.as_ptr())) }
}
fn acosh_d(x: f64) -> f64 {
    if x.is_nan() || x < 1.0 {
        return (x - x) / (x - x);
    }
    if x < 2.0 {
        let t = x - 1.0;
        rlx_libm::log1p(t + (t * (x + 1.0)).sqrt())
    } else {
        rlx_libm::log(x + (x * x - 1.0).sqrt())
    }
}
type F = (&'static str, fn(f64) -> f64, &'static core::ffi::CStr);
fn main() {
    use rlx_libm as l;
    let all: Vec<F> = vec![
        ("acosf", l::acos, c"acosf"),
        ("asinf", l::asin, c"asinf"),
        ("atanf", l::atan, c"atanf"),
        ("acoshf", acosh_d, c"acoshf"),
        ("asinhf", l::asinh, c"asinhf"),
        ("atanhf", l::atanh, c"atanhf"),
        ("cbrtf", l::cbrt, c"cbrtf"),
        ("coshf", l::cosh, c"coshf"),
        ("sinhf", l::sinh, c"sinhf"),
        ("tanf", l::tan, c"tanf"),
        ("expm1f", l::expm1, c"expm1f"),
        ("log10f", l::log10, c"log10f"),
        ("erff", l::erf, c"erff"),
        ("erfcf", l::erfc, c"erfcf"),
        ("lgammaf", l::lgamma, c"lgammaf"),
        ("tanhf", l::tanh, c"tanhf"),
        ("tgammaf", l::tgamma, c"tgammaf"),
    ];
    for (name, ours, sym) in all {
        let glibc = g(sym);
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get()) as u64;
        let count = AtomicU64::new(0);
        let found = std::sync::Mutex::new(Vec::new());
        std::thread::scope(|s| {
            for th in 0..threads {
                let (count, found) = (&count, &found);
                s.spawn(move || {
                    let span = (1u64 << 32) / threads;
                    let (lo, hi) = (
                        th * span,
                        if th + 1 == threads {
                            1 << 32
                        } else {
                            (th + 1) * span
                        },
                    );
                    let mut local = Vec::new();
                    for b in lo..hi {
                        let x = f32::from_bits(b as u32);
                        let a = ours(x as f64) as f32;
                        let gv = glibc(std::hint::black_box(x));
                        if a.to_bits() != gv.to_bits() && !(a.is_nan() && gv.is_nan()) {
                            if local.len() < 400 {
                                local.push((b as u32, gv.to_bits()));
                            }
                            count.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    found.lock().unwrap().extend(local);
                });
            }
        });
        let mut f = found.into_inner().unwrap();
        f.sort();
        let n = count.load(Ordering::Relaxed);
        println!("{name} {n}");
        for (x, gv) in &f {
            println!(
                "  {name} (0x{:04x}_{:04x}, 0x{:04x}_{:04x}),",
                x >> 16,
                x & 0xffff,
                gv >> 16,
                gv & 0xffff
            );
        }
    }
}
