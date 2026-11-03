// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The platform's C libm, looked up by name in its own library (`dlsym` on
//! `libSystem` / `libm.so.6`), for validation only: the `libm` feature's
//! Apple reference and the tests against glibc. A plain `extern "C"` call or
//! a std method would not do: with `export-glibc` the binary defines `sin` &
//! co. itself, and in a Rust binary on Linux some symbols bind to
//! compiler-builtins' weak musl copies instead of glibc's.
#![allow(unsafe_code, dead_code)]

use std::ffi::{CStr, c_char, c_int, c_void};
use std::sync::OnceLock;

unsafe extern "C" {
    fn dlopen(path: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
}

#[cfg(target_vendor = "apple")]
const LIB: &CStr = c"/usr/lib/libSystem.B.dylib";
#[cfg(not(target_vendor = "apple"))]
const LIB: &CStr = c"libm.so.6";

/// The address of `name` in the platform libm (panics if it has none).
pub fn sym(name: &CStr) -> usize {
    static HANDLE: OnceLock<usize> = OnceLock::new();
    let h = *HANDLE.get_or_init(|| {
        // SAFETY: a NUL-terminated path; RTLD_NOW (2 on Linux and Darwin).
        let h = unsafe { dlopen(LIB.as_ptr(), 2) };
        assert!(!h.is_null(), "dlopen {LIB:?} failed");
        h as usize
    });
    // SAFETY: a live handle and a NUL-terminated name.
    let p = unsafe { dlsym(h as *mut c_void, name.as_ptr()) };
    assert!(!p.is_null(), "{name:?} is not in {LIB:?}");
    p as usize
}

macro_rules! sys_fns {
    ($($name:ident($($a:ident: $t:ty),*) -> $r:ty;)*) => {$(
        #[doc = concat!("The platform's `", stringify!($name), "`.")]
        pub fn $name($($a: $t),*) -> $r {
            static P: OnceLock<usize> = OnceLock::new();
            let p = *P.get_or_init(|| sym(
                CStr::from_bytes_with_nul(concat!(stringify!($name), "\0").as_bytes()).unwrap(),
            ));
            // SAFETY: the libm function of this name has this C signature.
            let f: unsafe extern "C" fn($($t),*) -> $r = unsafe { core::mem::transmute(p) };
            // SAFETY: a pure function of its arguments.
            unsafe { f($($a),*) }
        }
    )*};
}

sys_fns! {
    exp(x: f64) -> f64; log(x: f64) -> f64; log2(x: f64) -> f64; log10(x: f64) -> f64;
    pow(x: f64, y: f64) -> f64; sin(x: f64) -> f64; cos(x: f64) -> f64; tan(x: f64) -> f64;
    expm1(x: f64) -> f64; log1p(x: f64) -> f64; hypot(x: f64, y: f64) -> f64;
    erfc(x: f64) -> f64; erf(x: f64) -> f64; tanh(x: f64) -> f64; cbrt(x: f64) -> f64;
    expf(x: f32) -> f32; logf(x: f32) -> f32; log1pf(x: f32) -> f32; tanhf(x: f32) -> f32;
    sinf(x: f32) -> f32; cosf(x: f32) -> f32; asin(x: f64) -> f64;
}

/// `(sin(x), cos(x))` from the platform's merged call (`__sincos_stret` on
/// Apple, `sincos` on glibc).
pub fn sincos(x: f64) -> (f64, f64) {
    #[cfg(target_vendor = "apple")]
    {
        #[repr(C)]
        struct SinCos(f64, f64);
        static P: OnceLock<usize> = OnceLock::new();
        let p = *P.get_or_init(|| sym(c"__sincos_stret"));
        // SAFETY: Apple's `__sincos_stret` returns the pair by value.
        let f: unsafe extern "C" fn(f64) -> SinCos = unsafe { core::mem::transmute(p) };
        // SAFETY: a pure function of one double.
        let r = unsafe { f(x) };
        (r.0, r.1)
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        static P: OnceLock<usize> = OnceLock::new();
        let p = *P.get_or_init(|| sym(c"sincos"));
        // SAFETY: glibc's `sincos` writes its two outputs.
        let f: unsafe extern "C" fn(f64, *mut f64, *mut f64) = unsafe { core::mem::transmute(p) };
        let (mut s, mut c) = (0.0, 0.0);
        // SAFETY: two valid out-pointers.
        unsafe { f(x, &mut s, &mut c) };
        (s, c)
    }
}

/// `(sinf(x), cosf(x))` from Apple's merged `__sincosf_stret`.
#[cfg(target_vendor = "apple")]
pub fn sincosf_stret(x: f32) -> (f32, f32) {
    #[repr(C)]
    struct SinCos(f32, f32);
    static P: OnceLock<usize> = OnceLock::new();
    let p = *P.get_or_init(|| sym(c"__sincosf_stret"));
    // SAFETY: Apple's `__sincosf_stret` returns the pair by value.
    let f: unsafe extern "C" fn(f32) -> SinCos = unsafe { core::mem::transmute(p) };
    // SAFETY: a pure function of one float.
    let r = unsafe { f(x) };
    (r.0, r.1)
}

/// A complex double, as C passes `double complex`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Cplx {
    pub re: f64,
    pub im: f64,
}

/// The platform's `cexp` / `csqrt`.
pub fn cfn(name: &CStr, z: Cplx) -> Cplx {
    // SAFETY: `cexp` / `csqrt` take and return a `double complex`.
    let f: unsafe extern "C" fn(Cplx) -> Cplx = unsafe { core::mem::transmute(sym(name)) };
    // SAFETY: a pure function of one complex double.
    unsafe { f(z) }
}
