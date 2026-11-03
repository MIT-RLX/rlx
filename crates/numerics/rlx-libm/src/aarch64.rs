// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! glibc 2.43 as built for Linux aarch64, where it differs from the x86-64
//! build the crate's root functions reproduce. Every other function is the
//! same on both (checked: all 2^32 inputs of each f32 function, and the f64
//! vectors, `tests/glibc_2_43_vectors.rs`); these five are not:
//!
//! * [`expf`], [`exp10`]: aarch64 defines `TOINT_INTRINSICS`, so `k` comes
//!   from `round` / `lround` (ties away from zero) instead of adding the
//!   `0x1.8p52` shift (ties to even).
//! * [`exp10`], [`exp2`], [`cbrt`]: x86-64 runs builds without FMA (no
//!   `-mfma` variant of these); gcc for aarch64 fuses every product whose
//!   uses are all adds.
//! * [`hypot`]: aarch64 defines `__FP_FAST_FMA`, which selects a different
//!   kernel (`fma(x, x, y²)`).
//!
//! `rlx_libm::reference` and the root functions stay the x86-64 build (the
//! reproducible reference on every target); call these to reproduce a Linux
//! aarch64 run (torch / NumPy on Graviton, Ampere, Apple-silicon Linux).

/// glibc `expf` on aarch64.
pub fn expf(x: f32) -> f32 {
    crate::expf_impl::<true>(x)
}
/// glibc `exp2` on aarch64.
pub fn exp2(x: f64) -> f64 {
    crate::dbl64::exp2_impl::<true>(x)
}
/// glibc `exp10` on aarch64.
pub fn exp10(x: f64) -> f64 {
    crate::dbl64::exp10_impl::<true>(x)
}
/// glibc `cbrt` on aarch64.
pub fn cbrt(x: f64) -> f64 {
    crate::dbl64::cbrt_impl::<true>(x)
}
/// glibc `hypot` on aarch64.
pub fn hypot(x: f64, y: f64) -> f64 {
    crate::complex::hypot_impl::<true>(x, y)
}
