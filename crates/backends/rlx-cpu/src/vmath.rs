// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Vector transcendentals — RLX equivalents of Accelerate vForce
//! `vvexpf` / `vvtanhf` / `vvrecf` / `vvlogf` / `vvsqrtf` / `vvrsqrtf`.
//!
//! Host API (this module) is shared across GPU backends via
//! `rlx_<backend>::vmath` re-exports. Device kernels:
//! - CUDA / ROCm / wgpu: `unary` op ids 3=exp, 2=tanh, 17=recip
//! - Metal: `exp_inplace` / `tanh_inplace` / `rec_inplace`
//! - Vulkan / oneAPI: Activation-order unary ids + 17=recip
//!
//! | API | Behavior |
//! |-----|----------|
//! | [`vvexpf`] / [`vvtanhf`] / [`vvlogf`] | Accurate: public Accelerate vForce on Apple, libm elsewhere |
//! | [`vvrecf`] | NEON on aarch64, vectorized/libm fallback elsewhere |
//! | [`vvsqrtf`] / [`vvrsqrtf`] | Hardware SIMD by default; Apple vForce when `RLX_VMATH_ACCURATE=1` |
//! | [`vvexpf_fast`] / [`vvtanhf_fast`] | Portable SIMD polynomial (~2e-7 rel) |
//! | [`vvexpf_hot`] / [`vvtanhf_hot`] | SIMD `*_fast` by default; accurate path when `RLX_VMATH_ACCURATE=1` |
//!
//! Callers depend on this module — never link Accelerate vForce directly.

/// Select accurate CPU vector math instead of the default SIMD exp/tanh path.
#[inline]
pub fn vmath_accurate() -> bool {
    rlx_ir::env::flag("RLX_VMATH_ACCURATE")
}

/// `y[i] = exp(x[i])`. Lengths must match. Aliasing OK.
pub fn vvexpf(y: &mut [f32], x: &[f32]) {
    assert_eq!(y.len(), x.len());
    #[cfg(target_vendor = "apple")]
    {
        let n = y.len() as i32;
        unsafe {
            accelerate::vvexpf(y.as_mut_ptr(), x.as_ptr(), &n);
        }
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        for (yi, &xi) in y.iter_mut().zip(x.iter()) {
            *yi = xi.exp();
        }
    }
}

/// In-place `y[i] = exp(y[i])`.
#[inline]
pub fn vvexpf_inplace(y: &mut [f32]) {
    #[cfg(target_vendor = "apple")]
    {
        let n = y.len() as i32;
        unsafe {
            accelerate::vvexpf(y.as_mut_ptr(), y.as_ptr(), &n);
        }
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        for yi in y.iter_mut() {
            *yi = yi.exp();
        }
    }
}

/// Fast SIMD `exp` (~2e-7 relative). Prefer [`vvexpf`] when ULPs matter.
pub fn vvexpf_fast(y: &mut [f32], x: &[f32]) {
    assert_eq!(y.len(), x.len());
    #[cfg(target_arch = "aarch64")]
    {
        vvexpf_neon(y, x);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                unsafe { vvexpf_avx2(y, x) };
                return;
            }
        }
        vvexpf(y, x);
    }
}

/// In-place fast SIMD `exp`.
#[inline]
pub fn vvexpf_fast_inplace(y: &mut [f32]) {
    let x = unsafe { &*(y as *const [f32]) };
    vvexpf_fast(y, x);
}

/// `exp` selected for CPU activation hot paths.
#[inline]
pub fn vvexpf_hot(y: &mut [f32], x: &[f32]) {
    if vmath_accurate() {
        vvexpf(y, x);
    } else {
        vvexpf_fast(y, x);
    }
}

/// In-place `exp` selected for CPU activation hot paths.
#[inline]
pub fn vvexpf_hot_inplace(y: &mut [f32]) {
    if vmath_accurate() {
        vvexpf_inplace(y);
    } else {
        vvexpf_fast_inplace(y);
    }
}

/// `y[i] = tanh(x[i])`. Lengths must match. Aliasing OK.
pub fn vvtanhf(y: &mut [f32], x: &[f32]) {
    assert_eq!(y.len(), x.len());
    #[cfg(target_vendor = "apple")]
    {
        let n = y.len() as i32;
        unsafe {
            accelerate::vvtanhf(y.as_mut_ptr(), x.as_ptr(), &n);
        }
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        for (yi, &xi) in y.iter_mut().zip(x.iter()) {
            *yi = xi.tanh();
        }
    }
}

/// In-place `y[i] = tanh(y[i])`.
#[inline]
pub fn vvtanhf_inplace(y: &mut [f32]) {
    #[cfg(target_vendor = "apple")]
    {
        let n = y.len() as i32;
        unsafe {
            accelerate::vvtanhf(y.as_mut_ptr(), y.as_ptr(), &n);
        }
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        for yi in y.iter_mut() {
            *yi = yi.tanh();
        }
    }
}

/// Fast SIMD `tanh` via `exp(2x)` poly. Prefer [`vvtanhf`] when ULPs matter.
pub fn vvtanhf_fast(y: &mut [f32], x: &[f32]) {
    assert_eq!(y.len(), x.len());
    #[cfg(target_arch = "aarch64")]
    {
        vvtanhf_neon(y, x);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                unsafe { vvtanhf_avx2(y, x) };
                return;
            }
        }
        vvtanhf(y, x);
    }
}

/// In-place fast SIMD `tanh`.
#[inline]
pub fn vvtanhf_fast_inplace(y: &mut [f32]) {
    let x = unsafe { &*(y as *const [f32]) };
    vvtanhf_fast(y, x);
}

/// `tanh` selected for CPU activation hot paths.
#[inline]
pub fn vvtanhf_hot(y: &mut [f32], x: &[f32]) {
    if vmath_accurate() {
        vvtanhf(y, x);
    } else {
        vvtanhf_fast(y, x);
    }
}

/// In-place `tanh` selected for CPU activation hot paths.
#[inline]
pub fn vvtanhf_hot_inplace(y: &mut [f32]) {
    if vmath_accurate() {
        vvtanhf_inplace(y);
    } else {
        vvtanhf_fast_inplace(y);
    }
}

/// `y[i] = 1 / x[i]`. Lengths must match. Aliasing OK.
pub fn vvrecf(y: &mut [f32], x: &[f32]) {
    assert_eq!(y.len(), x.len());
    #[cfg(target_arch = "aarch64")]
    {
        vvrecf_neon(y, x);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                unsafe { vvrecf_avx2(y, x) };
                return;
            }
        }
        for (yi, &xi) in y.iter_mut().zip(x.iter()) {
            *yi = 1.0 / xi;
        }
    }
}

/// In-place `y[i] = 1 / y[i]`.
#[inline]
pub fn vvrecf_inplace(y: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        let x = unsafe { &*(y as *const [f32]) };
        vvrecf_neon(y, x);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let x = unsafe { &*(y as *const [f32]) };
        vvrecf(y, x);
    }
}

/// `y[i] = ln(x[i])`. Lengths must match. Aliasing OK.
pub fn vvlogf(y: &mut [f32], x: &[f32]) {
    assert_eq!(y.len(), x.len());
    #[cfg(target_vendor = "apple")]
    {
        let n = y.len() as i32;
        unsafe {
            accelerate::vvlogf(y.as_mut_ptr(), x.as_ptr(), &n);
        }
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        for (yi, &xi) in y.iter_mut().zip(x.iter()) {
            *yi = xi.ln();
        }
    }
}

/// In-place `y[i] = ln(y[i])`.
#[inline]
pub fn vvlogf_inplace(y: &mut [f32]) {
    #[cfg(target_vendor = "apple")]
    {
        let n = y.len() as i32;
        unsafe {
            accelerate::vvlogf(y.as_mut_ptr(), y.as_ptr(), &n);
        }
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        for yi in y.iter_mut() {
            *yi = yi.ln();
        }
    }
}

/// `y[i] = sqrt(x[i])`. Lengths must match. Aliasing OK.
pub fn vvsqrtf(y: &mut [f32], x: &[f32]) {
    assert_eq!(y.len(), x.len());
    #[cfg(target_vendor = "apple")]
    if vmath_accurate() {
        let n = y.len() as i32;
        unsafe {
            accelerate::vvsqrtf(y.as_mut_ptr(), x.as_ptr(), &n);
        }
        return;
    }
    vvsqrtf_simd(y, x);
}

/// In-place `y[i] = sqrt(y[i])`.
#[inline]
pub fn vvsqrtf_inplace(y: &mut [f32]) {
    #[cfg(target_vendor = "apple")]
    if vmath_accurate() {
        let n = y.len() as i32;
        unsafe {
            accelerate::vvsqrtf(y.as_mut_ptr(), y.as_ptr(), &n);
        }
        return;
    }
    let x = unsafe { &*(y as *const [f32]) };
    vvsqrtf_simd(y, x);
}

/// `y[i] = 1 / sqrt(x[i])`. Lengths must match. Aliasing OK.
pub fn vvrsqrtf(y: &mut [f32], x: &[f32]) {
    assert_eq!(y.len(), x.len());
    #[cfg(target_vendor = "apple")]
    if vmath_accurate() {
        let n = y.len() as i32;
        unsafe {
            accelerate::vvrsqrtf(y.as_mut_ptr(), x.as_ptr(), &n);
        }
        return;
    }
    vvrsqrtf_simd(y, x);
}

/// In-place `y[i] = 1 / sqrt(y[i])`.
#[inline]
pub fn vvrsqrtf_inplace(y: &mut [f32]) {
    #[cfg(target_vendor = "apple")]
    if vmath_accurate() {
        let n = y.len() as i32;
        unsafe {
            accelerate::vvrsqrtf(y.as_mut_ptr(), y.as_ptr(), &n);
        }
        return;
    }
    let x = unsafe { &*(y as *const [f32]) };
    vvrsqrtf_simd(y, x);
}

/// `y[i] = 1 / (1 + exp(-x[i]))` (logistic sigmoid).
pub fn vvsigmoidf(y: &mut [f32], x: &[f32]) {
    assert_eq!(y.len(), x.len());
    let mut tmp = vec![0.0f32; x.len()];
    for (t, &xi) in tmp.iter_mut().zip(x.iter()) {
        *t = -xi;
    }
    vvexpf_hot_inplace(&mut tmp);
    for t in tmp.iter_mut() {
        *t += 1.0;
    }
    vvrecf(y, &tmp);
}

#[cfg(target_vendor = "apple")]
mod accelerate {
    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        pub fn vvexpf(y: *mut f32, x: *const f32, n: *const i32);
        pub fn vvtanhf(y: *mut f32, x: *const f32, n: *const i32);
        pub fn vvlogf(y: *mut f32, x: *const f32, n: *const i32);
        pub fn vvsqrtf(y: *mut f32, x: *const f32, n: *const i32);
        pub fn vvrsqrtf(y: *mut f32, x: *const f32, n: *const i32);
    }
}

fn vvsqrtf_simd(y: &mut [f32], x: &[f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        vvsqrtf_neon(y, x);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx") {
            unsafe { vvsqrtf_avx(y, x) };
            return;
        }
        for (yi, &xi) in y.iter_mut().zip(x.iter()) {
            *yi = xi.sqrt();
        }
    }
}

fn vvrsqrtf_simd(y: &mut [f32], x: &[f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        vvrsqrtf_neon(y, x);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx") {
            unsafe { vvrsqrtf_avx(y, x) };
            return;
        }
        for (yi, &xi) in y.iter_mut().zip(x.iter()) {
            *yi = 1.0 / xi.sqrt();
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn vvexpf_neon(y: &mut [f32], x: &[f32]) {
    use crate::kernels::neon_exp4;
    use std::arch::aarch64::*;
    let n = x.len();
    let chunks = n / 4;
    unsafe {
        for c in 0..chunks {
            let off = c * 4;
            let v = vld1q_f32(x.as_ptr().add(off));
            vst1q_f32(y.as_mut_ptr().add(off), neon_exp4(v));
        }
    }
    for i in chunks * 4..n {
        y[i] = x[i].exp();
    }
}

#[cfg(target_arch = "aarch64")]
fn vvtanhf_neon(y: &mut [f32], x: &[f32]) {
    use crate::kernels::neon_exp4;
    use std::arch::aarch64::*;
    let n = x.len();
    let chunks = n / 4;
    unsafe {
        let two = vdupq_n_f32(2.0);
        let one = vdupq_n_f32(1.0);
        for c in 0..chunks {
            let off = c * 4;
            let v = vld1q_f32(x.as_ptr().add(off));
            // tanh(x) = (e^{2x} - 1) / (e^{2x} + 1)
            let e = neon_exp4(vmulq_f32(v, two));
            let num = vsubq_f32(e, one);
            let den = vaddq_f32(e, one);
            vst1q_f32(y.as_mut_ptr().add(off), vdivq_f32(num, den));
        }
    }
    for i in chunks * 4..n {
        y[i] = x[i].tanh();
    }
}

#[cfg(target_arch = "aarch64")]
fn vvrecf_neon(y: &mut [f32], x: &[f32]) {
    use std::arch::aarch64::*;
    let n = x.len();
    let chunks = n / 4;
    unsafe {
        let one = vdupq_n_f32(1.0);
        for c in 0..chunks {
            let off = c * 4;
            let v = vld1q_f32(x.as_ptr().add(off));
            vst1q_f32(y.as_mut_ptr().add(off), vdivq_f32(one, v));
        }
    }
    for i in chunks * 4..n {
        y[i] = 1.0 / x[i];
    }
}

#[cfg(target_arch = "aarch64")]
fn vvsqrtf_neon(y: &mut [f32], x: &[f32]) {
    use std::arch::aarch64::*;
    let n = x.len();
    let chunks = n / 4;
    unsafe {
        for c in 0..chunks {
            let off = c * 4;
            let v = vld1q_f32(x.as_ptr().add(off));
            vst1q_f32(y.as_mut_ptr().add(off), vsqrtq_f32(v));
        }
    }
    for i in chunks * 4..n {
        y[i] = x[i].sqrt();
    }
}

#[cfg(target_arch = "aarch64")]
fn vvrsqrtf_neon(y: &mut [f32], x: &[f32]) {
    use std::arch::aarch64::*;
    let n = x.len();
    let chunks = n / 4;
    unsafe {
        let one = vdupq_n_f32(1.0);
        for c in 0..chunks {
            let off = c * 4;
            let v = vld1q_f32(x.as_ptr().add(off));
            vst1q_f32(y.as_mut_ptr().add(off), vdivq_f32(one, vsqrtq_f32(v)));
        }
    }
    for i in chunks * 4..n {
        y[i] = 1.0 / x[i].sqrt();
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vvexpf_avx2(y: &mut [f32], x: &[f32]) {
    use crate::kernels::avx2_exp8;
    use std::arch::x86_64::*;
    let n = x.len();
    let chunks = n / 8;
    for c in 0..chunks {
        let off = c * 8;
        let v = _mm256_loadu_ps(x.as_ptr().add(off));
        _mm256_storeu_ps(y.as_mut_ptr().add(off), avx2_exp8(v));
    }
    for i in chunks * 8..n {
        y[i] = x[i].exp();
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vvtanhf_avx2(y: &mut [f32], x: &[f32]) {
    use crate::kernels::avx2_exp8;
    use std::arch::x86_64::*;
    let n = x.len();
    let chunks = n / 8;
    let two = _mm256_set1_ps(2.0);
    let one = _mm256_set1_ps(1.0);
    for c in 0..chunks {
        let off = c * 8;
        let v = _mm256_loadu_ps(x.as_ptr().add(off));
        let e = avx2_exp8(_mm256_mul_ps(v, two));
        let num = _mm256_sub_ps(e, one);
        let den = _mm256_add_ps(e, one);
        _mm256_storeu_ps(y.as_mut_ptr().add(off), _mm256_div_ps(num, den));
    }
    for i in chunks * 8..n {
        y[i] = x[i].tanh();
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vvrecf_avx2(y: &mut [f32], x: &[f32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let chunks = n / 8;
    let one = _mm256_set1_ps(1.0);
    for c in 0..chunks {
        let off = c * 8;
        let v = _mm256_loadu_ps(x.as_ptr().add(off));
        _mm256_storeu_ps(y.as_mut_ptr().add(off), _mm256_div_ps(one, v));
    }
    for i in chunks * 8..n {
        y[i] = 1.0 / x[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vvsqrtf_avx(y: &mut [f32], x: &[f32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let chunks = n / 8;
    for c in 0..chunks {
        let off = c * 8;
        let v = _mm256_loadu_ps(x.as_ptr().add(off));
        _mm256_storeu_ps(y.as_mut_ptr().add(off), _mm256_sqrt_ps(v));
    }
    for i in chunks * 8..n {
        y[i] = x[i].sqrt();
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn vvrsqrtf_avx(y: &mut [f32], x: &[f32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let chunks = n / 8;
    let one = _mm256_set1_ps(1.0);
    for c in 0..chunks {
        let off = c * 8;
        let v = _mm256_loadu_ps(x.as_ptr().add(off));
        _mm256_storeu_ps(
            y.as_mut_ptr().add(off),
            _mm256_div_ps(one, _mm256_sqrt_ps(v)),
        );
    }
    for i in chunks * 8..n {
        y[i] = 1.0 / x[i].sqrt();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn vvexpf_close_to_libm() {
        let x: Vec<f32> = (-40..40).map(|i| i as f32 * 0.17).collect();
        let mut y = vec![0.0f32; x.len()];
        vvexpf(&mut y, &x);
        let ref_y: Vec<f32> = x.iter().map(|v| v.exp()).collect();
        // Apple vForce may differ from libm by a few ULPs on large |x|.
        assert!(max_abs_err(&y, &ref_y) < 1e-4, "vvexpf vs libm");

        let mut yf = vec![0.0f32; x.len()];
        vvexpf_fast(&mut yf, &x);
        assert!(max_abs_err(&yf, &ref_y) < 1e-4, "vvexpf_fast maxabs");
    }

    #[test]
    fn vvtanhf_close_to_libm() {
        let x: Vec<f32> = (-50..50).map(|i| i as f32 * 0.11).collect();
        let mut y = vec![0.0f32; x.len()];
        vvtanhf(&mut y, &x);
        let ref_y: Vec<f32> = x.iter().map(|v| v.tanh()).collect();
        assert!(max_abs_err(&y, &ref_y) < 1e-4, "vvtanhf vs libm");

        let mut yf = vec![0.0f32; x.len()];
        vvtanhf_fast(&mut yf, &x);
        assert!(max_abs_err(&yf, &ref_y) < 1e-4, "vvtanhf_fast maxabs");
    }

    #[test]
    fn vvrecf_and_inplace_exp() {
        let x = vec![0.5f32, 1.0, 2.0, 4.0, -1.0];
        let mut y = vec![0.0f32; x.len()];
        vvrecf(&mut y, &x);
        assert!((y[1] - 1.0).abs() < 1e-6);
        assert!((y[2] - 0.5).abs() < 1e-6);

        let mut recip_inplace = x.clone();
        vvrecf_inplace(&mut recip_inplace);
        assert_eq!(recip_inplace, y);

        let mut z = vec![0.0f32, 1.0, -0.5];
        vvexpf_inplace(&mut z);
        assert!((z[0] - 1.0).abs() < 1e-5);
        assert!((z[1] - 1.0f32.exp()).abs() < 1e-5);
    }

    #[test]
    fn vvlogf_sqrtf_and_rsqrtf_close_to_libm() {
        let x: Vec<f32> = (1..100).map(|i| i as f32 * 0.13).collect();
        let mut y = vec![0.0f32; x.len()];

        vvlogf(&mut y, &x);
        let log_ref: Vec<f32> = x.iter().map(|v| v.ln()).collect();
        assert!(max_abs_err(&y, &log_ref) < 1e-5, "vvlogf vs libm");
        let mut inplace = x.clone();
        vvlogf_inplace(&mut inplace);
        assert!(max_abs_err(&inplace, &log_ref) < 1e-5, "vvlogf inplace");

        vvsqrtf(&mut y, &x);
        let sqrt_ref: Vec<f32> = x.iter().map(|v| v.sqrt()).collect();
        assert!(max_abs_err(&y, &sqrt_ref) < 1e-5, "vvsqrtf vs libm");
        let mut inplace = x.clone();
        vvsqrtf_inplace(&mut inplace);
        assert!(max_abs_err(&inplace, &sqrt_ref) < 1e-5, "vvsqrtf inplace");

        vvrsqrtf(&mut y, &x);
        let rsqrt_ref: Vec<f32> = x.iter().map(|v| 1.0 / v.sqrt()).collect();
        assert!(max_abs_err(&y, &rsqrt_ref) < 1e-5, "vvrsqrtf vs libm");
        let mut inplace = x;
        vvrsqrtf_inplace(&mut inplace);
        assert!(max_abs_err(&inplace, &rsqrt_ref) < 1e-5, "vvrsqrtf inplace");
    }

    #[test]
    fn vvsigmoidf_basic() {
        let x = vec![0.0f32, 10.0, -10.0];
        let mut y = vec![0.0f32; 3];
        vvsigmoidf(&mut y, &x);
        assert!((y[0] - 0.5).abs() < 1e-5);
        assert!(y[1] > 0.999);
        assert!(y[2] < 0.001);
    }
}
