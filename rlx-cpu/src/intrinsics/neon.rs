// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! NEON intrinsic wrappers (plan #85).
//!
//! Thin typed wrappers around `std::arch::aarch64::*`. No algorithm
//! logic lives here; this is the dictionary of "what NEON ops do
//! we use" so kernels can be read as algorithms.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*;

/// 4-lane f32 vector. Aliasing makes the kernel signatures more
/// compact than carrying `float32x4_t` everywhere.
pub type F32x4 = float32x4_t;

#[inline(always)]
pub unsafe fn splat(x: f32) -> F32x4 {
    unsafe { vdupq_n_f32(x) }
}

#[inline(always)]
pub unsafe fn zero() -> F32x4 {
    unsafe { vdupq_n_f32(0.0) }
}

#[inline(always)]
pub unsafe fn load(p: *const f32) -> F32x4 {
    unsafe { vld1q_f32(p) }
}

#[inline(always)]
pub unsafe fn store(p: *mut f32, v: F32x4) {
    unsafe {
        vst1q_f32(p, v);
    }
}

/// Fused multiply-add: `acc + a*b`.
#[inline(always)]
pub unsafe fn fma(acc: F32x4, a: F32x4, b: F32x4) -> F32x4 {
    unsafe { vfmaq_f32(acc, a, b) }
}

/// Fused multiply-sub: `acc - a*b`.
#[inline(always)]
pub unsafe fn fms(acc: F32x4, a: F32x4, b: F32x4) -> F32x4 {
    unsafe { vfmsq_f32(acc, a, b) }
}

#[inline(always)]
pub unsafe fn mul(a: F32x4, b: F32x4) -> F32x4 {
    unsafe { vmulq_f32(a, b) }
}

#[inline(always)]
pub unsafe fn add(a: F32x4, b: F32x4) -> F32x4 {
    unsafe { vaddq_f32(a, b) }
}

/// Horizontal sum.
#[inline(always)]
pub unsafe fn hsum(v: F32x4) -> f32 {
    unsafe { vaddvq_f32(v) }
}

/// Strided f32 dot product over `len` elements via 4-wide FMA chain.
/// `lhs_stride` and `rhs_stride` are in elements (not bytes).
/// Tail handled scalar.
#[inline(always)]
pub unsafe fn strided_dot_f32(
    lhs: *const f32,
    lhs_stride: usize,
    rhs: *const f32,
    rhs_stride: usize,
    len: usize,
) -> f32 {
    unsafe {
        let mut acc = zero();
        let mut i = 0usize;
        // Stride-1 fast path covers 99% of callers (Q@K^T, etc.).
        if lhs_stride == 1 && rhs_stride == 1 {
            let chunks = len / 4;
            while i < chunks {
                let off = i * 4;
                acc = fma(acc, load(lhs.add(off)), load(rhs.add(off)));
                i += 1;
            }
            let mut sum = hsum(acc);
            for d in (chunks * 4)..len {
                sum += *lhs.add(d) * *rhs.add(d);
            }
            return sum;
        }
        // Strided: scalar fallback (no NEON gather on aarch64).
        let mut sum = 0.0f32;
        for d in 0..len {
            sum += *lhs.add(d * lhs_stride) * *rhs.add(d * rhs_stride);
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_scalar() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let scalar: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let vec = unsafe { strided_dot_f32(a.as_ptr(), 1, b.as_ptr(), 1, 8) };
        assert!((scalar - vec).abs() < 1e-5);
    }

    #[test]
    fn strided_dot_matches_scalar() {
        let a = [1.0f32, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0, 99.0];
        let b = [4.0f32, 99.0, 3.0, 99.0, 2.0, 99.0, 1.0, 99.0];
        // Stride 2 picks every other element.
        let scalar = 1.0 * 4.0 + 2.0 * 3.0 + 3.0 * 2.0 + 4.0 * 1.0;
        let vec = unsafe { strided_dot_f32(a.as_ptr(), 2, b.as_ptr(), 2, 4) };
        assert!((scalar - vec).abs() < 1e-5);
    }
}
