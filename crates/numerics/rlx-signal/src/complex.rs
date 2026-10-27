// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! NumPy's `complex128` arithmetic, spelled the way NumPy computes it on a
//! clang/FMA build (macOS arm64 wheels): each operation is one ufunc loop or
//! scalar-math routine whose C expressions are FMA-contracted, so the spelling —
//! which product is fused — decides the last bit.
//!
//! * `a * b` → `(fma(ar, br, -(ai·bi)), fma(ar, bi, ai·br))`: the left product fused.
//! * `a / b` → Smith's algorithm (`CDOUBLE_divide` / `cdouble_ctype_divide`), contracted.
//! * `np.exp`, `np.sqrt` → libm `cexp` / `csqrt` (NumPy's `npy_cexp`/`npy_csqrt` defer
//!   to the platform when it has them), called through the C ABI.
//! * `np.abs` → `hypot`.
//! * `np.convolve` (hence `np.poly`) multiplies **without** contraction.

/// A `complex128`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct C {
    pub re: f64,
    pub im: f64,
}

#[cfg(target_vendor = "apple")]
unsafe extern "C" {
    fn cexp(z: C) -> C;
    fn csqrt(z: C) -> C;
}

impl C {
    pub const fn new(re: f64, im: f64) -> Self {
        C { re, im }
    }
    /// A real value as NumPy promotes it (`x + 0j`).
    pub const fn real(re: f64) -> Self {
        C { re, im: 0.0 }
    }
    pub fn conj(self) -> Self {
        C::new(self.re, -self.im)
    }
    /// `np.abs(z)` = `hypot(re, im)`.
    pub fn abs(self) -> f64 {
        rlx_libm::reference::hypot(self.re, self.im)
    }
    /// `np.isreal(z)`: imaginary part exactly zero.
    pub fn is_real(self) -> bool {
        self.im == 0.0
    }
}

/// `a + b`.
pub fn add(a: C, b: C) -> C {
    C::new(a.re + b.re, a.im + b.im)
}
/// `a - b`.
pub fn sub(a: C, b: C) -> C {
    C::new(a.re - b.re, a.im - b.im)
}
/// `-a`.
pub fn neg(a: C) -> C {
    C::new(-a.re, -a.im)
}
/// Real `x` minus complex `z` (`x - z` with `x` promoted to `x + 0j`).
pub fn rsub(x: f64, z: C) -> C {
    C::new(x - z.re, 0.0 - z.im)
}
/// Real `x` plus complex `z`.
pub fn radd(x: f64, z: C) -> C {
    C::new(x + z.re, 0.0 + z.im)
}

/// Whether NumPy's baseline-compiled complex loops fuse `x ± a*b` on this target's
/// reference. On macOS arm64 clang fuses them (left product first); NumPy's Linux
/// x86-64 wheel builds them for the SSE baseline, so they round twice — measured on
/// msi (NumPy 2.3.5, 3000 random cases each): `np.prod`'s complex `multiply.reduce`,
/// complex scalar `*` and `/`, and the array `/` loop are all plain, while the array
/// `*` and `**2` loops (AVX2/FMA3 dispatch) are fused, as on macOS.
pub const BASELINE_FUSED: bool = cfg!(target_vendor = "apple");

#[inline(always)]
fn fma_b(a: f64, b: f64, c: f64) -> f64 {
    if BASELINE_FUSED {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

/// The elementwise array `a * b` loop (and `z**2`): FMA-contracted, left product
/// fused, on every reference.
pub fn mul(a: C, b: C) -> C {
    C::new(
        a.re.mul_add(b.re, -(a.im * b.im)),
        a.re.mul_add(b.im, a.im * b.re),
    )
}

/// Uncontracted `a * b` (`np.convolve`'s inner product).
pub fn mul_plain(a: C, b: C) -> C {
    C::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
}

/// A complex scalar `a * b`, or one step of `np.prod`'s `multiply.reduce`: fused
/// as [`mul`] on macOS, plain on the Linux reference ([`BASELINE_FUSED`]).
pub fn mul_baseline(a: C, b: C) -> C {
    C::new(
        fma_b(a.re, b.re, -(a.im * b.im)),
        fma_b(a.re, b.im, a.im * b.re),
    )
}

/// `a / b` (array or scalar): Smith's algorithm as NumPy spells it, contracted on
/// macOS, plain on the Linux reference ([`BASELINE_FUSED`]).
pub fn div(a: C, b: C) -> C {
    let (br, bi) = (b.re.abs(), b.im.abs());
    if br >= bi {
        if br == 0.0 && bi == 0.0 {
            return C::new(a.re / br, a.im / br);
        }
        let rat = b.im / b.re;
        let scl = 1.0 / fma_b(b.im, rat, b.re);
        C::new(fma_b(a.im, rat, a.re) * scl, fma_b(-a.re, rat, a.im) * scl)
    } else {
        let rat = b.re / b.im;
        let scl = 1.0 / fma_b(b.re, rat, b.im);
        C::new(fma_b(a.re, rat, a.im) * scl, fma_b(a.im, rat, -a.re) * scl)
    }
}

/// `np.exp(z)` → libm `cexp`: Apple's, or glibc's ([`rlx_libm::cexp`]) elsewhere.
pub fn exp(z: C) -> C {
    #[cfg(target_vendor = "apple")]
    // SAFETY: `cexp` is a pure libm function; `C` is `repr(C)` with the layout and
    // calling convention of `double _Complex` (an HFA of two doubles).
    return unsafe { cexp(z) };
    #[cfg(not(target_vendor = "apple"))]
    {
        let (re, im) = rlx_libm::cexp(z.re, z.im);
        C::new(re, im)
    }
}

/// `np.sqrt(z)` → libm `csqrt`: Apple's, or glibc's ([`rlx_libm::csqrt`]) elsewhere.
pub fn sqrt(z: C) -> C {
    #[cfg(target_vendor = "apple")]
    // SAFETY: as for `exp`.
    return unsafe { csqrt(z) };
    #[cfg(not(target_vendor = "apple"))]
    {
        let (re, im) = rlx_libm::csqrt(z.re, z.im);
        C::new(re, im)
    }
}

/// `np.prod(v)` of a non-empty complex array: a left-to-right `multiply.reduce`
/// starting from `v[0]`; `1 + 0j` when empty.
pub fn prod(v: &[C]) -> C {
    let mut it = v.iter();
    match it.next() {
        None => C::real(1.0),
        Some(&first) => it.fold(first, |acc, &x| mul_baseline(acc, x)),
    }
}
