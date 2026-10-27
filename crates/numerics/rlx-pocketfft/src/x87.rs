// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The few `long double` operations pocketfft performs, as x86-64 Linux gcc
//! computes them: x87 extended precision (64-bit significand, round-to-nearest-even)
//! and then rounded again to `double`. On Apple targets `long double` is `double`,
//! so these are plain `f64` operations there.
//!
//! Used by `Thigh(0.25L*pi/n)` (the twiddle angle) and `pypocketfft.cxx`'s
//! `norm_fct` (`T(1/ldbl_t(N))`, `T(1/sqrt(ldbl_t(N)))`).

use rlx_linux_ref::x87::F80;

/// `0.25L * pi`: pi's 64-bit significand `0xC90FDAA22168C235`, exponent −1.
fn quarter_pi() -> F80 {
    F80::from_bits80((16383 - 1) as u16, 0xC90F_DAA2_2168_C235)
}

/// Whether this target's reference (`scipy`'s wheel) has an 80-bit `long double`:
/// the Linux x86-64 reference does, Apple arm64's `long double` is `double`.
pub(crate) const X87: bool = !cfg!(target_vendor = "apple");

/// `Thigh(0.25L*pi/n)`.
pub(crate) fn quarter_pi_over(n: usize) -> f64 {
    if X87 {
        quarter_pi().div(F80::from_u64(n as u64)).to_f64()
    } else {
        0.25 * std::f64::consts::PI / n as f64
    }
}

/// `T(1/ldbl_t(n))`.
pub(crate) fn recip(n: usize) -> f64 {
    if X87 {
        F80::from_u64(1).div(F80::from_u64(n as u64)).to_f64()
    } else {
        1.0 / n as f64
    }
}

/// `T(1/sqrt(ldbl_t(n)))`.
pub(crate) fn recip_sqrt(n: usize) -> f64 {
    if X87 {
        F80::from_u64(1)
            .div(F80::from_u64(n as u64).sqrt())
            .to_f64()
    } else {
        1.0 / (n as f64).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants() {
        assert_eq!(quarter_pi().to_f64(), std::f64::consts::FRAC_PI_4);
        for n in [1usize, 2, 4, 1 << 20] {
            assert_eq!(recip(n), 1.0 / n as f64);
        }
    }
}
