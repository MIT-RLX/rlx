// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Boost.Math `tools/`: polynomial and rational evaluation, series summation,
//! continued fractions, and the `double` precision constants.
//!
//! Everything here reproduces what clang emits for the Boost headers under its
//! default `-ffp-contract=on`, which is how SciPy builds them: a product and a sum
//! **in one C++ expression** become one `fma`, the *left* product of an addition
//! first, and nothing fuses across statements. The fused steps are spelled `fmac`
//! (fused on Apple, plain on the Linux reference); every other operation rounds
//! separately, as written.

/// `tools::epsilon<double>()` and `policies::get_epsilon<double, Policy>()`: 2⁻⁵².
use crate::Contract;

pub const EPSILON: f64 = f64::EPSILON;
/// `tools::root_epsilon<double>()`: 2⁻²⁶ (Boost stores the literal, not `sqrt(eps)`).
pub const ROOT_EPSILON: f64 = 1.490_116_119_384_765_625e-8;
/// `tools::log_max_value<double>()`.
pub const LOG_MAX_VALUE: f64 = 709.0;
/// `tools::log_min_value<double>()`.
pub const LOG_MIN_VALUE: f64 = -708.0;
/// `tools::min_value<double>()`: the smallest *normal* double.
pub const MIN_VALUE: f64 = f64::MIN_POSITIVE;
/// `tools::max_value<double>()`.
pub const MAX_VALUE: f64 = f64::MAX;
/// `policies::get_max_series_iterations<Policy>()` for the default policy.
pub const MAX_SERIES_ITERATIONS: u64 = 1_000_000;
/// `max_factorial<double>::value`.
pub const MAX_FACTORIAL: u32 = 170;

/// `tools::evaluate_polynomial` for a fixed-size coefficient array, as the
/// `BOOST_MATH_POLY_METHOD 3` (second-order Horner) headers spell it.
///
/// `a[0]` is the constant term. Sizes 2–4 are one nested Horner expression, so
/// every step is fused. From 5 up the even and odd coefficients are accumulated
/// separately in `x²`; only each accumulator's *first* step is one expression
/// (`a[n]*x2 + a[n-2]`), the rest are `t *= x2; t += a[i];` statements.
#[inline]
pub fn evaluate_polynomial(a: &[f64], x: f64) -> f64 {
    let n = a.len();
    match n {
        0 => 0.0,
        1 => a[0],
        2..=4 => {
            let mut r = a[n - 1];
            for &c in a[..n - 1].iter().rev() {
                r = r.fmac(x, c);
            }
            r
        }
        _ => {
            let x2 = x * x;
            let (e, o) = second_order(|i| a[i], n, x2);
            e + o * x
        }
    }
}

/// The two `x²` accumulators of the second-order scheme over `n` coefficients
/// `c(0..n)`: returns `(ends_at_0, ends_at_1)` — the caller multiplies the second
/// by `x` (or `z`), which Boost does as its own statement.
#[inline]
fn second_order(c: impl Fn(usize) -> f64, n: usize, x2: f64) -> (f64, f64) {
    debug_assert!(n >= 5);
    let hi = n - 1;
    let mut acc = [c(hi).fmac(x2, c(hi - 2)), c(hi - 1).fmac(x2, c(hi - 3))];
    // `acc[0]` walks hi, hi-2, …; `acc[1]` walks hi-1, hi-3, …
    let mut next = [hi as isize - 4, hi as isize - 5];
    for (k, slot) in acc.iter_mut().enumerate() {
        while next[k] >= 0 {
            *slot *= x2;
            *slot += c(next[k] as usize);
            next[k] -= 2;
        }
    }
    // The accumulator whose last coefficient was c(0) is the even one.
    if hi.is_multiple_of(2) {
        (acc[0], acc[1])
    } else {
        (acc[1], acc[0])
    }
}

/// `tools::evaluate_rational(num, denom, z)` for equal-length fixed-size arrays
/// (`BOOST_MATH_RATIONAL_METHOD 3`).
///
/// Sizes 2–4 are a single fused Horner quotient with no range split. From 5 up,
/// `|x| <= 1` uses the second-order scheme in `x`; otherwise the coefficients are
/// taken in reverse at `z = 1/x`, with `z² = 1/(x*x)` (not `z*z`).
#[inline]
pub fn evaluate_rational(num: &[f64], denom: &[f64], x: f64) -> f64 {
    debug_assert_eq!(num.len(), denom.len());
    let n = num.len();
    match n {
        0 => 0.0,
        1 => num[0] / denom[0],
        2..=4 => evaluate_polynomial(num, x) / evaluate_polynomial(denom, x),
        _ => {
            if (-1.0..=1.0).contains(&x) {
                let x2 = x * x;
                let (ne, no) = second_order(|i| num[i], n, x2);
                let (de, do_) = second_order(|i| denom[i], n, x2);
                (ne + no * x) / (de + do_ * x)
            } else {
                let z = 1.0 / x;
                let z2 = 1.0 / (x * x);
                let (ne, no) = second_order(|i| num[n - 1 - i], n, z2);
                let (de, do_) = second_order(|i| denom[n - 1 - i], n, z2);
                (ne + no * z) / (de + do_ * z)
            }
        }
    }
}

/// `tools::sum_series(func, factor, max_terms, init_value)`.
///
/// Returns the sum and the number of terms used. Boost tests
/// `|factor * result| < |next_term|` *after* adding the term, and decrements the
/// budget only while that test holds.
#[inline]
pub fn sum_series(
    mut term: impl FnMut() -> f64,
    factor: f64,
    max_terms: u64,
    init: f64,
) -> (f64, u64) {
    let mut counter = max_terms;
    let mut result = init;
    loop {
        let next = term();
        result += next;
        if !((factor * result).abs() < next.abs()) {
            break;
        }
        counter -= 1;
        if counter == 0 {
            break;
        }
    }
    (result, max_terms - counter)
}

/// `detail::tiny_value<double>::get()`.
const TINY: f64 = 16.0 * f64::MIN_POSITIVE;

/// Modified Lentz, `tools::continued_fraction_b`: evaluates
/// `b0 + a1/(b1 + a2/(b2 + …))` from a generator of `(a_n, b_n)` pairs.
#[inline]
pub fn continued_fraction_b(mut g: impl FnMut() -> (f64, f64), factor: f64, max_terms: u64) -> f64 {
    let (_, b0) = g();
    lentz(&mut g, b0, factor, max_terms)
}

/// `tools::continued_fraction_a`: `a1/(b1 + a2/(b2 + …))`.
#[inline]
pub fn continued_fraction_a(mut g: impl FnMut() -> (f64, f64), factor: f64, max_terms: u64) -> f64 {
    let (a0, b0) = g();
    a0 / lentz(&mut g, b0, factor, max_terms)
}

#[inline]
fn lentz(g: &mut impl FnMut() -> (f64, f64), b0: f64, factor: f64, max_terms: u64) -> f64 {
    let terminator = factor.abs();
    let mut f = if b0 == 0.0 { TINY } else { b0 };
    let mut c = f;
    let mut d = 0.0f64;
    let mut counter = max_terms;
    loop {
        let (a, b) = g();
        // `D = b + a * D` is one expression: fused.
        d = a.fmac(d, b);
        if d == 0.0 {
            d = TINY;
        }
        c = b + a / c;
        if c == 0.0 {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if !((delta - 1.0).abs() > terminator) {
            break;
        }
        counter -= 1;
        if counter == 0 {
            break;
        }
    }
    f
}

/// `itrunc` / `ltrunc`: truncation toward zero.
#[inline]
pub(crate) fn itrunc(x: f64) -> i64 {
    x.trunc() as i64
}
