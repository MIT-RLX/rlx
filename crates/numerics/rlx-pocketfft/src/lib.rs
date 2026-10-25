// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # rlx-pocketfft — `scipy.fft` bit-for-bit
//!
//! SciPy's FFTs are [pocketfft](https://github.com/scipy/pocketfft)'s C++ header
//! (scipy/_lib/pocketfft @ `9367142748fcc9696a1c9e5a99b76ed9897c9daa`, the
//! submodule SciPy 1.17.1 pins), compiled by clang with `-ffp-contract=on`. A
//! generic FFT factors the length, orders the butterflies and computes the twiddles
//! differently, so it lands a few ULP away. [`engine`] is that header in pure Rust,
//! operation for operation, including where clang fuses and how the twiddles' `sin`
//! / `cos` are evaluated; both are selectable through [`Options`] (the `*_with`
//! entry points). The defaults reproduce SciPy on the target it runs on when that
//! is macOS arm64 (measured: every `tests/vectors` output bit-identical).
//!
//! The entry points mirror `scipy.fft` for 1-D input with `norm=None`
//! ("backward": forward unscaled, inverse scaled by `1/n`) including SciPy's
//! zero-pad / truncate to `n`. Complex values are `[re, im]`.

// pocketfft's literals are written to 34 digits; copying them verbatim is what
// guarantees the same rounding (a 16-digit spelling rounds some differently).
// The allow sits here, not in engine.rs, so it survives that file being rewritten.
#[allow(clippy::excessive_precision)]
pub mod engine;

pub use engine::{Contraction, Options, Twiddles};

/// A complex value as `[re, im]` (the layout of `std::complex<double>`).
pub type C64 = [f64; 2];

/// `scipy.fft`'s `norm` argument.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Norm {
    /// `None` / `"backward"`: forward unscaled, inverse `1/n`.
    #[default]
    Backward,
    /// `"ortho"`: both `1/sqrt(n)`.
    Ortho,
    /// `"forward"`: forward `1/n`, inverse unscaled.
    Forward,
}

/// `norm_fct(inorm, N)` of `pypocketfft.cxx` (`long double` is `double` on the
/// platforms SciPy's arm64/x86-64 macOS and Linux wheels target with clang/gcc —
/// on x86-64 Linux gcc `long double` is 80-bit, where `1/N` can round differently).
fn fct(norm: Norm, inverse: bool, n: usize) -> f64 {
    let inorm = match (norm, inverse) {
        (Norm::Backward, false) | (Norm::Forward, true) => 0,
        (Norm::Ortho, _) => 1,
        (Norm::Backward, true) | (Norm::Forward, false) => 2,
    };
    match inorm {
        0 => 1.0,
        1 => 1.0 / (n as f64).sqrt(),
        _ => 1.0 / n as f64,
    }
}

/// `x` zero-padded or truncated to `n` (SciPy's `_fix_shape`).
fn fix<T: Copy + Default>(x: &[T], n: usize) -> Vec<T> {
    let mut v: Vec<T> = x.iter().take(n).copied().collect();
    v.resize(n, T::default());
    v
}

/// `scipy.fft.rfft(x, n, norm)` → `n/2 + 1` bins.
///
/// # Panics
/// If `n == 0` (SciPy raises `ValueError`).
pub fn rfft(x: &[f64], n: Option<usize>, norm: Norm) -> Vec<C64> {
    rfft_with(x, n, norm, Options::default())
}

/// [`rfft`] with an explicit rounding.
pub fn rfft_with(x: &[f64], n: Option<usize>, norm: Norm, opt: Options) -> Vec<C64> {
    let n = n.unwrap_or(x.len());
    assert!(n > 0, "invalid number of data points (0) specified");
    let input = fix(x, n);
    flat(engine::r2c(&input, true, fct(norm, false, n), opt))
}

/// `scipy.fft.irfft(x, n, norm)` → `n` reals (`n` defaults to `2 * (len - 1)`).
/// The input is padded/truncated to `n/2 + 1` bins first.
///
/// # Panics
/// If the output length is 0.
pub fn irfft(x: &[C64], n: Option<usize>, norm: Norm) -> Vec<f64> {
    irfft_with(x, n, norm, Options::default())
}

/// [`irfft`] with an explicit rounding.
pub fn irfft_with(x: &[C64], n: Option<usize>, norm: Norm, opt: Options) -> Vec<f64> {
    let n = n.unwrap_or(2 * x.len().saturating_sub(1));
    assert!(n > 0, "Invalid number of data points (0) specified");
    let input = cx(&fix(x, n / 2 + 1));
    engine::c2r(&input, n, false, fct(norm, true, n), opt)
}

fn c2c(x: &[C64], n: Option<usize>, norm: Norm, inverse: bool, opt: Options) -> Vec<C64> {
    let n = n.unwrap_or(x.len());
    assert!(n > 0, "invalid number of data points (0) specified");
    let input = cx(&fix(x, n));
    flat(engine::c2c(&input, !inverse, fct(norm, inverse, n), opt))
}

fn cx(x: &[C64]) -> Vec<engine::Cx<f64>> {
    x.iter().map(|c| engine::Cx::new(c[0], c[1])).collect()
}
fn flat(x: Vec<engine::Cx<f64>>) -> Vec<C64> {
    x.into_iter().map(|c| [c.r, c.i]).collect()
}

/// `scipy.fft.fft(x, n, norm)` of complex input.
pub fn fft(x: &[C64], n: Option<usize>, norm: Norm) -> Vec<C64> {
    c2c(x, n, norm, false, Options::default())
}

/// [`fft`] with an explicit rounding.
pub fn fft_with(x: &[C64], n: Option<usize>, norm: Norm, opt: Options) -> Vec<C64> {
    c2c(x, n, norm, false, opt)
}

/// `scipy.fft.ifft(x, n, norm)`.
pub fn ifft(x: &[C64], n: Option<usize>, norm: Norm) -> Vec<C64> {
    c2c(x, n, norm, true, Options::default())
}

/// [`ifft`] with an explicit rounding.
pub fn ifft_with(x: &[C64], n: Option<usize>, norm: Norm, opt: Options) -> Vec<C64> {
    c2c(x, n, norm, true, opt)
}

/// `scipy.fft.next_fast_len(n, real=True)` — pocketfft's `good_size_real`
/// (the smallest `2^a 3^b 5^c ≥ n`).
pub fn next_fast_len_real(n: usize) -> usize {
    engine::good_size_real(n)
}

/// `scipy.fft.next_fast_len(n, real=False)` — `good_size_cmplx`
/// (`2^a 3^b 5^c 7^d 11^e ≥ n`).
pub fn next_fast_len_cmplx(n: usize) -> usize {
    engine::good_size_cmplx(n)
}
