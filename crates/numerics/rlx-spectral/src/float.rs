//! The precision a front-end computes in. Implemented for `f32` (production)
//! and `f64` (reference-grade); tables (twiddles, windows, mel weights) are
//! built in f64 and rounded once into the working type.

use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

pub trait Float:
    Copy
    + Send
    + Sync
    + PartialOrd
    + Default
    + std::fmt::Debug
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
{
    const ZERO: Self;
    const ONE: Self;
    /// Machine epsilon (`f32::EPSILON` / `f64::EPSILON`).
    const EPSILON: Self;
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
    fn mul_add(self, a: Self, b: Self) -> Self;
    fn max(self, o: Self) -> Self;
    fn sqrt(self) -> Self;
    /// Natural log of every element, in place.
    fn ln_slice(x: &mut [Self]);
    /// `log10` of every element, in place.
    fn log10_slice(x: &mut [Self]);
}

impl Float for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const EPSILON: Self = f32::EPSILON;
    #[inline(always)]
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        f32::mul_add(self, a, b)
    }
    #[inline(always)]
    fn max(self, o: Self) -> Self {
        f32::max(self, o)
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
    fn ln_slice(x: &mut [Self]) {
        rlx_fastmath::log_inplace(x);
    }
    fn log10_slice(x: &mut [Self]) {
        rlx_fastmath::log_inplace(x);
        let k = std::f32::consts::LOG10_E;
        x.iter_mut().for_each(|v| *v *= k);
    }
}

impl Float for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const EPSILON: Self = f64::EPSILON;
    #[inline(always)]
    fn from_f64(x: f64) -> Self {
        x
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        f64::mul_add(self, a, b)
    }
    #[inline(always)]
    fn max(self, o: Self) -> Self {
        f64::max(self, o)
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    fn ln_slice(x: &mut [Self]) {
        x.iter_mut().for_each(|v| *v = libm::log(*v));
    }
    fn log10_slice(x: &mut [Self]) {
        x.iter_mut().for_each(|v| *v = libm::log10(*v));
    }
}

/// A complex number in the working precision.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex<T> {
    pub re: T,
    pub im: T,
}

impl<T: Float> Complex<T> {
    #[inline(always)]
    pub fn new(re: T, im: T) -> Self {
        Complex { re, im }
    }
    #[inline(always)]
    pub fn add(self, o: Self) -> Self {
        Complex::new(self.re + o.re, self.im + o.im)
    }
    #[inline(always)]
    pub fn sub(self, o: Self) -> Self {
        Complex::new(self.re - o.re, self.im - o.im)
    }
    #[inline(always)]
    pub fn mul(self, o: Self) -> Self {
        Complex::new(
            self.re.mul_add(o.re, -(self.im * o.im)),
            self.re.mul_add(o.im, self.im * o.re),
        )
    }
    #[inline(always)]
    pub fn scale(self, s: T) -> Self {
        Complex::new(self.re * s, self.im * s)
    }
    #[inline(always)]
    pub fn conj(self) -> Self {
        Complex::new(self.re, -self.im)
    }
    /// `self · (-i)`
    #[inline(always)]
    pub fn mul_neg_i(self) -> Self {
        Complex::new(self.im, -self.re)
    }
    #[inline(always)]
    pub fn norm_sqr(self) -> T {
        self.re.mul_add(self.re, self.im * self.im)
    }
    /// `e^{iθ}` built in f64.
    pub fn cis(theta: f64) -> Self {
        Complex::new(T::from_f64(libm::cos(theta)), T::from_f64(libm::sin(theta)))
    }
}
