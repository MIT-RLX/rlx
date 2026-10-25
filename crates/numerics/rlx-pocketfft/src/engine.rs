// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! pocketfft's 1-D transforms (`scipy/_lib/pocketfft` @ `9367142`, the header SciPy
//! 1.17.1 and torch build `scipy.fft` / `torch.fft` from) in Rust, operation for
//! operation as clang compiles them on macOS arm64 (`-ffp-contract=on`):
//!
//! * an FMA exactly where clang fuses — a `x ± a*b` written in **one scalar
//!   expression**, the left product first (`a*b + c*d` → `fma(a, b, c*d)`). The
//!   `cmplx` operators are separate functions, so complex expressions
//!   (`t0 + t1*twr`) are never fused; compound `x += a*b` is, `x += a*b + c*d` is
//!   fused inside the right-hand side only;
//! * the FFTPACK passes (radix 2, 3, 4, 5, 7, 8, 11 and generic for complex; 2, 3, 4,
//!   5 and generic for real), Bluestein for lengths with a large prime factor, the
//!   plan choice by `cost_guess`, and the two-table twiddles (`sincos_2pibyn`,
//!   computed in `f64` and rounded to `T`);
//! * pocketfft's signed zeros (`akf[0]*T0(0)`, `T0(0)*c[0]` as padding).
//!
//! Two choices decide the rounding, and [`Options`] selects both:
//!
//! * [`Contraction`]: whether `x ± a*b` is fused as clang `-ffp-contract=on`
//!   compiles it (SciPy/torch on macOS arm64), or rounded twice (an x86-64 baseline
//!   build without FMA);
//! * [`Twiddles`]: how each `cos(a)`, `sin(a)` pair is evaluated: one merged
//!   `__sincos_stret` call (what clang emits on Darwin), or separate libm calls.
//!
//! Measured: SciPy 1.17.1 on macOS arm64 is `Contraction::Clang` +
//! `Twiddles::AppleSinCos` (every `tests/vectors` output bit-identical); with
//! separate `sin` / `cos` 286 outputs differ.

use std::ops::{Add, Div, Mul, Neg, Sub};

/// The scalar types pocketfft is instantiated for here.
pub trait Real:
    Copy
    + PartialEq
    + Default
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + std::fmt::Debug
{
    /// Whether `fma` fuses (also decides the twiddle products' rounding).
    const FUSED: bool = true;
    /// `self*b + c` where the reference fuses it.
    fn fma(self, b: Self, c: Self) -> Self;
    fn of(x: f64) -> Self;
    const ZERO: Self;
    const ONE: Self;
}
impl Real for f64 {
    #[inline(always)]
    fn fma(self, b: f64, c: f64) -> f64 {
        self.mul_add(b, c)
    }
    #[inline(always)]
    fn of(x: f64) -> f64 {
        x
    }
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
}
impl Real for f32 {
    #[inline(always)]
    fn fma(self, b: f32, c: f32) -> f32 {
        self.mul_add(b, c)
    }
    #[inline(always)]
    fn of(x: f64) -> f32 {
        x as f32
    }
    const ZERO: f32 = 0.0;
    const ONE: f32 = 1.0;
}

/// `T` with every contraction rounded twice: `fma(a, b, c)` is `a*b + c`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(transparent)]
pub struct NoFma<T>(pub T);

macro_rules! nofma_op {
    ($tr:ident, $f:ident) => {
        impl<T: Real> $tr for NoFma<T> {
            type Output = Self;
            #[inline(always)]
            fn $f(self, o: Self) -> Self {
                NoFma($tr::$f(self.0, o.0))
            }
        }
    };
}
nofma_op!(Add, add);
nofma_op!(Sub, sub);
nofma_op!(Mul, mul);
nofma_op!(Div, div);
impl<T: Real> Neg for NoFma<T> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        NoFma(-self.0)
    }
}
impl<T: Real> Real for NoFma<T> {
    const FUSED: bool = false;
    #[inline(always)]
    fn fma(self, b: Self, c: Self) -> Self {
        NoFma(self.0 * b.0 + c.0)
    }
    #[inline(always)]
    fn of(x: f64) -> Self {
        NoFma(T::of(x))
    }
    const ZERO: Self = NoFma(T::ZERO);
    const ONE: Self = NoFma(T::ONE);
}

/// `cmplx<T>`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cx<T> {
    pub r: T,
    pub i: T,
}

impl<T: Real> Cx<T> {
    #[inline(always)]
    pub fn new(r: T, i: T) -> Self {
        Cx { r, i }
    }
    #[inline(always)]
    fn add(self, o: Cx<T>) -> Cx<T> {
        Cx::new(self.r + o.r, self.i + o.i)
    }
    #[inline(always)]
    fn sub(self, o: Cx<T>) -> Cx<T> {
        Cx::new(self.r - o.r, self.i - o.i)
    }
    /// `cmplx * T` (not fused: a separate operator).
    #[inline(always)]
    fn scale(self, s: T) -> Cx<T> {
        Cx::new(self.r * s, self.i * s)
    }
}

/// `special_mul<fwd>(v1, v2)`: `v1 · conj(v2)` (fwd) or `v1 · v2`, each component one
/// fused expression.
#[inline(always)]
fn smul<T: Real>(fwd: bool, v1: Cx<T>, v2: Cx<T>) -> Cx<T> {
    if fwd {
        Cx::new(v1.r.fma(v2.r, v1.i * v2.i), v1.i.fma(v2.r, -(v1.r * v2.i)))
    } else {
        Cx::new(v1.r.fma(v2.r, -(v1.i * v2.i)), v1.r.fma(v2.i, v1.i * v2.r))
    }
}
/// `PM(a, b, c, d)`: `(c + d, c - d)`.
#[inline(always)]
fn pm<T: Real>(c: Cx<T>, d: Cx<T>) -> (Cx<T>, Cx<T>) {
    (c.add(d), c.sub(d))
}
/// `ROTX90<fwd>`.
#[inline(always)]
fn rotx90<T: Real>(fwd: bool, a: Cx<T>) -> Cx<T> {
    if fwd {
        Cx::new(a.i, -a.r)
    } else {
        Cx::new(-a.i, a.r)
    }
}

// ── twiddles ──────────────────────────────────────────────────────────────────

/// How the twiddles' `cos(a)`, `sin(a)` pairs are evaluated (in `f64`, then
/// rounded to `T`). The two disagree in the last bit for a few angles because
/// neither Apple's `sin`/`cos` nor its `__sincos_stret` is correctly rounded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Twiddles {
    /// What a clang `-O3` build does on this target: [`Twiddles::AppleSinCos`] on
    /// macOS, [`Twiddles::Separate`] elsewhere.
    #[default]
    Platform,
    /// One `__sincos_stret` call per angle, as clang merges `cos`+`sin` on Darwin
    /// (SciPy's and torch's macOS wheels). Only macOS has it; on other targets this
    /// is [`Twiddles::Separate`], which can differ in the last bit.
    AppleSinCos,
    /// Separate `cos` and `sin` calls to this target's libm (glibc on Linux, the
    /// `libm` port on wasm32).
    Separate,
}

impl Twiddles {
    /// Whether this choice reproduces its reference on the current target.
    pub fn exact_here(self) -> bool {
        !matches!(self, Twiddles::AppleSinCos) || cfg!(target_os = "macos")
    }
    fn merged(self) -> bool {
        cfg!(target_os = "macos") && matches!(self, Twiddles::Platform | Twiddles::AppleSinCos)
    }
}

/// Whether `x ± a*b` is fused.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Contraction {
    /// What SciPy's wheel for this target does: [`Contraction::Clang`] on Apple
    /// targets, [`Contraction::None`] elsewhere — the Linux x86-64 wheel (the Linux
    /// and wasm reference) is an SSE2 baseline build: `pypocketfft*.so` contains no
    /// FMA instruction.
    #[default]
    Platform,
    /// Exactly where clang `-ffp-contract=on` fuses: one scalar expression, the
    /// left product first (SciPy/torch builds on arm64, and on x86-64 with FMA).
    Clang,
    /// Never: every product rounded (an x86-64 SSE2 baseline build).
    None,
}

impl Contraction {
    /// Whether this choice fuses on the current target.
    pub fn fused(self) -> bool {
        match self {
            Contraction::Platform => cfg!(target_vendor = "apple"),
            Contraction::Clang => true,
            Contraction::None => false,
        }
    }
}

/// The rounding of a transform.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Options {
    pub contraction: Contraction,
    pub twiddles: Twiddles,
}

impl Options {
    /// `scipy.fft` / `torch.fft` (CPU) on macOS arm64, measured bit-identical.
    pub const SCIPY_MACOS_ARM64: Options = Options {
        contraction: Contraction::Clang,
        twiddles: Twiddles::AppleSinCos,
    };
    /// A build without FMA, separate libm calls.
    pub const NO_FMA: Options = Options {
        contraction: Contraction::None,
        twiddles: Twiddles::Separate,
    };
}

#[cfg(target_os = "macos")]
#[repr(C)]
struct SinCos {
    sin: f64,
    cos: f64,
}
#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn __sincos_stret(x: f64) -> SinCos;
}

/// `(cos a, sin a)`.
#[inline(never)]
fn cos_sin(a: f64, merged: bool) -> (f64, f64) {
    #[cfg(target_os = "macos")]
    if merged {
        // SAFETY: a pure libm function of a double.
        let r = unsafe { __sincos_stret(a) };
        return (r.cos, r.sin);
    }
    let _ = merged;
    (std::hint::black_box(a).cos(), std::hint::black_box(a).sin())
}

/// `sincos_2pibyn<T>::calc(x, n, ang)` (in `f64`).
fn calc(x: usize, n: usize, ang: f64, merged: bool) -> Cx<f64> {
    let mut x = x << 3;
    let c = |v: usize| cos_sin(v as f64 * ang, merged);
    if x < 4 * n {
        if x < 2 * n {
            if x < n {
                let (co, si) = c(x);
                return Cx { r: co, i: si };
            }
            let (co, si) = c(2 * n - x);
            return Cx { r: si, i: co };
        }
        x -= 2 * n;
        if x < n {
            let (co, si) = c(x);
            return Cx { r: -si, i: co };
        }
        let (co, si) = c(2 * n - x);
        Cx { r: -co, i: si }
    } else {
        x = 8 * n - x;
        if x < 2 * n {
            if x < n {
                let (co, si) = c(x);
                return Cx { r: co, i: -si };
            }
            let (co, si) = c(2 * n - x);
            return Cx { r: si, i: -co };
        }
        x -= 2 * n;
        if x < n {
            let (co, si) = c(x);
            return Cx { r: -si, i: -co };
        }
        let (co, si) = c(2 * n - x);
        Cx { r: -co, i: -si }
    }
}

/// `sincos_2pibyn<T>`: `e^{2πi·k/n}` from two tables of `f64` values.
struct SinCos2PiByN {
    n: usize,
    mask: usize,
    shift: u32,
    v1: Vec<Cx<f64>>,
    v2: Vec<Cx<f64>>,
}

impl SinCos2PiByN {
    fn new(n: usize, tw: Twiddles) -> Self {
        let merged = tw.merged();
        // Thigh(0.25L*pi/n): long double is double on Apple arm64
        let ang = 0.25 * std::f64::consts::PI / n as f64;
        let nval = (n + 2) / 2;
        let mut shift = 1u32;
        while (1usize << shift) * (1usize << shift) < nval {
            shift += 1;
        }
        let mask = (1usize << shift) - 1;
        let mut v1 = vec![Cx { r: 1.0, i: 0.0 }; mask + 1];
        for (i, v) in v1.iter_mut().enumerate().skip(1) {
            *v = calc(i, n, ang, merged);
        }
        let mut v2 = vec![Cx { r: 1.0, i: 0.0 }; nval.div_ceil(mask + 1)];
        for (i, v) in v2.iter_mut().enumerate().skip(1) {
            *v = calc(i * (mask + 1), n, ang, merged);
        }
        SinCos2PiByN {
            n,
            mask,
            shift,
            v1,
            v2,
        }
    }
    fn get<T: Real>(&self, idx: usize) -> Cx<T> {
        let (idx, neg) = if 2 * idx <= self.n {
            (idx, false)
        } else {
            (self.n - idx, true)
        };
        let (x1, x2) = (self.v1[idx & self.mask], self.v2[idx >> self.shift]);
        let (r, i) = if T::FUSED {
            (
                x1.r.mul_add(x2.r, -(x1.i * x2.i)),
                x1.r.mul_add(x2.i, x1.i * x2.r),
            )
        } else {
            (x1.r * x2.r - x1.i * x2.i, x1.r * x2.i + x1.i * x2.r)
        };
        let (r, i) = (T::of(r), T::of(i));
        Cx::new(r, if neg { -i } else { i })
    }
}

// ── util ──────────────────────────────────────────────────────────────────────

fn largest_prime_factor(mut n: usize) -> usize {
    let mut res = 1;
    while n & 1 == 0 {
        res = 2;
        n >>= 1;
    }
    let mut x = 3;
    while x * x <= n {
        while n.is_multiple_of(x) {
            res = x;
            n /= x;
        }
        x += 2;
    }
    if n > 1 {
        res = n;
    }
    res
}

fn cost_guess(mut n: usize) -> f64 {
    const LFP: f64 = 1.1;
    let ni = n;
    let mut result = 0.0f64;
    while n & 1 == 0 {
        result += 2.0;
        n >>= 1;
    }
    let mut x = 3;
    while x * x <= n {
        while n.is_multiple_of(x) {
            result += if x <= 5 { x as f64 } else { LFP * x as f64 };
            n /= x;
        }
        x += 2;
    }
    if n > 1 {
        result += if n <= 5 { n as f64 } else { LFP * n as f64 };
    }
    result * ni as f64
}

/// `util::good_size_cmplx`: the smallest 2·3·5·7·11-smooth number `≥ n`.
pub fn good_size_cmplx(n: usize) -> usize {
    if n <= 12 {
        return n;
    }
    let mut bestfac = 2 * n;
    let mut f11 = 1;
    while f11 < bestfac {
        let mut f117 = f11;
        while f117 < bestfac {
            let mut f1175 = f117;
            while f1175 < bestfac {
                let mut x = f1175;
                while x < n {
                    x *= 2;
                }
                loop {
                    if x < n {
                        x *= 3;
                    } else if x > n {
                        if x < bestfac {
                            bestfac = x;
                        }
                        if x & 1 == 1 {
                            break;
                        }
                        x >>= 1;
                    } else {
                        return n;
                    }
                }
                f1175 *= 5;
            }
            f117 *= 7;
        }
        f11 *= 11;
    }
    bestfac
}

/// `util::good_size_real`: the smallest 2·3·5-smooth number `≥ n`.
pub fn good_size_real(n: usize) -> usize {
    if n <= 6 {
        return n;
    }
    let mut bestfac = 2 * n;
    let mut f5 = 1;
    while f5 < bestfac {
        let mut x = f5;
        while x < n {
            x *= 2;
        }
        loop {
            if x < n {
                x *= 3;
            } else if x > n {
                if x < bestfac {
                    bestfac = x;
                }
                if x & 1 == 1 {
                    break;
                }
                x >>= 1;
            } else {
                return n;
            }
        }
        f5 *= 5;
    }
    bestfac
}

// ── complex FFTPACK (cfftp) ───────────────────────────────────────────────────

/// A long-double literal as `T0(…L)` converts it on arm64 (via `double`).
#[inline(always)]
fn lit<T: Real>(x: f64) -> T {
    T::of(x)
}

struct CFact<T> {
    fct: usize,
    tw: Vec<Cx<T>>,
    tws: Vec<Cx<T>>,
}

pub struct Cfftp<T> {
    length: usize,
    fact: Vec<CFact<T>>,
}

impl<T: Real> Cfftp<T> {
    pub fn new(length: usize, tw: Twiddles) -> Self {
        assert!(length > 0, "zero-length FFT requested");
        let mut p = Cfftp {
            length,
            fact: Vec::new(),
        };
        if length == 1 {
            return p;
        }
        // factorize
        let mut len = length;
        let mut f = Vec::new();
        while len & 7 == 0 {
            f.push(8);
            len >>= 3;
        }
        while len & 3 == 0 {
            f.push(4);
            len >>= 2;
        }
        if len & 1 == 0 {
            len >>= 1;
            f.push(2);
            let last = f.len() - 1;
            f.swap(0, last);
        }
        let mut d = 3;
        while d * d <= len {
            while len.is_multiple_of(d) {
                f.push(d);
                len /= d;
            }
            d += 2;
        }
        if len > 1 {
            f.push(len);
        }
        // twiddles
        let tw = SinCos2PiByN::new(length, tw);
        let mut l1 = 1;
        for &ip in &f {
            let ido = length / (l1 * ip);
            let mut t = vec![Cx::default(); (ip - 1) * (ido - 1)];
            for j in 1..ip {
                for i in 1..ido {
                    t[(j - 1) * (ido - 1) + i - 1] = tw.get(j * l1 * i);
                }
            }
            let tws = if ip > 11 {
                (0..ip).map(|j| tw.get(j * l1 * ido)).collect()
            } else {
                Vec::new()
            };
            p.fact.push(CFact {
                fct: ip,
                tw: t,
                tws,
            });
            l1 *= ip;
        }
        p
    }

    pub fn len(&self) -> usize {
        self.length
    }
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// `exec(c, fct, fwd)`.
    pub fn exec(&self, c: &mut [Cx<T>], fct: T, fwd: bool) {
        let n = self.length;
        if n == 1 {
            c[0] = c[0].scale(fct);
            return;
        }
        let mut l1 = 1;
        let mut ch = vec![Cx::<T>::default(); n];
        let mut in_c = true; // p1 == c
        for f in &self.fact {
            let ip = f.fct;
            let l2 = ip * l1;
            let ido = n / l2;
            let (src, dst): (&mut [Cx<T>], &mut [Cx<T>]) = if in_c {
                (&mut *c, &mut ch[..])
            } else {
                (&mut ch[..], &mut *c)
            };
            match ip {
                4 => pass4(fwd, ido, l1, src, dst, &f.tw),
                8 => pass8(fwd, ido, l1, src, dst, &f.tw),
                2 => pass2(fwd, ido, l1, src, dst, &f.tw),
                3 => pass3(fwd, ido, l1, src, dst, &f.tw),
                5 => pass5(fwd, ido, l1, src, dst, &f.tw),
                7 => pass7(fwd, ido, l1, src, dst, &f.tw),
                11 => pass11(fwd, ido, l1, src, dst, &f.tw),
                _ => {
                    passg(fwd, ido, ip, l1, src, dst, &f.tw, &f.tws);
                    in_c = !in_c; // the result is back in `src`
                }
            }
            in_c = !in_c;
            l1 = l2;
        }
        if !in_c {
            if fct != T::ONE {
                for i in 0..n {
                    c[i] = ch[i].scale(fct);
                }
            } else {
                c.copy_from_slice(&ch);
            }
        } else if fct != T::ONE {
            for v in c.iter_mut() {
                *v = v.scale(fct);
            }
        }
    }
}

fn pass2<T: Real>(fwd: bool, ido: usize, l1: usize, cc: &[Cx<T>], ch: &mut [Cx<T>], wa: &[Cx<T>]) {
    let cc_ = |a: usize, b: usize, c: usize| cc[a + ido * (b + 2 * c)];
    let chi = |a: usize, b: usize, c: usize| a + ido * (b + l1 * c);
    let w = |x: usize, i: usize| wa[i - 1 + x * (ido - 1)];
    for k in 0..l1 {
        ch[chi(0, k, 0)] = cc_(0, 0, k).add(cc_(0, 1, k));
        ch[chi(0, k, 1)] = cc_(0, 0, k).sub(cc_(0, 1, k));
        for i in 1..ido {
            ch[chi(i, k, 0)] = cc_(i, 0, k).add(cc_(i, 1, k));
            ch[chi(i, k, 1)] = smul(fwd, cc_(i, 0, k).sub(cc_(i, 1, k)), w(0, i));
        }
    }
}

fn pass3<T: Real>(fwd: bool, ido: usize, l1: usize, cc: &[Cx<T>], ch: &mut [Cx<T>], wa: &[Cx<T>]) {
    let tw1r: T = lit(-0.5);
    let tw1i: T =
        lit::<T>(0.8660254037844386467637231707529362) * if fwd { -T::ONE } else { T::ONE };
    let cc_ = |a: usize, b: usize, c: usize| cc[a + ido * (b + 3 * c)];
    let chi = |a: usize, b: usize, c: usize| a + ido * (b + l1 * c);
    let w = |x: usize, i: usize| wa[i - 1 + x * (ido - 1)];
    for k in 0..l1 {
        for i in 0..ido {
            let t0 = cc_(i, 0, k);
            let (t1, t2) = pm(cc_(i, 1, k), cc_(i, 2, k));
            ch[chi(i, k, 0)] = t0.add(t1);
            let ca = t0.add(t1.scale(tw1r));
            let cb = Cx::new((-t2.i) * tw1i, t2.r * tw1i);
            if i == 0 {
                let (a, b) = pm(ca, cb);
                ch[chi(0, k, 1)] = a;
                ch[chi(0, k, 2)] = b;
            } else {
                ch[chi(i, k, 1)] = smul(fwd, ca.add(cb), w(0, i));
                ch[chi(i, k, 2)] = smul(fwd, ca.sub(cb), w(1, i));
            }
        }
    }
}

fn pass4<T: Real>(fwd: bool, ido: usize, l1: usize, cc: &[Cx<T>], ch: &mut [Cx<T>], wa: &[Cx<T>]) {
    let cc_ = |a: usize, b: usize, c: usize| cc[a + ido * (b + 4 * c)];
    let chi = |a: usize, b: usize, c: usize| a + ido * (b + l1 * c);
    let w = |x: usize, i: usize| wa[i - 1 + x * (ido - 1)];
    for k in 0..l1 {
        {
            let (t2, t1) = pm(cc_(0, 0, k), cc_(0, 2, k));
            let (t3, t4) = pm(cc_(0, 1, k), cc_(0, 3, k));
            let t4 = rotx90(fwd, t4);
            let (a, b) = pm(t2, t3);
            ch[chi(0, k, 0)] = a;
            ch[chi(0, k, 2)] = b;
            let (a, b) = pm(t1, t4);
            ch[chi(0, k, 1)] = a;
            ch[chi(0, k, 3)] = b;
        }
        for i in 1..ido {
            let (cc0, cc1, cc2, cc3) = (cc_(i, 0, k), cc_(i, 1, k), cc_(i, 2, k), cc_(i, 3, k));
            let (t2, t1) = pm(cc0, cc2);
            let (t3, t4) = pm(cc1, cc3);
            let t4 = rotx90(fwd, t4);
            ch[chi(i, k, 0)] = t2.add(t3);
            ch[chi(i, k, 1)] = smul(fwd, t1.add(t4), w(0, i));
            ch[chi(i, k, 2)] = smul(fwd, t2.sub(t3), w(1, i));
            ch[chi(i, k, 3)] = smul(fwd, t1.sub(t4), w(2, i));
        }
    }
}

/// `y1*a ± y2*b ± …` as clang contracts it: the first two products as
/// `fma(y1, a, ±(y2*b))`, each further one `fma(±yk, v, acc)`. `terms[0]`'s sign is
/// the unary sign on its coefficient.
#[inline(always)]
fn signed_chain<T: Real>(terms: &[(T, T)]) -> T {
    let (c0, v0) = terms[0];
    let (c1, v1) = terms[1];
    let mut acc = c0.fma(v0, c1 * v1);
    for &(c, v) in &terms[2..] {
        acc = c.fma(v, acc);
    }
    acc
}

fn pass5<T: Real>(fwd: bool, ido: usize, l1: usize, cc: &[Cx<T>], ch: &mut [Cx<T>], wa: &[Cx<T>]) {
    let s = if fwd { -T::ONE } else { T::ONE };
    let tw1r: T = lit(0.3090169943749474241022934171828191);
    let tw1i: T = s * lit(0.9510565162951535721164393333793821);
    let tw2r: T = lit(-0.8090169943749474241022934171828191);
    let tw2i: T = s * lit(0.5877852522924731291687059546390728);
    let cc_ = |a: usize, b: usize, c: usize| cc[a + ido * (b + 5 * c)];
    let chi = |a: usize, b: usize, c: usize| a + ido * (b + l1 * c);
    let w = |x: usize, i: usize| wa[i - 1 + x * (ido - 1)];
    for k in 0..l1 {
        for i in 0..ido {
            let t0 = cc_(i, 0, k);
            let (t1, t4) = pm(cc_(i, 1, k), cc_(i, 4, k));
            let (t2, t3) = pm(cc_(i, 2, k), cc_(i, 3, k));
            ch[chi(i, k, 0)] = Cx::new(t0.r + t1.r + t2.r, t0.i + t1.i + t2.i);
            // (u1, u2, twar, twbr, twai, ±twbi): the sign is the binary ± in the source
            for (u1, u2, twar, twbr, twai, twbi) in [
                (1, 4, tw1r, tw2r, tw1i, tw2i),
                (2, 3, tw2r, tw1r, tw2i, -tw1i),
            ] {
                let ca = Cx::new(
                    twbr.fma(t2.r, twar.fma(t1.r, t0.r)),
                    twbr.fma(t2.i, twar.fma(t1.i, t0.i)),
                );
                let cb = Cx::new(
                    -signed_chain(&[(twai, t4.i), (twbi, t3.i)]),
                    signed_chain(&[(twai, t4.r), (twbi, t3.r)]),
                );
                if i == 0 {
                    let (a, b) = pm(ca, cb);
                    ch[chi(0, k, u1)] = a;
                    ch[chi(0, k, u2)] = b;
                } else {
                    ch[chi(i, k, u1)] = smul(fwd, ca.add(cb), w(u1 - 1, i));
                    ch[chi(i, k, u2)] = smul(fwd, ca.sub(cb), w(u2 - 1, i));
                }
            }
        }
    }
}

fn pass7<T: Real>(fwd: bool, ido: usize, l1: usize, cc: &[Cx<T>], ch: &mut [Cx<T>], wa: &[Cx<T>]) {
    let s = if fwd { -T::ONE } else { T::ONE };
    let tw1r: T = lit(0.6234898018587335305250048840042398);
    let tw1i: T = s * lit(0.7818314824680298087084445266740578);
    let tw2r: T = lit(-0.2225209339563144042889025644967948);
    let tw2i: T = s * lit(0.9749279121818236070181316829939312);
    let tw3r: T = lit(-0.9009688679024191262361023195074451);
    let tw3i: T = s * lit(0.433883739117558120475768332848359);
    let cc_ = |a: usize, b: usize, c: usize| cc[a + ido * (b + 7 * c)];
    let chi = |a: usize, b: usize, c: usize| a + ido * (b + l1 * c);
    let w = |x: usize, i: usize| wa[i - 1 + x * (ido - 1)];
    for k in 0..l1 {
        for i in 0..ido {
            let t1 = cc_(i, 0, k);
            let (t2, t7) = pm(cc_(i, 1, k), cc_(i, 6, k));
            let (t3, t6) = pm(cc_(i, 2, k), cc_(i, 5, k));
            let (t4, t5) = pm(cc_(i, 3, k), cc_(i, 4, k));
            ch[chi(i, k, 0)] = Cx::new(t1.r + t2.r + t3.r + t4.r, t1.i + t2.i + t3.i + t4.i);
            let steps = [
                (1, 6, [tw1r, tw2r, tw3r], [tw1i, tw2i, tw3i]),
                (2, 5, [tw2r, tw3r, tw1r], [tw2i, -tw3i, -tw1i]),
                (3, 4, [tw3r, tw1r, tw2r], [tw3i, -tw1i, tw2i]),
            ];
            for (u1, u2, x, y) in steps {
                let ca = Cx::new(
                    x[2].fma(t4.r, x[1].fma(t3.r, x[0].fma(t2.r, t1.r))),
                    x[2].fma(t4.i, x[1].fma(t3.i, x[0].fma(t2.i, t1.i))),
                );
                let cb = Cx::new(
                    -signed_chain(&[(y[0], t7.i), (y[1], t6.i), (y[2], t5.i)]),
                    signed_chain(&[(y[0], t7.r), (y[1], t6.r), (y[2], t5.r)]),
                );
                let (da, db) = pm(ca, cb);
                if i == 0 {
                    ch[chi(0, k, u1)] = da;
                    ch[chi(0, k, u2)] = db;
                } else {
                    ch[chi(i, k, u1)] = smul(fwd, da, w(u1 - 1, i));
                    ch[chi(i, k, u2)] = smul(fwd, db, w(u2 - 1, i));
                }
            }
        }
    }
}

fn rotx45<T: Real>(fwd: bool, a: Cx<T>) -> Cx<T> {
    let h: T = lit(std::f64::consts::FRAC_1_SQRT_2);
    if fwd {
        Cx::new(h * (a.r + a.i), h * (a.i - a.r))
    } else {
        Cx::new(h * (a.r - a.i), h * (a.i + a.r))
    }
}
fn rotx135<T: Real>(fwd: bool, a: Cx<T>) -> Cx<T> {
    let h: T = lit(std::f64::consts::FRAC_1_SQRT_2);
    if fwd {
        Cx::new(h * (a.i - a.r), h * ((-a.r) - a.i))
    } else {
        Cx::new(h * ((-a.r) - a.i), h * (a.r - a.i))
    }
}

fn pass8<T: Real>(fwd: bool, ido: usize, l1: usize, cc: &[Cx<T>], ch: &mut [Cx<T>], wa: &[Cx<T>]) {
    let cc_ = |a: usize, b: usize, c: usize| cc[a + ido * (b + 8 * c)];
    let chi = |a: usize, b: usize, c: usize| a + ido * (b + l1 * c);
    let w = |x: usize, i: usize| wa[i - 1 + x * (ido - 1)];
    for k in 0..l1 {
        {
            let (a1, a5) = pm(cc_(0, 1, k), cc_(0, 5, k));
            let (a3, a7) = pm(cc_(0, 3, k), cc_(0, 7, k));
            let (a1, a3) = (a1.add(a3), a1.sub(a3)); // PMINPLACE
            let a3 = rotx90(fwd, a3);
            let a7 = rotx90(fwd, a7);
            let (a5, a7) = (a5.add(a7), a5.sub(a7));
            let a5 = rotx45(fwd, a5);
            let a7 = rotx135(fwd, a7);
            let (a0, a4) = pm(cc_(0, 0, k), cc_(0, 4, k));
            let (a2, a6) = pm(cc_(0, 2, k), cc_(0, 6, k));
            let (x, y) = pm(a0.add(a2), a1);
            ch[chi(0, k, 0)] = x;
            ch[chi(0, k, 4)] = y;
            let (x, y) = pm(a0.sub(a2), a3);
            ch[chi(0, k, 2)] = x;
            ch[chi(0, k, 6)] = y;
            let a6 = rotx90(fwd, a6);
            let (x, y) = pm(a4.add(a6), a5);
            ch[chi(0, k, 1)] = x;
            ch[chi(0, k, 5)] = y;
            let (x, y) = pm(a4.sub(a6), a7);
            ch[chi(0, k, 3)] = x;
            ch[chi(0, k, 7)] = y;
        }
        for i in 1..ido {
            let (a1, a5) = pm(cc_(i, 1, k), cc_(i, 5, k));
            let (a3, a7) = pm(cc_(i, 3, k), cc_(i, 7, k));
            let a7 = rotx90(fwd, a7);
            let (a1, a3) = (a1.add(a3), a1.sub(a3));
            let a3 = rotx90(fwd, a3);
            let (a5, a7) = (a5.add(a7), a5.sub(a7));
            let a5 = rotx45(fwd, a5);
            let a7 = rotx135(fwd, a7);
            let (a0, a4) = pm(cc_(i, 0, k), cc_(i, 4, k));
            let (a2, a6) = pm(cc_(i, 2, k), cc_(i, 6, k));
            let (a0, a2) = (a0.add(a2), a0.sub(a2));
            ch[chi(i, k, 0)] = a0.add(a1);
            ch[chi(i, k, 4)] = smul(fwd, a0.sub(a1), w(3, i));
            ch[chi(i, k, 2)] = smul(fwd, a2.add(a3), w(1, i));
            ch[chi(i, k, 6)] = smul(fwd, a2.sub(a3), w(5, i));
            let a6 = rotx90(fwd, a6);
            let (a4, a6) = (a4.add(a6), a4.sub(a6));
            ch[chi(i, k, 1)] = smul(fwd, a4.add(a5), w(0, i));
            ch[chi(i, k, 5)] = smul(fwd, a4.sub(a5), w(4, i));
            ch[chi(i, k, 3)] = smul(fwd, a6.add(a7), w(2, i));
            ch[chi(i, k, 7)] = smul(fwd, a6.sub(a7), w(6, i));
        }
    }
}

fn pass11<T: Real>(fwd: bool, ido: usize, l1: usize, cc: &[Cx<T>], ch: &mut [Cx<T>], wa: &[Cx<T>]) {
    let s = if fwd { -T::ONE } else { T::ONE };
    let r: [T; 5] = [
        lit(0.8412535328311811688618116489193677),
        lit(0.4154150130018864255292741492296232),
        lit(-0.1423148382732851404437926686163697),
        lit(-0.6548607339452850640569250724662936),
        lit(-0.9594929736144973898903680570663277),
    ];
    let im: [T; 5] = [
        s * lit(0.5406408174555975821076359543186917),
        s * lit(0.9096319953545183714117153830790285),
        s * lit(0.9898214418809327323760920377767188),
        s * lit(0.7557495743542582837740358439723444),
        s * lit(0.2817325568414296977114179153466169),
    ];
    let (tw1r, tw2r, tw3r, tw4r, tw5r) = (r[0], r[1], r[2], r[3], r[4]);
    let (tw1i, tw2i, tw3i, tw4i, tw5i) = (im[0], im[1], im[2], im[3], im[4]);
    let cc_ = |a: usize, b: usize, c: usize| cc[a + ido * (b + 11 * c)];
    let chi = |a: usize, b: usize, c: usize| a + ido * (b + l1 * c);
    let w = |x: usize, i: usize| wa[i - 1 + x * (ido - 1)];
    let steps = [
        (
            1,
            10,
            [tw1r, tw2r, tw3r, tw4r, tw5r],
            [tw1i, tw2i, tw3i, tw4i, tw5i],
        ),
        (
            2,
            9,
            [tw2r, tw4r, tw5r, tw3r, tw1r],
            [tw2i, tw4i, -tw5i, -tw3i, -tw1i],
        ),
        (
            3,
            8,
            [tw3r, tw5r, tw2r, tw1r, tw4r],
            [tw3i, -tw5i, -tw2i, tw1i, tw4i],
        ),
        (
            4,
            7,
            [tw4r, tw3r, tw1r, tw5r, tw2r],
            [tw4i, -tw3i, tw1i, tw5i, -tw2i],
        ),
        (
            5,
            6,
            [tw5r, tw1r, tw4r, tw2r, tw3r],
            [tw5i, -tw1i, tw4i, -tw2i, tw3i],
        ),
    ];
    for k in 0..l1 {
        for i in 0..ido {
            let t1 = cc_(i, 0, k);
            let (t2, t11) = pm(cc_(i, 1, k), cc_(i, 10, k));
            let (t3, t10) = pm(cc_(i, 2, k), cc_(i, 9, k));
            let (t4, t9) = pm(cc_(i, 3, k), cc_(i, 8, k));
            let (t5, t8) = pm(cc_(i, 4, k), cc_(i, 7, k));
            let (t6, t7) = pm(cc_(i, 5, k), cc_(i, 6, k));
            ch[chi(i, k, 0)] = Cx::new(
                t1.r + t2.r + t3.r + t4.r + t5.r + t6.r,
                t1.i + t2.i + t3.i + t4.i + t5.i + t6.i,
            );
            for (u1, u2, x, y) in steps {
                // complex operators: never fused
                let ca = t1
                    .add(t2.scale(x[0]))
                    .add(t3.scale(x[1]))
                    .add(t4.scale(x[2]))
                    .add(t5.scale(x[3]))
                    .add(t6.scale(x[4]));
                let cb = Cx::new(
                    -signed_chain(&[
                        (y[0], t11.i),
                        (y[1], t10.i),
                        (y[2], t9.i),
                        (y[3], t8.i),
                        (y[4], t7.i),
                    ]),
                    signed_chain(&[
                        (y[0], t11.r),
                        (y[1], t10.r),
                        (y[2], t9.r),
                        (y[3], t8.r),
                        (y[4], t7.r),
                    ]),
                );
                let (da, db) = pm(ca, cb);
                if i == 0 {
                    ch[chi(0, k, u1)] = da;
                    ch[chi(0, k, u2)] = db;
                } else {
                    ch[chi(i, k, u1)] = smul(fwd, da, w(u1 - 1, i));
                    ch[chi(i, k, u2)] = smul(fwd, db, w(u2 - 1, i));
                }
            }
        }
    }
}

/// `passg`: a generic odd prime factor; the result is left in `cc`.
#[allow(clippy::too_many_arguments)]
fn passg<T: Real>(
    fwd: bool,
    ido: usize,
    ip: usize,
    l1: usize,
    cc: &mut [Cx<T>],
    ch: &mut [Cx<T>],
    wa: &[Cx<T>],
    csarr: &[Cx<T>],
) {
    let cdim = ip;
    let ipph = ip.div_ceil(2);
    let idl1 = ido * l1;
    let chi = |a: usize, b: usize, c: usize| a + ido * (b + l1 * c);
    let cci = |a: usize, b: usize, c: usize| a + ido * (b + cdim * c);
    let cxi = |a: usize, b: usize, c: usize| a + ido * (b + l1 * c);
    let mut wal = vec![Cx::new(T::ONE, T::ZERO); ip];
    for i in 1..ip {
        wal[i] = Cx::new(csarr[i].r, if fwd { -csarr[i].i } else { csarr[i].i });
    }
    for k in 0..l1 {
        for i in 0..ido {
            ch[chi(i, k, 0)] = cc[cci(i, 0, k)];
        }
    }
    let (mut j, mut jc) = (1, ip - 1);
    while j < ipph {
        for k in 0..l1 {
            for i in 0..ido {
                let (a, b) = pm(cc[cci(i, j, k)], cc[cci(i, jc, k)]);
                ch[chi(i, k, j)] = a;
                ch[chi(i, k, jc)] = b;
            }
        }
        j += 1;
        jc -= 1;
    }
    for k in 0..l1 {
        for i in 0..ido {
            let mut tmp = ch[chi(i, k, 0)];
            for j in 1..ipph {
                tmp = tmp.add(ch[chi(i, k, j)]);
            }
            cc[cxi(i, k, 0)] = tmp;
        }
    }
    let (mut l, mut lc) = (1, ip - 1);
    while l < ipph {
        for ik in 0..idl1 {
            let c = |b: usize| ch[ik + idl1 * b];
            cc[ik + idl1 * l] = Cx::new(
                wal[2 * l].r.fma(c(2).r, wal[l].r.fma(c(1).r, c(0).r)),
                wal[2 * l].r.fma(c(2).i, wal[l].r.fma(c(1).i, c(0).i)),
            );
            cc[ik + idl1 * lc] = Cx::new(
                (-wal[l].i).fma(c(ip - 1).i, -(wal[2 * l].i * c(ip - 2).i)),
                wal[l].i.fma(c(ip - 1).r, wal[2 * l].i * c(ip - 2).r),
            );
        }
        let mut iwal = 2 * l;
        let (mut j, mut jc) = (3, ip - 3);
        while j + 1 < ipph {
            iwal += l;
            if iwal > ip {
                iwal -= ip;
            }
            let xwal = wal[iwal];
            iwal += l;
            if iwal > ip {
                iwal -= ip;
            }
            let xwal2 = wal[iwal];
            for ik in 0..idl1 {
                let c = |b: usize| ch[ik + idl1 * b];
                let (cj, cj1, cjc, cjc1) = (c(j), c(j + 1), c(jc), c(jc - 1));
                let x = &mut cc[ik + idl1 * l];
                x.r = x.r + cj.r.fma(xwal.r, cj1.r * xwal2.r);
                x.i = x.i + cj.i.fma(xwal.r, cj1.i * xwal2.r);
                let y = &mut cc[ik + idl1 * lc];
                y.r = y.r - cjc.i.fma(xwal.i, cjc1.i * xwal2.i);
                y.i = y.i + cjc.r.fma(xwal.i, cjc1.r * xwal2.i);
            }
            j += 2;
            jc -= 2;
        }
        while j < ipph {
            iwal += l;
            if iwal > ip {
                iwal -= ip;
            }
            let xwal = wal[iwal];
            for ik in 0..idl1 {
                let (cj, cjc) = (ch[ik + idl1 * j], ch[ik + idl1 * jc]);
                let x = &mut cc[ik + idl1 * l];
                x.r = cj.r.fma(xwal.r, x.r);
                x.i = cj.i.fma(xwal.r, x.i);
                let y = &mut cc[ik + idl1 * lc];
                y.r = (-cjc.i).fma(xwal.i, y.r);
                y.i = cjc.r.fma(xwal.i, y.i);
            }
            j += 1;
            jc -= 1;
        }
        l += 1;
        lc -= 1;
    }
    // shuffling and twiddling
    if ido == 1 {
        let (mut j, mut jc) = (1, ip - 1);
        while j < ipph {
            for ik in 0..idl1 {
                let (t1, t2) = (cc[ik + idl1 * j], cc[ik + idl1 * jc]);
                let (a, b) = pm(t1, t2);
                cc[ik + idl1 * j] = a;
                cc[ik + idl1 * jc] = b;
            }
            j += 1;
            jc -= 1;
        }
    } else {
        let (mut j, mut jc) = (1, ip - 1);
        while j < ipph {
            for k in 0..l1 {
                let (t1, t2) = (cc[cxi(0, k, j)], cc[cxi(0, k, jc)]);
                let (a, b) = pm(t1, t2);
                cc[cxi(0, k, j)] = a;
                cc[cxi(0, k, jc)] = b;
                for i in 1..ido {
                    let (x1, x2) = pm(cc[cxi(i, k, j)], cc[cxi(i, k, jc)]);
                    cc[cxi(i, k, j)] = smul(fwd, x1, wa[(j - 1) * (ido - 1) + i - 1]);
                    cc[cxi(i, k, jc)] = smul(fwd, x2, wa[(jc - 1) * (ido - 1) + i - 1]);
                }
            }
            j += 1;
            jc -= 1;
        }
    }
}

// ── real FFTPACK (rfftp) ──────────────────────────────────────────────────────

struct RFact<T> {
    fct: usize,
    tw: Vec<T>,
    tws: Vec<T>,
}

pub struct Rfftp<T> {
    length: usize,
    fact: Vec<RFact<T>>,
}

/// `MULPM(a, b, c, d, e, f)`: `a = c*e + d*f`, `b = c*f - d*e`, each fused.
#[inline(always)]
fn mulpm<T: Real>(c: T, d: T, e: T, f: T) -> (T, T) {
    (c.fma(e, d * f), c.fma(f, -(d * e)))
}

impl<T: Real> Rfftp<T> {
    pub fn new(length: usize, tw: Twiddles) -> Self {
        assert!(length > 0, "zero-length FFT requested");
        let mut p = Rfftp {
            length,
            fact: Vec::new(),
        };
        if length == 1 {
            return p;
        }
        let mut len = length;
        let mut f = Vec::new();
        while len.is_multiple_of(4) {
            f.push(4);
            len >>= 2;
        }
        if len.is_multiple_of(2) {
            len >>= 1;
            f.push(2);
            let last = f.len() - 1;
            f.swap(0, last);
        }
        let mut d = 3;
        while d * d <= len {
            while len.is_multiple_of(d) {
                f.push(d);
                len /= d;
            }
            d += 2;
        }
        if len > 1 {
            f.push(len);
        }
        let twid = SinCos2PiByN::new(length, tw);
        let mut l1 = 1;
        let nf = f.len();
        for (k, &ip) in f.iter().enumerate() {
            let ido = length / (l1 * ip);
            let mut tw = Vec::new();
            if k < nf - 1 {
                tw = vec![T::ZERO; (ip - 1) * (ido - 1)];
                for j in 1..ip {
                    for i in 1..=(ido - 1) / 2 {
                        let t: Cx<T> = twid.get(j * l1 * i);
                        tw[(j - 1) * (ido - 1) + 2 * i - 2] = t.r;
                        tw[(j - 1) * (ido - 1) + 2 * i - 1] = t.i;
                    }
                }
            }
            let mut tws = Vec::new();
            if ip > 5 {
                tws = vec![T::ZERO; 2 * ip];
                tws[0] = T::ONE;
                tws[1] = T::ZERO;
                let (mut i, mut ic) = (2, 2 * ip - 2);
                while i <= ic {
                    let t: Cx<T> = twid.get(i / 2 * (length / ip));
                    tws[i] = t.r;
                    tws[i + 1] = t.i;
                    tws[ic] = t.r;
                    tws[ic + 1] = -t.i;
                    i += 2;
                    // pocketfft decrements an unsigned `ic` past 0 on the last step; it is
                    // never read after, so wrap as C does instead of panicking in debug.
                    ic = ic.wrapping_sub(2);
                }
            }
            p.fact.push(RFact { fct: ip, tw, tws });
            l1 *= ip;
        }
        p
    }

    pub fn len(&self) -> usize {
        self.length
    }
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// `exec(c, fct, r2hc)`.
    pub fn exec(&self, c: &mut [T], fct: T, r2hc: bool) {
        let n = self.length;
        if n == 1 {
            c[0] = c[0] * fct;
            return;
        }
        let nf = self.fact.len();
        let mut ch = vec![T::ZERO; n];
        let mut in_c = true;
        if r2hc {
            let mut l1 = n;
            for k1 in 0..nf {
                let k = nf - k1 - 1;
                let f = &self.fact[k];
                let ip = f.fct;
                let ido = n / l1;
                l1 /= ip;
                let (src, dst): (&mut [T], &mut [T]) = if in_c {
                    (&mut *c, &mut ch[..])
                } else {
                    (&mut ch[..], &mut *c)
                };
                match ip {
                    4 => radf4(ido, l1, src, dst, &f.tw),
                    2 => radf2(ido, l1, src, dst, &f.tw),
                    3 => radf3(ido, l1, src, dst, &f.tw),
                    5 => radf5(ido, l1, src, dst, &f.tw),
                    _ => {
                        radfg(ido, ip, l1, src, dst, &f.tw, &f.tws);
                        in_c = !in_c;
                    }
                }
                in_c = !in_c;
            }
        } else {
            let mut l1 = 1;
            for f in &self.fact {
                let ip = f.fct;
                let ido = n / (ip * l1);
                let (src, dst): (&mut [T], &mut [T]) = if in_c {
                    (&mut *c, &mut ch[..])
                } else {
                    (&mut ch[..], &mut *c)
                };
                match ip {
                    4 => radb4(ido, l1, src, dst, &f.tw),
                    2 => radb2(ido, l1, src, dst, &f.tw),
                    3 => radb3(ido, l1, src, dst, &f.tw),
                    5 => radb5(ido, l1, src, dst, &f.tw),
                    _ => radbg(ido, ip, l1, src, dst, &f.tw, &f.tws),
                }
                in_c = !in_c;
                l1 *= ip;
            }
        }
        // copy_and_norm
        if !in_c {
            if fct != T::ONE {
                for i in 0..n {
                    c[i] = fct * ch[i];
                }
            } else {
                c.copy_from_slice(&ch);
            }
        } else if fct != T::ONE {
            for v in c.iter_mut() {
                *v = *v * fct;
            }
        }
    }
}

fn radf2<T: Real>(ido: usize, l1: usize, cc: &[T], ch: &mut [T], wa: &[T]) {
    let w = |x: usize, i: usize| wa[i + x * (ido - 1)];
    let cc_ = |a: usize, b: usize, c: usize| cc[a + ido * (b + l1 * c)];
    let chi = |a: usize, b: usize, c: usize| a + ido * (b + 2 * c);
    for k in 0..l1 {
        ch[chi(0, 0, k)] = cc_(0, k, 0) + cc_(0, k, 1);
        ch[chi(ido - 1, 1, k)] = cc_(0, k, 0) - cc_(0, k, 1);
    }
    if ido & 1 == 0 {
        for k in 0..l1 {
            ch[chi(0, 1, k)] = -cc_(ido - 1, k, 1);
            ch[chi(ido - 1, 0, k)] = cc_(ido - 1, k, 0);
        }
    }
    if ido <= 2 {
        return;
    }
    for k in 0..l1 {
        let mut i = 2;
        while i < ido {
            let ic = ido - i;
            let (tr2, ti2) = mulpm(w(0, i - 2), w(0, i - 1), cc_(i - 1, k, 1), cc_(i, k, 1));
            ch[chi(i - 1, 0, k)] = cc_(i - 1, k, 0) + tr2;
            ch[chi(ic - 1, 1, k)] = cc_(i - 1, k, 0) - tr2;
            ch[chi(i, 0, k)] = ti2 + cc_(i, k, 0);
            ch[chi(ic, 1, k)] = ti2 - cc_(i, k, 0);
            i += 2;
        }
    }
}

fn radf3<T: Real>(ido: usize, l1: usize, cc: &[T], ch: &mut [T], wa: &[T]) {
    let taur: T = lit(-0.5);
    let taui: T = lit(0.8660254037844386467637231707529362);
    let w = |x: usize, i: usize| wa[i + x * (ido - 1)];
    let cc_ = |a: usize, b: usize, c: usize| cc[a + ido * (b + l1 * c)];
    let chi = |a: usize, b: usize, c: usize| a + ido * (b + 3 * c);
    for k in 0..l1 {
        let cr2 = cc_(0, k, 1) + cc_(0, k, 2);
        ch[chi(0, 0, k)] = cc_(0, k, 0) + cr2;
        ch[chi(0, 2, k)] = taui * (cc_(0, k, 2) - cc_(0, k, 1));
        ch[chi(ido - 1, 1, k)] = taur.fma(cr2, cc_(0, k, 0));
    }
    if ido == 1 {
        return;
    }
    for k in 0..l1 {
        let mut i = 2;
        while i < ido {
            let ic = ido - i;
            let (dr2, di2) = mulpm(w(0, i - 2), w(0, i - 1), cc_(i - 1, k, 1), cc_(i, k, 1));
            let (dr3, di3) = mulpm(w(1, i - 2), w(1, i - 1), cc_(i - 1, k, 2), cc_(i, k, 2));
            // REARRANGE(dr2, di2, dr3, di3)
            let (t1, t2, t3, t4) = (dr2 + dr3, dr3 - dr2, di2 + di3, di2 - di3);
            let (dr2, di2, dr3, di3) = (t1, t3, t4, t2);
            ch[chi(i - 1, 0, k)] = cc_(i - 1, k, 0) + dr2;
            ch[chi(i, 0, k)] = cc_(i, k, 0) + di2;
            let tr2 = taur.fma(dr2, cc_(i - 1, k, 0));
            let ti2 = taur.fma(di2, cc_(i, k, 0));
            let tr3 = taui * dr3;
            let ti3 = taui * di3;
            ch[chi(i - 1, 2, k)] = tr2 + tr3;
            ch[chi(ic - 1, 1, k)] = tr2 - tr3;
            ch[chi(i, 2, k)] = ti3 + ti2;
            ch[chi(ic, 1, k)] = ti3 - ti2;
            i += 2;
        }
    }
}

fn radf4<T: Real>(ido: usize, l1: usize, cc: &[T], ch: &mut [T], wa: &[T]) {
    let hsqt2: T = lit(std::f64::consts::FRAC_1_SQRT_2);
    let w = |x: usize, i: usize| wa[i + x * (ido - 1)];
    let cc_ = |a: usize, b: usize, c: usize| cc[a + ido * (b + l1 * c)];
    let chi = |a: usize, b: usize, c: usize| a + ido * (b + 4 * c);
    for k in 0..l1 {
        let tr1 = cc_(0, k, 3) + cc_(0, k, 1);
        ch[chi(0, 2, k)] = cc_(0, k, 3) - cc_(0, k, 1);
        let tr2 = cc_(0, k, 0) + cc_(0, k, 2);
        ch[chi(ido - 1, 1, k)] = cc_(0, k, 0) - cc_(0, k, 2);
        ch[chi(0, 0, k)] = tr2 + tr1;
        ch[chi(ido - 1, 3, k)] = tr2 - tr1;
    }
    if ido & 1 == 0 {
        for k in 0..l1 {
            let ti1 = (-hsqt2) * (cc_(ido - 1, k, 1) + cc_(ido - 1, k, 3));
            let tr1 = hsqt2 * (cc_(ido - 1, k, 1) - cc_(ido - 1, k, 3));
            ch[chi(ido - 1, 0, k)] = cc_(ido - 1, k, 0) + tr1;
            ch[chi(ido - 1, 2, k)] = cc_(ido - 1, k, 0) - tr1;
            ch[chi(0, 3, k)] = ti1 + cc_(ido - 1, k, 2);
            ch[chi(0, 1, k)] = ti1 - cc_(ido - 1, k, 2);
        }
    }
    if ido <= 2 {
        return;
    }
    for k in 0..l1 {
        let mut i = 2;
        while i < ido {
            let ic = ido - i;
            let (cr2, ci2) = mulpm(w(0, i - 2), w(0, i - 1), cc_(i - 1, k, 1), cc_(i, k, 1));
            let (cr3, ci3) = mulpm(w(1, i - 2), w(1, i - 1), cc_(i - 1, k, 2), cc_(i, k, 2));
            let (cr4, ci4) = mulpm(w(2, i - 2), w(2, i - 1), cc_(i - 1, k, 3), cc_(i, k, 3));
            let (tr1, tr4) = (cr4 + cr2, cr4 - cr2);
            let (ti1, ti4) = (ci2 + ci4, ci2 - ci4);
            let (tr2, tr3) = (cc_(i - 1, k, 0) + cr3, cc_(i - 1, k, 0) - cr3);
            let (ti2, ti3) = (cc_(i, k, 0) + ci3, cc_(i, k, 0) - ci3);
            ch[chi(i - 1, 0, k)] = tr2 + tr1;
            ch[chi(ic - 1, 3, k)] = tr2 - tr1;
            ch[chi(i, 0, k)] = ti1 + ti2;
            ch[chi(ic, 3, k)] = ti1 - ti2;
            ch[chi(i - 1, 2, k)] = tr3 + ti4;
            ch[chi(ic - 1, 1, k)] = tr3 - ti4;
            ch[chi(i, 2, k)] = tr4 + ti3;
            ch[chi(ic, 1, k)] = tr4 - ti3;
            i += 2;
        }
    }
}

fn radf5<T: Real>(ido: usize, l1: usize, cc: &[T], ch: &mut [T], wa: &[T]) {
    let tr11: T = lit(0.3090169943749474241022934171828191);
    let ti11: T = lit(0.9510565162951535721164393333793821);
    let tr12: T = lit(-0.8090169943749474241022934171828191);
    let ti12: T = lit(0.5877852522924731291687059546390728);
    let w = |x: usize, i: usize| wa[i + x * (ido - 1)];
    let cc_ = |a: usize, b: usize, c: usize| cc[a + ido * (b + l1 * c)];
    let chi = |a: usize, b: usize, c: usize| a + ido * (b + 5 * c);
    for k in 0..l1 {
        let (cr2, ci5) = (cc_(0, k, 4) + cc_(0, k, 1), cc_(0, k, 4) - cc_(0, k, 1));
        let (cr3, ci4) = (cc_(0, k, 3) + cc_(0, k, 2), cc_(0, k, 3) - cc_(0, k, 2));
        let c0 = cc_(0, k, 0);
        ch[chi(0, 0, k)] = c0 + cr2 + cr3;
        ch[chi(ido - 1, 1, k)] = tr12.fma(cr3, tr11.fma(cr2, c0));
        ch[chi(0, 2, k)] = ti11.fma(ci5, ti12 * ci4);
        ch[chi(ido - 1, 3, k)] = tr11.fma(cr3, tr12.fma(cr2, c0));
        ch[chi(0, 4, k)] = ti12.fma(ci5, -(ti11 * ci4));
    }
    if ido == 1 {
        return;
    }
    for k in 0..l1 {
        let (mut i, mut ic) = (2, ido - 2);
        while i < ido {
            let (dr2, di2) = mulpm(w(0, i - 2), w(0, i - 1), cc_(i - 1, k, 1), cc_(i, k, 1));
            let (dr3, di3) = mulpm(w(1, i - 2), w(1, i - 1), cc_(i - 1, k, 2), cc_(i, k, 2));
            let (dr4, di4) = mulpm(w(2, i - 2), w(2, i - 1), cc_(i - 1, k, 3), cc_(i, k, 3));
            let (dr5, di5) = mulpm(w(3, i - 2), w(3, i - 1), cc_(i - 1, k, 4), cc_(i, k, 4));
            let (dr2, di2, dr5, di5) = (dr2 + dr5, di2 + di5, di2 - di5, dr5 - dr2);
            let (dr3, di3, dr4, di4) = (dr3 + dr4, di3 + di4, di3 - di4, dr4 - dr3);
            let (c0, c1) = (cc_(i - 1, k, 0), cc_(i, k, 0));
            ch[chi(i - 1, 0, k)] = c0 + dr2 + dr3;
            ch[chi(i, 0, k)] = c1 + di2 + di3;
            let tr2 = tr12.fma(dr3, tr11.fma(dr2, c0));
            let ti2 = tr12.fma(di3, tr11.fma(di2, c1));
            let tr3 = tr11.fma(dr3, tr12.fma(dr2, c0));
            let ti3 = tr11.fma(di3, tr12.fma(di2, c1));
            let tr5 = ti11.fma(dr5, ti12 * dr4);
            let ti5 = ti11.fma(di5, ti12 * di4);
            let tr4 = ti12.fma(dr5, -(ti11 * dr4));
            let ti4 = ti12.fma(di5, -(ti11 * di4));
            ch[chi(i - 1, 2, k)] = tr2 + tr5;
            ch[chi(ic - 1, 1, k)] = tr2 - tr5;
            ch[chi(i, 2, k)] = ti5 + ti2;
            ch[chi(ic, 1, k)] = ti5 - ti2;
            ch[chi(i - 1, 4, k)] = tr3 + tr4;
            ch[chi(ic - 1, 3, k)] = tr3 - tr4;
            ch[chi(i, 4, k)] = ti4 + ti3;
            ch[chi(ic, 3, k)] = ti4 - ti3;
            i += 2;
            // pocketfft decrements an unsigned `ic` past 0 on the last step; it is
            // never read after, so wrap as C does instead of panicking in debug.
            ic = ic.wrapping_sub(2);
        }
    }
}

/// `radfg`: a generic odd factor; the result is left in `cc`.
#[allow(clippy::too_many_arguments)]
fn radfg<T: Real>(
    ido: usize,
    ip: usize,
    l1: usize,
    cc: &mut [T],
    ch: &mut [T],
    wa: &[T],
    csarr: &[T],
) {
    let cdim = ip;
    let ipph = ip.div_ceil(2);
    let idl1 = ido * l1;
    let cci = |a: usize, b: usize, c: usize| a + ido * (b + cdim * c);
    let chi = |a: usize, b: usize, c: usize| a + ido * (b + l1 * c);
    let c1i = |a: usize, b: usize, c: usize| a + ido * (b + l1 * c);
    if ido > 1 {
        let (mut j, mut jc) = (1, ip - 1);
        while j < ipph {
            let (is, is2) = ((j - 1) * (ido - 1), (jc - 1) * (ido - 1));
            for k in 0..l1 {
                let (mut idij, mut idij2) = (is, is2);
                let mut i = 1;
                while i + 1 < ido {
                    let (t1, t2, t3, t4) = (
                        cc[c1i(i, k, j)],
                        cc[c1i(i + 1, k, j)],
                        cc[c1i(i, k, jc)],
                        cc[c1i(i + 1, k, jc)],
                    );
                    let x1 = wa[idij].fma(t1, wa[idij + 1] * t2);
                    let x2 = wa[idij].fma(t2, -(wa[idij + 1] * t1));
                    let x3 = wa[idij2].fma(t3, wa[idij2 + 1] * t4);
                    let x4 = wa[idij2].fma(t4, -(wa[idij2 + 1] * t3));
                    cc[c1i(i, k, j)] = x3 + x1;
                    cc[c1i(i + 1, k, jc)] = x3 - x1;
                    cc[c1i(i + 1, k, j)] = x2 + x4;
                    cc[c1i(i, k, jc)] = x2 - x4;
                    idij += 2;
                    idij2 += 2;
                    i += 2;
                }
            }
            j += 1;
            jc -= 1;
        }
    }
    let (mut j, mut jc) = (1, ip - 1);
    while j < ipph {
        for k in 0..l1 {
            // MPINPLACE(C1(0,k,jc), C1(0,k,j))
            let (a, b) = (cc[c1i(0, k, jc)], cc[c1i(0, k, j)]);
            cc[c1i(0, k, jc)] = a - b;
            cc[c1i(0, k, j)] = a + b;
        }
        j += 1;
        jc -= 1;
    }
    let (mut l, mut lc) = (1, ip - 1);
    while l < ipph {
        for ik in 0..idl1 {
            let c2 = |b: usize| cc[ik + idl1 * b];
            ch[ik + idl1 * l] = csarr[4 * l].fma(c2(2), csarr[2 * l].fma(c2(1), c2(0)));
            ch[ik + idl1 * lc] = csarr[2 * l + 1].fma(c2(ip - 1), csarr[4 * l + 1] * c2(ip - 2));
        }
        let mut iang = 2 * l;
        let (mut j, mut jc) = (3, ip - 3);
        while j + 3 < ipph {
            let mut ar = [T::ZERO; 4];
            let mut ai = [T::ZERO; 4];
            for q in 0..4 {
                iang += l;
                if iang >= ip {
                    iang -= ip;
                }
                ar[q] = csarr[2 * iang];
                ai[q] = csarr[2 * iang + 1];
            }
            for ik in 0..idl1 {
                let c2 = |b: usize| cc[ik + idl1 * b];
                let s = ar[3].fma(
                    c2(j + 3),
                    ar[2].fma(c2(j + 2), ar[0].fma(c2(j), ar[1] * c2(j + 1))),
                );
                ch[ik + idl1 * l] = ch[ik + idl1 * l] + s;
                let s = ai[3].fma(
                    c2(jc - 3),
                    ai[2].fma(c2(jc - 2), ai[0].fma(c2(jc), ai[1] * c2(jc - 1))),
                );
                ch[ik + idl1 * lc] = ch[ik + idl1 * lc] + s;
            }
            j += 4;
            jc -= 4;
        }
        while j + 1 < ipph {
            let mut ar = [T::ZERO; 2];
            let mut ai = [T::ZERO; 2];
            for q in 0..2 {
                iang += l;
                if iang >= ip {
                    iang -= ip;
                }
                ar[q] = csarr[2 * iang];
                ai[q] = csarr[2 * iang + 1];
            }
            for ik in 0..idl1 {
                let c2 = |b: usize| cc[ik + idl1 * b];
                let s = ar[0].fma(c2(j), ar[1] * c2(j + 1));
                ch[ik + idl1 * l] = ch[ik + idl1 * l] + s;
                let s = ai[0].fma(c2(jc), ai[1] * c2(jc - 1));
                ch[ik + idl1 * lc] = ch[ik + idl1 * lc] + s;
            }
            j += 2;
            jc -= 2;
        }
        while j < ipph {
            iang += l;
            if iang >= ip {
                iang -= ip;
            }
            let (ar, ai) = (csarr[2 * iang], csarr[2 * iang + 1]);
            for ik in 0..idl1 {
                let (cj, cjc) = (cc[ik + idl1 * j], cc[ik + idl1 * jc]);
                ch[ik + idl1 * l] = ar.fma(cj, ch[ik + idl1 * l]);
                ch[ik + idl1 * lc] = ai.fma(cjc, ch[ik + idl1 * lc]);
            }
            j += 1;
            jc -= 1;
        }
        l += 1;
        lc -= 1;
    }
    ch[..idl1].copy_from_slice(&cc[..idl1]);
    for j in 1..ipph {
        for ik in 0..idl1 {
            ch[ik] = ch[ik] + cc[ik + idl1 * j];
        }
    }
    for k in 0..l1 {
        for i in 0..ido {
            cc[cci(i, 0, k)] = ch[chi(i, k, 0)];
        }
    }
    let (mut j, mut jc) = (1, ip - 1);
    while j < ipph {
        let j2 = 2 * j - 1;
        for k in 0..l1 {
            cc[cci(ido - 1, j2, k)] = ch[chi(0, k, j)];
            cc[cci(0, j2 + 1, k)] = ch[chi(0, k, jc)];
        }
        j += 1;
        jc -= 1;
    }
    if ido == 1 {
        return;
    }
    let (mut j, mut jc) = (1, ip - 1);
    while j < ipph {
        let j2 = 2 * j - 1;
        for k in 0..l1 {
            let (mut i, mut ic) = (1, ido - 3);
            while i + 1 < ido {
                cc[cci(i, j2 + 1, k)] = ch[chi(i, k, j)] + ch[chi(i, k, jc)];
                cc[cci(ic, j2, k)] = ch[chi(i, k, j)] - ch[chi(i, k, jc)];
                cc[cci(i + 1, j2 + 1, k)] = ch[chi(i + 1, k, j)] + ch[chi(i + 1, k, jc)];
                cc[cci(ic + 1, j2, k)] = ch[chi(i + 1, k, jc)] - ch[chi(i + 1, k, j)];
                i += 2;
                ic = ic.wrapping_sub(2);
            }
        }
        j += 1;
        jc -= 1;
    }
}

fn radb2<T: Real>(ido: usize, l1: usize, cc: &[T], ch: &mut [T], wa: &[T]) {
    let w = |x: usize, i: usize| wa[i + x * (ido - 1)];
    let cc_ = |a: usize, b: usize, c: usize| cc[a + ido * (b + 2 * c)];
    let chi = |a: usize, b: usize, c: usize| a + ido * (b + l1 * c);
    let two: T = lit(2.0);
    for k in 0..l1 {
        ch[chi(0, k, 0)] = cc_(0, 0, k) + cc_(ido - 1, 1, k);
        ch[chi(0, k, 1)] = cc_(0, 0, k) - cc_(ido - 1, 1, k);
    }
    if ido & 1 == 0 {
        for k in 0..l1 {
            ch[chi(ido - 1, k, 0)] = two * cc_(ido - 1, 0, k);
            ch[chi(ido - 1, k, 1)] = (-two) * cc_(0, 1, k);
        }
    }
    if ido <= 2 {
        return;
    }
    for k in 0..l1 {
        let mut i = 2;
        while i < ido {
            let ic = ido - i;
            ch[chi(i - 1, k, 0)] = cc_(i - 1, 0, k) + cc_(ic - 1, 1, k);
            let tr2 = cc_(i - 1, 0, k) - cc_(ic - 1, 1, k);
            let ti2 = cc_(i, 0, k) + cc_(ic, 1, k);
            ch[chi(i, k, 0)] = cc_(i, 0, k) - cc_(ic, 1, k);
            let (a, b) = mulpm(w(0, i - 2), w(0, i - 1), ti2, tr2);
            ch[chi(i, k, 1)] = a;
            ch[chi(i - 1, k, 1)] = b;
            i += 2;
        }
    }
}

fn radb3<T: Real>(ido: usize, l1: usize, cc: &[T], ch: &mut [T], wa: &[T]) {
    let taur: T = lit(-0.5);
    let taui: T = lit(0.8660254037844386467637231707529362);
    let two: T = lit(2.0);
    let w = |x: usize, i: usize| wa[i + x * (ido - 1)];
    let cc_ = |a: usize, b: usize, c: usize| cc[a + ido * (b + 3 * c)];
    let chi = |a: usize, b: usize, c: usize| a + ido * (b + l1 * c);
    for k in 0..l1 {
        let tr2 = two * cc_(ido - 1, 1, k);
        let cr2 = taur.fma(tr2, cc_(0, 0, k));
        ch[chi(0, k, 0)] = cc_(0, 0, k) + tr2;
        let ci3 = (two * taui) * cc_(0, 2, k); // 2*taui folded at compile time
        ch[chi(0, k, 2)] = cr2 + ci3;
        ch[chi(0, k, 1)] = cr2 - ci3;
    }
    if ido == 1 {
        return;
    }
    for k in 0..l1 {
        let (mut i, mut ic) = (2, ido - 2);
        while i < ido {
            let tr2 = cc_(i - 1, 2, k) + cc_(ic - 1, 1, k);
            let ti2 = cc_(i, 2, k) - cc_(ic, 1, k);
            let cr2 = taur.fma(tr2, cc_(i - 1, 0, k));
            let ci2 = taur.fma(ti2, cc_(i, 0, k));
            ch[chi(i - 1, k, 0)] = cc_(i - 1, 0, k) + tr2;
            ch[chi(i, k, 0)] = cc_(i, 0, k) + ti2;
            let cr3 = taui * (cc_(i - 1, 2, k) - cc_(ic - 1, 1, k));
            let ci3 = taui * (cc_(i, 2, k) + cc_(ic, 1, k));
            let (dr3, dr2) = (cr2 + ci3, cr2 - ci3);
            let (di2, di3) = (ci2 + cr3, ci2 - cr3);
            let (a, b) = mulpm(w(0, i - 2), w(0, i - 1), di2, dr2);
            ch[chi(i, k, 1)] = a;
            ch[chi(i - 1, k, 1)] = b;
            let (a, b) = mulpm(w(1, i - 2), w(1, i - 1), di3, dr3);
            ch[chi(i, k, 2)] = a;
            ch[chi(i - 1, k, 2)] = b;
            i += 2;
            // pocketfft decrements an unsigned `ic` past 0 on the last step; it is
            // never read after, so wrap as C does instead of panicking in debug.
            ic = ic.wrapping_sub(2);
        }
    }
}

fn radb4<T: Real>(ido: usize, l1: usize, cc: &[T], ch: &mut [T], wa: &[T]) {
    let sqrt2: T = lit(std::f64::consts::SQRT_2);
    let two: T = lit(2.0);
    let w = |x: usize, i: usize| wa[i + x * (ido - 1)];
    let cc_ = |a: usize, b: usize, c: usize| cc[a + ido * (b + 4 * c)];
    let chi = |a: usize, b: usize, c: usize| a + ido * (b + l1 * c);
    for k in 0..l1 {
        let (tr2, tr1) = (
            cc_(0, 0, k) + cc_(ido - 1, 3, k),
            cc_(0, 0, k) - cc_(ido - 1, 3, k),
        );
        let tr3 = two * cc_(ido - 1, 1, k);
        let tr4 = two * cc_(0, 2, k);
        ch[chi(0, k, 0)] = tr2 + tr3;
        ch[chi(0, k, 2)] = tr2 - tr3;
        ch[chi(0, k, 3)] = tr1 + tr4;
        ch[chi(0, k, 1)] = tr1 - tr4;
    }
    if ido & 1 == 0 {
        for k in 0..l1 {
            let (ti1, ti2) = (cc_(0, 3, k) + cc_(0, 1, k), cc_(0, 3, k) - cc_(0, 1, k));
            let (tr2, tr1) = (
                cc_(ido - 1, 0, k) + cc_(ido - 1, 2, k),
                cc_(ido - 1, 0, k) - cc_(ido - 1, 2, k),
            );
            ch[chi(ido - 1, k, 0)] = tr2 + tr2;
            ch[chi(ido - 1, k, 1)] = sqrt2 * (tr1 - ti1);
            ch[chi(ido - 1, k, 2)] = ti2 + ti2;
            ch[chi(ido - 1, k, 3)] = (-sqrt2) * (tr1 + ti1);
        }
    }
    if ido <= 2 {
        return;
    }
    for k in 0..l1 {
        let mut i = 2;
        while i < ido {
            let ic = ido - i;
            let (tr2, tr1) = (
                cc_(i - 1, 0, k) + cc_(ic - 1, 3, k),
                cc_(i - 1, 0, k) - cc_(ic - 1, 3, k),
            );
            let (ti1, ti2) = (cc_(i, 0, k) + cc_(ic, 3, k), cc_(i, 0, k) - cc_(ic, 3, k));
            let (tr4, ti3) = (cc_(i, 2, k) + cc_(ic, 1, k), cc_(i, 2, k) - cc_(ic, 1, k));
            let (tr3, ti4) = (
                cc_(i - 1, 2, k) + cc_(ic - 1, 1, k),
                cc_(i - 1, 2, k) - cc_(ic - 1, 1, k),
            );
            ch[chi(i - 1, k, 0)] = tr2 + tr3;
            let cr3 = tr2 - tr3;
            ch[chi(i, k, 0)] = ti2 + ti3;
            let ci3 = ti2 - ti3;
            let (cr4, cr2) = (tr1 + tr4, tr1 - tr4);
            let (ci2, ci4) = (ti1 + ti4, ti1 - ti4);
            let (a, b) = mulpm(w(0, i - 2), w(0, i - 1), ci2, cr2);
            ch[chi(i, k, 1)] = a;
            ch[chi(i - 1, k, 1)] = b;
            let (a, b) = mulpm(w(1, i - 2), w(1, i - 1), ci3, cr3);
            ch[chi(i, k, 2)] = a;
            ch[chi(i - 1, k, 2)] = b;
            let (a, b) = mulpm(w(2, i - 2), w(2, i - 1), ci4, cr4);
            ch[chi(i, k, 3)] = a;
            ch[chi(i - 1, k, 3)] = b;
            i += 2;
        }
    }
}

fn radb5<T: Real>(ido: usize, l1: usize, cc: &[T], ch: &mut [T], wa: &[T]) {
    let tr11: T = lit(0.3090169943749474241022934171828191);
    let ti11: T = lit(0.9510565162951535721164393333793821);
    let tr12: T = lit(-0.8090169943749474241022934171828191);
    let ti12: T = lit(0.5877852522924731291687059546390728);
    let w = |x: usize, i: usize| wa[i + x * (ido - 1)];
    let cc_ = |a: usize, b: usize, c: usize| cc[a + ido * (b + 5 * c)];
    let chi = |a: usize, b: usize, c: usize| a + ido * (b + l1 * c);
    for k in 0..l1 {
        let ti5 = cc_(0, 2, k) + cc_(0, 2, k);
        let ti4 = cc_(0, 4, k) + cc_(0, 4, k);
        let tr2 = cc_(ido - 1, 1, k) + cc_(ido - 1, 1, k);
        let tr3 = cc_(ido - 1, 3, k) + cc_(ido - 1, 3, k);
        let c0 = cc_(0, 0, k);
        ch[chi(0, k, 0)] = c0 + tr2 + tr3;
        let cr2 = tr12.fma(tr3, tr11.fma(tr2, c0));
        let cr3 = tr11.fma(tr3, tr12.fma(tr2, c0));
        let (ci5, ci4) = mulpm(ti5, ti4, ti11, ti12);
        ch[chi(0, k, 4)] = cr2 + ci5;
        ch[chi(0, k, 1)] = cr2 - ci5;
        ch[chi(0, k, 3)] = cr3 + ci4;
        ch[chi(0, k, 2)] = cr3 - ci4;
    }
    if ido == 1 {
        return;
    }
    for k in 0..l1 {
        let (mut i, mut ic) = (2, ido - 2);
        while i < ido {
            let (tr2, tr5) = (
                cc_(i - 1, 2, k) + cc_(ic - 1, 1, k),
                cc_(i - 1, 2, k) - cc_(ic - 1, 1, k),
            );
            let (ti5, ti2) = (cc_(i, 2, k) + cc_(ic, 1, k), cc_(i, 2, k) - cc_(ic, 1, k));
            let (tr3, tr4) = (
                cc_(i - 1, 4, k) + cc_(ic - 1, 3, k),
                cc_(i - 1, 4, k) - cc_(ic - 1, 3, k),
            );
            let (ti4, ti3) = (cc_(i, 4, k) + cc_(ic, 3, k), cc_(i, 4, k) - cc_(ic, 3, k));
            let (c0, c1) = (cc_(i - 1, 0, k), cc_(i, 0, k));
            ch[chi(i - 1, k, 0)] = c0 + tr2 + tr3;
            ch[chi(i, k, 0)] = c1 + ti2 + ti3;
            let cr2 = tr12.fma(tr3, tr11.fma(tr2, c0));
            let ci2 = tr12.fma(ti3, tr11.fma(ti2, c1));
            let cr3 = tr11.fma(tr3, tr12.fma(tr2, c0));
            let ci3 = tr11.fma(ti3, tr12.fma(ti2, c1));
            let (cr5, cr4) = mulpm(tr5, tr4, ti11, ti12);
            let (ci5, ci4) = mulpm(ti5, ti4, ti11, ti12);
            let (dr4, dr3) = (cr3 + ci4, cr3 - ci4);
            let (di3, di4) = (ci3 + cr4, ci3 - cr4);
            let (dr5, dr2) = (cr2 + ci5, cr2 - ci5);
            let (di2, di5) = (ci2 + cr5, ci2 - cr5);
            for (q, di, dr) in [(1, di2, dr2), (2, di3, dr3), (3, di4, dr4), (4, di5, dr5)] {
                let (a, b) = mulpm(w(q - 1, i - 2), w(q - 1, i - 1), di, dr);
                ch[chi(i, k, q)] = a;
                ch[chi(i - 1, k, q)] = b;
            }
            i += 2;
            // pocketfft decrements an unsigned `ic` past 0 on the last step; it is
            // never read after, so wrap as C does instead of panicking in debug.
            ic = ic.wrapping_sub(2);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn radbg<T: Real>(
    ido: usize,
    ip: usize,
    l1: usize,
    cc: &mut [T],
    ch: &mut [T],
    wa: &[T],
    csarr: &[T],
) {
    let cdim = ip;
    let ipph = ip.div_ceil(2);
    let idl1 = ido * l1;
    let cci = |a: usize, b: usize, c: usize| a + ido * (b + cdim * c);
    let chi = |a: usize, b: usize, c: usize| a + ido * (b + l1 * c);
    let c1i = |a: usize, b: usize, c: usize| a + ido * (b + l1 * c);
    let two: T = lit(2.0);
    for k in 0..l1 {
        for i in 0..ido {
            ch[chi(i, k, 0)] = cc[cci(i, 0, k)];
        }
    }
    let (mut j, mut jc) = (1, ip - 1);
    while j < ipph {
        let j2 = 2 * j - 1;
        for k in 0..l1 {
            ch[chi(0, k, j)] = two * cc[cci(ido - 1, j2, k)];
            ch[chi(0, k, jc)] = two * cc[cci(0, j2 + 1, k)];
        }
        j += 1;
        jc -= 1;
    }
    if ido != 1 {
        let (mut j, mut jc) = (1, ip - 1);
        while j < ipph {
            let j2 = 2 * j - 1;
            for k in 0..l1 {
                let (mut i, mut ic) = (1, ido - 3);
                while i + 1 < ido {
                    ch[chi(i, k, j)] = cc[cci(i, j2 + 1, k)] + cc[cci(ic, j2, k)];
                    ch[chi(i, k, jc)] = cc[cci(i, j2 + 1, k)] - cc[cci(ic, j2, k)];
                    ch[chi(i + 1, k, j)] = cc[cci(i + 1, j2 + 1, k)] - cc[cci(ic + 1, j2, k)];
                    ch[chi(i + 1, k, jc)] = cc[cci(i + 1, j2 + 1, k)] + cc[cci(ic + 1, j2, k)];
                    i += 2;
                    ic = ic.wrapping_sub(2);
                }
            }
            j += 1;
            jc -= 1;
        }
    }
    let (mut l, mut lc) = (1, ip - 1);
    while l < ipph {
        for ik in 0..idl1 {
            let h = |b: usize| ch[ik + idl1 * b];
            cc[ik + idl1 * l] = csarr[4 * l].fma(h(2), csarr[2 * l].fma(h(1), h(0)));
            cc[ik + idl1 * lc] = csarr[2 * l + 1].fma(h(ip - 1), csarr[4 * l + 1] * h(ip - 2));
        }
        let mut iang = 2 * l;
        let (mut j, mut jc) = (3, ip - 3);
        while j + 3 < ipph {
            let mut ar = [T::ZERO; 4];
            let mut ai = [T::ZERO; 4];
            for q in 0..4 {
                iang += l;
                if iang > ip {
                    iang -= ip;
                }
                ar[q] = csarr[2 * iang];
                ai[q] = csarr[2 * iang + 1];
            }
            for ik in 0..idl1 {
                let h = |b: usize| ch[ik + idl1 * b];
                let s = ar[3].fma(
                    h(j + 3),
                    ar[2].fma(h(j + 2), ar[0].fma(h(j), ar[1] * h(j + 1))),
                );
                cc[ik + idl1 * l] = cc[ik + idl1 * l] + s;
                let s = ai[3].fma(
                    h(jc - 3),
                    ai[2].fma(h(jc - 2), ai[0].fma(h(jc), ai[1] * h(jc - 1))),
                );
                cc[ik + idl1 * lc] = cc[ik + idl1 * lc] + s;
            }
            j += 4;
            jc -= 4;
        }
        while j + 1 < ipph {
            let mut ar = [T::ZERO; 2];
            let mut ai = [T::ZERO; 2];
            for q in 0..2 {
                iang += l;
                if iang > ip {
                    iang -= ip;
                }
                ar[q] = csarr[2 * iang];
                ai[q] = csarr[2 * iang + 1];
            }
            for ik in 0..idl1 {
                let h = |b: usize| ch[ik + idl1 * b];
                let s = ar[0].fma(h(j), ar[1] * h(j + 1));
                cc[ik + idl1 * l] = cc[ik + idl1 * l] + s;
                let s = ai[0].fma(h(jc), ai[1] * h(jc - 1));
                cc[ik + idl1 * lc] = cc[ik + idl1 * lc] + s;
            }
            j += 2;
            jc -= 2;
        }
        while j < ipph {
            iang += l;
            if iang > ip {
                iang -= ip;
            }
            let (war, wai) = (csarr[2 * iang], csarr[2 * iang + 1]);
            for ik in 0..idl1 {
                let (hj, hjc) = (ch[ik + idl1 * j], ch[ik + idl1 * jc]);
                cc[ik + idl1 * l] = war.fma(hj, cc[ik + idl1 * l]);
                cc[ik + idl1 * lc] = wai.fma(hjc, cc[ik + idl1 * lc]);
            }
            j += 1;
            jc -= 1;
        }
        l += 1;
        lc -= 1;
    }
    for j in 1..ipph {
        for ik in 0..idl1 {
            ch[ik] = ch[ik] + ch[ik + idl1 * j];
        }
    }
    let (mut j, mut jc) = (1, ip - 1);
    while j < ipph {
        for k in 0..l1 {
            let (a, b) = (cc[c1i(0, k, j)], cc[c1i(0, k, jc)]);
            ch[chi(0, k, jc)] = a + b;
            ch[chi(0, k, j)] = a - b;
        }
        j += 1;
        jc -= 1;
    }
    if ido == 1 {
        return;
    }
    let (mut j, mut jc) = (1, ip - 1);
    while j < ipph {
        for k in 0..l1 {
            let mut i = 1;
            while i + 1 < ido {
                ch[chi(i, k, j)] = cc[c1i(i, k, j)] - cc[c1i(i + 1, k, jc)];
                ch[chi(i, k, jc)] = cc[c1i(i, k, j)] + cc[c1i(i + 1, k, jc)];
                ch[chi(i + 1, k, j)] = cc[c1i(i + 1, k, j)] + cc[c1i(i, k, jc)];
                ch[chi(i + 1, k, jc)] = cc[c1i(i + 1, k, j)] - cc[c1i(i, k, jc)];
                i += 2;
            }
        }
        j += 1;
        jc -= 1;
    }
    for j in 1..ip {
        let is = (j - 1) * (ido - 1);
        for k in 0..l1 {
            let mut idij = is;
            let mut i = 1;
            while i + 1 < ido {
                let (t1, t2) = (ch[chi(i, k, j)], ch[chi(i + 1, k, j)]);
                ch[chi(i, k, j)] = wa[idij].fma(t1, -(wa[idij + 1] * t2));
                ch[chi(i + 1, k, j)] = wa[idij].fma(t2, wa[idij + 1] * t1);
                idij += 2;
                i += 2;
            }
        }
    }
}

// ── Bluestein ────────────────────────────────────────────────────────────────

pub struct FftBlue<T> {
    n: usize,
    n2: usize,
    plan: Cfftp<T>,
    bk: Vec<Cx<T>>,
    bkf: Vec<Cx<T>>,
}

impl<T: Real> FftBlue<T> {
    pub fn new(n: usize, tw: Twiddles) -> Self {
        let n2 = good_size_cmplx(n * 2 - 1);
        let plan = Cfftp::new(n2, tw);
        let tmp = SinCos2PiByN::new(2 * n, tw);
        let mut bk = vec![Cx::new(T::ONE, T::ZERO); n];
        let mut coeff = 0;
        for (m, b) in bk.iter_mut().enumerate().skip(1) {
            coeff += 2 * m - 1;
            if coeff >= 2 * n {
                coeff -= 2 * n;
            }
            *b = tmp.get(coeff);
        }
        let mut tbkf = vec![Cx::<T>::default(); n2];
        let xn2 = T::ONE / T::of(n2 as f64);
        tbkf[0] = bk[0].scale(xn2);
        for m in 1..n {
            let v = bk[m].scale(xn2);
            tbkf[m] = v;
            tbkf[n2 - m] = v;
        }
        for v in tbkf.iter_mut().take(n2 - n + 1).skip(n) {
            *v = Cx::new(T::ZERO, T::ZERO);
        }
        plan.exec(&mut tbkf, T::ONE, true);
        let bkf = tbkf[..n2 / 2 + 1].to_vec();
        FftBlue {
            n,
            n2,
            plan,
            bk,
            bkf,
        }
    }

    fn fft(&self, fwd: bool, c: &mut [Cx<T>], fct: T) {
        let (n, n2) = (self.n, self.n2);
        let mut akf = vec![Cx::<T>::default(); n2];
        for m in 0..n {
            akf[m] = smul(fwd, c[m], self.bk[m]);
        }
        let zero = akf[0].scale(T::ZERO);
        for v in akf.iter_mut().skip(n) {
            *v = zero;
        }
        self.plan.exec(&mut akf, T::ONE, true);
        akf[0] = smul(!fwd, akf[0], self.bkf[0]);
        for m in 1..n2.div_ceil(2) {
            akf[m] = smul(!fwd, akf[m], self.bkf[m]);
            akf[n2 - m] = smul(!fwd, akf[n2 - m], self.bkf[m]);
        }
        if n2 & 1 == 0 {
            akf[n2 / 2] = smul(!fwd, akf[n2 / 2], self.bkf[n2 / 2]);
        }
        self.plan.exec(&mut akf, T::ONE, false);
        for m in 0..n {
            c[m] = smul(fwd, akf[m], self.bk[m]).scale(fct);
        }
    }

    pub fn exec(&self, c: &mut [Cx<T>], fct: T, fwd: bool) {
        self.fft(fwd, c, fct)
    }

    pub fn exec_r(&self, c: &mut [T], fct: T, fwd: bool) {
        let n = self.n;
        let mut tmp = vec![Cx::<T>::default(); n];
        if fwd {
            let zero = T::ZERO * c[0];
            for m in 0..n {
                tmp[m] = Cx::new(c[m], zero);
            }
            self.fft(true, &mut tmp, fct);
            c[0] = tmp[0].r;
            // std::copy_n(&tmp[1].r, n-1, &c[1]): the interleaved (r, i) stream
            for k in 0..n - 1 {
                let z = tmp[1 + k / 2];
                c[1 + k] = if k % 2 == 0 { z.r } else { z.i };
            }
        } else {
            tmp[0] = Cx::new(c[0], c[0] * T::ZERO);
            // std::copy_n(c+1, n-1, &tmp[1].r)
            for k in 0..n - 1 {
                let z = &mut tmp[1 + k / 2];
                if k % 2 == 0 {
                    z.r = c[1 + k];
                } else {
                    z.i = c[1 + k];
                }
            }
            if n & 1 == 0 {
                tmp[n / 2].i = T::ZERO * c[0];
            }
            let mut m = 1;
            while 2 * m < n {
                tmp[n - m] = Cx::new(tmp[m].r, -tmp[m].i);
                m += 1;
            }
            self.fft(false, &mut tmp, fct);
            for m in 0..n {
                c[m] = tmp[m].r;
            }
        }
    }
}

// ── plans ─────────────────────────────────────────────────────────────────────

/// `pocketfft_c<T>`.
pub enum PlanC<T> {
    Pack(Cfftp<T>),
    Blue(FftBlue<T>),
}

impl<T: Real> PlanC<T> {
    pub fn new(length: usize, tw: Twiddles) -> Self {
        assert!(length > 0, "zero-length FFT requested");
        let tmp = if length < 50 {
            0
        } else {
            largest_prime_factor(length)
        };
        if tmp * tmp <= length {
            return PlanC::Pack(Cfftp::new(length, tw));
        }
        let comp1 = cost_guess(length);
        let comp2 = 2.0 * cost_guess(good_size_cmplx(2 * length - 1)) * 1.5;
        if comp2 < comp1 {
            PlanC::Blue(FftBlue::new(length, tw))
        } else {
            PlanC::Pack(Cfftp::new(length, tw))
        }
    }
    pub fn exec(&self, c: &mut [Cx<T>], fct: T, fwd: bool) {
        match self {
            PlanC::Pack(p) => p.exec(c, fct, fwd),
            PlanC::Blue(b) => b.exec(c, fct, fwd),
        }
    }
}

/// `pocketfft_r<T>`.
pub enum PlanR<T> {
    Pack(Rfftp<T>),
    Blue(FftBlue<T>),
}

impl<T: Real> PlanR<T> {
    pub fn new(length: usize, tw: Twiddles) -> Self {
        assert!(length > 0, "zero-length FFT requested");
        let tmp = if length < 50 {
            0
        } else {
            largest_prime_factor(length)
        };
        if tmp * tmp <= length {
            return PlanR::Pack(Rfftp::new(length, tw));
        }
        let comp1 = 0.5 * cost_guess(length);
        let comp2 = 2.0 * cost_guess(good_size_cmplx(2 * length - 1)) * 1.5;
        if comp2 < comp1 {
            PlanR::Blue(FftBlue::new(length, tw))
        } else {
            PlanR::Pack(Rfftp::new(length, tw))
        }
    }
    pub fn exec(&mut self, c: &mut [T], fct: T, fwd: bool) {
        match self {
            PlanR::Pack(p) => p.exec(c, fct, fwd),
            PlanR::Blue(b) => b.exec_r(c, fct, fwd),
        }
    }
}

// ── the 1-D entry points (`general_r2c` / `general_c2r` / `c2c`) ──────────────

/// `T` as [`NoFma<T>`] and back (`repr(transparent)`, so a copy).
fn wrap<T: Real>(x: &[T]) -> Vec<NoFma<T>> {
    x.iter().map(|&v| NoFma(v)).collect()
}
fn wrap_c<T: Real>(x: &[Cx<T>]) -> Vec<Cx<NoFma<T>>> {
    x.iter().map(|c| Cx::new(NoFma(c.r), NoFma(c.i))).collect()
}
fn unwrap_c<T: Real>(x: Vec<Cx<NoFma<T>>>) -> Vec<Cx<T>> {
    x.into_iter().map(|c| Cx::new(c.r.0, c.i.0)).collect()
}

/// `pocketfft::r2c` of `x.len()` reals → `n/2 + 1` complex, scaled by `fct`.
pub fn r2c<T: Real>(x: &[T], forward: bool, fct: T, opt: Options) -> Vec<Cx<T>> {
    match opt.contraction.fused() {
        true => r2c_in(x, forward, fct, opt.twiddles),
        false => unwrap_c(r2c_in(&wrap(x), forward, NoFma(fct), opt.twiddles)),
    }
}

/// `pocketfft::c2r`: `n/2 + 1` complex → `n` reals, scaled by `fct`.
pub fn c2r<T: Real>(x: &[Cx<T>], n: usize, forward: bool, fct: T, opt: Options) -> Vec<T> {
    match opt.contraction.fused() {
        true => c2r_in(x, n, forward, fct, opt.twiddles),
        false => c2r_in(&wrap_c(x), n, forward, NoFma(fct), opt.twiddles)
            .into_iter()
            .map(|v| v.0)
            .collect(),
    }
}

/// `pocketfft::c2c` of `x.len()` complex values, scaled by `fct`.
pub fn c2c<T: Real>(x: &[Cx<T>], forward: bool, fct: T, opt: Options) -> Vec<Cx<T>> {
    match opt.contraction.fused() {
        true => c2c_in(x, forward, fct, opt.twiddles),
        false => unwrap_c(c2c_in(&wrap_c(x), forward, NoFma(fct), opt.twiddles)),
    }
}

fn r2c_in<T: Real>(x: &[T], forward: bool, fct: T, tw: Twiddles) -> Vec<Cx<T>> {
    let len = x.len();
    let mut t = x.to_vec();
    PlanR::new(len, tw).exec(&mut t, fct, true);
    let mut out = Vec::with_capacity(len / 2 + 1);
    out.push(Cx::new(t[0], T::ZERO));
    let mut i = 1;
    while i + 1 < len {
        out.push(Cx::new(t[i], if forward { t[i + 1] } else { -t[i + 1] }));
        i += 2;
    }
    if i < len {
        out.push(Cx::new(t[i], T::ZERO));
    }
    out
}

fn c2r_in<T: Real>(x: &[Cx<T>], n: usize, forward: bool, fct: T, tw: Twiddles) -> Vec<T> {
    let mut t = vec![T::ZERO; n];
    t[0] = x[0].r;
    let (mut i, mut ii) = (1, 1);
    while i + 1 < n {
        t[i] = x[ii].r;
        t[i + 1] = if forward { -x[ii].i } else { x[ii].i };
        i += 2;
        ii += 1;
    }
    if i < n {
        t[i] = x[ii].r;
    }
    PlanR::new(n, tw).exec(&mut t, fct, false);
    t
}

fn c2c_in<T: Real>(x: &[Cx<T>], forward: bool, fct: T, tw: Twiddles) -> Vec<Cx<T>> {
    let mut c = x.to_vec();
    PlanC::new(c.len(), tw).exec(&mut c, fct, forward);
    c
}

/// A reusable complex plan of one length under one [`Options`] (twiddles built
/// once; `exec` is `&self`, so one plan serves many rows and threads).
pub enum C2cPlan<T> {
    Fused(PlanC<T>),
    Unfused(PlanC<NoFma<T>>),
}

impl<T: Real> C2cPlan<T> {
    pub fn new(length: usize, opt: Options) -> Self {
        match opt.contraction.fused() {
            true => C2cPlan::Fused(PlanC::new(length, opt.twiddles)),
            false => C2cPlan::Unfused(PlanC::new(length, opt.twiddles)),
        }
    }

    /// In place: `c ← fct · DFT±(c)` (`forward` is the `e^{-2πi…}` sign).
    pub fn exec(&self, c: &mut [Cx<T>], forward: bool, fct: T) {
        match self {
            C2cPlan::Fused(p) => p.exec(c, fct, forward),
            C2cPlan::Unfused(p) => {
                let mut w = wrap_c(c);
                p.exec(&mut w, NoFma(fct), forward);
                for (d, s) in c.iter_mut().zip(w) {
                    *d = Cx::new(s.r.0, s.i.0);
                }
            }
        }
    }
}

// ── plan export (for lowering the transform into another IR) ─────────────────

/// One complex pass of a [`Cfftp`] plan: factor `ip` at stride `l1`, inner length
/// `ido`. `tw[(j-1)*(ido-1) + i-1]` is pass twiddle `WA(j-1, i)`; `tws` (factors
/// above 11 only) the generic pass's `csarr`.
#[derive(Clone, Debug)]
pub struct PassSpec<T> {
    pub ip: usize,
    pub l1: usize,
    pub ido: usize,
    pub tw: Vec<Cx<T>>,
    pub tws: Vec<Cx<T>>,
}

/// A complex plan as data: the passes in execution order, or Bluestein around an
/// inner plan of length `n2` (`bk` the chirp, `bkf` its scaled transform).
#[derive(Clone, Debug)]
pub enum C2cSpec<T> {
    Pack(Vec<PassSpec<T>>),
    Blue {
        n2: usize,
        bk: Vec<Cx<T>>,
        bkf: Vec<Cx<T>>,
        plan: Vec<PassSpec<T>>,
    },
}

impl<T: Real> Cfftp<T> {
    fn spec(&self) -> Vec<PassSpec<T>> {
        let mut l1 = 1;
        self.fact
            .iter()
            .map(|f| {
                let ido = self.length / (l1 * f.fct);
                let p = PassSpec {
                    ip: f.fct,
                    l1,
                    ido,
                    tw: f.tw.clone(),
                    tws: f.tws.clone(),
                };
                l1 *= f.fct;
                p
            })
            .collect()
    }
}

impl<T: Real> PlanC<T> {
    pub fn spec(&self) -> C2cSpec<T> {
        match self {
            PlanC::Pack(p) => C2cSpec::Pack(p.spec()),
            PlanC::Blue(b) => C2cSpec::Blue {
                n2: b.n2,
                bk: b.bk.clone(),
                bkf: b.bkf.clone(),
                plan: b.plan.spec(),
            },
        }
    }
}

impl<T: Real> C2cPlan<T> {
    /// The plan's passes and constants, rounded exactly as `exec` uses them.
    pub fn spec(&self) -> C2cSpec<T> {
        fn un<T: Real>(v: &[Cx<NoFma<T>>]) -> Vec<Cx<T>> {
            v.iter().map(|c| Cx::new(c.r.0, c.i.0)).collect()
        }
        fn un_p<T: Real>(p: &[PassSpec<NoFma<T>>]) -> Vec<PassSpec<T>> {
            p.iter()
                .map(|p| PassSpec {
                    ip: p.ip,
                    l1: p.l1,
                    ido: p.ido,
                    tw: un(&p.tw),
                    tws: un(&p.tws),
                })
                .collect()
        }
        match self {
            C2cPlan::Fused(p) => p.spec(),
            C2cPlan::Unfused(p) => match p.spec() {
                C2cSpec::Pack(v) => C2cSpec::Pack(un_p(&v)),
                C2cSpec::Blue { n2, bk, bkf, plan } => C2cSpec::Blue {
                    n2,
                    bk: un(&bk),
                    bkf: un(&bkf),
                    plan: un_p(&plan),
                },
            },
        }
    }
}
