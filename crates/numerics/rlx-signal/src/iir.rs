// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `scipy.signal` IIR design and filtering (SciPy 1.17.1), bit-for-bit: Butterworth
//! prototypes, the `lp2*_zpk` transforms, `bilinear_zpk`, `zpk2tf`, `zpk2sos`
//! (`pairing="nearest"`), `lfilter` / `sosfilt` (SciPy's own C loops, vendored),
//! `lfilter_zi` / `sosfilt_zi` (`np.linalg.solve` → LAPACK `dgesv`), `filtfilt` and
//! `sosfiltfilt` (`method="pad"`).
//!
//! Arrays carry NumPy's dtype where it changes the arithmetic: a zero/pole vector
//! is [`Roots::Real`] or [`Roots::Complex`], because `np.poly` of a real array is a
//! real convolution and of a complex one a complex convolution.

use crate::complex::{self as cx, C};

/// `a*b + c` as SciPy's compiled C/Cython computes it on this target's reference:
/// fused where clang `-ffp-contract=on` fuses on Apple arm64, two roundings on the
/// Linux x86-64 wheel (an SSE2 baseline build — measured: no FMA instruction in
/// `_sigtools*.so` / `_sosfilt*.so`).
#[inline(always)]
fn fma_c(a: f64, b: f64, c: f64) -> f64 {
    if cfg!(target_vendor = "apple") {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

/// SciPy 1.17.1's `filt<NPY_DOUBLE>` (`scipy/signal/_lfilter.cc`, one 1-D slice) with
/// the rounding its macOS arm64 wheel's clang (`-ffp-contract=on`) gives each source
/// expression: one `a*b + c` per expression is fused, the LEFT product first.
/// `b`, `a` have `len_b` entries (zero-padded to equal length) and are normalised by
/// `a[0]` in place, as the original does; `z` holds `len_b - 1` states.
fn lfilter_kernel(b: &mut [f64], a: &mut [f64], x: &[f64], y: &mut [f64], z: &mut [f64]) {
    let len_b = b.len();
    let a0 = a[0];
    for n in 0..len_b {
        b[n] /= a0;
        a[n] /= a0;
    }
    for (k, &xn) in x.iter().enumerate() {
        if len_b > 1 {
            // *yn = *ptr_Z + *ptr_b * *xn;
            let yn = fma_c(b[0], xn, z[0]);
            y[k] = yn;
            for n in 0..len_b - 2 {
                // *ptr_Z = ptr_Z[1] + *xn * (*ptr_b) - *yn * (*ptr_a);
                z[n] = fma_c(-yn, a[n + 1], fma_c(xn, b[n + 1], z[n + 1]));
            }
            // *ptr_Z = *xn * (*ptr_b) - *yn * (*ptr_a);
            z[len_b - 2] = fma_c(xn, b[len_b - 1], -(yn * a[len_b - 1]));
        } else {
            y[k] = xn * b[0];
        }
    }
}

/// SciPy 1.17.1's `_sosfilt_float` (`scipy/signal/_sosfilt.pyx`, one signal),
/// contracted as above. `sos` is `n_sections × 6`, `zi` `n_sections × 2`; `x` is
/// filtered in place.
fn sosfilt_kernel(sos: &[f64], x: &mut [f64], zi: &mut [f64]) {
    let n_sections = sos.len() / 6;
    for xv in x.iter_mut() {
        // x_cur = const_1 * x[n];
        let mut x_cur = 1.0 * *xv;
        for s in 0..n_sections {
            let c = &sos[s * 6..s * 6 + 6];
            // x_new = sos[0] * x_cur + zi[0];
            let x_new = fma_c(c[0], x_cur, zi[s * 2]);
            // zi[0] = (sos[1] * x_cur - sos[4] * x_new + zi[1]);
            zi[s * 2] = fma_c(c[1], x_cur, -(c[4] * x_new)) + zi[s * 2 + 1];
            // zi[1] = sos[2] * x_cur - sos[5] * x_new;
            zi[s * 2 + 1] = fma_c(c[2], x_cur, -(c[5] * x_new));
            x_cur = x_new;
        }
        *xv = x_cur;
    }
}

/// One second-order section `[b0, b1, b2, a0, a1, a2]`.
pub type Sos = [f64; 6];

/// A zero or pole vector with its NumPy dtype.
#[derive(Clone, Debug, PartialEq)]
pub enum Roots {
    Real(Vec<f64>),
    Complex(Vec<C>),
}

impl Roots {
    pub fn len(&self) -> usize {
        match self {
            Roots::Real(v) => v.len(),
            Roots::Complex(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// The values as complex (`astype(complex128)`).
    pub fn to_complex(&self) -> Vec<C> {
        match self {
            Roots::Real(v) => v.iter().map(|&x| C::real(x)).collect(),
            Roots::Complex(v) => v.clone(),
        }
    }
    /// `np.concatenate((self, extra))` with NumPy's dtype promotion.
    fn concat_real(&self, extra: &[f64]) -> Roots {
        match self {
            Roots::Real(v) => Roots::Real(v.iter().chain(extra).copied().collect()),
            Roots::Complex(v) => Roots::Complex(
                v.iter()
                    .copied()
                    .chain(extra.iter().map(|&x| C::real(x)))
                    .collect(),
            ),
        }
    }
}

/// A zero-pole-gain system.
#[derive(Clone, Debug, PartialEq)]
pub struct Zpk {
    pub z: Roots,
    pub p: Roots,
    pub k: f64,
}

/// `scipy.signal.buttap(N)`: no zeros, poles `-exp(1j·π·m/(2N))`, `m = -N+1, -N+3, …, N-1`.
pub fn buttap(n: usize) -> Zpk {
    // `1j * xp.pi` is a Python complex product: (0, π) exactly.
    let ipi = C::new(0.0, std::f64::consts::PI);
    let two_n = C::real((2 * n) as f64);
    let p = (0..n)
        .map(|i| {
            let m = -(n as f64) + 1.0 + 2.0 * i as f64;
            cx::neg(cx::exp(cx::div(cx::mul(ipi, C::real(m)), two_n)))
        })
        .collect();
    Zpk {
        z: Roots::Real(Vec::new()),
        p: Roots::Complex(p),
        k: 1.0,
    }
}

/// `p.shape[0] - z.shape[0]`.
fn relative_degree(z: &Roots, p: &Roots) -> Result<usize, String> {
    p.len().checked_sub(z.len()).ok_or_else(|| {
        "Improper transfer function. Must have at least as many poles as zeros.".to_string()
    })
}

/// Python `float ** int` (libm `pow`).
fn pypow(x: f64, e: usize) -> f64 {
    rlx_libm::reference::pow(x, e as f64)
}

/// `lp2lp_zpk(z, p, k, wo)`.
pub fn lp2lp_zpk(s: &Zpk, wo: f64) -> Result<Zpk, String> {
    let degree = relative_degree(&s.z, &s.p)?;
    let scale = |r: &Roots| match r {
        Roots::Real(v) => Roots::Real(v.iter().map(|x| wo * x).collect()),
        Roots::Complex(v) => Roots::Complex(v.iter().map(|&x| cx::mul(C::real(wo), x)).collect()),
    };
    Ok(Zpk {
        z: scale(&s.z),
        p: scale(&s.p),
        k: s.k * pypow(wo, degree),
    })
}

/// `np.prod(v)` of a real array: a left fold from `v[0]`; `1.0` when empty.
fn prod_real(v: &[f64]) -> f64 {
    match v.split_first() {
        None => 1.0,
        Some((&first, rest)) => rest.iter().fold(first, |a, &x| a * x),
    }
}

/// `np.prod(-v)` as NumPy types it: real → real, complex → complex.
fn prod_neg(r: &Roots) -> C {
    match r {
        Roots::Real(v) => C::real(prod_real(&v.iter().map(|x| -x).collect::<Vec<_>>())),
        Roots::Complex(v) => cx::prod(&v.iter().map(|&x| cx::neg(x)).collect::<Vec<_>>()),
    }
}

/// `lp2hp_zpk(z, p, k, wo)`.
pub fn lp2hp_zpk(s: &Zpk, wo: f64) -> Result<Zpk, String> {
    let degree = relative_degree(&s.z, &s.p)?;
    let inv = |r: &Roots| match r {
        Roots::Real(v) => Roots::Real(v.iter().map(|x| wo / x).collect()),
        Roots::Complex(v) => Roots::Complex(v.iter().map(|&x| cx::div(C::real(wo), x)).collect()),
    };
    let z = inv(&s.z).concat_real(&vec![0.0; degree]);
    let p = inv(&s.p);
    let k = s.k * cx::div(prod_neg(&s.z), prod_neg(&s.p)).re;
    Ok(Zpk { z, p, k })
}

/// `lp2bp_zpk(z, p, k, wo, bw)`.
pub fn lp2bp_zpk(s: &Zpk, wo: f64, bw: f64) -> Result<Zpk, String> {
    let degree = relative_degree(&s.z, &s.p)?;
    let wo2 = pypow(wo, 2);
    // z * bw / 2 → complex128
    let half = |r: &Roots| -> Vec<C> {
        match r {
            Roots::Real(v) => v.iter().map(|&x| C::real(x * bw / 2.0)).collect(),
            Roots::Complex(v) => v
                .iter()
                .map(|&x| cx::div(cx::mul(x, C::real(bw)), C::real(2.0)))
                .collect(),
        }
    };
    let split = |v: Vec<C>| -> Vec<C> {
        let r: Vec<C> = v
            .iter()
            .map(|&s| cx::sqrt(cx::sub(cx::mul(s, s), C::real(wo2))))
            .collect();
        let mut out: Vec<C> = v.iter().zip(&r).map(|(&s, &q)| cx::add(s, q)).collect();
        out.extend(v.iter().zip(&r).map(|(&s, &q)| cx::sub(s, q)));
        out
    };
    let mut z = split(half(&s.z));
    z.extend(std::iter::repeat_n(C::real(0.0), degree));
    let p = split(half(&s.p));
    Ok(Zpk {
        z: Roots::Complex(z),
        p: Roots::Complex(p),
        k: s.k * pypow(bw, degree),
    })
}

/// `bilinear_zpk(z, p, k, fs)`.
pub fn bilinear_zpk(s: &Zpk, fs: f64) -> Result<Zpk, String> {
    let degree = relative_degree(&s.z, &s.p)?;
    let fs2 = 2.0 * fs;
    let map = |r: &Roots| match r {
        Roots::Real(v) => Roots::Real(v.iter().map(|&x| (fs2 + x) / (fs2 - x)).collect()),
        Roots::Complex(v) => Roots::Complex(
            v.iter()
                .map(|&x| cx::div(cx::radd(fs2, x), cx::rsub(fs2, x)))
                .collect(),
        ),
    };
    let z = map(&s.z).concat_real(&vec![-1.0; degree]);
    let p = map(&s.p);
    let prod = |r: &Roots| -> C {
        match r {
            Roots::Real(v) => C::real(prod_real(&v.iter().map(|&x| fs2 - x).collect::<Vec<_>>())),
            Roots::Complex(v) => cx::prod(&v.iter().map(|&x| cx::rsub(fs2, x)).collect::<Vec<_>>()),
        }
    };
    let k = s.k * cx::div(prod(&s.z), prod(&s.p)).re;
    Ok(Zpk { z, p, k })
}

/// The (x, y) operand pairs of `np.convolve(a, v)`'s output `k`, in the order NumPy's
/// `correlate` hands them to the dot product: `x` runs over the longer array
/// ascending (NumPy swaps the arguments when `v` is longer), `y` over the reversed
/// shorter one.
fn convolve_terms<T: Copy>(a: &[T], v: &[T], k: usize) -> Vec<(T, T)> {
    if v.len() > a.len() {
        // swapped: x over v, y over a
        return (0..v.len())
            .filter(|&j| k >= j && k - j < a.len())
            .map(|j| (v[j], a[k - j]))
            .collect();
    }
    (0..a.len())
        .filter(|&j| k >= j && k - j < v.len())
        .map(|j| (a[j], v[k - j]))
        .collect()
}

/// `np.convolve(a, [1, -r])` of real arrays: each output is `cblas_ddot` of its
/// overlap (a plain left-to-right sum for these lengths, measured), added to `0.0`.
fn convolve_root_real(a: &[f64], r: f64) -> Vec<f64> {
    let v = [1.0, -r];
    (0..=a.len())
        .map(|k| {
            convolve_terms(a, &v, k)
                .iter()
                .fold(0.0, |s, &(x, y)| s + x * y)
        })
        .collect()
}

/// `np.convolve(a, [1, -r])` of complex arrays, each output spelled as measured
/// against NumPy 2.3.5 / Accelerate (600 random cases per position and length, all
/// equal): a two-term overlap is `cblas_zdotu_sub`'s four separate FMA accumulators
/// (`re = Σxr·yr − Σxi·yi`, `im = Σxr·yi + Σxi·yr`); a one-term overlap is a
/// **plain** product at the right edge but the **left-fused** product when NumPy has
/// swapped the operands (`len(v) > len(a)`). Every output is then `0.0 + tmp`.
fn convolve_root_complex(a: &[C], r: C) -> Vec<C> {
    let v = [C::real(1.0), cx::neg(r)];
    let swapped = v.len() > a.len();
    (0..=a.len())
        .map(|k| {
            let t = convolve_terms(a, &v, k);
            let d = match t.as_slice() {
                [(x, y)] if swapped && cx::BASELINE_FUSED => cx::mul(*x, *y),
                [(x, y)] => cx::mul_plain(*x, *y),
                // OpenBLAS `zdotu` (Linux reference): four plain accumulators from 0
                [(x0, y0), (x1, y1)] if !cx::BASELINE_FUSED => {
                    let s1 = (0.0 + x0.re * y0.re) + x1.re * y1.re;
                    let s2 = (0.0 + x0.im * y0.im) + x1.im * y1.im;
                    let s3 = (0.0 + x0.re * y0.im) + x1.re * y1.im;
                    let s4 = (0.0 + x0.im * y0.re) + x1.im * y1.re;
                    C::new(s1 - s2, s3 + s4)
                }
                [(x0, y0), (x1, y1)] => {
                    let s1 = x1.re.mul_add(y1.re, x0.re * y0.re);
                    let s2 = x1.im.mul_add(y1.im, x0.im * y0.im);
                    let s3 = x1.re.mul_add(y1.im, x0.re * y0.im);
                    let s4 = x1.im.mul_add(y1.re, x0.im * y0.re);
                    C::new(s1 - s2, s3 + s4)
                }
                _ => unreachable!("a root factor has two coefficients"),
            };
            C::new(0.0 + d.re, 0.0 + d.im)
        })
        .collect()
}

/// `scipy.signal._polyutils.poly(roots)` → real coefficients. A complex result is
/// returned as its real part, which is what NumPy keeps when the roots come in
/// conjugate pairs — and what assigning it into a real array keeps otherwise.
pub fn poly(roots: &Roots) -> Vec<f64> {
    match roots {
        Roots::Real(v) => v.iter().fold(vec![1.0], |a, &r| convolve_root_real(&a, r)),
        Roots::Complex(v) => v
            .iter()
            .fold(vec![C::real(1.0)], |a, &r| convolve_root_complex(&a, r))
            .into_iter()
            .map(|c| c.re)
            .collect(),
    }
}

/// `zpk2tf(z, p, k)` → `(b, a)`.
pub fn zpk2tf(s: &Zpk) -> (Vec<f64>, Vec<f64>) {
    let b = poly(&s.z).into_iter().map(|c| s.k * c).collect();
    (b, poly(&s.p))
}

/// `_cplxreal(z)` → (one of each conjugate pair, positive imaginary part; reals).
pub fn cplxreal(z: &[C]) -> Result<(Vec<C>, Vec<f64>), String> {
    if z.is_empty() {
        return Ok((vec![], vec![]));
    }
    let tol = 100.0 * f64::EPSILON;
    let mut idx: Vec<usize> = (0..z.len()).collect();
    // lexsort((abs(imag), real)) — stable, primary key real
    idx.sort_by(|&a, &b| {
        z[a].re
            .partial_cmp(&z[b].re)
            .unwrap()
            .then(z[a].im.abs().partial_cmp(&z[b].im.abs()).unwrap())
    });
    let z: Vec<C> = idx.iter().map(|&i| z[i]).collect();
    let is_r: Vec<bool> = z.iter().map(|v| v.im.abs() <= tol * v.abs()).collect();
    let zr: Vec<f64> = z
        .iter()
        .zip(&is_r)
        .filter(|(_, r)| **r)
        .map(|(v, _)| v.re)
        .collect();
    if zr.len() == z.len() {
        return Ok((vec![], zr));
    }
    let zc: Vec<C> = z
        .iter()
        .zip(&is_r)
        .filter(|(_, r)| !**r)
        .map(|(v, _)| *v)
        .collect();
    let mut zp: Vec<C> = zc.iter().copied().filter(|v| v.im > 0.0).collect();
    let mut zn: Vec<C> = zc.iter().copied().filter(|v| v.im < 0.0).collect();
    if zp.len() != zn.len() {
        return Err("Array contains complex value with no matching conjugate.".into());
    }
    let same: Vec<bool> = (1..zp.len())
        .map(|i| zp[i].re - zp[i - 1].re <= tol * zp[i - 1].abs())
        .collect();
    let mut i = 0;
    while i < same.len() {
        if same[i] {
            let start = i;
            while i < same.len() && same[i] {
                i += 1;
            }
            let stop = i + 1;
            for chunk in [&mut zp[start..stop], &mut zn[start..stop]] {
                let mut c = chunk.to_vec();
                c.sort_by(|a, b| a.im.abs().partial_cmp(&b.im.abs()).unwrap());
                chunk.copy_from_slice(&c);
            }
        } else {
            i += 1;
        }
    }
    for (a, b) in zp.iter().zip(&zn) {
        if cx::sub(*a, b.conj()).abs() > tol * b.abs() {
            return Err("Array contains complex value with no matching conjugate.".into());
        }
    }
    let zc = zp
        .iter()
        .zip(&zn)
        .map(|(&a, &b)| cx::div(cx::add(a, b.conj()), C::real(2.0)))
        .collect();
    Ok((zc, zr))
}

/// `_single_zpksos(z, p, 1)`.
fn single_zpksos(z: &Roots, p: &Roots) -> Sos {
    let (b, a) = zpk2tf(&Zpk {
        z: z.clone(),
        p: p.clone(),
        k: 1.0,
    });
    let mut s = [0.0; 6];
    s[3 - b.len()..3].copy_from_slice(&b);
    s[6 - a.len()..6].copy_from_slice(&a);
    s
}

/// `np.asarray([a, b, …])` of NumPy scalars of one dtype.
fn roots_of(vals: &[C], complex: bool) -> Roots {
    if complex {
        Roots::Complex(vals.to_vec())
    } else {
        Roots::Real(vals.iter().map(|c| c.re).collect())
    }
}

/// `_nearest_real_complex_idx(fro, to, which)`: `argsort(abs(fro - to))` (stable for
/// the short arrays a filter has — NumPy's small-array path is an insertion sort).
fn nearest(fro: &[C], to: C, which: &str) -> usize {
    let mut order: Vec<usize> = (0..fro.len()).collect();
    let d: Vec<f64> = fro.iter().map(|&f| cx::sub(f, to).abs()).collect();
    order.sort_by(|&a, &b| d[a].partial_cmp(&d[b]).unwrap());
    match which {
        "any" => order[0],
        "real" => *order.iter().find(|&&i| fro[i].is_real()).unwrap(),
        _ => *order.iter().find(|&&i| !fro[i].is_real()).unwrap(),
    }
}

/// `np.argmin(np.abs(1 - np.abs(p)))` (first minimum).
fn idx_worst(p: &[C]) -> usize {
    let d: Vec<f64> = p.iter().map(|v| (1.0 - v.abs()).abs()).collect();
    let mut best = 0;
    for i in 1..d.len() {
        if d[i] < d[best] {
            best = i;
        }
    }
    best
}

/// `zpk2sos(z, p, k)` for a digital system (`pairing="nearest"`).
pub fn zpk2sos(s: &Zpk) -> Result<Vec<Sos>, String> {
    let k = s.k;
    if s.z.is_empty() && s.p.is_empty() {
        return Ok(vec![[k, 0.0, 0.0, 1.0, 0.0, 0.0]]);
    }
    let (mut z, mut p) = (s.z.clone(), s.p.clone());
    let (nz, np_) = (z.len(), p.len());
    p = p.concat_real(&vec![0.0; nz.saturating_sub(np_)]);
    z = z.concat_real(&vec![0.0; np_.saturating_sub(nz)]);
    let n_sections = p.len().max(z.len()).div_ceil(2);
    if p.len() % 2 == 1 {
        p = p.concat_real(&[0.0]);
        z = z.concat_real(&[0.0]);
    }
    // np.concatenate(_cplxreal(z)): complex if any complex pair survived, else real
    let cat = |r: &Roots| -> Result<(Vec<C>, bool), String> {
        let (c, re) = cplxreal(&r.to_complex())?;
        let complex = !c.is_empty();
        Ok((
            c.into_iter().chain(re.into_iter().map(C::real)).collect(),
            complex,
        ))
    };
    let (mut z, z_cplx) = cat(&z)?;
    let (mut p, p_cplx) = cat(&p)?;
    let zero = C::real(0.0);
    let mut sos = vec![[0.0; 6]; n_sections];
    for si in (0..n_sections).rev() {
        let p1 = p.remove(idx_worst(&p));
        let n_real_p = p.iter().filter(|v| v.is_real()).count();
        if p1.is_real() && n_real_p == 0 {
            let z1 = z.remove(nearest(&z, p1, "real"));
            sos[si] = single_zpksos(
                &roots_of(&[z1, zero], z_cplx),
                &roots_of(&[p1, zero], p_cplx),
            );
        } else if p.len() + 1 == z.len()
            && !p1.is_real()
            && n_real_p == 1
            && z.iter().filter(|v| v.is_real()).count() == 1
        {
            let z1 = z.remove(nearest(&z, p1, "complex"));
            sos[si] = single_zpksos(
                &roots_of(&[z1, z1.conj()], z_cplx),
                &roots_of(&[p1, p1.conj()], p_cplx),
            );
        } else {
            let p2 = if p1.is_real() {
                let reals: Vec<usize> = (0..p.len()).filter(|&i| p[i].is_real()).collect();
                let sub: Vec<C> = reals.iter().map(|&i| p[i]).collect();
                p.remove(reals[idx_worst(&sub)])
            } else {
                p1.conj()
            };
            let pp = roots_of(&[p1, p2], p_cplx);
            if !z.is_empty() {
                let z1 = z.remove(nearest(&z, p1, "any"));
                if !z1.is_real() {
                    sos[si] = single_zpksos(&roots_of(&[z1, z1.conj()], z_cplx), &pp);
                } else if !z.is_empty() {
                    let z2 = z.remove(nearest(&z, p1, "real"));
                    sos[si] = single_zpksos(&roots_of(&[z1, z2], z_cplx), &pp);
                } else {
                    sos[si] = single_zpksos(&roots_of(&[z1], z_cplx), &pp);
                }
            } else {
                sos[si] = single_zpksos(&Roots::Real(vec![]), &pp);
            }
        }
    }
    for v in &mut sos[0][..3] {
        *v *= k;
    }
    Ok(sos)
}

/// `iirfilter`'s `btype`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BType {
    Lowpass,
    Highpass,
    Bandpass,
}

/// `scipy.signal.butter(N, Wn, btype)` (digital, `fs=None`: `Wn` in half-cycles per
/// sample) as zeros/poles/gain — feed [`zpk2tf`] for `output="ba"` or [`zpk2sos`]
/// for `output="sos"`.
pub fn butter_zpk(n: usize, wn: &[f64], btype: BType) -> Result<Zpk, String> {
    if wn.iter().any(|&w| w <= 0.0) {
        return Err("filter critical frequencies must be greater than 0".into());
    }
    // `not Wn[0] < Wn[1]`: NaN edges refuse too
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    let unordered = wn.len() > 1 && !(wn[0] < wn[1]);
    if unordered {
        return Err("Wn[0] must be less than Wn[1]".into());
    }
    if wn.iter().any(|&w| w >= 1.0) {
        return Err("Digital filter critical frequencies must be 0 < Wn < 1".into());
    }
    let proto = buttap(n);
    let fs = 2.0;
    let warped: Vec<f64> = wn
        .iter()
        .map(|&w| 2.0 * fs * rlx_libm::reference::tan(std::f64::consts::PI * w / fs))
        .collect();
    let analog = match btype {
        BType::Lowpass | BType::Highpass => {
            if wn.len() != 1 {
                return Err(
                    "Must specify a single critical frequency Wn for lowpass or highpass filter"
                        .into(),
                );
            }
            if btype == BType::Lowpass {
                lp2lp_zpk(&proto, warped[0])?
            } else {
                lp2hp_zpk(&proto, warped[0])?
            }
        }
        BType::Bandpass => {
            if wn.len() != 2 {
                return Err(
                    "Wn must specify start and stop frequencies for bandpass or bandstop filter"
                        .into(),
                );
            }
            let bw = warped[1] - warped[0];
            let wo = (warped[0] * warped[1]).sqrt();
            lp2bp_zpk(&proto, wo, bw)?
        }
    };
    bilinear_zpk(&analog, fs)
}

/// `scipy.signal.lfilter(b, a, x, zi=zi)` for `len(a) > 1` (SciPy's C loop) → `(y, zf)`;
/// `zi = None` starts from rest. `b`, `a` are zero-padded to equal length.
///
/// # Panics
/// If `len(a) == 1` — SciPy filters that case by `np.convolve`, which is not ported.
pub fn lfilter(b: &[f64], a: &[f64], x: &[f64], zi: Option<&[f64]>) -> (Vec<f64>, Vec<f64>) {
    assert!(
        a.len() > 1,
        "lfilter with len(a) == 1 (SciPy's np.convolve path) is not ported"
    );
    let n = a.len().max(b.len());
    let (mut bb, mut aa) = (b.to_vec(), a.to_vec());
    bb.resize(n, 0.0);
    aa.resize(n, 0.0);
    let mut z = zi
        .map(<[f64]>::to_vec)
        .unwrap_or_else(|| vec![0.0; n.saturating_sub(1)]);
    z.resize(n.saturating_sub(1).max(1), 0.0);
    let mut y = vec![0.0; x.len()];
    lfilter_kernel(&mut bb, &mut aa, x, &mut y, &mut z);
    z.truncate(n.saturating_sub(1));
    (y, z)
}

/// `scipy.signal.sosfilt(sos, x, zi=zi)` → `(y, zf)`; `zi` is `n_sections × 2`.
pub fn sosfilt(sos: &[Sos], x: &[f64], zi: Option<&[[f64; 2]]>) -> (Vec<f64>, Vec<[f64; 2]>) {
    let flat: Vec<f64> = sos.iter().flatten().copied().collect();
    let mut z: Vec<f64> = match zi {
        Some(z) => z.iter().flatten().copied().collect(),
        None => vec![0.0; 2 * sos.len()],
    };
    let mut y = x.to_vec();
    sosfilt_kernel(&flat, &mut y, &mut z);
    (y, z.chunks(2).map(|c| [c[0], c[1]]).collect())
}

/// `scipy.signal.lfilter_zi(b, a)`: `solve(I - companion(a).T, b[1:] - a[1:]·b[0])`
/// through NumPy's `dgesv`.
pub fn lfilter_zi(b: &[f64], a: &[f64]) -> Result<Vec<f64>, String> {
    let mut a: Vec<f64> = a.to_vec();
    while a.len() > 1 && a[0] == 0.0 {
        a.remove(0);
    }
    if a.is_empty() {
        return Err("There must be at least one nonzero `a` coefficient.".into());
    }
    let mut b = b.to_vec();
    if a[0] != 1.0 {
        let a0 = a[0];
        b = b.iter().map(|v| v / a0).collect();
        a = a.iter().map(|v| v / a0).collect();
    }
    let n = a.len().max(b.len());
    a.resize(n, 0.0);
    b.resize(n, 0.0);
    let m = n - 1;
    if m == 0 {
        return Ok(Vec::new());
    }
    // companion(a): first row -a[1:] / (1.0 * a[0]); IminusA = eye - companion.T
    let first: Vec<f64> = a[1..].iter().map(|v| -v / (1.0 * a[0])).collect();
    let mut ima = vec![0.0f64; m * m];
    for i in 0..m {
        for j in 0..m {
            let ct = if j == 0 {
                first[i]
            } else if j == i + 1 {
                1.0
            } else {
                0.0
            };
            let eye = if i == j { 1.0 } else { 0.0 };
            ima[i * m + j] = eye - ct;
        }
    }
    let rhs: Vec<f64> = (1..n).map(|i| b[i] - a[i] * b[0]).collect();
    rlx_cpu::numpy_blas::dgesv_numpy(&ima, &rhs, m)
        .map_err(|info| format!("Singular matrix (dgesv info {info})"))
}

/// `scipy.signal.sosfilt_zi(sos)`.
pub fn sosfilt_zi(sos: &[Sos]) -> Result<Vec<[f64; 2]>, String> {
    let mut scale = 1.0f64;
    let mut out = Vec::with_capacity(sos.len());
    for s in sos {
        let (b, a) = (&s[..3], &s[3..]);
        let z = lfilter_zi(b, a)?;
        out.push([scale * z[0], scale * z[1]]);
        scale *= ((b[0] + b[1]) + b[2]) / ((a[0] + a[1]) + a[2]);
    }
    Ok(out)
}

/// `filtfilt` / `sosfiltfilt`'s `padtype`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PadType {
    Odd,
    Even,
    Constant,
    None,
}

/// `_validate_pad(padtype, padlen, x, ntaps)` → `(edge, ext)`.
fn validate_pad(
    padtype: PadType,
    padlen: Option<usize>,
    x: &[f64],
    ntaps: usize,
) -> Result<(usize, Vec<f64>), String> {
    let edge = if padtype == PadType::None {
        0
    } else {
        padlen.unwrap_or(ntaps * 3)
    };
    if x.len() <= edge {
        return Err(format!(
            "The length of the input vector x must be greater than padlen, which is {edge}."
        ));
    }
    if padtype == PadType::None || edge == 0 {
        return Ok((edge, x.to_vec()));
    }
    let n = x.len();
    let (l, r) = (x[0], x[n - 1]);
    let left: Vec<f64> = (1..=edge).rev().map(|i| x[i]).collect();
    let right: Vec<f64> = (1..=edge).map(|i| x[n - 1 - i]).collect();
    let (left, right) = match padtype {
        PadType::Odd => (
            left.iter().map(|v| 2.0 * l - v).collect(),
            right.iter().map(|v| 2.0 * r - v).collect(),
        ),
        PadType::Even => (left, right),
        _ => (vec![l; edge], vec![r; edge]),
    };
    Ok((
        edge,
        left.into_iter()
            .chain(x.iter().copied())
            .chain(right)
            .collect(),
    ))
}

/// `scipy.signal.filtfilt(b, a, x, padtype, padlen)` (`method="pad"`).
pub fn filtfilt(
    b: &[f64],
    a: &[f64],
    x: &[f64],
    padtype: PadType,
    padlen: Option<usize>,
) -> Result<Vec<f64>, String> {
    let (edge, ext) = validate_pad(padtype, padlen, x, a.len().max(b.len()))?;
    let zi = lfilter_zi(b, a)?;
    let z0: Vec<f64> = zi.iter().map(|z| z * ext[0]).collect();
    let (y, _) = lfilter(b, a, &ext, Some(&z0));
    let y0 = *y.last().unwrap();
    let z1: Vec<f64> = zi.iter().map(|z| z * y0).collect();
    let rev: Vec<f64> = y.into_iter().rev().collect();
    let (mut y, _) = lfilter(b, a, &rev, Some(&z1));
    y.reverse();
    Ok(if edge > 0 {
        y[edge..y.len() - edge].to_vec()
    } else {
        y
    })
}

/// `scipy.signal.sosfiltfilt(sos, x, padtype, padlen)`.
pub fn sosfiltfilt(
    sos: &[Sos],
    x: &[f64],
    padtype: PadType,
    padlen: Option<usize>,
) -> Result<Vec<f64>, String> {
    let zeros2 = sos.iter().filter(|s| s[2] == 0.0).count();
    let zeros5 = sos.iter().filter(|s| s[5] == 0.0).count();
    let ntaps = 2 * sos.len() + 1 - zeros2.min(zeros5);
    let (edge, ext) = validate_pad(padtype, padlen, x, ntaps)?;
    let zi = sosfilt_zi(sos)?;
    let scaled = |v: f64| -> Vec<[f64; 2]> { zi.iter().map(|z| [z[0] * v, z[1] * v]).collect() };
    let (y, _) = sosfilt(sos, &ext, Some(&scaled(ext[0])));
    let y0 = *y.last().unwrap();
    let rev: Vec<f64> = y.into_iter().rev().collect();
    let (mut y, _) = sosfilt(sos, &rev, Some(&scaled(y0)));
    y.reverse();
    Ok(if edge > 0 {
        y[edge..y.len() - edge].to_vec()
    } else {
        y
    })
}
