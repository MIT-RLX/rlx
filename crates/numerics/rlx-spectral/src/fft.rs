//! FFT plans in the working precision.
//!
//! * [`FftPlan`] — complex DFT of any length: a Stockham autosort
//!   decimation-in-frequency transform over radices 4, 2, 3 and 5 (ping-pong
//!   buffers, no bit reversal), Bluestein's chirp-z over a power-of-two plan
//!   when the length has a larger prime factor.
//! * [`RealFftPlan`] — real input, `n/2 + 1` bins: one complex transform of
//!   half the length plus a split pass for even `n`.
//!
//! Twiddles are computed in f64 and rounded once. Forward is
//! `X[k] = Σ x[t]·e^{-2πi·tk/n}`; the inverse is unnormalized.

use crate::float::{Complex, Float};

#[derive(Clone, Debug)]
struct Stage<T> {
    radix: usize,
    /// sub-transform length at this stage
    len: usize,
    /// `ω_len^{q·j}` for `q < len/radix`, `j = 1..radix`
    tw: Vec<Complex<T>>,
}

#[derive(Clone, Debug)]
enum Kind<T> {
    Stockham(Vec<Stage<T>>),
    Bluestein {
        inner: Box<FftPlan<T>>,
        m: usize,
        /// `e^{-πi k²/n}`, `k < n`
        chirp: Vec<Complex<T>>,
        /// forward transform of the conjugate chirp, length `m`
        kernel: Vec<Complex<T>>,
    },
}

/// A complex FFT of fixed length.
#[derive(Clone, Debug)]
pub struct FftPlan<T> {
    n: usize,
    kind: Kind<T>,
}

fn factor(mut n: usize) -> (Vec<usize>, usize) {
    let mut f = Vec::new();
    for p in [8usize, 4, 2, 3, 5] {
        while n.is_multiple_of(p) && n > 1 {
            f.push(p);
            n /= p;
        }
    }
    (f, n)
}

impl<T: Float> FftPlan<T> {
    pub fn new(n: usize) -> Self {
        assert!(n > 0, "FFT length must be positive");
        let (radices, rest) = factor(n);
        if rest == 1 {
            let mut stages = Vec::with_capacity(radices.len());
            let mut len = n;
            for &radix in &radices {
                let m = len / radix;
                let mut tw = Vec::with_capacity(m * (radix - 1));
                for q in 0..m {
                    for j in 1..radix {
                        let e = ((q * j) % len) as f64;
                        tw.push(Complex::cis(-2.0 * std::f64::consts::PI * e / len as f64));
                    }
                }
                stages.push(Stage { radix, len, tw });
                len = m;
            }
            return FftPlan {
                n,
                kind: Kind::Stockham(stages),
            };
        }
        // Bluestein: X_k = w_k · Σ_t (x_t w_t) · conj(w_{k-t}), w_k = e^{-πi k²/n}
        let m = (2 * n - 1).next_power_of_two();
        let inner = FftPlan::new(m);
        let two_n = 2 * n as u128;
        let chirp: Vec<Complex<T>> = (0..n)
            .map(|k| {
                let k2 = ((k as u128 * k as u128) % two_n) as f64;
                Complex::cis(-std::f64::consts::PI * k2 / n as f64)
            })
            .collect();
        let mut kernel = vec![Complex::default(); m];
        for k in 0..n {
            kernel[k] = chirp[k].conj();
            if k > 0 {
                kernel[m - k] = chirp[k].conj();
            }
        }
        let mut scratch = vec![Complex::default(); m];
        inner.forward_with(&mut kernel, &mut scratch);
        FftPlan {
            n,
            kind: Kind::Bluestein {
                inner: Box::new(inner),
                m,
                chirp,
                kernel,
            },
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Scratch length [`forward_with`](Self::forward_with) needs.
    pub fn scratch_len(&self) -> usize {
        match &self.kind {
            Kind::Stockham(_) => self.n,
            Kind::Bluestein { m, .. } => 2 * m,
        }
    }

    /// In-place forward transform (allocates scratch).
    pub fn forward(&self, buf: &mut [Complex<T>]) {
        let mut s = vec![Complex::default(); self.scratch_len()];
        self.forward_with(buf, &mut s);
    }

    /// In-place inverse transform, unnormalized (allocates scratch).
    pub fn inverse(&self, buf: &mut [Complex<T>]) {
        buf.iter_mut().for_each(|v| *v = v.conj());
        self.forward(buf);
        buf.iter_mut().for_each(|v| *v = v.conj());
    }

    /// In-place forward transform; `scratch.len() >= scratch_len()`.
    pub fn forward_with(&self, buf: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        assert_eq!(buf.len(), self.n);
        match &self.kind {
            Kind::Stockham(stages) => stockham(stages, buf, &mut scratch[..self.n]),
            Kind::Bluestein {
                inner,
                m,
                chirp,
                kernel,
            } => {
                let (a, rest) = scratch.split_at_mut(*m);
                let s = &mut rest[..*m];
                for (k, v) in a.iter_mut().enumerate() {
                    *v = if k < self.n {
                        buf[k].mul(chirp[k])
                    } else {
                        Complex::default()
                    };
                }
                inner.forward_with(a, s);
                for (v, &kv) in a.iter_mut().zip(kernel) {
                    *v = v.mul(kv).conj(); // conj for the inverse below
                }
                inner.forward_with(a, s);
                let inv_m = T::from_f64(1.0 / *m as f64);
                for k in 0..self.n {
                    buf[k] = a[k].conj().scale(inv_m).mul(chirp[k]);
                }
            }
        }
    }
}

/// Stockham DIF: stage with radix `p` on length-`len` sub-transforms at
/// stride `s` reads `x[k + s·(q + m·t)]`, writes `y[k + s·(p·q + j)] =
/// (Σ_t x_t ω_p^{jt}) · ω_len^{qj}`.
fn stockham<T: Float>(stages: &[Stage<T>], buf: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
    let n = buf.len();
    let mut s = 1usize;
    let mut in_buf = true; // current data in `buf` (else in `scratch`)
    for st in stages {
        let m = st.len / st.radix;
        let (x, y): (&[Complex<T>], &mut [Complex<T>]) = if in_buf {
            // SAFETY-free split: read buf, write scratch
            (
                unsafe { std::slice::from_raw_parts(buf.as_ptr(), n) },
                &mut *scratch,
            )
        } else {
            (
                unsafe { std::slice::from_raw_parts(scratch.as_ptr(), n) },
                &mut *buf,
            )
        };
        match st.radix {
            2 => radix2(x, y, s, m, &st.tw),
            8 => radix8(x, y, s, m, &st.tw),
            3 => radix3(x, y, s, m, &st.tw),
            4 => radix4(x, y, s, m, &st.tw),
            5 => radix5(x, y, s, m, &st.tw),
            _ => unreachable!(),
        }
        in_buf = !in_buf;
        s *= st.radix;
    }
    if !in_buf {
        buf.copy_from_slice(scratch);
    }
}

#[inline(always)]
fn radix2<T: Float>(x: &[Complex<T>], y: &mut [Complex<T>], s: usize, m: usize, tw: &[Complex<T>]) {
    for q in 0..m {
        let w1 = tw[q];
        for k in 0..s {
            let a0 = x[k + s * q];
            let a1 = x[k + s * (q + m)];
            y[k + s * 2 * q] = a0.add(a1);
            y[k + s * (2 * q + 1)] = a0.sub(a1).mul(w1);
        }
    }
}

#[inline(always)]
fn radix4<T: Float>(x: &[Complex<T>], y: &mut [Complex<T>], s: usize, m: usize, tw: &[Complex<T>]) {
    for q in 0..m {
        let (w1, w2, w3) = (tw[3 * q], tw[3 * q + 1], tw[3 * q + 2]);
        for k in 0..s {
            let a0 = x[k + s * q];
            let a1 = x[k + s * (q + m)];
            let a2 = x[k + s * (q + 2 * m)];
            let a3 = x[k + s * (q + 3 * m)];
            let t0 = a0.add(a2);
            let t1 = a0.sub(a2);
            let t2 = a1.add(a3);
            let t3 = a1.sub(a3).mul_neg_i();
            let o = k + s * 4 * q;
            y[o] = t0.add(t2);
            if m == 1 {
                y[o + s] = t1.add(t3);
                y[o + 2 * s] = t0.sub(t2);
                y[o + 3 * s] = t1.sub(t3);
            } else {
                y[o + s] = t1.add(t3).mul(w1);
                y[o + 2 * s] = t0.sub(t2).mul(w2);
                y[o + 3 * s] = t1.sub(t3).mul(w3);
            }
        }
    }
}

/// Radix 8 = radix 2 over two radix-4 butterflies (even / odd inputs), the
/// odd half rotated by `ω_8^j`.
#[inline(always)]
fn radix8<T: Float>(x: &[Complex<T>], y: &mut [Complex<T>], s: usize, m: usize, tw: &[Complex<T>]) {
    let h = T::from_f64(std::f64::consts::FRAC_1_SQRT_2);
    // ω_8 = (1 − i)/√2, ω_8^2 = −i, ω_8^3 = −(1 + i)/√2
    let w8_1 = |c: Complex<T>| Complex::new((c.re + c.im) * h, (c.im - c.re) * h);
    let w8_3 = |c: Complex<T>| Complex::new((c.im - c.re) * h, -(c.re + c.im) * h);
    let last = m == 1;
    for q in 0..m {
        let w = &tw[7 * q..7 * q + 7];
        for k in 0..s {
            let a = |t: usize| x[k + s * (q + m * t)];
            let (a0, a1, a2, a3, a4, a5, a6, a7) = (a(0), a(1), a(2), a(3), a(4), a(5), a(6), a(7));
            // radix-4 on the even inputs (a0, a2, a4, a6) and odd (a1, a3, a5, a7)
            let (e0, e1, e2, e3) = {
                let t0 = a0.add(a4);
                let t1 = a0.sub(a4);
                let t2 = a2.add(a6);
                let t3 = a2.sub(a6).mul_neg_i();
                (t0.add(t2), t1.add(t3), t0.sub(t2), t1.sub(t3))
            };
            let (o0, o1, o2, o3) = {
                let t0 = a1.add(a5);
                let t1 = a1.sub(a5);
                let t2 = a3.add(a7);
                let t3 = a3.sub(a7).mul_neg_i();
                (t0.add(t2), t1.add(t3), t0.sub(t2), t1.sub(t3))
            };
            let (o1, o2, o3) = (w8_1(o1), o2.mul_neg_i(), w8_3(o3));
            let b = [
                e0.add(o0),
                e1.add(o1),
                e2.add(o2),
                e3.add(o3),
                e0.sub(o0),
                e1.sub(o1),
                e2.sub(o2),
                e3.sub(o3),
            ];
            let o = k + s * 8 * q;
            y[o] = b[0];
            if last {
                for j in 1..8 {
                    y[o + j * s] = b[j];
                }
            } else {
                for j in 1..8 {
                    y[o + j * s] = b[j].mul(w[j - 1]);
                }
            }
        }
    }
}

#[inline(always)]
fn radix3<T: Float>(x: &[Complex<T>], y: &mut [Complex<T>], s: usize, m: usize, tw: &[Complex<T>]) {
    let c = T::from_f64(-0.5);
    let sn = T::from_f64(3f64.sqrt() / 2.0);
    for q in 0..m {
        let (w1, w2) = (tw[2 * q], tw[2 * q + 1]);
        for k in 0..s {
            let a0 = x[k + s * q];
            let a1 = x[k + s * (q + m)];
            let a2 = x[k + s * (q + 2 * m)];
            let t1 = a1.add(a2);
            let t2 = Complex::new(t1.re.mul_add(c, a0.re), t1.im.mul_add(c, a0.im));
            let t3 = a1.sub(a2).mul_neg_i().scale(sn);
            let o = k + s * 3 * q;
            y[o] = a0.add(t1);
            y[o + s] = t2.add(t3).mul(w1);
            y[o + 2 * s] = t2.sub(t3).mul(w2);
        }
    }
}

#[inline(always)]
fn radix5<T: Float>(x: &[Complex<T>], y: &mut [Complex<T>], s: usize, m: usize, tw: &[Complex<T>]) {
    let pi = std::f64::consts::PI;
    let c1 = T::from_f64(libm::cos(2.0 * pi / 5.0));
    let c2 = T::from_f64(libm::cos(4.0 * pi / 5.0));
    let s1 = T::from_f64(libm::sin(2.0 * pi / 5.0));
    let s2 = T::from_f64(libm::sin(4.0 * pi / 5.0));
    for q in 0..m {
        let (w1, w2, w3, w4) = (tw[4 * q], tw[4 * q + 1], tw[4 * q + 2], tw[4 * q + 3]);
        for k in 0..s {
            let a0 = x[k + s * q];
            let a1 = x[k + s * (q + m)];
            let a2 = x[k + s * (q + 2 * m)];
            let a3 = x[k + s * (q + 3 * m)];
            let a4 = x[k + s * (q + 4 * m)];
            let p14 = a1.add(a4);
            let m14 = a1.sub(a4);
            let p23 = a2.add(a3);
            let m23 = a2.sub(a3);
            let r1 = Complex::new(
                p23.re.mul_add(c2, p14.re.mul_add(c1, a0.re)),
                p23.im.mul_add(c2, p14.im.mul_add(c1, a0.im)),
            );
            let r2 = Complex::new(
                p23.re.mul_add(c1, p14.re.mul_add(c2, a0.re)),
                p23.im.mul_add(c1, p14.im.mul_add(c2, a0.im)),
            );
            // -i·(m14·s1 + m23·s2) and -i·(m14·s2 − m23·s1)
            let i1 = Complex::new(
                m23.re.mul_add(s2, m14.re * s1),
                m23.im.mul_add(s2, m14.im * s1),
            )
            .mul_neg_i();
            let i2 = Complex::new(
                m14.re.mul_add(s2, -(m23.re * s1)),
                m14.im.mul_add(s2, -(m23.im * s1)),
            )
            .mul_neg_i();
            let o = k + s * 5 * q;
            y[o] = a0.add(p14).add(p23);
            y[o + s] = r1.add(i1).mul(w1);
            y[o + 2 * s] = r2.add(i2).mul(w2);
            y[o + 3 * s] = r2.sub(i2).mul(w3);
            y[o + 4 * s] = r1.sub(i1).mul(w4);
        }
    }
}

/// A real-input FFT of fixed length: `n/2 + 1` bins.
#[derive(Clone, Debug)]
pub struct RealFftPlan<T> {
    n: usize,
    /// even n: half-length complex plan + split twiddles `ω_n^k`
    half: Option<(FftPlan<T>, Vec<Complex<T>>)>,
    /// odd n: full complex plan
    full: Option<FftPlan<T>>,
}

impl<T: Float> RealFftPlan<T> {
    pub fn new(n: usize) -> Self {
        assert!(n > 0);
        if n.is_multiple_of(2) && n >= 2 {
            let h = n / 2;
            let tw = (0..=h)
                .map(|k| Complex::cis(-2.0 * std::f64::consts::PI * k as f64 / n as f64))
                .collect();
            RealFftPlan {
                n,
                half: Some((FftPlan::new(h), tw)),
                full: None,
            }
        } else {
            RealFftPlan {
                n,
                half: None,
                full: Some(FftPlan::new(n)),
            }
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn bins(&self) -> usize {
        self.n / 2 + 1
    }

    /// A reusable work buffer for [`forward_with`](Self::forward_with).
    pub fn make_scratch(&self) -> Vec<Complex<T>> {
        match (&self.half, &self.full) {
            (Some((p, _)), _) => vec![Complex::default(); p.len() + p.scratch_len()],
            (_, Some(p)) => vec![Complex::default(); p.len() + p.scratch_len()],
            _ => unreachable!(),
        }
    }

    /// `out[k] = Σ x[t]·e^{-2πi·tk/n}` for `k ≤ n/2`; `x.len() == n`.
    pub fn forward_with(&self, x: &[T], out: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        assert_eq!(x.len(), self.n);
        assert_eq!(out.len(), self.bins());
        if let Some((plan, tw)) = &self.half {
            let h = plan.len();
            let (z, s) = scratch.split_at_mut(h);
            for (k, v) in z.iter_mut().enumerate() {
                *v = Complex::new(x[2 * k], x[2 * k + 1]);
            }
            plan.forward_with(z, s);
            let half = T::from_f64(0.5);
            for k in 0..=h {
                let zk = z[k % h];
                let zc = z[(h - k) % h].conj();
                let e = zk.add(zc).scale(half);
                let o = zk.sub(zc).scale(half).mul_neg_i();
                out[k] = e.add(tw[k].mul(o));
            }
        } else if let Some(plan) = &self.full {
            let n = self.n;
            let (z, s) = scratch.split_at_mut(n);
            for (v, &xi) in z.iter_mut().zip(x) {
                *v = Complex::new(xi, T::ZERO);
            }
            plan.forward_with(z, s);
            out.copy_from_slice(&z[..self.bins()]);
        }
    }

    /// Allocating convenience form.
    pub fn forward(&self, x: &[T]) -> Vec<Complex<T>> {
        let mut out = vec![Complex::default(); self.bins()];
        let mut s = self.make_scratch();
        self.forward_with(x, &mut out, &mut s);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dft(x: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let n = x.len();
        (0..n)
            .map(|k| {
                let mut acc = Complex::new(0.0, 0.0);
                for (t, v) in x.iter().enumerate() {
                    let w: Complex<f64> =
                        Complex::cis(-2.0 * std::f64::consts::PI * ((t * k) % n) as f64 / n as f64);
                    acc = acc.add(v.mul(w));
                }
                acc
            })
            .collect()
    }

    fn signal(n: usize) -> Vec<Complex<f64>> {
        let mut s = 0x2545_f491u64;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let a = (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                Complex::new(a, (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5)
            })
            .collect()
    }

    /// Every radix path, mixed lengths and Bluestein lengths against an f64
    /// DFT, in f64 (to ~1e-13) and f32 (to ~1e-5 relative to the norm).
    #[test]
    fn complex_matches_dft() {
        for n in [
            1, 2, 3, 4, 5, 6, 8, 9, 12, 15, 16, 25, 30, 64, 100, 128, 200, 243, 256, 400, 512, 7,
            11, 13, 49, 97, 1000, 1021,
        ] {
            let x = signal(n);
            let want = dft(&x);
            let scale = want
                .iter()
                .map(|c| c.norm_sqr())
                .sum::<f64>()
                .sqrt()
                .max(1e-300);
            let mut got = x.clone();
            FftPlan::<f64>::new(n).forward(&mut got);
            let err = got
                .iter()
                .zip(&want)
                .map(|(a, b)| a.sub(*b).norm_sqr())
                .sum::<f64>()
                .sqrt()
                / scale;
            assert!(err < 1e-13, "f64 n={n}: rel err {err:e}");
            let mut g32: Vec<Complex<f32>> = x
                .iter()
                .map(|c| Complex::new(c.re as f32, c.im as f32))
                .collect();
            FftPlan::<f32>::new(n).forward(&mut g32);
            let err32 = g32
                .iter()
                .zip(&want)
                .map(|(a, b)| Complex::new(a.re as f64 - b.re, a.im as f64 - b.im).norm_sqr())
                .sum::<f64>()
                .sqrt()
                / scale;
            assert!(
                err32 < 2e-6 * (n as f64).log2().max(1.0),
                "f32 n={n}: rel err {err32:e}"
            );
            // inverse round trip
            let mut back = got.clone();
            FftPlan::<f64>::new(n).inverse(&mut back);
            let rt = back
                .iter()
                .zip(&x)
                .map(|(a, b)| a.scale(1.0 / n as f64).sub(*b).norm_sqr())
                .sum::<f64>()
                .sqrt();
            assert!(rt < 1e-12, "round trip n={n}: {rt:e}");
        }
    }

    #[test]
    fn real_matches_complex() {
        for n in [1, 2, 3, 4, 10, 16, 31, 400, 512, 1000] {
            let x: Vec<f64> = signal(n).iter().map(|c| c.re).collect();
            let cx: Vec<Complex<f64>> = x.iter().map(|&v| Complex::new(v, 0.0)).collect();
            let want = dft(&cx);
            let got = RealFftPlan::<f64>::new(n).forward(&x);
            assert_eq!(got.len(), n / 2 + 1);
            for (k, (a, b)) in got.iter().zip(&want).enumerate() {
                assert!(
                    a.sub(*b).norm_sqr().sqrt() < 1e-12 * (n as f64),
                    "n={n} bin {k}: {a:?} vs {b:?}"
                );
            }
        }
    }
}
