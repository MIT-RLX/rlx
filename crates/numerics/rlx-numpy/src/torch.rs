// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! torch CPU float32 reductions, reproduced bit-for-bit.
//!
//! `Tensor.sum()` on CPU is not a pairwise or a sequential sum: ATen's
//! `cascade_sum` (`ReduceOps/SumKernel.cpp`) loads `Vectorized<float>` lanes
//! — four on arm64 — sums the vectors four at a time (`ilp_factor`) through a
//! four-level cascade of accumulators, folds the four partial vectors
//! left-to-right, then adds the scalar tail and the lanes in order. Measured
//! against torch 2.11 (macOS arm64) on every length 1–199 and a few up to
//! 5000: bit-identical (float64: two lanes, likewise measured). Beyond the parallel grain (32768 elements) torch
//! splits the range over threads and the result depends on the thread count;
//! these functions are the single-thread spelling.

const LANES: usize = 4;
const ILP: usize = 4;

fn ceil_log2(x: usize) -> usize {
    if x <= 1 {
        0
    } else {
        (usize::BITS - (x - 1).leading_zeros()) as usize
    }
}

/// `multi_row_sum<acc_t, 4>` over `size` rows of four accumulators each;
/// `load(i, k)` is element `k` of row `i`.
fn multi_row_sum<A: Copy + Default + std::ops::Add<Output = A>>(
    size: usize,
    load: impl Fn(usize, usize) -> A,
) -> [A; ILP] {
    const LEVELS: usize = 4;
    let level_power = 4usize.max(ceil_log2(size) / LEVELS);
    let level_step = 1usize << level_power;
    let level_mask = level_step - 1;
    let mut acc = [[A::default(); ILP]; LEVELS];
    let mut i = 0;
    while i + level_step <= size {
        for _ in 0..level_step {
            for (k, a) in acc[0].iter_mut().enumerate() {
                *a = *a + load(i, k);
            }
            i += 1;
        }
        for j in 1..LEVELS {
            for k in 0..ILP {
                acc[j][k] = acc[j][k] + acc[j - 1][k];
                acc[j - 1][k] = A::default();
            }
            if i & (level_mask << (j * level_power)) != 0 {
                break;
            }
        }
    }
    while i < size {
        for (k, a) in acc[0].iter_mut().enumerate() {
            *a = *a + load(i, k);
        }
        i += 1;
    }
    for j in 1..LEVELS {
        for k in 0..ILP {
            acc[0][k] = acc[0][k] + acc[j][k];
        }
    }
    acc[0]
}

/// `row_sum`: `n` values `load(i)`, four interleaved partial sums.
fn row_sum<A: Copy + Default + std::ops::Add<Output = A>>(
    n: usize,
    load: impl Fn(usize) -> A,
) -> A {
    let size_ilp = n / ILP;
    let mut ps = multi_row_sum(size_ilp, |i, k| load(i * ILP + k));
    for i in size_ilp * ILP..n {
        ps[0] = ps[0] + load(i);
    }
    for k in 1..ILP {
        ps[0] = ps[0] + ps[k];
    }
    ps[0]
}

#[derive(Clone, Copy, Default)]
struct V4([f32; LANES]);

impl std::ops::Add for V4 {
    type Output = V4;
    fn add(self, o: V4) -> V4 {
        let mut r = self.0;
        for (a, b) in r.iter_mut().zip(o.0) {
            *a += b;
        }
        V4(r)
    }
}

/// `x.sum()` for a contiguous float32 tensor (`vectorized_inner_sum`).
pub fn sum_f32(x: &[f32]) -> f32 {
    inner_sum_f32(x.len(), |i| x[i])
}

/// [`sum_f32`] over `size` contiguous elements given by `value(i)` — e.g. one row
/// of `X.sum(-1)` read in place.
pub fn inner_sum_f32(size: usize, value: impl Fn(usize) -> f32) -> f32 {
    let vecs = size / LANES;
    let v = row_sum(vecs, |i| V4(std::array::from_fn(|l| value(i * LANES + l))));
    let mut acc = 0f32;
    for k in vecs * LANES..size {
        acc += value(k);
    }
    for l in v.0 {
        acc += l;
    }
    acc
}

/// One column of `multi_row_sum` (`cascade_sum`'s four-level accumulator) over
/// `size` rows: the reduction ATen runs for each output of a sum over a
/// non-contiguous dimension (see [`outer_sum_f32`]).
pub fn cascade_sum_f32(size: usize, value: impl Fn(usize) -> f32) -> f32 {
    const LEVELS: usize = 4;
    let level_power = 4usize.max(ceil_log2(size) / LEVELS);
    let level_step = 1usize << level_power;
    let level_mask = level_step - 1;
    let mut acc = [0f32; LEVELS];
    let mut i = 0;
    while i + level_step <= size {
        for _ in 0..level_step {
            acc[0] += value(i);
            i += 1;
        }
        for j in 1..LEVELS {
            acc[j] += acc[j - 1];
            acc[j - 1] = 0.0;
            if i & (level_mask << (j * level_power)) != 0 {
                break;
            }
        }
    }
    while i < size {
        acc[0] += value(i);
        i += 1;
    }
    for j in 1..LEVELS {
        acc[0] += acc[j];
    }
    acc[0]
}

/// ATen's scalar `row_sum` over `size` values `value(i)`: four interleaved
/// [`cascade_sum_f32`] accumulators, the tail into the first, then
/// `((a0 + a1) + a2) + a3`.
pub fn row_sum_f32(size: usize, value: impl Fn(usize) -> f32) -> f32 {
    row_sum(size, value)
}

/// Output `j` of `X.sum(dim)` over a non-contiguous dimension of length `size`
/// with `n_out` contiguous outputs (`vectorized_outer_sum`, four-lane vectors):
/// the first `⌊n_out / 16⌋ · 16` outputs are reduced by [`cascade_sum_f32`], the
/// rest by [`row_sum_f32`].
pub fn outer_sum_f32(size: usize, j: usize, n_out: usize, value: impl Fn(usize) -> f32) -> f32 {
    if j < n_out / 16 * 16 {
        cascade_sum_f32(size, value)
    } else {
        row_sum_f32(size, value)
    }
}

#[derive(Clone, Copy, Default)]
struct V2([f64; 2]);

impl std::ops::Add for V2 {
    type Output = V2;
    fn add(self, o: V2) -> V2 {
        V2([self.0[0] + o.0[0], self.0[1] + o.0[1]])
    }
}

/// `x.sum()` for a contiguous float64 tensor — [`sum_f32`] with
/// `Vectorized<double>`'s two lanes.
pub fn sum_f64(x: &[f64]) -> f64 {
    let n = x.len();
    let vecs = n / 2;
    let v = row_sum(vecs, |i| V2([x[2 * i], x[2 * i + 1]]));
    let mut acc = 0f64;
    for &t in &x[vecs * 2..] {
        acc += t;
    }
    acc + v.0[0] + v.0[1]
}

/// `x.mean()` for a contiguous float32 tensor: [`sum_f32`] divided by `n`.
pub fn mean_f32(x: &[f32]) -> f32 {
    sum_f32(x) / x.len() as f32
}

/// The total of a strided 2-D view, e.g. `d2_sorted[:, 1:k+1].sum()`:
/// ATen reduces each (contiguous) row — vectorised when it has at least four
/// elements, as a scalar `row_sum` when shorter — and accumulates the row
/// results into the output one row at a time.
pub fn sum_rows_f32<'a>(rows: impl IntoIterator<Item = &'a [f32]>) -> f32 {
    let mut out = 0f32;
    for r in rows {
        out += if r.len() >= LANES {
            sum_f32(r)
        } else {
            row_sum(r.len(), |i| r[i])
        };
    }
    out
}

/// [`sum_rows_f32`] for float64 (vectorised from two elements).
pub fn sum_rows_f64<'a>(rows: impl IntoIterator<Item = &'a [f64]>) -> f64 {
    let mut out = 0f64;
    for r in rows {
        out += if r.len() >= 2 {
            sum_f64(r)
        } else {
            row_sum(r.len(), |i| r[i])
        };
    }
    out
}

/// `torch.exp` on a float32 CPU tensor (arm64): SLEEF's `xexpf` (u10), which
/// torch's `Vectorized<float>::exp` calls for every element — 0-d tensors and
/// vector tails included. It differs from libm's `expf` in about one value
/// in ten (measured on 2000 values in `[-3, 0]`).
// Constants are SLEEF's source literals, kept verbatim for traceability.
#[allow(clippy::approx_constant, clippy::excessive_precision)]
pub fn exp_f32(d: f32) -> f32 {
    const R_LN2F: f32 = 1.442_695_f32;
    const L2UF: f32 = 0.693_145_751_953_125_f32;
    const L2LF: f32 = 1.428_606_765_330_187_045e-6_f32;
    if d.is_nan() {
        return d;
    }
    let q = (d * R_LN2F).round_ties_even() as i32;
    let qf = q as f32;
    let mut s = qf.mul_add(-L2UF, d);
    s = qf.mul_add(-L2LF, s);
    let mut u = 0.000_198_527_617_612_853_646_278_381_f32;
    u = u.mul_add(s, 0.001_393_043_552_525_341_510_772_71);
    u = u.mul_add(s, 0.008_333_360_776_305_198_669_433_59);
    u = u.mul_add(s, 0.041_666_485_369_205_474_853_515_6);
    u = u.mul_add(s, 0.166_666_671_633_720_397_949_219);
    u = u.mul_add(s, 0.5);
    u = 1.0 + (s * s).mul_add(u, s);
    // vldexp2: u · 2^(q>>1) · 2^(q − (q>>1))
    let pow2i = |e: i32| f32::from_bits(((e + 0x7f) as u32) << 23);
    u = u * pow2i(q >> 1) * pow2i(q - (q >> 1));
    if d < -104.0 {
        u = 0.0;
    }
    if d > 100.0 {
        u = f32::INFINITY;
    }
    u
}

// ── torch.pow: SLEEF's xpowf ───────────────────────────────────────────────────
//
// Sleef's `xpowf` (u10, AdvSIMD build with FMA — the revision torch 2.10 vendors,
// `shibatch/sleef@5a1d179`), lane-scalar: what `Vectorized<float>::pow` computes for
// the vectorised part of torch's `pow(Tensor, Scalar)` on arm64. Not correctly
// rounded (1.7 % of `fl(k/n)^4` differ from the rounded exact value), so the
// double-float algorithm is reproduced operation for operation. Constants are
// Sleef's literals, kept verbatim.

/// `torch.pow(x, y)` for one float32 element on the vectorised path
/// (`Sleef_powf_u10`).
pub fn pow_f32(x: f32, y: f32) -> f32 {
    powf_u10(x, y)
}

#[allow(clippy::excessive_precision, clippy::approx_constant)]
mod sleef_pow {
    #[derive(Clone, Copy)]
    struct F2 {
        x: f32,
        y: f32,
    }

    #[inline]
    fn fma(a: f32, b: f32, c: f32) -> f32 {
        a.mul_add(b, c)
    }
    /// `vfmapn(x, y, z) = x·y − z`.
    #[inline]
    fn fmapn(x: f32, y: f32, z: f32) -> f32 {
        x.mul_add(y, -z)
    }
    /// `vfmanp(x, y, z) = z − x·y`.
    #[inline]
    fn fmanp(x: f32, y: f32, z: f32) -> f32 {
        (-x).mul_add(y, z)
    }

    fn dfnormalize(t: F2) -> F2 {
        let s = t.x + t.y;
        F2 {
            x: s,
            y: (t.x - s) + t.y,
        }
    }
    fn dfscale(d: F2, s: f32) -> F2 {
        F2 {
            x: d.x * s,
            y: d.y * s,
        }
    }
    fn dfadd2_ff(x: f32, y: f32) -> F2 {
        let s = x + y;
        let v = s - x;
        F2 {
            x: s,
            y: (x - (s - v)) + (y - v),
        }
    }
    fn dfadd2_f2f(x: F2, y: f32) -> F2 {
        let s = x.x + y;
        let v = s - x.x;
        let t = (x.x - (s - v)) + (y - v);
        F2 { x: s, y: t + x.y }
    }
    fn dfadd_ff2(x: f32, y: F2) -> F2 {
        let s = x + y.x;
        F2 {
            x: s,
            y: ((x - s) + y.x) + y.y,
        }
    }
    fn dfadd_f2f2(x: F2, y: F2) -> F2 {
        let s = x.x + y.x;
        F2 {
            x: s,
            y: (((x.x - s) + y.x) + x.y) + y.y,
        }
    }
    fn dfadd2_f2f2(x: F2, y: F2) -> F2 {
        let s = x.x + y.x;
        let v = s - x.x;
        let t = (x.x - (s - v)) + (y.x - v);
        F2 {
            x: s,
            y: t + (x.y + y.y),
        }
    }
    fn dfdiv(n: F2, d: F2) -> F2 {
        let t = 1.0f32 / d.x;
        let s = n.x * t;
        let u = fmapn(t, n.x, s);
        let v = fmanp(d.y, t, fmanp(d.x, t, 1.0));
        F2 {
            x: s,
            y: fma(s, v, fma(n.y, t, u)),
        }
    }
    fn dfsqu(x: F2) -> F2 {
        let s = x.x * x.x;
        F2 {
            x: s,
            y: fma(x.x + x.x, x.y, fmapn(x.x, x.x, s)),
        }
    }
    fn dfmul_f2f2(x: F2, y: F2) -> F2 {
        let s = x.x * y.x;
        F2 {
            x: s,
            y: fma(x.x, y.y, fma(x.y, y.x, fmapn(x.x, y.x, s))),
        }
    }
    fn dfmul_f2f(x: F2, y: f32) -> F2 {
        let s = x.x * y;
        F2 {
            x: s,
            y: fma(x.y, y, fmapn(x.x, y, s)),
        }
    }

    fn ilogb2k(d: f32) -> i32 {
        (((d.to_bits() >> 23) & 0xff) as i32) - 0x7f
    }
    fn ldexp3(d: f32, q: i32) -> f32 {
        f32::from_bits((d.to_bits() as i32).wrapping_add(q << 23) as u32)
    }
    fn ldexp(x: f32, q: i32) -> f32 {
        let mut m = q >> 31;
        m = (((m + q) >> 6) - m) << 4;
        let q = q - (m << 2);
        m += 0x7f;
        m = if m > 0 { m } else { 0 };
        m = if m > 0xff { 0xff } else { m };
        let u = f32::from_bits((m << 23) as u32);
        let x = x * u * u * u * u;
        let u = f32::from_bits(((q + 0x7f) << 23) as u32);
        x * u
    }

    fn logkf(d: f32) -> F2 {
        let o = d < f32::MIN_POSITIVE;
        let d = if o {
            d * ((1u64 << 32) as f32 * (1u64 << 32) as f32)
        } else {
            d
        };
        let mut e = ilogb2k(d * (1.0f32 / 0.75f32));
        let m = ldexp3(d, -e);
        if o {
            e -= 64;
        }
        let x = dfdiv(dfadd2_ff(-1.0, m), dfadd2_ff(1.0, m));
        let x2 = dfsqu(x);
        let mut t = 0.240320354700088500976562f32;
        t = fma(t, x2.x, 0.285112679004669189453125f32);
        t = fma(t, x2.x, 0.400007992982864379882812f32);
        let c = F2 {
            x: 0.66666662693023681640625f32,
            y: 3.69183861259614332084311e-09f32,
        };
        let mut s = dfmul_f2f(
            F2 {
                x: 0.69314718246459960938f32,
                y: -1.904654323148236017e-09f32,
            },
            e as f32,
        );
        s = dfadd_f2f2(s, dfscale(x, 2.0));
        s = dfadd_f2f2(
            s,
            dfmul_f2f2(dfmul_f2f2(x2, x), dfadd2_f2f2(dfmul_f2f(x2, t), c)),
        );
        s
    }

    fn expkf(d: F2) -> f32 {
        const R_LN2F: f32 = 1.442695040888963407359924681001892137426645954152985934135449406931f32;
        const L2UF: f32 = 0.693145751953125f32;
        const L2LF: f32 = 1.428606765330187045e-06f32;
        let u = (d.x + d.y) * R_LN2F;
        let q = u.round_ties_even() as i32;
        let mut s = dfadd2_f2f(d, q as f32 * -L2UF);
        s = dfadd2_f2f(s, q as f32 * -L2LF);
        s = dfnormalize(s);
        let mut u = 0.00136324646882712841033936f32;
        u = fma(u, s.x, 0.00836596917361021041870117f32);
        u = fma(u, s.x, 0.0416710823774337768554688f32);
        u = fma(u, s.x, 0.166665524244308471679688f32);
        u = fma(u, s.x, 0.499999850988388061523438f32);
        let mut t = dfadd_f2f2(s, dfmul_f2f(dfsqu(s), u));
        t = dfadd_ff2(1.0, t);
        let mut u = t.x + t.y;
        u = ldexp(u, q);
        if d.x < -104.0 { 0.0 } else { u }
    }

    /// `Sleef_powf_u10` for one lane (see [`pow_f32`]).
    pub fn powf_u10(x: f32, y: f32) -> f32 {
        let yisint = y.trunc() == y || y.abs() > (1 << 24) as f32;
        let yisodd = ((y as i32) & 1) == 1 && yisint && y.abs() < (1 << 24) as f32;
        let mut result = expkf(dfmul_f2f(logkf(x.abs()), y));
        if result.is_nan() {
            result = f32::INFINITY;
        }
        result *= if x > 0.0 {
            1.0
        } else if yisint {
            if yisodd { -1.0 } else { 1.0 }
        } else {
            f32::NAN
        };
        let efx = (x.abs() - 1.0).copysign(y);
        if y.is_infinite() {
            result = if efx < 0.0 {
                0.0
            } else if efx == 0.0 {
                1.0
            } else {
                f32::INFINITY
            };
        }
        if x.is_infinite() || x == 0.0 {
            // mulsign(signbit(y) ^ (x == 0) ? 0 : ∞, yisodd ? x : 1)
            let mag = if y.is_sign_negative() ^ (x == 0.0) {
                0.0f32
            } else {
                f32::INFINITY
            };
            let b = if yisodd { x } else { 1.0 };
            result = if b.is_sign_negative() { -mag } else { mag };
        }
        if x.is_nan() || y.is_nan() {
            result = f32::NAN;
        }
        if y == 0.0 || x == 1.0 {
            result = 1.0;
        }
        result
    }

    /// torch CPU `pow(t, exp)` for a float32 tensor and a scalar exponent outside the
    /// 2 / 3 / −2 / ±0.5 / −1 fast paths: `cpu_kernel_vec` runs the vectorised Sleef
    /// kernel over whole 8-element blocks (two 4-lane vectors) and `std::pow(float,
    /// double)` on the remainder.
    pub fn pow_scalar_f32(x: &[f32], exp: f64) -> Vec<f32> {
        let vec_end = x.len() / 8 * 8;
        x.iter()
            .enumerate()
            .map(|(i, &v)| {
                if i < vec_end {
                    powf_u10(v, exp as f32)
                } else {
                    (v as f64).powf(exp) as f32
                }
            })
            .collect()
    }
}
pub use sleef_pow::{pow_scalar_f32, powf_u10};

#[cfg(test)]
mod tests {
    use super::*;

    /// `torch.exp(x)` for 710 float32 inputs (torch 2.11, macOS arm64), as hex
    /// pairs `x y`.
    #[test]
    fn exp_matches_torch() {
        let mut bad = 0;
        for line in include_str!("torch_exp_ref.txt").lines() {
            let mut it = line.split(' ').map(|h| u32::from_str_radix(h, 16).unwrap());
            let (x, y) = (f32::from_bits(it.next().unwrap()), it.next().unwrap());
            if exp_f32(x).to_bits() != y {
                bad += 1;
                eprintln!(
                    "exp({x:e}) = {:e}, torch {:e}",
                    exp_f32(x),
                    f32::from_bits(y)
                );
            }
        }
        assert_eq!(bad, 0);
    }

    /// `torch.pow(x, 4.0)` on the vectorised path (torch 2.10, macOS arm64): 15 318
    /// inputs, including where torch differs from the correctly rounded result and
    /// special values, as hex pairs `x y`.
    #[test]
    fn powf_matches_torch() {
        let mut bad = 0;
        for l in include_str!("torch_pow4_ref.txt").lines() {
            let mut it = l.split(' ').map(|h| u32::from_str_radix(h, 16).unwrap());
            let (x, y) = (f32::from_bits(it.next().unwrap()), it.next().unwrap());
            if powf_u10(x, 4.0).to_bits() != y {
                bad += 1;
            }
        }
        assert_eq!(bad, 0);
    }

    /// The per-output cascade spellings agree with the contiguous one.
    #[test]
    fn closure_sums_agree() {
        let x: Vec<f32> = (0..1000)
            .map(|i| ((i * 7919 % 101) as f32 - 50.0) * 0.0137)
            .collect();
        assert_eq!(
            inner_sum_f32(x.len(), |i| x[i]).to_bits(),
            sum_f32(&x).to_bits()
        );
        assert_eq!(
            row_sum_f32(x.len(), |i| x[i]).to_bits(),
            row_sum(x.len(), |i| x[i]).to_bits()
        );
        assert_eq!(
            outer_sum_f32(37, 3, 20, |i| x[i]).to_bits(),
            cascade_sum_f32(37, |i| x[i]).to_bits()
        );
        assert_eq!(
            outer_sum_f32(37, 17, 20, |i| x[i]).to_bits(),
            row_sum_f32(37, |i| x[i]).to_bits()
        );
    }

    /// `torch.tensor(x).sum()` from torch 2.11, macOS arm64, as hex.
    #[test]
    fn matches_torch_examples() {
        let x: Vec<f32> = (0..37)
            .map(|i| ((i * 7919 % 101) as f32 - 50.0) * 0.137)
            .collect();
        assert_eq!(sum_f32(&x).to_bits(), 0x3f0c49b8);
        let y: Vec<f32> = (0..300).map(|i| 1.0 / (1.0 + i as f32)).collect();
        assert_eq!(sum_f32(&y).to_bits(), 0x40c90b94);
        let z: Vec<f64> = (0..301).map(|i| 1.0 / (3.0 + i as f64)).collect();
        assert_eq!(sum_f64(&z).to_bits(), 0x40132b9ebd1e3a0f);
    }
}
