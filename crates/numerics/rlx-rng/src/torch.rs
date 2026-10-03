// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! PyTorch's CPU generator.
//!
//! MT19937 seeded with the same Knuth initialiser numpy uses — the two agree on
//! the raw stream — plus `randperm`, which is Fisher–Yates walking FORWARD with
//! `random() % (n - i)`. The backward variant consumes the same draws in a
//! different order and yields a different permutation.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// MT19937 with numpy's legacy Gaussian cache.
pub struct RandomState {
    mt: [u32; N],
    mti: usize,
    /// The polar method yields TWO variates per rejection loop; numpy returns
    /// one and caches the other. Dropping it desynchronises the whole stream
    /// after the first draw.
    gauss: f64,
    has_gauss: bool,
}

impl RandomState {
    /// `np.random.RandomState(seed)` for a non-negative scalar seed.
    pub fn new(seed: u32) -> Self {
        let mut mt = [0u32; N];
        mt[0] = seed;
        for i in 1..N {
            // Knuth's initialiser, as `mt19937_init_genrand`.
            mt[i] = 1812433253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self {
            mt,
            mti: N,
            gauss: 0.0,
            has_gauss: false,
        }
    }

    pub fn genrand_u32(&mut self) -> u32 {
        if self.mti >= N {
            for k in 0..(N - M) {
                let y = (self.mt[k] & UPPER_MASK) | (self.mt[k + 1] & LOWER_MASK);
                self.mt[k] = self.mt[k + M] ^ (y >> 1) ^ if y & 1 != 0 { MATRIX_A } else { 0 };
            }
            for k in (N - M)..(N - 1) {
                let y = (self.mt[k] & UPPER_MASK) | (self.mt[k + 1] & LOWER_MASK);
                self.mt[k] = self.mt[k + M - N] ^ (y >> 1) ^ if y & 1 != 0 { MATRIX_A } else { 0 };
            }
            let y = (self.mt[N - 1] & UPPER_MASK) | (self.mt[0] & LOWER_MASK);
            self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ if y & 1 != 0 { MATRIX_A } else { 0 };
            self.mti = 0;
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// numpy's `random_double`: 53 bits from two 32-bit draws.
    pub fn random_double(&mut self) -> f64 {
        let a = (self.genrand_u32() >> 5) as f64;
        let b = (self.genrand_u32() >> 6) as f64;
        (a * 67_108_864.0 + b) / 9_007_199_254_740_992.0
    }

    /// numpy's `legacy_gauss` — Marsaglia polar with a cached second variate.
    ///
    /// The rejection condition is `r2 >= 1.0 || r2 == 0.0`, and the returned
    /// variate is `f * x2` with `f * x1` cached. Returning `f*x1` first instead
    /// produces a valid Gaussian sequence that does not match numpy's.
    ///
    /// **`r2` is computed with a fused multiply-add.** numpy's C is
    /// `r2 = x1*x1 + x2*x2`, and clang contracts that into
    /// `fma(x1, x1, x2*x2)` within the statement (`-ffp-contract=on`, its default).
    /// An unfused sum differs from numpy in the last bit of `r2` often enough to
    /// matter: on seed 0 it diverges on 4 of the first 40 draws, always in pairs,
    /// because both variates from a loop share `f`. The draws are still a valid
    /// Gaussian stream either way — only the bit-exact match with
    /// `np.random.RandomState(seed).normal(...)` is lost, which is the entire point
    /// of this type.
    pub fn standard_normal(&mut self) -> f64 {
        if self.has_gauss {
            self.has_gauss = false;
            return self.gauss;
        }
        loop {
            let x1 = 2.0 * self.random_double() - 1.0;
            let x2 = 2.0 * self.random_double() - 1.0;
            let r2 = x1.mul_add(x1, x2 * x2);
            if r2 < 1.0 && r2 != 0.0 {
                let f = (-2.0 * r2.ln() / r2).sqrt();
                self.gauss = f * x1;
                self.has_gauss = true;
                return f * x2;
            }
        }
    }

    /// `rng.normal(loc, scale, size)` in C order.
    pub fn normal(&mut self, loc: f64, scale: f64, size: usize) -> Vec<f64> {
        (0..size)
            .map(|_| loc + scale * self.standard_normal())
            .collect()
    }
}

impl RandomState {
    /// One `torch.rand(...)` draw at torch's **default dtype, float32**.
    ///
    /// ATen's `uniform_real_distribution<float>` is
    /// `(random() & ((1 << 24) - 1)) * 2^-24` — 24 bits, the float32 mantissa
    /// width, out of one 32-bit word.
    ///
    /// **`torch.rand(n)` is float32 unless you ask otherwise**, and
    /// `torch.rand(1, dtype=torch.float64)` consumes the stream differently and
    /// returns a different value from the same seed (measured: 0.8822692632675171
    /// vs 0.05815448596142969 at seed 42). Porting a `torch.rand(1).item()` call
    /// site as an f64 draw gives plausible numbers that diverge immediately.
    pub fn rand_f32(&mut self) -> f32 {
        ((self.genrand_u32() & 0x00FF_FFFF) as f32) * (1.0f32 / 16_777_216.0)
    }

    /// `torch.rand(1).item()` — [`Self::rand_f32`] widened to `f64`, which is what
    /// `.item()` hands back to Python. The value has a float32's precision; the
    /// widening is exact.
    pub fn rand(&mut self) -> f64 {
        self.rand_f32() as f64
    }

    /// `torch.randperm(n)` **continuing this generator's stream**.
    ///
    /// Use this rather than [`randperm`] whenever the call site passes a
    /// `torch.Generator` that has already been drawn from — torch's `randperm`
    /// shares the generator with any preceding `rand`, so restarting from the seed
    /// gives a different permutation (measured at seed 42, n=6:
    /// `[5, 2, 4, 3, 0, 1]` continuing vs `[0, 3, 2, 4, 1, 5]` fresh).
    pub fn randperm(&mut self, n: usize) -> Vec<usize> {
        let mut r: Vec<usize> = (0..n).collect();
        if n < 2 {
            return r;
        }
        for i in 0..(n - 1) {
            let z = (self.genrand_u32() as usize) % (n - i);
            r.swap(i, z + i);
        }
        r
    }
}

/// `torch.randperm(n)` after `torch.manual_seed(seed)`.
///
/// Fisher–Yates walking FORWARD with `random() % (n - i)`, which is the
/// direction torch uses; the more common backward variant consumes the same
/// draws in a different order and yields a different permutation.
///
/// This starts a **fresh** generator. If the call site shares a generator with
/// earlier draws, use [`RandomState::randperm`] instead.
pub fn randperm(n: usize, seed: u32) -> Vec<usize> {
    RandomState::new(seed).randperm(n)
}

/// The reference's `_get_subsample_indices`: the first `k` of a permutation.
pub fn subsample_indices(n: usize, k: usize, seed: u32) -> Vec<usize> {
    let mut p = randperm(n, seed);
    p.truncate(k.min(n));
    p
}
