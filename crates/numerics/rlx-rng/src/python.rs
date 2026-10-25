// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! CPython's `random` module (`Modules/_randommodule.c` + `Lib/random.py`).
//!
//! The same MT19937 as numpy and torch, but **seeded differently**: `random.seed(n)`
//! feeds the 32-bit words of `|n|` to `init_by_array`, where numpy's
//! `RandomState(n)` and torch's `manual_seed(n)` call `init_genrand(n)` directly —
//! so one integer seed gives three different streams.
//!
//! Integer draws reject rather than scale: `_randbelow(n)` takes
//! `getrandbits(n.bit_length())` until the value is below `n`, so `randint` and
//! `choice` consume a data-dependent number of words.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// `random.Random`.
#[derive(Clone)]
pub struct Random {
    mt: [u32; N],
    mti: usize,
}

impl Random {
    /// `random.Random(seed)` / `random.seed(seed)` for an integer seed.
    pub fn new(seed: u64) -> Self {
        // `random_seed`: the absolute value's 32-bit words, least significant
        // first; zero is the single word 0.
        let mut key = vec![seed as u32];
        if seed >> 32 != 0 {
            key.push((seed >> 32) as u32);
        }
        Self::from_key(&key)
    }

    /// `init_by_array(key)`.
    pub fn from_key(key: &[u32]) -> Self {
        let mut r = Self::init_genrand(19_650_218);
        let mt = &mut r.mt;
        let (mut i, mut j) = (1usize, 0usize);
        for _ in 0..N.max(key.len()) {
            mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1_664_525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..N - 1 {
            mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1_566_083_941))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
        }
        mt[0] = 0x8000_0000;
        r
    }

    fn init_genrand(s: u32) -> Self {
        let mut mt = [0u32; N];
        mt[0] = s;
        for i in 1..N {
            mt[i] = 1_812_433_253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { mt, mti: N }
    }

    /// `genrand_uint32`.
    pub fn genrand_u32(&mut self) -> u32 {
        if self.mti >= N {
            let mag = |y: u32| if y & 1 != 0 { MATRIX_A } else { 0 };
            for k in 0..N - M {
                let y = (self.mt[k] & UPPER_MASK) | (self.mt[k + 1] & LOWER_MASK);
                self.mt[k] = self.mt[k + M] ^ (y >> 1) ^ mag(y);
            }
            for k in N - M..N - 1 {
                let y = (self.mt[k] & UPPER_MASK) | (self.mt[k + 1] & LOWER_MASK);
                self.mt[k] = self.mt[k + M - N] ^ (y >> 1) ^ mag(y);
            }
            let y = (self.mt[N - 1] & UPPER_MASK) | (self.mt[0] & LOWER_MASK);
            self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ mag(y);
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

    /// `random.random()`: 53 bits, `(a·2²⁶ + b) / 2⁵³` with `a` 27 and `b` 26 bits.
    pub fn random(&mut self) -> f64 {
        let a = (self.genrand_u32() >> 5) as f64;
        let b = (self.genrand_u32() >> 6) as f64;
        (a * 67_108_864.0 + b) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// `random.getrandbits(k)` for `k ≤ 64` (words least significant first; the
    /// last word keeps its top bits).
    pub fn getrandbits(&mut self, k: u32) -> u64 {
        assert!(k <= 64, "getrandbits: k = {k} > 64 unsupported");
        if k == 0 {
            return 0;
        }
        if k <= 32 {
            return (self.genrand_u32() >> (32 - k)) as u64;
        }
        let lo = self.genrand_u32() as u64;
        let hi = (self.genrand_u32() >> (64 - k)) as u64;
        lo | (hi << 32)
    }

    /// `Random._randbelow(n)` (`_randbelow_with_getrandbits`), `n ≥ 1`.
    pub fn randbelow(&mut self, n: u64) -> u64 {
        assert!(n > 0, "randbelow(0)");
        let k = 64 - n.leading_zeros();
        loop {
            let r = self.getrandbits(k);
            if r < n {
                return r;
            }
        }
    }

    /// `random.randint(a, b)` (inclusive).
    pub fn randint(&mut self, a: i64, b: i64) -> i64 {
        assert!(b >= a, "randint({a}, {b}): empty range");
        a + self.randbelow((b - a + 1) as u64) as i64
    }

    /// Index `random.choice(seq)` picks for a sequence of length `len`.
    pub fn choice_index(&mut self, len: usize) -> usize {
        self.randbelow(len as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Measured on CPython 3.14: `r = random.Random(42)`, three `r.random()`,
    /// `r.randint(1, 50)`, `r.getrandbits(40)`, `r.choice(range(7))`,
    /// `r.randint(0, 1000)`; then a two-word seed and seed 0.
    #[test]
    fn matches_cpython() {
        let mut r = Random::new(42);
        assert_eq!(r.random().to_bits(), 0.6394267984578837f64.to_bits());
        assert_eq!(r.random().to_bits(), 0.025010755222666936f64.to_bits());
        assert_eq!(r.random().to_bits(), 0.27502931836911926f64.to_bits());
        assert_eq!(r.randint(1, 50), 15);
        assert_eq!(r.getrandbits(40), 808_053_162_473);
        assert_eq!(r.choice_index(7), 0);
        assert_eq!(r.randint(0, 1000), 692);
        let mut r = Random::new((1 << 40) + 5);
        assert_eq!(r.random().to_bits(), 0.5043802970418443f64.to_bits());
        assert_eq!(r.randint(0, 3), 2);
        assert_eq!(
            Random::new(0).random().to_bits(),
            0.8444218515250481f64.to_bits()
        );
    }
}
