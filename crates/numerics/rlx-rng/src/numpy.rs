// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! numpy's two generators.
//!
//! `RandomState` is the legacy MT19937 path (what `sklearn` still uses through
//! `check_random_state`); `Generator`/PCG64 is the modern one. They are not
//! interchangeable — same distribution, different stream.

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

    fn genrand_u32(&mut self) -> u32 {
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
    pub fn standard_normal(&mut self) -> f64 {
        if self.has_gauss {
            self.has_gauss = false;
            return self.gauss;
        }
        loop {
            let x1 = 2.0 * self.random_double() - 1.0;
            let x2 = 2.0 * self.random_double() - 1.0;
            let r2 = x1 * x1 + x2 * x2;
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

/// sklearn's `GaussianRandomProjection._make_random_matrix` — the `[k, d]`
/// matrix with entries `N(0, 1/k)`, i.e. scale `1/sqrt(k)`.
pub fn gaussian_random_matrix(n_components: usize, n_features: usize, seed: u32) -> Vec<f64> {
    let mut rs = RandomState::new(seed);
    let scale = 1.0 / (n_components as f64).sqrt();
    rs.normal(0.0, scale, n_components * n_features)
}

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;
const POOL_SIZE: usize = 4;

/// numpy's `SeedSequence` over a single non-negative integer entropy value.
pub struct SeedSequence {
    pool: [u32; POOL_SIZE],
}

impl SeedSequence {
    pub fn new(entropy: u128) -> Self {
        // Entropy is decomposed into 32-bit words, little end first.
        let mut words: Vec<u32> = Vec::new();
        let mut e = entropy;
        if e == 0 {
            words.push(0);
        }
        while e > 0 {
            words.push((e & 0xffff_ffff) as u32);
            e >>= 32;
        }
        let mut pool = [0u32; POOL_SIZE];
        let mut hash_const = INIT_A;
        let hashmix = |value: u32, hc: &mut u32| -> u32 {
            let mut v = value ^ *hc;
            *hc = hc.wrapping_mul(MULT_A);
            v = v.wrapping_mul(*hc);
            v ^= v >> XSHIFT;
            v
        };
        let mix = |x: u32, y: u32| -> u32 {
            let mut r = MIX_MULT_L
                .wrapping_mul(x)
                .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
            r ^= r >> XSHIFT;
            r
        };

        for i in 0..POOL_SIZE {
            let v = words.get(i).copied().unwrap_or(0);
            pool[i] = hashmix(v, &mut hash_const);
        }
        for i_src in 0..POOL_SIZE {
            for i_dst in 0..POOL_SIZE {
                if i_src != i_dst {
                    let h = hashmix(pool[i_src], &mut hash_const);
                    pool[i_dst] = mix(pool[i_dst], h);
                }
            }
        }
        for i_src in POOL_SIZE..words.len() {
            for i_dst in 0..POOL_SIZE {
                let h = hashmix(words[i_src], &mut hash_const);
                pool[i_dst] = mix(pool[i_dst], h);
            }
        }
        Self { pool }
    }

    /// `generate_state(n, dtype=uint32)`.
    pub fn generate_state(&self, n: usize) -> Vec<u32> {
        let mut hash_const = INIT_B;
        (0..n)
            .map(|i| {
                let mut v = self.pool[i % POOL_SIZE];
                v ^= hash_const;
                hash_const = hash_const.wrapping_mul(MULT_B);
                v = v.wrapping_mul(hash_const);
                v ^= v >> XSHIFT;
                v
            })
            .collect()
    }
}

const PCG_MULT: u128 = 0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645;

/// PCG64 (XSL-RR 128/64).
/// .
pub struct Generator {
    state: u128,
    inc: u128,
    /// PCG64 serves 32-bit requests by splitting one 64-bit draw and holding
    /// the high half for the next call. Drawing a fresh u64 each time would
    /// consume the stream twice as fast and diverge after the first value.
    buffered: Option<u32>,
}

impl Generator {
    /// `np.random.PCG64(seed)`.
    pub fn new(seed: u128) -> Self {
        let w = SeedSequence::new(seed).generate_state(8);
        let u64s: Vec<u64> = w
            .chunks(2)
            .map(|c| (c[1] as u64) << 32 | c[0] as u64)
            .collect();
        // `pcg64_set_seed` reads `seed[0]` as the HIGH word:
        //     s = ((pcg128_t)seed[0] << 64) | seed[1]
        // Assembling it the other way round produces a valid PCG64 stream that
        // is not numpy's — the generator still looks random, so only comparing
        // against the reference catches it.
        let initstate = (u64s[0] as u128) << 64 | u64s[1] as u128;
        let initseq = (u64s[2] as u128) << 64 | u64s[3] as u128;
        let mut s = Self {
            state: 0,
            inc: (initseq << 1) | 1,
            buffered: None,
        };
        s.step();
        s.state = s.state.wrapping_add(initstate);
        s.step();
        s
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
    }

    /// One 32-bit output — LOW half first, then the buffered high half.
    pub fn next_u32(&mut self) -> u32 {
        if let Some(hi) = self.buffered.take() {
            return hi;
        }
        let v = self.next_u64();
        self.buffered = Some((v >> 32) as u32);
        v as u32
    }

    /// One 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let s = self.state;
        // XSL-RR: xor the halves, then rotate by the top 6 bits.
        let xored = ((s >> 64) as u64) ^ (s as u64);
        let rot = (s >> 122) as u32;
        xored.rotate_right(rot)
    }
}

/// `Generator.integers(0, high, size, dtype=int64)` — Lemire's multiply-shift.
///
/// `Generator` uses Lemire with rejection (`use_masked=False`); the older
/// `RandomState` uses masked rejection instead. They are both unbiased and they
/// consume the stream DIFFERENTLY, so picking the wrong one gives plausible
/// draws that diverge from numpy immediately — which is how this was caught.
///
/// ```text
/// m = next_u64() * rng_excl            (128-bit)
/// leftover = low 64 bits of m
/// if leftover < rng_excl:
///     threshold = (2^64 - 1 - rng) % rng_excl
///     while leftover < threshold: redraw
/// return high 64 bits of m
/// ```
pub fn integers(g: &mut Generator, high: u64, size: usize) -> Vec<u64> {
    assert!(high > 0, "high must be positive");
    let rng = high - 1;
    if rng == 0 {
        return vec![0; size];
    }
    // numpy dispatches to a 32-BIT generator whenever the range fits in 32
    // bits, which is the usual case here (bucket counts are small). The 64-bit
    // path consumes the stream at a different rate, so choosing the wrong one
    // matches for a draw or two and then diverges.
    if rng <= u32::MAX as u64 {
        let rng32 = rng as u32;
        let rng_excl = rng32 as u64 + 1;
        return (0..size)
            .map(|_| {
                let mut m = (g.next_u32() as u64) * rng_excl;
                let mut leftover = m as u32;
                if (leftover as u64) < rng_excl {
                    let threshold = ((u32::MAX - rng32) as u64 % rng_excl) as u32;
                    while leftover < threshold {
                        m = (g.next_u32() as u64) * rng_excl;
                        leftover = m as u32;
                    }
                }
                m >> 32
            })
            .collect();
    }

    let rng_excl = rng as u128 + 1;
    (0..size)
        .map(|_| {
            let mut m = (g.next_u64() as u128) * rng_excl;
            let mut leftover = m as u64;
            if (leftover as u128) < rng_excl {
                let threshold = ((u64::MAX - rng) as u128 % rng_excl) as u64;
                while leftover < threshold {
                    m = (g.next_u64() as u128) * rng_excl;
                    leftover = m as u64;
                }
            }
            (m >> 64) as u64
        })
        .collect()
}

/// `Generator.choice([-1.0, 1.0], size)` — uniform over two items with
/// replacement, which numpy implements as `integers(0, 2, size)`.
pub fn choice_pm1(g: &mut Generator, size: usize) -> Vec<f32> {
    integers(g, 2, size)
        .into_iter()
        .map(|i| if i == 0 { -1.0 } else { 1.0 })
        .collect()
}

/// EEG-FM-Bench's `_deterministic_hash` — FNV-1a over `>qqI` + key bytes.
///
/// Deliberately not Python's `hash()`, which is salted by `PYTHONHASHSEED` and
/// so differs between runs; this is the reason the projection is reproducible
/// at all.
pub fn deterministic_hash(seed: i64, key: &str, length: i64) -> u64 {
    let mut data = Vec::with_capacity(20 + key.len());
    data.extend_from_slice(&seed.to_be_bytes());
    data.extend_from_slice(&length.to_be_bytes());
    data.extend_from_slice(&(key.len() as u32).to_be_bytes());
    data.extend_from_slice(key.as_bytes());

    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The bucket assignment and signs `HashingProjector` builds for `(key, length)`.
pub fn hashing_projection(
    seed: i64,
    key: &str,
    length: usize,
    proj_dim: usize,
) -> (Vec<u64>, Vec<f32>) {
    let mixed = deterministic_hash(seed, key, length as i64);
    let mut g = Generator::new(mixed as u128);
    let buckets = integers(&mut g, proj_dim.max(1) as u64, length);
    let signs = choice_pm1(&mut g, length);
    (buckets, signs)
}
