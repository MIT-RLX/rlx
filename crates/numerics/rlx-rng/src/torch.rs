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
    /// ATen's `next_double_normal_sample`: `normal_distribution<double>`
    /// returns `r·cos θ` and caches `r·sin θ` for the next call.
    torch_normal_cache: Option<f64>,
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
            torch_normal_cache: None,
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

/// `sincosf(x)` → `(sin, cos)` as the platform's merged call computes it.
#[cfg(target_os = "macos")]
fn sincosf(x: f32) -> (f32, f32) {
    #[repr(C)]
    struct SinCos {
        s: f32,
        c: f32,
    }
    unsafe extern "C" {
        fn __sincosf_stret(x: f32) -> SinCos;
    }
    // SAFETY: a pure libm function returning a two-float struct by value.
    let r = unsafe { __sincosf_stret(x) };
    (r.s, r.c)
}

/// `sincosf(x)` → `(sin, cos)` as the platform's merged call computes it.
#[cfg(not(target_os = "macos"))]
fn sincosf(x: f32) -> (f32, f32) {
    unsafe extern "C" {
        fn sincosf(x: f32, s: *mut f32, c: *mut f32);
    }
    let (mut s, mut c) = (0.0f32, 0.0f32);
    // SAFETY: libm's `sincosf` writes the two outputs.
    unsafe { sincosf(x, &mut s, &mut c) };
    (s, c)
}

impl RandomState {
    /// ATen `CPUGeneratorImpl::random64`: two 32-bit draws, high word first.
    pub fn random64(&mut self) -> u64 {
        let hi = self.genrand_u32() as u64;
        let lo = self.genrand_u32() as u64;
        (hi << 32) | lo
    }

    /// One element of `tensor.uniform_(from, to)` on a float32 CPU tensor:
    /// `from`/`to` are cast to float, `x = (random() & (2²⁴−1))·2⁻²⁴` and
    /// `x·(to − from) + from` is formed in double (`dist_acctype<float>`),
    /// then stored as float.
    pub fn uniform_f32(&mut self, from: f64, to: f64) -> f32 {
        let (f, t) = (from as f32, to as f32);
        let x = (self.genrand_u32() & 0x00FF_FFFF) as f64 / 16_777_216.0;
        (x * (t - f) as f64 + f as f64) as f32
    }

    /// `torch.empty(n).uniform_(from, to)` (contiguous, serial order).
    pub fn uniform_tensor_f32(&mut self, n: usize, from: f64, to: f64) -> Vec<f32> {
        (0..n).map(|_| self.uniform_f32(from, to)).collect()
    }

    /// `uniform_real_distribution<double>(0, 1)`: 53 bits of `random64()`.
    fn uniform_f64(&mut self) -> f64 {
        (self.random64() & ((1u64 << 53) - 1)) as f64 / 9_007_199_254_740_992.0
    }

    /// One element of `tensor.bernoulli_(p)` on a CPU tensor without MKL (the
    /// default kernel, e.g. arm64): `uniform_real_distribution<double>(0,1) < p`,
    /// one `random64()` per element in memory order. This is the draw
    /// `F.dropout(x, q, training=True)` makes for its keep mask with `p = 1 − q`.
    pub fn bernoulli(&mut self, p: f64) -> bool {
        self.uniform_f64() < p
    }

    /// `tensor.normal_(mean, std)` on a contiguous float32 CPU tensor, the
    /// portable (non-AVX2) kernel. With ≥ 16 elements, `normal_fill`: fill
    /// with float uniforms, Box–Muller in blocks of 16 pairing `j` with
    /// `j + 8` (`u1 = 1 − data[j]`), and when `n % 16 ≠ 0` redraw and redo the
    /// LAST 16. Fewer than 16: `normal_distribution<double>` per element
    /// (`r = √(−2·ln(1 − u2))`, `θ = 2π·u1`, return `r·cos θ`).
    ///
    /// x86 builds with AVX2 use a vectorised `normal_fill_AVX2` whose
    /// transcendental approximations differ in the last bits.
    pub fn normal_tensor_f32(&mut self, n: usize, mean: f64, std: f64) -> Vec<f32> {
        let (m, sd) = (mean as f32, std as f32);
        if n >= 16 {
            let mut d: Vec<f32> = (0..n).map(|_| self.uniform_f32(0.0, 1.0)).collect();
            let fill16 = |d: &mut [f32]| {
                for j in 0..8 {
                    let u1 = 1.0f32 - d[j];
                    let u2 = d[j + 8];
                    let radius = (-2.0f32 * u1.ln()).sqrt();
                    // `2.0f * c10::pi<double> * u2` is a DOUBLE product, narrowed on assignment
                    let theta = (2.0f64 * std::f64::consts::PI * u2 as f64) as f32;
                    // ATen's `std::cos(theta)` / `std::sin(theta)` are merged into ONE
                    // `sincosf` call by clang -O3; on Darwin `__sincosf_stret` differs
                    // from separate `sinf`/`cosf` in the last bit (measured: 64 of 6464
                    // `torch.randn` values). Calling it explicitly makes debug builds
                    // agree with torch (and with release builds, where LLVM merges too).
                    let (s, c) = sincosf(theta);
                    d[j] = radius * c * sd + m;
                    d[j + 8] = radius * s * sd + m;
                }
            };
            let mut i = 0;
            while i + 16 <= n {
                fill16(&mut d[i..i + 16]);
                i += 16;
            }
            if !n.is_multiple_of(16) {
                let tail = &mut d[n - 16..];
                for v in tail.iter_mut() {
                    *v = self.uniform_f32(0.0, 1.0);
                }
                fill16(tail);
            }
            d
        } else {
            (0..n)
                .map(|_| {
                    let z = match self.torch_normal_cache.take() {
                        Some(z) => z,
                        None => {
                            let u1 = self.uniform_f64();
                            let u2 = self.uniform_f64();
                            let r = (-2.0 * (-u2).ln_1p()).sqrt();
                            let theta = 2.0 * std::f64::consts::PI * u1;
                            self.torch_normal_cache = Some(r * theta.sin());
                            r * theta.cos()
                        }
                    };
                    (z * std + mean) as f32
                })
                .collect()
        }
    }
}

impl RandomState {
    /// `tensor.normal_(mean, std)` on a contiguous float64 CPU tensor — the
    /// double instantiation of the same `normal_fill` as
    /// [`Self::normal_tensor_f32`]: uniforms are 53-bit `random64()` draws,
    /// Box–Muller runs in double, and `cos`/`sin` are again one merged
    /// `sincos` call. Fewer than 16 elements: `normal_distribution<double>`.
    pub fn normal_tensor_f64(&mut self, n: usize, mean: f64, std: f64) -> Vec<f64> {
        if n >= 16 {
            let mut d: Vec<f64> = (0..n).map(|_| self.uniform_f64()).collect();
            let fill16 = |d: &mut [f64]| {
                for j in 0..8 {
                    let u1 = 1.0 - d[j];
                    let u2 = d[j + 8];
                    let radius = (-2.0 * u1.ln()).sqrt();
                    let theta = 2.0 * std::f64::consts::PI * u2;
                    let (s, c) = sincos(theta);
                    d[j] = radius * c * std + mean;
                    d[j + 8] = radius * s * std + mean;
                }
            };
            let mut i = 0;
            while i + 16 <= n {
                fill16(&mut d[i..i + 16]);
                i += 16;
            }
            if !n.is_multiple_of(16) {
                let tail = &mut d[n - 16..];
                for v in tail.iter_mut() {
                    *v = self.uniform_f64();
                }
                fill16(tail);
            }
            d
        } else {
            (0..n)
                .map(|_| {
                    let z = match self.torch_normal_cache.take() {
                        Some(z) => z,
                        None => {
                            let u1 = self.uniform_f64();
                            let u2 = self.uniform_f64();
                            let r = (-2.0 * (-u2).ln_1p()).sqrt();
                            let theta = 2.0 * std::f64::consts::PI * u1;
                            self.torch_normal_cache = Some(r * theta.sin());
                            r * theta.cos()
                        }
                    };
                    z * std + mean
                })
                .collect()
        }
    }
}

/// `sincos(x)` → `(sin, cos)` as the platform's merged call computes it.
#[cfg(target_os = "macos")]
fn sincos(x: f64) -> (f64, f64) {
    #[repr(C)]
    struct SinCos {
        s: f64,
        c: f64,
    }
    unsafe extern "C" {
        fn __sincos_stret(x: f64) -> SinCos;
    }
    // SAFETY: a pure libm function returning a two-double struct by value.
    let r = unsafe { __sincos_stret(x) };
    (r.s, r.c)
}

/// `sincos(x)` → `(sin, cos)` as the platform's merged call computes it.
#[cfg(not(target_os = "macos"))]
fn sincos(x: f64) -> (f64, f64) {
    unsafe extern "C" {
        fn sincos(x: f64, s: *mut f64, c: *mut f64);
    }
    let (mut s, mut c) = (0.0f64, 0.0f64);
    // SAFETY: libm's `sincos` writes the two outputs.
    unsafe { sincos(x, &mut s, &mut c) };
    (s, c)
}

/// How one parameter is initialised by its module's `reset_parameters` (or
/// constructor), drawn from torch's global CPU generator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Init {
    /// `kaiming_uniform_(w, a=sqrt(5))` (Conv*/Linear weights).
    KaimingA5 { fan_in: usize },
    /// `uniform_(b, -1/sqrt(fan_in), 1/sqrt(fan_in))` (Conv*/Linear biases).
    FanInBias { fan_in: usize },
    /// `uniform_(-bound, bound)` with an explicit bound (e.g. LSTM: `1/sqrt(hidden)`).
    Uniform { bound: f64 },
    /// `ones_` / `zeros_` (BatchNorm/LayerNorm affine) — no draws.
    Const(f32),
    /// `nn.Parameter(torch.randn(...))` at construction (never reset).
    Randn,
}

impl Init {
    /// Draw this parameter's `numel` values (C order), consuming the generator as torch does.
    pub fn draw(self, g: &mut RandomState, numel: usize) -> Vec<f32> {
        match self {
            Init::KaimingA5 { fan_in } => {
                // gain = calculate_gain("leaky_relu", sqrt(5)); std = gain/sqrt(fan); bound = sqrt(3)*std
                let a = 5f64.sqrt();
                let gain = (2.0 / (1.0 + a.powi(2))).sqrt();
                let std = gain / (fan_in as f64).sqrt();
                let bound = 3f64.sqrt() * std;
                g.uniform_tensor_f32(numel, -bound, bound)
            }
            Init::FanInBias { fan_in } => {
                let bound = if fan_in > 0 {
                    1.0 / (fan_in as f64).sqrt()
                } else {
                    0.0
                };
                g.uniform_tensor_f32(numel, -bound, bound)
            }
            Init::Uniform { bound } => g.uniform_tensor_f32(numel, -bound, bound),
            Init::Const(c) => vec![c; numel],
            Init::Randn => g.normal_tensor_f32(numel, 0.0, 1.0),
        }
    }
}

/// `torch.empty((), dtype=torch.int64).random_().item()`: `random64() % 2^63`.
pub fn random_int64(g: &mut RandomState) -> u64 {
    g.random64() % (1u64 << 63)
}

/// One pass of `DataLoader(ds, batch_size, shuffle=True)` (no explicit
/// generator, `num_workers=0`): `iter(loader)` draws the iterator's base seed,
/// the first `next()` draws `RandomSampler`'s seed for a fresh generator, and
/// the order is `randperm(n)` on that generator.
pub fn epoch_permutation(g: &mut RandomState, n: usize) -> Vec<usize> {
    let _base_seed = random_int64(g);
    let seed = random_int64(g);
    // torch.Generator().manual_seed(seed): mt19937 takes the low 32 bits
    let mut sg = RandomState::new(seed as u32);
    sg.randperm(n)
}

/// `F.dropout(x, p, training=True)` on CPU (`_dropout`): a `bernoulli_(1 − p)`
/// keep mask over `numel` elements, divided by `(float)(1 − p)`.
pub fn dropout_mask(g: &mut RandomState, numel: usize, p: f64) -> Vec<f32> {
    let keep = 1.0 - p;
    let scale = 1.0f32 / keep as f32;
    (0..numel)
        .map(|_| if g.bernoulli(keep) { scale } else { 0.0 })
        .collect()
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

#[cfg(test)]
mod tensor_init_tests {
    use super::RandomState;

    fn hex(v: &[f32]) -> Vec<String> {
        v.iter().map(|x| format!("{:08x}", x.to_bits())).collect()
    }

    /// torch 2.5.1 (macOS arm64, portable kernels), one stream after
    /// `torch.manual_seed(7)`: `empty(5).uniform_(-0.3, 0.7)`, `randn(37)`
    /// (normal_fill + tail redraw), `randn(5)` (double Box–Muller),
    /// `empty(3).uniform_(-1, 1)`.
    #[test]
    fn uniform_and_normal_match_torch_cpu() {
        let mut g = RandomState::new(7);
        assert_eq!(
            hex(&g.uniform_tensor_f32(5, -0.3, 0.7)),
            ["3e708f88", "bdcf4048", "3eb7ea98", "3eb6ba52", "bd89b450"]
        );
        let b = hex(&g.normal_tensor_f32(37, 0.0, 1.0));
        assert_eq!(&b[..3], ["3e91ff2d", "bf241e93", "bf64c695"]);
        assert_eq!(&b[34..], ["bdbbc13d", "3fcfe99a", "3acd6ef5"]);
        assert_eq!(
            hex(&g.normal_tensor_f32(5, 0.0, 1.0)),
            ["3f980292", "bea4e20b", "3f9c3058", "3f42427a", "be3e24fb"]
        );
        assert_eq!(
            hex(&g.uniform_tensor_f32(3, -1.0, 1.0)),
            ["3c659180", "bec72900", "3f0c2a7a"]
        );
    }
}
