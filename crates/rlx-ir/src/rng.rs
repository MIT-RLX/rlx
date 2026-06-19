// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Counter-based and ONNX Runtime–compatible RNG for in-graph random ops.
//!
//! # Behavioral contract
//!
//! [`Op::RngNormal`] / [`Op::RngUniform`] take an optional shape-template input
//! (ONNX `Random*Like`) or no inputs when the output shape is fixed at import
//! time (ONNX `Random*` with a `shape` attribute). The output tensor shape is
//! always the node's assigned shape; the template input is not copied into the
//! output.
//!
//! | Backend | Semantics |
//! |---------|-----------|
//! | [`RngBackend::Philox`] | Deterministic Philox4×32-10 stream keyed by [`RngOptions::seed`] + per-node `key`. Default for RLX-native runs. |
//! | [`RngBackend::Ort`] | Matches ONNX Runtime CPU `Random*` (`minstd_rand0` + polar normal / uniform). Use for import parity tests. Per-op ONNX `seed` (f32) overrides the mixed engine seed when set. |
//! | [`RngBackend::Zero`] | Writes zeros — useful when comparing against a stochastic reference without re-seeding ORT. |
//!
//! Policy is set at compile time via [`CompileOptions::rng`] and can be overridden
//! per session through [`rlx_runtime::CompiledGraph::set_rng`] without
//! recompiling. Each execute re-seeds from the current policy (ORT session state
//! is not advanced across runs today).

/// Which RNG implementation to use for [`Op::RngNormal`] / [`Op::RngUniform`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum RngBackend {
    /// Philox4×32-10 sequential stream (RLX native default).
    #[default]
    Philox,
    /// ONNX Runtime CPU `Random*Like` (`minstd_rand0` + `std::normal_distribution`).
    Ort,
    /// Fill with zero (deterministic parity vs stochastic reference runs).
    Zero,
}

/// Compile-time / execute-time RNG policy for graphs containing random ops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct RngOptions {
    /// Global seed mixed into per-node keys (maps to ORT session seed).
    pub seed: u64,
    pub backend: RngBackend,
}

impl Default for RngOptions {
    fn default() -> Self {
        Self {
            seed: 42,
            backend: RngBackend::Philox,
        }
    }
}

impl RngOptions {
    pub const fn new(seed: u64, backend: RngBackend) -> Self {
        Self { seed, backend }
    }

    pub fn philox(seed: u64) -> Self {
        Self {
            seed,
            backend: RngBackend::Philox,
        }
    }

    pub fn ort(seed: u64) -> Self {
        Self {
            seed,
            backend: RngBackend::Ort,
        }
    }

    pub fn zero() -> Self {
        Self {
            seed: 0,
            backend: RngBackend::Zero,
        }
    }
}

/// Mix a global compile seed with a per-node key (ONNX node name hash).
pub fn combine_seed(global: u64, key: u64) -> u64 {
    global.wrapping_add(key.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// ORT CPU engine seed: explicit ONNX `seed` attr cast to u32, else global+key.
pub fn ort_engine_seed(global: u64, key: u64, op_seed: Option<f32>) -> u32 {
    if let Some(s) = op_seed {
        s as u32
    } else {
        global.wrapping_add(key) as u32
    }
}

/// Fill `out` with `mean + scale * N(0,1)` samples.
pub fn fill_normal_like(
    out: &mut [f32],
    mean: f32,
    scale: f32,
    opts: RngOptions,
    key: u64,
    op_seed: Option<f32>,
) {
    match opts.backend {
        RngBackend::Zero => out.fill(0.0),
        RngBackend::Philox => {
            let mut rng = Philox4x32::new(combine_seed(opts.seed, key));
            for v in out.iter_mut() {
                *v = mean + scale * rng.normal();
            }
        }
        RngBackend::Ort => {
            let mut eng = MinstdRand0::new(ort_engine_seed(opts.seed, key, op_seed));
            let mut dist = StdNormalDist::new(mean, scale);
            for v in out.iter_mut() {
                *v = dist.sample(&mut eng);
            }
        }
    }
}

/// Fill `out` with uniform samples in `[low, high)`.
pub fn fill_uniform_like(
    out: &mut [f32],
    low: f32,
    high: f32,
    opts: RngOptions,
    key: u64,
    op_seed: Option<f32>,
) {
    match opts.backend {
        RngBackend::Zero => out.fill(0.0),
        RngBackend::Philox => {
            let mut rng = Philox4x32::new(combine_seed(opts.seed, key));
            for v in out.iter_mut() {
                *v = rng.uniform(low, high);
            }
        }
        RngBackend::Ort => {
            let mut eng = MinstdRand0::new(ort_engine_seed(opts.seed, key, op_seed));
            for v in out.iter_mut() {
                *v = low + (high - low) * eng.unit_f32();
            }
        }
    }
}

/// Philox4×32 counter-based RNG. Produces 4 u32s per round of the
/// core hash — we expose an iterator that yields one f32 per call.
#[derive(Debug, Clone, Copy)]
pub struct Philox4x32 {
    seed: [u32; 2],
    counter: [u32; 4],
    /// Cached output buffer + cursor into it.
    buffer: [u32; 4],
    cursor: u8,
}

impl Philox4x32 {
    pub const fn new(seed: u64) -> Self {
        let lo = (seed & 0xFFFF_FFFF) as u32;
        let hi = (seed >> 32) as u32;
        Self {
            seed: [lo, hi],
            counter: [0, 0, 0, 0],
            buffer: [0; 4],
            cursor: 4, // empty — next next_u32 fills the buffer
        }
    }

    fn round(state: &mut [u32; 4], key: [u32; 2]) {
        const M0: u64 = 0xD256_1A75;
        const M1: u64 = 0xCD9E_8D57;
        let p0 = (state[0] as u64) * M0;
        let p1 = (state[2] as u64) * M1;
        let hi0 = (p0 >> 32) as u32;
        let lo0 = p0 as u32;
        let hi1 = (p1 >> 32) as u32;
        let lo1 = p1 as u32;
        state[0] = hi1 ^ state[1] ^ key[0];
        state[1] = lo1;
        state[2] = hi0 ^ state[3] ^ key[1];
        state[3] = lo0;
    }

    fn fill_buffer(&mut self) {
        let mut state = self.counter;
        let mut key = self.seed;
        for _ in 0..10 {
            Self::round(&mut state, key);
            // Bump the key on every round (Philox key schedule).
            key[0] = key[0].wrapping_add(0x9E37_79B9);
            key[1] = key[1].wrapping_add(0xBB67_AE85);
        }
        self.buffer = state;
        self.cursor = 0;

        // Increment the 128-bit counter.
        let (c0, of0) = self.counter[0].overflowing_add(1);
        self.counter[0] = c0;
        if of0 {
            let (c1, of1) = self.counter[1].overflowing_add(1);
            self.counter[1] = c1;
            if of1 {
                let (c2, of2) = self.counter[2].overflowing_add(1);
                self.counter[2] = c2;
                if of2 {
                    self.counter[3] = self.counter[3].wrapping_add(1);
                }
            }
        }
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.cursor >= 4 {
            self.fill_buffer();
        }
        let v = self.buffer[self.cursor as usize];
        self.cursor += 1;
        v
    }

    /// Uniform `[0, 1)` f32 — the top 24 bits of a u32 give exactly
    /// f32 mantissa precision.
    pub fn next_f32(&mut self) -> f32 {
        let bits = self.next_u32() >> 8;
        bits as f32 / (1u32 << 24) as f32
    }

    /// Uniform `[lo, hi)` f32.
    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }

    /// Standard-normal `f32` via Box-Muller. Returns one sample;
    /// the second is discarded (we don't cache to keep the type
    /// `Copy`-able).
    pub fn normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(f32::MIN_POSITIVE);
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        r * theta.cos()
    }

    /// Fill `out` with uniform `[0, 1)` samples. Convenience for
    /// weight init.
    pub fn fill_uniform(&mut self, out: &mut [f32]) {
        for v in out {
            *v = self.next_f32();
        }
    }

    /// Fill `out` with N(0, 1) samples.
    pub fn fill_normal(&mut self, out: &mut [f32]) {
        for v in out {
            *v = self.normal();
        }
    }
}

/// C++11 `std::default_random_engine` on libstdc++/libc++ (`minstd_rand0`).
#[derive(Debug, Clone, Copy)]
struct MinstdRand0 {
    state: u32,
}

impl MinstdRand0 {
    const A: u32 = 48_271;
    const M: u32 = 2_147_483_647;

    fn new(seed: u32) -> Self {
        Self {
            state: seed % Self::M,
        }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = ((self.state as u64 * Self::A as u64) % Self::M as u64) as u32;
        self.state
    }

    /// Uniform in `[0, 1)` matching ORT's `RealType(g()) / (g.max() - g.min())`.
    fn unit_f32(&mut self) -> f32 {
        self.next_u32() as f32 / (Self::M - 1) as f32
    }
}

/// C++ `std::normal_distribution<float>` (polar method, caches spare sample).
#[derive(Debug, Clone, Copy)]
struct StdNormalDist {
    mean: f32,
    scale: f32,
    spare: f32,
    has_spare: bool,
}

impl StdNormalDist {
    fn new(mean: f32, scale: f32) -> Self {
        Self {
            mean,
            scale,
            spare: 0.0,
            has_spare: false,
        }
    }

    fn sample(&mut self, eng: &mut MinstdRand0) -> f32 {
        if self.has_spare {
            self.has_spare = false;
            return self.spare;
        }
        loop {
            let u1 = 2.0 * eng.unit_f32() - 1.0;
            let u2 = 2.0 * eng.unit_f32() - 1.0;
            let s = u1 * u1 + u2 * u2;
            if s >= 1.0 || s == 0.0 {
                continue;
            }
            let factor = (-2.0 * s.ln() / s).sqrt();
            self.spare = u2 * factor * self.scale + self.mean;
            self.has_spare = true;
            return u1 * factor * self.scale + self.mean;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_sequence() {
        let mut a = Philox4x32::new(0x1234_5678);
        let mut b = Philox4x32::new(0x1234_5678);
        for _ in 0..256 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn different_seed_different_sequence() {
        let mut a = Philox4x32::new(1);
        let mut b = Philox4x32::new(2);
        let mut diffs = 0usize;
        for _ in 0..16 {
            if a.next_u32() != b.next_u32() {
                diffs += 1;
            }
        }
        assert!(
            diffs >= 14,
            "two distinct seeds should disagree on >=14/16 samples"
        );
    }

    #[test]
    fn next_f32_in_unit_interval() {
        let mut r = Philox4x32::new(42);
        for _ in 0..1000 {
            let v = r.next_f32();
            assert!((0.0..1.0).contains(&v), "{v} not in [0, 1)");
        }
    }

    #[test]
    fn fill_uniform_is_deterministic() {
        let mut r1 = Philox4x32::new(7);
        let mut r2 = Philox4x32::new(7);
        let mut a = vec![0f32; 64];
        let mut b = vec![0f32; 64];
        r1.fill_uniform(&mut a);
        r2.fill_uniform(&mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn normal_mean_is_near_zero() {
        let mut r = Philox4x32::new(123);
        let n = 10_000;
        let mut sum = 0f32;
        for _ in 0..n {
            sum += r.normal();
        }
        let mean = sum / n as f32;
        assert!(mean.abs() < 0.1, "mean {mean} too far from 0");
    }

    #[test]
    fn zero_backend_fills_zeros() {
        let mut out = vec![1.0; 8];
        fill_normal_like(&mut out, 0.0, 1.0, RngOptions::zero(), 0xABC, None);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn philox_backend_is_deterministic() {
        let opts = RngOptions::philox(99);
        let mut a = vec![0f32; 32];
        let mut b = vec![0f32; 32];
        fill_normal_like(&mut a, 0.0, 0.5, opts, 123, None);
        fill_normal_like(&mut b, 0.0, 0.5, opts, 123, None);
        assert_eq!(a, b);
    }

    #[test]
    fn ort_backend_is_deterministic() {
        let opts = RngOptions::ort(7);
        let mut a = vec![0f32; 64];
        let mut b = vec![0f32; 64];
        fill_normal_like(&mut a, 0.1, 2.0, opts, 555, None);
        fill_normal_like(&mut b, 0.1, 2.0, opts, 555, None);
        assert_eq!(a, b);
    }

    #[test]
    fn backends_disagree() {
        let mut philox = vec![0f32; 16];
        let mut ort = vec![0f32; 16];
        fill_normal_like(&mut philox, 0.0, 1.0, RngOptions::philox(42), 1, None);
        fill_normal_like(&mut ort, 0.0, 1.0, RngOptions::ort(42), 1, None);
        assert_ne!(philox, ort);
    }
}
