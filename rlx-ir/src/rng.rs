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

//! Counter-based deterministic RNG (plan #43).
//!
//! Borrowed from MAX's `nn/randn.mojo` / `nn/rand_uniform.mojo`
//! pattern. Counter-based RNGs (Philox) are stateless besides the
//! seed and counter — given the same `(seed, counter)` you get the
//! same output bit-for-bit, which makes:
//!
//!   - init values deterministic for reproducible benches,
//!   - weight init reproducible across machines (CI vs laptop),
//!   - tests that depend on random data trivial to debug — replay
//!     with the same seed.
//!
//! Implementation is Philox 4×32-10 (the same family numpy / JAX
//! use). Pure Rust, no extern crate.

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
        // Sanity check: 10k samples of N(0,1) should average within 0.1 of 0.
        let mut r = Philox4x32::new(123);
        let n = 10_000;
        let mut sum = 0f32;
        for _ in 0..n {
            sum += r.normal();
        }
        let mean = sum / n as f32;
        assert!(mean.abs() < 0.1, "mean {mean} too far from 0");
    }
}
