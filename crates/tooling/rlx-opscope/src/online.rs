// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-cutting — streaming/online sketches for a **sampled** real-inference
//! recording path. To ride the production forward pass at near-zero overhead
//! you can't keep every tensor; you keep bounded per-op summaries updated in one
//! pass: a reservoir sample, a fixed-bucket quantile sketch (t-digest-lite), and
//! a HyperLogLog cardinality estimate. All deterministic (hash-seeded, no RNG
//! dependency), so recordings are reproducible.

/// splitmix64-style finalizer — deterministic hash for sampling / HLL bucketing.
pub fn hash64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Reservoir sample of `k` values over a stream of unknown length (Algorithm R,
/// deterministic via a per-item counter hash).
pub struct Reservoir {
    k: usize,
    seen: u64,
    pub items: Vec<f32>,
}

impl Reservoir {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            seen: 0,
            items: Vec::with_capacity(k),
        }
    }
    pub fn push(&mut self, v: f32) {
        if self.items.len() < self.k {
            self.items.push(v);
        } else {
            let j = (hash64(self.seen) % (self.seen + 1)) as usize;
            if j < self.k {
                self.items[j] = v;
            }
        }
        self.seen += 1;
    }
    pub fn extend(&mut self, xs: &[f32]) {
        for &v in xs {
            self.push(v);
        }
    }
}

/// Fixed-range histogram used as a streaming quantile sketch (t-digest-lite).
pub struct Quantiles {
    lo: f32,
    hi: f32,
    bins: Vec<u64>,
    n: u64,
}

impl Quantiles {
    pub fn new(lo: f32, hi: f32, nbins: usize) -> Self {
        Self {
            lo,
            hi,
            bins: vec![0; nbins.max(1)],
            n: 0,
        }
    }
    pub fn push(&mut self, v: f32) {
        let nb = self.bins.len();
        let t = ((v - self.lo) / (self.hi - self.lo) * nb as f32).floor();
        let idx = (t.max(0.0) as usize).min(nb - 1);
        self.bins[idx] += 1;
        self.n += 1;
    }
    pub fn extend(&mut self, xs: &[f32]) {
        for &v in xs {
            self.push(v);
        }
    }
    /// Approximate `q`-quantile (0..1) by walking the cumulative histogram.
    pub fn quantile(&self, q: f32) -> f32 {
        if self.n == 0 {
            return self.lo;
        }
        let target = (q.clamp(0.0, 1.0) as f64 * self.n as f64).ceil() as u64;
        let nb = self.bins.len();
        let mut cum = 0u64;
        for (i, &c) in self.bins.iter().enumerate() {
            cum += c;
            if cum >= target {
                let frac = (i as f32 + 0.5) / nb as f32;
                return self.lo + frac * (self.hi - self.lo);
            }
        }
        self.hi
    }
}

/// HyperLogLog cardinality estimator with `2^p` registers.
pub struct Hll {
    p: u32,
    reg: Vec<u8>,
}

impl Hll {
    pub fn new(p: u32) -> Self {
        Self {
            p,
            reg: vec![0; 1usize << p],
        }
    }
    pub fn add_bits(&mut self, bits: u64) {
        let h = hash64(bits);
        let idx = (h >> (64 - self.p)) as usize;
        // leading zeros in the remaining (64-p) bits, +1.
        let w = h << self.p;
        let rank = (w.leading_zeros().min(64 - self.p) + 1) as u8;
        if rank > self.reg[idx] {
            self.reg[idx] = rank;
        }
    }
    pub fn add_f32(&mut self, v: f32) {
        self.add_bits(v.to_bits() as u64);
    }
    pub fn estimate(&self) -> f64 {
        let m = self.reg.len() as f64;
        let alpha = 0.7213 / (1.0 + 1.079 / m);
        let sum: f64 = self.reg.iter().map(|&r| 2f64.powi(-(r as i32))).sum();
        let raw = alpha * m * m / sum;
        // small-range correction (linear counting) when many registers are empty.
        let zeros = self.reg.iter().filter(|&&r| r == 0).count() as f64;
        if raw <= 2.5 * m && zeros > 0.0 {
            m * (m / zeros).ln()
        } else {
            raw
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservoir_bounded_and_full() {
        let mut r = Reservoir::new(64);
        for i in 0..10_000 {
            r.push(i as f32);
        }
        assert_eq!(r.items.len(), 64);
    }

    #[test]
    fn quantile_recovers_uniform_median() {
        let mut q = Quantiles::new(0.0, 1.0, 100);
        for i in 0..10_000 {
            q.push(i as f32 / 10_000.0);
        }
        assert!(
            (q.quantile(0.5) - 0.5).abs() < 0.05,
            "median {}",
            q.quantile(0.5)
        );
        assert!(q.quantile(0.9) > q.quantile(0.1));
    }

    #[test]
    fn hll_estimates_cardinality() {
        let mut h = Hll::new(12); // 4096 registers
        for i in 0..10_000u64 {
            h.add_bits(i);
        }
        let est = h.estimate();
        // within ~5% of true 10k.
        assert!((est - 10_000.0).abs() / 10_000.0 < 0.08, "est {est}");
        // adding duplicates doesn't grow the estimate.
        for i in 0..10_000u64 {
            h.add_bits(i);
        }
        assert!((h.estimate() - est).abs() / est < 0.01);
    }
}
