// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A real specialized kernel behind a runtime **guard** — the actuation the
//! optimizer recommends. `opscope-optimize` says *"this op's input repeats over
//! time → memoize"*; [`MemoMatmul`] is that kernel: a cheap guard (input hash +
//! exact verify) skips the matmul on a cache hit and returns the stored output,
//! falling back to a dense compute on a miss. Correct by construction — the
//! verify guarantees a hit is the *identical* input, so the output is bit-exact.

use std::collections::HashMap;

/// FNV-1a over the raw bits — the cheap guard fingerprint.
pub fn hash_f32(x: &[f32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &v in x {
        h ^= v.to_bits() as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Dense `[m,k]·[k,n] → [m,n]`.
pub fn dense_matmul(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut o = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0f32;
            for p in 0..k {
                s += x[i * k + p] * w[p * n + j];
            }
            o[i * n + j] = s;
        }
    }
    o
}

/// A matmul with a fixed weight that memoizes outputs keyed by input, behind a
/// hash+verify guard and a bounded cache. Exploits temporal input recurrence.
pub struct MemoMatmul {
    w: Vec<f32>,
    m: usize,
    k: usize,
    n: usize,
    cache: HashMap<u64, (Vec<f32>, Vec<f32>)>, // hash → (input, output)
    pub hits: usize,
    pub misses: usize,
    cap: usize,
}

impl MemoMatmul {
    pub fn new(w: Vec<f32>, m: usize, k: usize, n: usize, cap: usize) -> Self {
        Self {
            w,
            m,
            k,
            n,
            cache: HashMap::new(),
            hits: 0,
            misses: 0,
            cap,
        }
    }

    /// Guarded run: on a verified cache hit, returns the stored output without
    /// touching the matmul; otherwise computes densely (fallback) and caches.
    pub fn run(&mut self, x: &[f32]) -> Vec<f32> {
        let h = hash_f32(x);
        if let Some((xi, out)) = self.cache.get(&h) {
            if xi.as_slice() == x {
                // GUARD passed — identical input → cached output is exact.
                self.hits += 1;
                return out.clone();
            }
            // hash collision on a *different* input → fall through (fallback).
        }
        self.misses += 1;
        let out = dense_matmul(x, &self.w, self.m, self.k, self.n);
        if self.cache.len() < self.cap {
            self.cache.insert(h, (x.to_vec(), out.clone()));
        }
        out
    }

    pub fn hit_rate(&self) -> f32 {
        self.hits as f32 / (self.hits + self.misses).max(1) as f32
    }
}
