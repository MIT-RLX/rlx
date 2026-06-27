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

//! Advanced token samplers — the long-tail of llama.cpp samplers ported to
//! a backend-agnostic `Sampler` chain.
//!
//! Why backend-agnostic: all samplers operate on a logits row (`&mut [f32]`)
//! plus a token history slice; nothing about the kernel cares which device
//! produced the logits. This file owns the algorithms; backend kernels just
//! emit logits and (for paths like Metal/CUDA) may call into a thin shim
//! that runs the chain on the CPU after a single device→host copy.
//!
//! Available samplers
//! ------------------
//!
//! - [`Temperature`] — uniform `logits[i] /= T` scaling (T → 0 ⇒ greedy).
//! - [`DynamicTemperature`] — entropy-aware temperature: T scales between
//!   `min` and `max` based on `entropy(p) / log(vocab)`.
//! - [`TopK`] — keep the K largest logits; mask the rest to `-inf`.
//! - [`TopP`] — nucleus sampling: smallest set whose cumulative prob ≥ p.
//! - [`TopNSigma`] — keep tokens whose logit ≥ `max - n * σ(logits)`.
//! - [`TypicalP`] — locally-typical sampling (Meister et al. 2022): keep
//!   tokens whose `|−log p − H(p)|` is smallest until cumulative prob ≥ p.
//! - [`MirostatV1`] — target-surprise mode-V1 (μ/τ control loop).
//! - [`MirostatV2`] — simpler logit-based variant: keep tokens with surprise ≤ μ.
//! - [`Xtc`] — eXclude Top Choices: with probability `prob`, drop top tokens
//!   above `threshold` so only the lower-rank tokens survive (Q1-style sampling).
//! - [`Dry`] — DRY (Don't Repeat Yourself) repetition penalty. Penalises
//!   tokens whose continuation matches an earlier n-gram in the history.
//! - [`RepetitionPenalty`] — classic frequency/presence penalty (kept here
//!   so the chain is self-contained).
//!
//! Composition: a `SamplerChain` runs samplers in order, each mutating the
//! logits in place, then the final `Sampler::sample` draws one token. All
//! samplers are deterministic given the same RNG state.

use rlx_ir::Philox4x32;

/// One row of vocabulary-shaped logits (`[vocab]`).
pub type Logits<'a> = &'a mut [f32];

/// State that some samplers update each step (e.g. Mirostat μ).
#[derive(Debug, Default, Clone)]
pub struct SamplerState {
    /// Mirostat estimator. NaN until the first apply.
    pub mirostat_mu: f32,
}

impl SamplerState {
    pub fn new() -> Self {
        Self {
            mirostat_mu: f32::NAN,
        }
    }
}

/// A logit transform + sample step. Samplers mutate the logits row in place;
/// `sample` is called at the end of the chain.
pub trait Sampler: std::fmt::Debug + Send + Sync {
    /// Apply this sampler's transformation to `logits`. `history` is the
    /// already-emitted token sequence (newest last). Some samplers (DRY,
    /// repetition penalty) need it; pure logit transforms ignore it.
    fn apply(
        &self,
        logits: Logits<'_>,
        history: &[u32],
        state: &mut SamplerState,
        rng: &mut Philox4x32,
    );

    /// Optional name for tracing.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

/// A linear chain of samplers terminated by a `softmax + multinomial`
/// draw. Build one with [`SamplerChain::builder`] or directly from a Vec.
#[derive(Debug)]
pub struct SamplerChain {
    pub steps: Vec<Box<dyn Sampler>>,
}

impl SamplerChain {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    pub fn builder() -> SamplerChainBuilder {
        SamplerChainBuilder::default()
    }

    /// Run every sampler in order over `logits`, then draw one token.
    pub fn sample(
        &self,
        logits: Logits<'_>,
        history: &[u32],
        state: &mut SamplerState,
        rng: &mut Philox4x32,
    ) -> u32 {
        for step in &self.steps {
            step.apply(logits, history, state, rng);
        }
        sample_from_logits(logits, rng)
    }
}

impl Default for SamplerChain {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Default)]
pub struct SamplerChainBuilder {
    steps: Vec<Box<dyn Sampler>>,
}

impl SamplerChainBuilder {
    pub fn push<S: Sampler + 'static>(mut self, s: S) -> Self {
        self.steps.push(Box::new(s));
        self
    }

    pub fn push_boxed(mut self, s: Box<dyn Sampler>) -> Self {
        self.steps.push(s);
        self
    }

    pub fn build(self) -> SamplerChain {
        SamplerChain { steps: self.steps }
    }
}

// ─── shared utilities ─────────────────────────────────────────────────

/// Stable softmax in place. Tokens at `-inf` get probability 0.
pub fn softmax_inplace(logits: &mut [f32]) {
    let mut maxv = f32::NEG_INFINITY;
    for &x in logits.iter() {
        if x > maxv {
            maxv = x;
        }
    }
    if !maxv.is_finite() {
        let inv = 1.0 / logits.len() as f32;
        for x in logits.iter_mut() {
            *x = inv;
        }
        return;
    }
    let mut s = 0.0f32;
    for x in logits.iter_mut() {
        let v = (*x - maxv).exp();
        *x = v;
        s += v;
    }
    let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
    for x in logits.iter_mut() {
        *x *= inv;
    }
}

/// Multinomial draw from a probability row (assumed already softmax-ed).
/// Robust against floating-point drift in the tail.
pub fn sample_from_probs(probs: &[f32], rng: &mut Philox4x32) -> u32 {
    let r = rng.next_f32();
    let mut acc = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if r <= acc {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

/// Apply softmax to logits in place then multinomial-sample one index.
pub fn sample_from_logits(logits: &mut [f32], rng: &mut Philox4x32) -> u32 {
    softmax_inplace(logits);
    sample_from_probs(logits, rng)
}

/// Sorted-descending index/value pairs. Used by every top-something sampler.
fn sorted_desc(logits: &[f32]) -> Vec<(usize, f32)> {
    let mut v: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    v
}

// ─── Temperature ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct Temperature {
    pub t: f32,
}

impl Sampler for Temperature {
    fn apply(&self, logits: Logits<'_>, _h: &[u32], _s: &mut SamplerState, _r: &mut Philox4x32) {
        let t = self.t.max(1e-6);
        for x in logits.iter_mut() {
            *x /= t;
        }
    }
}

// ─── DynamicTemperature ──────────────────────────────────────────────
//
// Scale temperature between [min, max] based on the entropy of the
// current logits' softmax: T = min + (max - min) * (H / Hmax)^exponent.
// `exponent=1.0` matches the original llama.cpp `dynatemp` behavior.

#[derive(Debug, Clone, Copy)]
pub struct DynamicTemperature {
    pub min: f32,
    pub max: f32,
    pub exponent: f32,
}

impl Sampler for DynamicTemperature {
    fn apply(&self, logits: Logits<'_>, _h: &[u32], _s: &mut SamplerState, _r: &mut Philox4x32) {
        let v = logits.len();
        if v == 0 {
            return;
        }
        let mut tmp: Vec<f32> = logits.to_vec();
        softmax_inplace(&mut tmp);
        // Shannon entropy in nats.
        let mut h = 0.0f32;
        for &p in tmp.iter() {
            if p > 0.0 {
                h -= p * p.ln();
            }
        }
        let hmax = (v as f32).ln().max(1e-6);
        let norm = (h / hmax).clamp(0.0, 1.0);
        let t = self.min + (self.max - self.min) * norm.powf(self.exponent);
        let t = t.max(1e-6);
        for x in logits.iter_mut() {
            *x /= t;
        }
    }
}

// ─── TopK ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct TopK {
    pub k: usize,
}

impl Sampler for TopK {
    fn apply(&self, logits: Logits<'_>, _h: &[u32], _s: &mut SamplerState, _r: &mut Philox4x32) {
        let v = logits.len();
        if self.k == 0 || self.k >= v {
            return;
        }
        let sorted = sorted_desc(logits);
        let cutoff = sorted[self.k - 1].1;
        for x in logits.iter_mut() {
            if *x < cutoff {
                *x = f32::NEG_INFINITY;
            }
        }
    }
}

// ─── TopP (nucleus) ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct TopP {
    pub p: f32,
    /// Always keep at least this many tokens. Avoids degenerate single-token
    /// nucleus when one logit dominates.
    pub min_keep: usize,
}

impl Sampler for TopP {
    fn apply(&self, logits: Logits<'_>, _h: &[u32], _s: &mut SamplerState, _r: &mut Philox4x32) {
        if self.p >= 1.0 {
            return;
        }
        let v = logits.len();
        if v == 0 {
            return;
        }
        let mut probs: Vec<f32> = logits.to_vec();
        softmax_inplace(&mut probs);
        let sorted = sorted_desc(&probs);
        let mut keep = vec![false; v];
        let mut cum = 0.0f32;
        for (rank, (idx, p)) in sorted.iter().enumerate() {
            keep[*idx] = true;
            cum += *p;
            if cum >= self.p && rank + 1 >= self.min_keep {
                break;
            }
        }
        for (i, x) in logits.iter_mut().enumerate() {
            if !keep[i] {
                *x = f32::NEG_INFINITY;
            }
        }
    }
}

// ─── MinP ────────────────────────────────────────────────────────────
//
// Keep tokens whose probability ≥ p · p_max, where p_max is the top
// token's probability (Nguyen et al. 2024, "min-p sampling"). The cutoff
// scales with the model's confidence: permissive when the distribution is
// flat, strict when one token dominates. A cheaper, often better-behaved
// alternative to top-p.

#[derive(Debug, Clone, Copy)]
pub struct MinP {
    pub p: f32,
    /// Always keep at least this many tokens (avoids an empty nucleus when
    /// one token dominates and `p` is large).
    pub min_keep: usize,
}

impl Sampler for MinP {
    fn apply(&self, logits: Logits<'_>, _h: &[u32], _s: &mut SamplerState, _r: &mut Philox4x32) {
        // Reject p outside (0, 1]; the explicit is_nan keeps a NaN p from
        // slipping past the bounds checks (NaN compares false to everything).
        if self.p <= 0.0 || self.p > 1.0 || self.p.is_nan() {
            return;
        }
        let v = logits.len();
        if v == 0 {
            return;
        }
        let mut probs: Vec<f32> = logits.to_vec();
        softmax_inplace(&mut probs);
        let p_max = probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if p_max <= 0.0 || p_max.is_nan() {
            return;
        }
        let threshold = self.p * p_max;
        let mut keep: Vec<bool> = probs.iter().map(|&p| p >= threshold).collect();
        // Guarantee at least `min_keep` survivors: re-admit the top probs.
        let floor = self.min_keep.max(1);
        if keep.iter().filter(|&&k| k).count() < floor {
            for (idx, _) in sorted_desc(&probs).into_iter().take(floor) {
                keep[idx] = true;
            }
        }
        for (i, x) in logits.iter_mut().enumerate() {
            if !keep[i] {
                *x = f32::NEG_INFINITY;
            }
        }
    }
}

/// Apply an additive per-token logit bias in place (OpenAI `logit_bias`
/// semantics): `logits[id] += bias`. Out-of-range ids are ignored. This is
/// a host-side logits processor — run it on the raw logits row *before*
/// the sampler chain. Kept as a free function (not a `SampleOpts` field) so
/// `SampleOpts` stays `Copy`.
pub fn apply_logit_bias(logits: &mut [f32], bias: &[(u32, f32)]) {
    let v = logits.len();
    for &(id, b) in bias {
        let i = id as usize;
        if i < v {
            logits[i] += b;
        }
    }
}

// ─── TopNSigma ───────────────────────────────────────────────────────
//
// Keep tokens whose logit ≥ max_logit − n × σ(logits). Works directly on
// the logit space, so no softmax is required to mask. (Ref: Hewitt et al.
// "Top-n-σ" 2024.)

#[derive(Debug, Clone, Copy)]
pub struct TopNSigma {
    pub n: f32,
}

impl Sampler for TopNSigma {
    fn apply(&self, logits: Logits<'_>, _h: &[u32], _s: &mut SamplerState, _r: &mut Philox4x32) {
        let v = logits.len();
        if v == 0 || !self.n.is_finite() || self.n <= 0.0 {
            return;
        }
        let mut maxv = f32::NEG_INFINITY;
        let mut count = 0usize;
        let mut sum = 0.0f32;
        for &x in logits.iter() {
            if x.is_finite() {
                if x > maxv {
                    maxv = x;
                }
                sum += x;
                count += 1;
            }
        }
        if count == 0 || !maxv.is_finite() {
            return;
        }
        let mean = sum / count as f32;
        let mut var = 0.0f32;
        for &x in logits.iter() {
            if x.is_finite() {
                let d = x - mean;
                var += d * d;
            }
        }
        let sigma = (var / count as f32).sqrt();
        let cutoff = maxv - self.n * sigma;
        for x in logits.iter_mut() {
            if *x < cutoff {
                *x = f32::NEG_INFINITY;
            }
        }
    }
}

// ─── TypicalP (locally typical sampling) ────────────────────────────
//
// Meister et al. 2022: rank tokens by deviation of their surprisal
// (−log p) from the distribution's entropy H, ascending. Keep the
// smallest set whose cumulative prob ≥ p.

#[derive(Debug, Clone, Copy)]
pub struct TypicalP {
    pub p: f32,
    pub min_keep: usize,
}

impl Sampler for TypicalP {
    fn apply(&self, logits: Logits<'_>, _h: &[u32], _s: &mut SamplerState, _r: &mut Philox4x32) {
        if self.p >= 1.0 {
            return;
        }
        let v = logits.len();
        if v == 0 {
            return;
        }
        let mut probs: Vec<f32> = logits.to_vec();
        softmax_inplace(&mut probs);
        let mut h = 0.0f32;
        for &p in probs.iter() {
            if p > 0.0 {
                h -= p * p.ln();
            }
        }
        // Score each token by |−log p − H|.
        let mut scored: Vec<(usize, f32, f32)> = probs
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let neg_log = if p > 0.0 { -p.ln() } else { f32::INFINITY };
                let dev = (neg_log - h).abs();
                (i, p, dev)
            })
            .collect();
        scored.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        let mut keep = vec![false; v];
        let mut cum = 0.0f32;
        for (rank, (idx, p, _)) in scored.iter().enumerate() {
            keep[*idx] = true;
            cum += *p;
            if cum >= self.p && rank + 1 >= self.min_keep {
                break;
            }
        }
        for (i, x) in logits.iter_mut().enumerate() {
            if !keep[i] {
                *x = f32::NEG_INFINITY;
            }
        }
    }
}

// ─── Mirostat v1 ─────────────────────────────────────────────────────
//
// Original (NeurIPS 2021) paper: maintain a running μ that is the target
// surprise. Top-k is set so the average surprise of the truncated head
// matches μ, then μ is updated by the error against τ.

#[derive(Debug, Clone, Copy)]
pub struct MirostatV1 {
    pub tau: f32,
    pub eta: f32,
    /// Pareto‐tail estimator window. Original paper uses 100.
    pub m: usize,
}

impl Default for MirostatV1 {
    fn default() -> Self {
        Self {
            tau: 5.0,
            eta: 0.1,
            m: 100,
        }
    }
}

impl Sampler for MirostatV1 {
    fn apply(
        &self,
        logits: Logits<'_>,
        _h: &[u32],
        state: &mut SamplerState,
        rng: &mut Philox4x32,
    ) {
        let v = logits.len();
        if v == 0 {
            return;
        }
        if !state.mirostat_mu.is_finite() {
            state.mirostat_mu = 2.0 * self.tau;
        }
        let mu = state.mirostat_mu.max(1e-6);
        // Softmax to get probabilities and sort descending.
        let mut probs = logits.to_vec();
        softmax_inplace(&mut probs);
        let sorted = sorted_desc(&probs);
        // Pareto-tail s-hat from top-m. Surprise[i] = −log p[i].
        let m = self.m.min(sorted.len()).max(2);
        let mut num = 0.0f32;
        let mut den = 0.0f32;
        for i in 0..(m - 1) {
            let t = ((i + 2) as f32 / (i + 1) as f32).ln();
            let b = (sorted[i].1 / sorted[i + 1].1).ln().max(1e-9);
            num += t * b;
            den += t * t;
        }
        let s_hat = if den > 0.0 { num / den } else { 1.0 };
        // k = ((eps * 2^μ) / (1 − N^(−eps))) ^ (1 / s_hat)
        let eps = (s_hat - 1.0).abs().max(1e-3);
        let k_real = ((eps * (2.0f32.powf(mu))) / (1.0 - (v as f32).powf(-eps)))
            .powf(1.0 / s_hat)
            .clamp(1.0, v as f32);
        let k = k_real as usize;
        if k < sorted.len() {
            let cutoff = sorted[k - 1].1;
            for (i, p) in probs.iter_mut().enumerate() {
                if *p < cutoff {
                    *p = 0.0;
                }
                let _ = i;
            }
            let s: f32 = probs.iter().sum();
            if s > 0.0 {
                for p in probs.iter_mut() {
                    *p /= s;
                }
            }
        }
        // Draw the token now; convert back into logits by setting one-hot
        // (downstream samplers softmax+sample again, which is OK since one-hot).
        let tok = sample_from_probs(&probs, rng) as usize;
        let surprise = if probs[tok] > 0.0 {
            -probs[tok].ln() / 2.0f32.ln()
        } else {
            mu
        };
        state.mirostat_mu = (mu - self.eta * (surprise - self.tau)).max(0.0);
        for (i, x) in logits.iter_mut().enumerate() {
            *x = if i == tok {
                f32::INFINITY
            } else {
                f32::NEG_INFINITY
            };
        }
    }
}

// ─── Mirostat v2 ─────────────────────────────────────────────────────
//
// Simpler variant (commonly shipped by llama.cpp): keep tokens whose
// surprise (in bits) is ≤ μ; sample, then update μ ← μ − η(s − τ).

#[derive(Debug, Clone, Copy)]
pub struct MirostatV2 {
    pub tau: f32,
    pub eta: f32,
}

impl Default for MirostatV2 {
    fn default() -> Self {
        Self { tau: 5.0, eta: 0.1 }
    }
}

impl Sampler for MirostatV2 {
    fn apply(
        &self,
        logits: Logits<'_>,
        _h: &[u32],
        state: &mut SamplerState,
        rng: &mut Philox4x32,
    ) {
        let v = logits.len();
        if v == 0 {
            return;
        }
        if !state.mirostat_mu.is_finite() {
            state.mirostat_mu = 2.0 * self.tau;
        }
        let mu = state.mirostat_mu;
        let mut probs = logits.to_vec();
        softmax_inplace(&mut probs);
        // Sort descending; keep tokens with surprise ≤ μ.
        let mut sorted = sorted_desc(&probs);
        let ln2 = 2.0f32.ln();
        let mut keep_n = 0usize;
        for (i, (_, p)) in sorted.iter().enumerate() {
            let s = if *p > 0.0 {
                -p.ln() / ln2
            } else {
                f32::INFINITY
            };
            if s > mu {
                break;
            }
            keep_n = i + 1;
        }
        if keep_n == 0 {
            keep_n = 1;
        }
        let kept: std::collections::HashSet<usize> =
            sorted.drain(..keep_n).map(|(i, _)| i).collect();
        for (i, p) in probs.iter_mut().enumerate() {
            if !kept.contains(&i) {
                *p = 0.0;
            }
        }
        let s: f32 = probs.iter().sum();
        if s > 0.0 {
            for p in probs.iter_mut() {
                *p /= s;
            }
        }
        let tok = sample_from_probs(&probs, rng) as usize;
        let surprise = if probs[tok] > 0.0 {
            -probs[tok].ln() / ln2
        } else {
            mu
        };
        state.mirostat_mu = (mu - self.eta * (surprise - self.tau)).max(0.0);
        for (i, x) in logits.iter_mut().enumerate() {
            *x = if i == tok {
                f32::INFINITY
            } else {
                f32::NEG_INFINITY
            };
        }
    }
}

// ─── XTC (eXclude Top Choices) ───────────────────────────────────────
//
// Drops top tokens whose prob > `threshold`, with probability `prob`, as
// long as at least one such "high-confidence" token would remain. Forces
// the model into long-tail tokens to break repetition / mode collapse.

#[derive(Debug, Clone, Copy)]
pub struct Xtc {
    pub threshold: f32,
    pub prob: f32,
    /// Keep this many top tokens minimum even after exclusion.
    pub min_keep: usize,
}

impl Sampler for Xtc {
    fn apply(&self, logits: Logits<'_>, _h: &[u32], _s: &mut SamplerState, rng: &mut Philox4x32) {
        if self.prob <= 0.0 {
            return;
        }
        if rng.next_f32() > self.prob {
            return;
        }
        let v = logits.len();
        if v == 0 {
            return;
        }
        let mut probs = logits.to_vec();
        softmax_inplace(&mut probs);
        let sorted = sorted_desc(&probs);
        // Count tokens above threshold.
        let n_above = sorted.iter().filter(|(_, p)| *p > self.threshold).count();
        if n_above < 2 {
            return; // not enough to exclude
        }
        // Exclude all but the lowest-ranked one above threshold so
        // exactly one survives.
        let to_kill = n_above.saturating_sub(self.min_keep.max(1));
        for (idx, _) in sorted.iter().take(to_kill) {
            logits[*idx] = f32::NEG_INFINITY;
        }
    }
}

// ─── DRY (Don't Repeat Yourself) ─────────────────────────────────────
//
// For each suffix of `history` of length n ≥ `allowed_length`, if the
// next token would extend the suffix into a known n-gram match, scale
// its logit down by `multiplier * base^(n − allowed_length)`. Caps at
// `max_ngram` to bound work.

#[derive(Debug, Clone)]
pub struct Dry {
    pub multiplier: f32,
    pub base: f32,
    pub allowed_length: usize,
    pub max_ngram: usize,
    /// Tokens that break a repetition run (e.g. newlines for prose, EOS).
    pub sequence_breakers: Vec<u32>,
}

impl Default for Dry {
    fn default() -> Self {
        Self {
            multiplier: 0.8,
            base: 1.75,
            allowed_length: 2,
            max_ngram: 32,
            sequence_breakers: Vec::new(),
        }
    }
}

impl Sampler for Dry {
    fn apply(
        &self,
        logits: Logits<'_>,
        history: &[u32],
        _s: &mut SamplerState,
        _r: &mut Philox4x32,
    ) {
        if self.multiplier <= 0.0 || history.is_empty() {
            return;
        }
        let n = history.len();
        let max_ngram = self.max_ngram.min(n);
        let breakers: std::collections::HashSet<u32> =
            self.sequence_breakers.iter().copied().collect();
        // For each candidate next token t (only tokens that actually appear
        // in history can match — bail early on the rest), find the longest
        // suffix of history that matches a prefix ending at some earlier i.
        // Track the longest match per token.
        let mut longest: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for i in 0..n.saturating_sub(1) {
            if breakers.contains(&history[i]) {
                continue;
            }
            // Length of common tail starting at history[..=i] (matching i with
            // n-1, i-1 with n-2, …).
            let mut l = 0usize;
            while l < max_ngram && i >= l && n > l && history[i - l] == history[n - 1 - l] {
                l += 1;
            }
            if l >= self.allowed_length && i + 1 < n {
                let next = history[i + 1];
                let cur = longest.entry(next).or_insert(0);
                if l > *cur {
                    *cur = l;
                }
            }
        }
        for (tok, l) in longest {
            let pen = self.multiplier * self.base.powi((l - self.allowed_length) as i32);
            let idx = tok as usize;
            if idx < logits.len() {
                logits[idx] -= pen;
            }
        }
    }
}

// ─── Repetition penalty ──────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct RepetitionPenalty {
    pub penalty: f32,
    pub frequency: f32,
    pub presence: f32,
    /// Last N tokens of history to consider.
    pub last_n: usize,
}

impl Default for RepetitionPenalty {
    fn default() -> Self {
        Self {
            penalty: 1.0,
            frequency: 0.0,
            presence: 0.0,
            last_n: 64,
        }
    }
}

impl Sampler for RepetitionPenalty {
    fn apply(
        &self,
        logits: Logits<'_>,
        history: &[u32],
        _s: &mut SamplerState,
        _r: &mut Philox4x32,
    ) {
        if history.is_empty() {
            return;
        }
        let start = history.len().saturating_sub(self.last_n);
        let window = &history[start..];
        let mut counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        for &t in window {
            *counts.entry(t).or_insert(0) += 1;
        }
        for (tok, c) in counts {
            let idx = tok as usize;
            if idx >= logits.len() {
                continue;
            }
            // OpenAI-style: subtract presence + frequency * count.
            logits[idx] -= self.presence + self.frequency * c as f32;
            // Anthropic/llama.cpp-style: divide (for positive logits) or
            // multiply (negative) by `penalty`.
            if (self.penalty - 1.0).abs() > 1e-6 {
                if logits[idx] > 0.0 {
                    logits[idx] /= self.penalty;
                } else {
                    logits[idx] *= self.penalty;
                }
            }
        }
    }
}

// ─── tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rng() -> Philox4x32 {
        Philox4x32::new(0xDEAD_BEEF)
    }

    #[test]
    fn temperature_zero_is_greedy_after_chain() {
        let chain = SamplerChain::builder()
            .push(Temperature { t: 1e-6 })
            .build();
        let mut state = SamplerState::new();
        let mut r = rng();
        let mut logits = vec![1.0, 5.0, 2.0, 3.0];
        let tok = chain.sample(&mut logits, &[], &mut state, &mut r);
        assert_eq!(tok, 1);
    }

    #[test]
    fn top_k_masks_below_kth() {
        let mut logits = vec![1.0, 5.0, 2.0, 3.0];
        let mut s = SamplerState::new();
        let mut r = rng();
        TopK { k: 2 }.apply(&mut logits, &[], &mut s, &mut r);
        assert_eq!(logits[1], 5.0);
        assert_eq!(logits[3], 3.0);
        assert!(logits[0].is_infinite() && logits[0] < 0.0);
        assert!(logits[2].is_infinite() && logits[2] < 0.0);
    }

    #[test]
    fn top_p_keeps_nucleus() {
        let mut logits = vec![0.0f32; 4];
        logits[0] = 10.0;
        logits[1] = 5.0;
        let mut s = SamplerState::new();
        let mut r = rng();
        TopP {
            p: 0.5,
            min_keep: 1,
        }
        .apply(&mut logits, &[], &mut s, &mut r);
        assert!(logits[0].is_finite());
        // Lowest-probability tokens should be masked.
        assert!(logits[2].is_infinite() && logits[2] < 0.0);
        assert!(logits[3].is_infinite() && logits[3] < 0.0);
    }

    #[test]
    fn min_p_scales_cutoff_by_top_prob() {
        // Dominant token: prob ~ exp(10)/Z ≈ 1; a 0.1 cutoff keeps only it.
        let mut logits = vec![0.0f32; 4];
        logits[0] = 10.0;
        logits[1] = 2.0;
        let mut s = SamplerState::new();
        let mut r = rng();
        MinP {
            p: 0.1,
            min_keep: 1,
        }
        .apply(&mut logits, &[], &mut s, &mut r);
        assert!(logits[0].is_finite());
        assert!(logits[1].is_infinite() && logits[1] < 0.0);
        assert!(logits[2].is_infinite() && logits[2] < 0.0);
    }

    #[test]
    fn min_p_min_keep_admits_top_tokens() {
        // Even with an aggressive cutoff, min_keep survivors are guaranteed.
        let mut logits = vec![10.0, 9.0, 1.0, 0.0];
        let mut s = SamplerState::new();
        let mut r = rng();
        MinP {
            p: 0.99,
            min_keep: 2,
        }
        .apply(&mut logits, &[], &mut s, &mut r);
        assert!(logits[0].is_finite() && logits[1].is_finite());
    }

    #[test]
    fn logit_bias_is_additive_and_bounds_checked() {
        let mut logits = vec![0.0f32, 1.0, 2.0];
        apply_logit_bias(&mut logits, &[(1, 5.0), (99, 1.0)]); // 99 ignored
        assert_eq!(logits, vec![0.0, 6.0, 2.0]);
    }

    #[test]
    fn top_n_sigma_keeps_top_logits() {
        // One peak, rest noise.
        let mut logits = vec![0.0f32; 32];
        logits[0] = 10.0;
        logits[1] = 9.5;
        let mut s = SamplerState::new();
        let mut r = rng();
        TopNSigma { n: 1.0 }.apply(&mut logits, &[], &mut s, &mut r);
        // Peak survives, the long flat tail is killed.
        assert!(logits[0].is_finite());
        assert!(logits[5].is_infinite() && logits[5] < 0.0);
    }

    #[test]
    fn dynamic_temperature_scales_with_entropy() {
        // Uniform input → maximum entropy → T = max.
        let mut logits = vec![1.0f32; 16];
        let before = logits.clone();
        let mut s = SamplerState::new();
        let mut r = rng();
        DynamicTemperature {
            min: 0.5,
            max: 2.0,
            exponent: 1.0,
        }
        .apply(&mut logits, &[], &mut s, &mut r);
        assert!((logits[0] - before[0] / 2.0).abs() < 1e-5);
    }

    #[test]
    fn typical_p_keeps_typical_token() {
        let mut logits = vec![5.0, 4.0, 0.0, -10.0];
        let mut s = SamplerState::new();
        let mut r = rng();
        TypicalP {
            p: 0.5,
            min_keep: 1,
        }
        .apply(&mut logits, &[], &mut s, &mut r);
        // At least one token survives.
        assert!(logits.iter().any(|x| x.is_finite()));
    }

    #[test]
    fn mirostat_v2_keeps_at_least_one() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        let mut s = SamplerState::new();
        let mut r = rng();
        MirostatV2 { tau: 5.0, eta: 0.1 }.apply(&mut logits, &[], &mut s, &mut r);
        // Result is a one-hot encoding of the drawn token.
        let n_inf = logits
            .iter()
            .filter(|x| x.is_infinite() && **x > 0.0)
            .count();
        assert_eq!(n_inf, 1);
    }

    #[test]
    fn xtc_disabled_when_prob_zero() {
        let mut logits = vec![10.0, 5.0, 1.0];
        let before = logits.clone();
        let mut s = SamplerState::new();
        let mut r = rng();
        Xtc {
            threshold: 0.5,
            prob: 0.0,
            min_keep: 1,
        }
        .apply(&mut logits, &[], &mut s, &mut r);
        assert_eq!(logits, before);
    }

    #[test]
    fn dry_penalises_repeat_continuation() {
        // History "A B A B A" → continuing the "B" pattern after "A" should be penalised.
        let history = vec![0u32, 1, 0, 1, 0];
        let mut logits = vec![0.0, 0.0];
        let mut s = SamplerState::new();
        let mut r = rng();
        Dry {
            multiplier: 1.0,
            base: 2.0,
            allowed_length: 2,
            max_ngram: 8,
            sequence_breakers: vec![],
        }
        .apply(&mut logits, &history, &mut s, &mut r);
        assert!(logits[1] < 0.0, "B should be penalised; got {}", logits[1]);
    }

    #[test]
    fn repetition_penalty_lowers_repeated_token() {
        let history = vec![0u32; 8];
        let mut logits = vec![1.0, 1.0];
        let mut s = SamplerState::new();
        let mut r = rng();
        RepetitionPenalty {
            penalty: 2.0,
            frequency: 0.0,
            presence: 0.0,
            last_n: 64,
        }
        .apply(&mut logits, &history, &mut s, &mut r);
        assert!(logits[0] < logits[1]);
    }
}
