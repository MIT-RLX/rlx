// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sampling helpers for LM runners.
//!
//! `SampleOpts` itself lives in `rlx-runtime::lm` so backend code can
//! reference it without depending on `rlx-text`. This module adds the
//! actual sampling routines that consume those options.

pub use rlx_runtime::SampleOpts;

/// Greedy argmax over `logits`.
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

/// Sample one token from `logits` according to `opts`.
///
/// Greedy when `opts.temperature == 0`. Otherwise applies temperature,
/// optional top-k, top-p (nucleus), and a repetition penalty against the
/// `history` ids (set empty to disable).
///
/// `rng` is a 32-bit Lehmer LCG seed; pass any non-zero value. Returns
/// the new seed so callers can chain.
pub fn sample_next(logits: &[f32], history: &[u32], opts: &SampleOpts, rng: &mut u32) -> u32 {
    if opts.is_greedy() {
        return argmax(logits);
    }
    let mut probs: Vec<f32> = logits.to_vec();

    // Repetition penalty (greater than 1 discourages repeats).
    if (opts.repetition_penalty - 1.0).abs() > 1e-6 {
        for &id in history {
            let i = id as usize;
            if i < probs.len() {
                let p = probs[i];
                probs[i] = if p > 0.0 {
                    p / opts.repetition_penalty
                } else {
                    p * opts.repetition_penalty
                };
            }
        }
    }

    // Temperature.
    if opts.temperature > 0.0 && (opts.temperature - 1.0).abs() > 1e-6 {
        let inv = 1.0 / opts.temperature;
        for v in probs.iter_mut() {
            *v *= inv;
        }
    }

    // Softmax in-place.
    let max = probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for v in probs.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum <= 0.0 {
        return argmax(logits);
    }
    for v in probs.iter_mut() {
        *v /= sum;
    }

    // Sort indices by probability desc.
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|a, b| {
        probs[*b]
            .partial_cmp(&probs[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Top-k.
    let k = opts.top_k.unwrap_or(order.len() as u32) as usize;
    let k = k.min(order.len());

    // Top-p.
    let mut acc = 0f32;
    let mut keep = 0usize;
    for &idx in order.iter().take(k) {
        acc += probs[idx];
        keep += 1;
        if acc >= opts.top_p {
            break;
        }
    }
    let keep = keep.max(1);

    // Renormalise.
    let mut total = 0f32;
    for &idx in order.iter().take(keep) {
        total += probs[idx];
    }
    if total <= 0.0 {
        return order[0] as u32;
    }

    // Lehmer LCG: state * 48271 mod 2^31-1.
    *rng = ((*rng as u64 * 48271) % 0x7FFFFFFF) as u32;
    let r = (*rng as f32) / (0x7FFFFFFF as f32);
    let target = r * total;
    let mut acc = 0f32;
    for &idx in order.iter().take(keep) {
        acc += probs[idx];
        if acc >= target {
            return idx as u32;
        }
    }
    order[keep - 1] as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_picks_largest() {
        assert_eq!(argmax(&[1.0, 5.0, 3.0, -1.0]), 1);
    }

    #[test]
    fn greedy_short_circuit() {
        let mut rng = 1u32;
        let logits = vec![1.0, 9.0, 3.0];
        let opts = SampleOpts::greedy();
        assert_eq!(sample_next(&logits, &[], &opts, &mut rng), 1);
    }

    #[test]
    fn high_temperature_widens_distribution() {
        let logits = vec![0.0, 1.0, 0.0];
        let opts = SampleOpts::nucleus(2.0, 1.0);
        let mut rng = 42u32;
        // Pull a few samples; should hit at least two distinct ids.
        let mut seen = std::collections::HashSet::new();
        for _ in 0..50 {
            seen.insert(sample_next(&logits, &[], &opts, &mut rng));
        }
        assert!(seen.len() >= 2);
    }
}
