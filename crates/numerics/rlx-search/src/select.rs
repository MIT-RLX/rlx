// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-row selection: argmax, top-k, log-softmax, log-sum-exp.
//!
//! Every decoder reads a `[vocab]` row per step and picks from it. These are
//! the shared spellings, with one tie rule everywhere: **equal values go to
//! the lowest index**, so a result never depends on iteration order or on
//! which backend produced the row. NaN is never selected.

/// Lanes of the max reduction in [`argmax`]. Sixteen independent
/// accumulators let the loop vectorize on baseline SSE2/NEON without
/// `-C target-cpu=native`.
const LANES: usize = 16;

/// Index of the largest value; ties go to the lowest index, NaN is skipped.
/// An empty row, or one with no value above `-inf`, gives `0`.
///
/// Two passes — a lane-parallel max, then the first position holding it —
/// about 3.5× faster than the single branchy loop on a 152k-entry vocabulary.
pub fn argmax(row: &[f32]) -> usize {
    let mut lanes = [f32::NEG_INFINITY; LANES];
    let (chunks, rest) = row.as_chunks::<LANES>();
    for c in chunks {
        for (m, &v) in lanes.iter_mut().zip(c) {
            // `v > m` is false for NaN, so NaN never enters a lane.
            *m = if v > *m { v } else { *m };
        }
    }
    let mut best = f32::NEG_INFINITY;
    for &v in lanes.iter().chain(rest) {
        if v > best {
            best = v;
        }
    }
    if best == f32::NEG_INFINITY {
        return 0;
    }
    row.iter().position(|&v| v == best).unwrap_or(0)
}

/// [`argmax`] as a token id.
pub fn argmax_u32(row: &[f32]) -> u32 {
    argmax(row) as u32
}

/// The `k` largest **finite** values as `(index, value)`, best first; ties go
/// to the lowest index. Fewer than `k` come back when the row has fewer
/// finite values.
///
/// One pass with a sorted `k`-entry buffer: a value enters only if it beats
/// the current `k`-th, which after the first few thousand entries is rare, so
/// small `k` (beam widths) cost about one compare per element.
pub fn top_k(row: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut best: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
    if k == 0 {
        return best;
    }
    for (i, &v) in row.iter().enumerate() {
        if !v.is_finite() {
            continue;
        }
        if best.len() == k && v <= best[k - 1].1 {
            continue;
        }
        // Insert after every entry >= v: an equal value seen earlier has the
        // lower index and stays ahead.
        let at = best.partition_point(|&(_, b)| b >= v);
        best.insert(at, (i as u32, v));
        best.truncate(k);
    }
    best
}

/// `ln Σ exp(x)`, max-shifted; `-inf` for an empty row or one that is all
/// `-inf`.
pub fn log_sum_exp(row: &[f32]) -> f32 {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max == f32::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    max + row.iter().map(|&v| (v - max).exp()).sum::<f32>().ln()
}

/// `ln(exp(a) + exp(b))`, exact when either side is `-inf`.
#[inline]
pub fn log_add(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// Logits to log-probabilities in place. `-inf` entries (masked tokens) stay
/// `-inf`; a row with nothing finite is left as is.
pub fn log_softmax(row: &mut [f32]) {
    let lse = log_sum_exp(row);
    if lse == f32::NEG_INFINITY {
        return;
    }
    for v in row.iter_mut() {
        *v -= lse;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argmax_ref(row: &[f32]) -> usize {
        let mut best = 0;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in row.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        best
    }

    #[test]
    fn argmax_matches_the_scalar_loop() {
        let mut row: Vec<f32> = (0..1000)
            .map(|i| ((i * 2_654_435_761u64 as usize) % 997) as f32)
            .collect();
        assert_eq!(argmax(&row), argmax_ref(&row));
        // Ties anywhere, including across lanes and in the remainder.
        row[3] = 5000.0;
        row[19] = 5000.0;
        row[999] = 5000.0;
        assert_eq!(argmax(&row), 3);
        row[3] = f32::NAN;
        assert_eq!(argmax(&row), 19);
        for n in 0..40 {
            let r: Vec<f32> = (0..n).map(|i| ((i * 7) % 5) as f32).collect();
            assert_eq!(argmax(&r), argmax_ref(&r), "n={n}");
        }
    }

    #[test]
    fn argmax_degenerate_rows() {
        assert_eq!(argmax(&[]), 0);
        assert_eq!(argmax(&[f32::NEG_INFINITY; 20]), 0);
        assert_eq!(argmax(&[f32::NAN, f32::NAN]), 0);
        assert_eq!(argmax(&[f32::NAN, -3.0]), 1);
        assert_eq!(argmax(&[-0.0, 0.0]), 0);
    }

    #[test]
    fn top_k_is_sorted_with_low_index_ties() {
        let row = [1.0, 3.0, 3.0, f32::NEG_INFINITY, 2.0, f32::NAN, 3.0];
        assert_eq!(top_k(&row, 3), vec![(1, 3.0), (2, 3.0), (6, 3.0)]);
        assert_eq!(
            top_k(&row, 5),
            vec![(1, 3.0), (2, 3.0), (6, 3.0), (4, 2.0), (0, 1.0)]
        );
        assert_eq!(top_k(&row, 9).len(), 5, "only finite values");
        assert!(top_k(&row, 0).is_empty());
    }

    #[test]
    fn top_k_matches_a_full_sort() {
        let row: Vec<f32> = (0..5000)
            .map(|i| ((i * 2_654_435_761u64 as usize) % 300) as f32 * 0.5)
            .collect();
        let mut idx: Vec<u32> = (0..row.len() as u32).collect();
        idx.sort_by(|&a, &b| row[b as usize].total_cmp(&row[a as usize]).then(a.cmp(&b)));
        for k in [1, 2, 5, 17] {
            let want: Vec<(u32, f32)> = idx[..k].iter().map(|&i| (i, row[i as usize])).collect();
            assert_eq!(top_k(&row, k), want, "k={k}");
        }
    }

    #[test]
    fn log_softmax_normalizes_and_keeps_masks() {
        let mut row = [1.0f32, 2.0, f32::NEG_INFINITY, 0.5];
        log_softmax(&mut row);
        let total: f32 = row.iter().map(|v| v.exp()).sum();
        assert!((total - 1.0).abs() < 1e-6);
        assert_eq!(row[2], f32::NEG_INFINITY);
        let mut masked = [f32::NEG_INFINITY; 3];
        log_softmax(&mut masked);
        assert_eq!(masked, [f32::NEG_INFINITY; 3]);
    }

    #[test]
    fn log_add_handles_infinities() {
        assert_eq!(log_add(f32::NEG_INFINITY, -2.0), -2.0);
        assert_eq!(log_add(-2.0, f32::NEG_INFINITY), -2.0);
        assert!((log_add(0.0, 0.0) - 2f32.ln()).abs() < 1e-7);
        assert_eq!(log_sum_exp(&[]), f32::NEG_INFINITY);
    }
}
