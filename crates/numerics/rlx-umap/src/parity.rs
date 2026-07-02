// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Parity metrics vs a reference k-NN run.

/// Summary of how closely two k-NN results match.
#[derive(Debug, Clone, Copy)]
pub struct KnnParityReport {
    pub n: usize,
    pub k: usize,
    /// Fraction of `(row, slot)` index entries that match exactly.
    pub index_match_rate: f64,
    /// Max absolute distance error over matching index entries.
    pub max_dist_error: f32,
    /// Mean absolute distance error over matching index entries.
    pub mean_dist_error: f32,
    /// L1 distance between normalized histograms of all k-NN distances.
    pub dist_hist_l1: f64,
}

/// Compare flat row-major k-NN outputs (`indices` as f32-encoded cols).
pub fn compare_knn(
    ref_idx: &[f32],
    ref_dist: &[f32],
    got_idx: &[f32],
    got_dist: &[f32],
    n: usize,
    k: usize,
) -> KnnParityReport {
    assert_eq!(ref_idx.len(), n * k);
    assert_eq!(got_idx.len(), n * k);
    assert_eq!(ref_dist.len(), n * k);
    assert_eq!(got_dist.len(), n * k);

    let mut index_matches = 0usize;
    let mut dist_err_sum = 0f32;
    let mut dist_err_max = 0f32;
    let mut dist_count = 0usize;

    for i in 0..n {
        for slot in 0..k {
            let p = i * k + slot;
            let ri = ref_idx[p] as i32;
            let gi = got_idx[p] as i32;
            if ri == gi {
                index_matches += 1;
                let err = (ref_dist[p] - got_dist[p]).abs();
                dist_err_sum += err;
                dist_err_max = dist_err_max.max(err);
                dist_count += 1;
            }
        }
    }

    let slots = n * k;
    let index_match_rate = index_matches as f64 / slots as f64;
    let mean_dist_error = if dist_count > 0 {
        dist_err_sum / dist_count as f32
    } else {
        f32::NAN
    };

    let dist_hist_l1 = histogram_l1(ref_dist, got_dist, 32);

    KnnParityReport {
        n,
        k,
        index_match_rate,
        max_dist_error: dist_err_max,
        mean_dist_error,
        dist_hist_l1,
    }
}

/// L1 between normalized histograms (same binning).
fn histogram_l1(a: &[f32], b: &[f32], bins: usize) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let min_v = a
        .iter()
        .chain(b.iter())
        .copied()
        .fold(f32::INFINITY, f32::min);
    let max_v = a
        .iter()
        .chain(b.iter())
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let span = (max_v - min_v).max(1e-8);
    let mut ha = vec![0f64; bins];
    let mut hb = vec![0f64; bins];
    for &v in a {
        let bin = ((v - min_v) / span * (bins - 1) as f32).round() as usize;
        ha[bin.min(bins - 1)] += 1.0;
    }
    for &v in b {
        let bin = ((v - min_v) / span * (bins - 1) as f32).round() as usize;
        hb[bin.min(bins - 1)] += 1.0;
    }
    let na: f64 = ha.iter().sum();
    let nb: f64 = hb.iter().sum();
    if na > 0.0 {
        for h in &mut ha {
            *h /= na;
        }
    }
    if nb > 0.0 {
        for h in &mut hb {
            *h /= nb;
        }
    }
    ha.iter().zip(hb.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// Compare two flat pairwise matrices (max abs diff).
pub fn max_pairwise_error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}
