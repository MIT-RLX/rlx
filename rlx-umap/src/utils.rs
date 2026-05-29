// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Data normalization and layout helpers.

const SMALL_STD: f64 = 1e-8;

/// Column-wise z-score statistics from row-major `data` (`n × d`).
#[derive(Debug, Clone)]
pub struct NormStats {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

impl NormStats {
    pub fn compute(data: &[f64], n: usize, d: usize) -> Self {
        let mut mean = vec![0.0; d];
        let mut std = vec![SMALL_STD; d];
        for feature in 0..d {
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            for sample in 0..n {
                let v = data[sample * d + feature];
                sum += v;
                sum_sq += v * v;
            }
            let m = sum / n as f64;
            let variance = sum_sq / n as f64 - m * m;
            mean[feature] = m;
            std[feature] = variance.sqrt() + SMALL_STD;
        }
        Self { mean, std }
    }

    /// Apply training statistics to `data` (`n × d`, in-place).
    pub fn apply(&self, data: &mut [f64], n: usize, d: usize) {
        assert_eq!(self.mean.len(), d);
        for feature in 0..d {
            let m = self.mean[feature];
            let s = self.std[feature];
            for sample in 0..n {
                let idx = sample * d + feature;
                data[idx] = (data[idx] - m) / s;
            }
        }
    }
}

/// Column-wise z-score normalization (in-place, row-major `n × d`).
pub fn normalize_data_f64(data: &mut [f64], n: usize, d: usize) {
    NormStats::compute(data, n, d).apply(data, n, d);
}

pub fn flatten_f64(data: &[Vec<f64>]) -> (Vec<f64>, usize, usize) {
    let n = data.len();
    let d = data.first().map(|r| r.len()).unwrap_or(0);
    let flat: Vec<f64> = data.iter().flatten().copied().collect();
    (flat, n, d)
}

pub fn unflatten_f64(flat: &[f64], n: usize, d: usize) -> Vec<Vec<f64>> {
    (0..n).map(|i| flat[i * d..(i + 1) * d].to_vec()).collect()
}

pub fn f64_to_f32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

pub fn f32_to_f64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

/// Deterministic synthetic data in `[0, 1)` for benchmarks.
pub fn generate_test_data(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            (0..d)
                .map(|_| {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                    const M: u64 = (1 << 21) - 1;
                    ((s >> 11) & M) as f64 / M as f64
                })
                .collect()
        })
        .collect()
}
