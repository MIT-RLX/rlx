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
//! Numerical parity helpers shared with slang-splat-rs tests and RLX model parity suites.

/// Mean absolute error gate used by GPU vs CPU reference render tests.
pub const MEAN_ABS_ERROR_GPU_CPU: f32 = 5e-3;

/// Strict cosine distance for deterministic CPU reference paths (gradients, params).
pub const COSINE_DISTANCE_STRICT: f64 = 1e-7;

/// Render-image cosine distance gate (full RGBA frames; legacy GPU paths).
pub const COSINE_DISTANCE_RENDER: f64 = 1e-3;

/// Max per-element |Δ| vs `reference_cpu.py` (f32 Rust vs f64 Python projection; ~1.5e-6 on tiny frame).
pub const PARITY_MAX_ABS: f32 = 2e-6;
/// Cosine distance vs Python baseline (effectively zero on shared scene).
pub const PARITY_MAX_COSINE: f64 = 1e-12;

/// Projection alpha cutoff tolerance in outline tests.
pub const PROJECTION_ALPHA_TOL: f32 = 5e-4;

pub fn cosine_distance(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len() {
        let av = a[i] as f64;
        let bv = b[i] as f64;
        dot += av * bv;
        na += av * av;
        nb += bv * bv;
    }
    let denom = (na * nb).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        (1.0 - dot / denom).max(0.0)
    }
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    assert_eq!(a.len(), b.len());
    let mut max = 0.0f32;
    let mut idx = 0usize;
    for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
        let d = (av - bv).abs();
        if d > max {
            max = d;
            idx = i;
        }
    }
    (max, idx)
}

pub fn mean_abs_error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return 0.0;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f32>()
        / a.len() as f32
}

/// Strict parity with [`PARITY_MAX_ABS`] / [`PARITY_MAX_COSINE`] vs `reference_cpu.py`.
pub fn assert_parity_exact(a: &[f32], b: &[f32]) -> Result<(), String> {
    let mae = mean_abs_error(a, b);
    let cos = cosine_distance(a, b);
    let (mad, idx) = max_abs_diff(a, b);
    if mae > PARITY_MAX_ABS || cos > PARITY_MAX_COSINE || mad > PARITY_MAX_ABS {
        Err(format!(
            "parity failed: mean_abs={mae:.6e} (limit {PARITY_MAX_ABS:.6e}), \
             cosine={cos:.6e} (limit {PARITY_MAX_COSINE:.6e}), max_abs={mad:.6e} @ {idx}"
        ))
    } else {
        Ok(())
    }
}

pub fn assert_parity(a: &[f32], b: &[f32], max_mean_abs: f32, max_cosine: f64) -> Result<(), String> {
    let mae = mean_abs_error(a, b);
    let cos = cosine_distance(a, b);
    if mae > max_mean_abs || cos > max_cosine {
        let (mad, idx) = max_abs_diff(a, b);
        Err(format!(
            "parity failed: mean_abs={mae:.6e} (limit {max_mean_abs:.6e}), cosine={cos:.6e} (limit {max_cosine:.6e}), max_abs={mad:.6e} @ {idx}"
        ))
    } else {
        Ok(())
    }
}
