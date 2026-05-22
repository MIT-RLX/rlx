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
//! COLMAP / point-cloud Gaussian init hyperparameters (parity with slang-splat Python).

use anyhow::{Result, bail};

use super::types::{ColmapReconstruction, GaussianInitHyperParams};
use super::point_tables;

const MIN_SCALE: f32 = 1e-4;
const INIT_BASE_SCALE_SPACING_RATIO: f32 = 0.25;
/// `1 / sqrt(12)` — uniform jitter RMS for unit spacing.
const INIT_JITTER_SPACING_RATIO: f32 = 0.2886751345948129;
const INIT_REPLACEMENT_JITTER_BOOST: f32 = 1.5;
const INIT_SCALE_JITTER_BASE: f32 = 0.03;
const INIT_SCALE_JITTER_VARIABILITY: f32 = 0.10;
const INIT_SCALE_JITTER_MIN: f32 = 0.01;
const INIT_SCALE_JITTER_MAX: f32 = 0.16;
const INIT_OPACITY_BASE: f32 = 0.22;
const INIT_OPACITY_MIN: f32 = 0.10;
const INIT_OPACITY_MAX: f32 = 0.35;

fn clip(v: f32, lo: f32, hi: f32) -> f32 {
    v.clamp(lo, hi)
}

fn subsample_points(points: &[[f32; 3]], max_points: usize) -> Vec<[f32; 3]> {
    let n = points.len();
    if n <= max_points {
        return points.to_vec();
    }
    let mut out = Vec::with_capacity(max_points);
    for i in 0..max_points {
        let idx = (i as f64 * (n - 1) as f64 / (max_points - 1).max(1) as f64).round() as usize;
        out.push(points[idx.min(n - 1)]);
    }
    out
}

fn estimate_point_spacing(points: &[[f32; 3]]) -> (f32, f32) {
    if points.len() <= 1 {
        return (1.0, 0.15);
    }
    let sample = subsample_points(points, 2048);
    let m = sample.len();
    let mut nearest = Vec::with_capacity(m);
    for (i, pi) in sample.iter().enumerate() {
        let mut best = f32::INFINITY;
        for (j, pj) in sample.iter().enumerate() {
            if i == j {
                continue;
            }
            let d2 = (pi[0] - pj[0]).powi(2) + (pi[1] - pj[1]).powi(2) + (pi[2] - pj[2]).powi(2);
            best = best.min(d2);
        }
        if best.is_finite() && best > 0.0 {
            nearest.push(best.sqrt());
        }
    }
    if nearest.is_empty() {
        return (1.0, 0.15);
    }
    nearest.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let spacing = nearest[nearest.len() / 2].max(1e-4);
    let q1 = nearest[nearest.len() / 4];
    let q3 = nearest[(nearest.len() * 3) / 4];
    let variability = ((q3 - q1) / spacing.max(1e-6)).clamp(0.0, 1.0);
    (spacing, variability)
}

/// Suggest init hyperparameters from a point table (Nx3 positions).
pub fn suggest_points_init_hparams(points: &[[f32; 3]], max_gaussians: i32) -> Result<GaussianInitHyperParams> {
    if points.is_empty() {
        bail!("point initialization requires a non-empty point table");
    }
    let point_count = points.len();
    let chosen_count = if max_gaussians <= 0 {
        point_count
    } else {
        point_count.min(max_gaussians.max(1) as usize)
    };
    let (spacing, variability) = estimate_point_spacing(points);
    let density_scale = (point_count as f32 / chosen_count.max(1) as f32).powf(1.0 / 3.0);
    let target_spacing = (spacing * density_scale).max(1e-4);
    let replacement_factor = if chosen_count > point_count {
        INIT_REPLACEMENT_JITTER_BOOST
    } else {
        1.0
    };
    Ok(GaussianInitHyperParams {
        position_jitter_std: Some(clip(
            INIT_JITTER_SPACING_RATIO * target_spacing * replacement_factor,
            0.0,
            10.0,
        )),
        base_scale: Some(clip(
            INIT_BASE_SCALE_SPACING_RATIO * target_spacing,
            MIN_SCALE,
            10.0,
        )),
        scale_jitter_ratio: Some(clip(
            INIT_SCALE_JITTER_BASE + INIT_SCALE_JITTER_VARIABILITY * variability,
            INIT_SCALE_JITTER_MIN,
            INIT_SCALE_JITTER_MAX,
        )),
        initial_opacity: Some(clip(
            INIT_OPACITY_BASE * density_scale.sqrt(),
            INIT_OPACITY_MIN,
            INIT_OPACITY_MAX,
        )),
        color_jitter_std: Some(0.0),
    })
}

fn resolve_init_hparams(
    suggested: &GaussianInitHyperParams,
    init_hparams: Option<&GaussianInitHyperParams>,
) -> GaussianInitHyperParams {
    let Some(over) = init_hparams else {
        return suggested.clone();
    };
    GaussianInitHyperParams {
        position_jitter_std: over
            .position_jitter_std
            .or(suggested.position_jitter_std),
        base_scale: over.base_scale.or(suggested.base_scale),
        scale_jitter_ratio: over
            .scale_jitter_ratio
            .or(suggested.scale_jitter_ratio),
        initial_opacity: over.initial_opacity.or(suggested.initial_opacity),
        color_jitter_std: over.color_jitter_std.or(suggested.color_jitter_std),
    }
}

/// Suggest init hyperparameters from COLMAP sparse points.
pub fn suggest_colmap_init_hparams(
    recon: &ColmapReconstruction,
    max_gaussians: i32,
    min_track_length: i32,
) -> Result<GaussianInitHyperParams> {
    let (xyz, _) = point_tables(recon, min_track_length);
    if xyz.is_empty() {
        return Err(anyhow::anyhow!("{}", min_track_length_error(min_track_length)));
    }
    let count = xyz.len() / 3;
    let mut points = Vec::with_capacity(count);
    for i in 0..count {
        let p = [
            xyz[i * 3],
            xyz[i * 3 + 1],
            xyz[i * 3 + 2],
        ];
        if p[0].is_finite() && p[1].is_finite() && p[2].is_finite() {
            points.push(p);
        }
    }
    if points.is_empty() {
        return Err(anyhow::anyhow!("{}", min_track_length_error(min_track_length)));
    }
    suggest_points_init_hparams(&points, max_gaussians)
}

/// Merge suggested COLMAP init params with optional overrides.
pub fn resolve_colmap_init_hparams(
    recon: &ColmapReconstruction,
    max_gaussians: i32,
    init_hparams: Option<&GaussianInitHyperParams>,
    min_track_length: i32,
) -> Result<GaussianInitHyperParams> {
    let suggested = suggest_colmap_init_hparams(recon, max_gaussians, min_track_length)?;
    Ok(resolve_init_hparams(&suggested, init_hparams))
}

pub(crate) fn min_track_length_error(min_track_length: i32) -> String {
    let threshold = min_track_length.max(0);
    if threshold <= 0 {
        "COLMAP reconstruction has no 3D points.".to_string()
    } else {
        format!(
            "COLMAP reconstruction has no 3D points observed by at least {threshold} cameras."
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suggest_points_init_hparams_nonempty() {
        let pts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let h = suggest_points_init_hparams(&pts, 2).unwrap();
        assert!(h.base_scale.unwrap() > 0.0);
        assert!(h.initial_opacity.unwrap() > 0.0);
    }
}
