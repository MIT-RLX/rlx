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
//! Initialize Gaussian scenes from point clouds.

use anyhow::{Result, ensure};

use crate::core::GaussianScene;
use crate::core::math::quaternion_from_rotation_matrix;
use crate::core::sh::{SUPPORTED_SH_COEFF_COUNT, pad_sh_coeffs, rgb_to_sh0};

use super::colmap::GaussianInitHyperParams;

const MIN_SCALE: f32 = 1e-4;
const MAX_SCALE: f32 = 1e4;
const NEIGHBOR_COUNT: usize = 8;

pub fn build_scene_from_positions_colors(
    mut positions: Vec<f32>,
    colors: Vec<f32>,
    seed: u64,
    init_hparams: Option<&GaussianInitHyperParams>,
) -> Result<GaussianScene> {
    let count = positions.len() / 3;
    ensure!(
        count > 0 && colors.len() == count * 3,
        "position/color count mismatch"
    );
    let mut rng = Rng::new(seed);
    if let Some(h) = init_hparams {
        if let Some(std) = h.position_jitter_std {
            if std > 0.0 {
                for v in &mut positions {
                    *v += rng.normal(0.0, std);
                }
            }
        }
    }
    let (mut scales, rotations) = point_local_gaussian_axes(&positions, count);
    if let Some(h) = init_hparams {
        if let Some(base) = h.base_scale {
            let median = median_max_axis(&scales).max(1e-6);
            let factor = base.max(MIN_SCALE) / median;
            for axis in &mut scales {
                *axis *= factor;
            }
        }
        if let Some(ratio) = h.scale_jitter_ratio {
            if ratio > 0.0 {
                let lo = (1.0 - ratio).max(MIN_SCALE);
                let hi = 1.0 + ratio;
                for v in &mut scales {
                    *v *= rng.uniform(lo, hi);
                }
            }
        }
    }
    for v in &mut scales {
        *v = v.clamp(MIN_SCALE, MAX_SCALE).ln();
    }
    let opacity = init_hparams
        .and_then(|h| h.initial_opacity)
        .unwrap_or(0.1)
        .clamp(1e-4, 0.9999);
    let mut sh0 = Vec::with_capacity(count * 3);
    for splat in 0..count {
        let rgb = [
            colors[splat * 3],
            colors[splat * 3 + 1],
            colors[splat * 3 + 2],
        ];
        sh0.extend_from_slice(&rgb_to_sh0(rgb));
    }
    let sh_coeffs = pad_sh_coeffs(&sh0, count, SUPPORTED_SH_COEFF_COUNT);
    Ok(GaussianScene::new(
        positions,
        scales,
        rotations,
        vec![opacity; count],
        colors,
        sh_coeffs,
        SUPPORTED_SH_COEFF_COUNT,
    ))
}

fn median_max_axis(scales: &[f32]) -> f32 {
    let count = scales.len() / 3;
    if count == 0 {
        return 1.0;
    }
    let mut vals: Vec<f32> = (0..count)
        .map(|s| scales[s * 3].max(scales[s * 3 + 1]).max(scales[s * 3 + 2]))
        .collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    vals[vals.len() / 2]
}

fn point_local_gaussian_axes(positions: &[f32], count: usize) -> (Vec<f32>, Vec<f32>) {
    if count == 0 {
        return (Vec::new(), Vec::new());
    }
    if count == 1 {
        return (vec![MIN_SCALE; 3], vec![1.0, 0.0, 0.0, 0.0]);
    }
    let (axis_scales, rotation_mats, nearest_scales) =
        point_local_covariance_frames(positions, count, NEIGHBOR_COUNT);
    let mut scales = vec![0.0f32; count * 3];
    let mut rotations = vec![0.0f32; count * 4];
    for splat in 0..count {
        let clipped = [
            axis_scales[splat * 3].clamp(MIN_SCALE, MAX_SCALE),
            axis_scales[splat * 3 + 1].clamp(MIN_SCALE, MAX_SCALE),
            axis_scales[splat * 3 + 2].clamp(MIN_SCALE, MAX_SCALE),
        ];
        let reference = clipped[0].max(MIN_SCALE);
        let factor = nearest_scales[splat] / reference;
        scales[splat * 3] = clipped[0] * factor;
        scales[splat * 3 + 1] = clipped[1] * factor;
        scales[splat * 3 + 2] = clipped[2] * factor;
        let rot = [
            [
                rotation_mats[splat * 9],
                rotation_mats[splat * 9 + 1],
                rotation_mats[splat * 9 + 2],
            ],
            [
                rotation_mats[splat * 9 + 3],
                rotation_mats[splat * 9 + 4],
                rotation_mats[splat * 9 + 5],
            ],
            [
                rotation_mats[splat * 9 + 6],
                rotation_mats[splat * 9 + 7],
                rotation_mats[splat * 9 + 8],
            ],
        ];
        rotations[splat * 4..splat * 4 + 4].copy_from_slice(&quaternion_from_rotation_matrix(rot));
    }
    (scales, rotations)
}

fn point_local_covariance_frames(
    positions: &[f32],
    count: usize,
    neighbor_count: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let k = neighbor_count.min(count).max(2);
    let mut axis_scales = vec![0.0f32; count * 3];
    let mut rotation_mats = vec![0.0f32; count * 9];
    let mut nearest_scales = vec![MIN_SCALE; count];
    for i in 0..count {
        let pi = [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]];
        let mut dists: Vec<(f32, usize)> = (0..count)
            .filter(|&j| j != i)
            .map(|j| {
                let pj = [positions[j * 3], positions[j * 3 + 1], positions[j * 3 + 2]];
                let d =
                    ((pi[0] - pj[0]).powi(2) + (pi[1] - pj[1]).powi(2) + (pi[2] - pj[2]).powi(2))
                        .sqrt();
                (d, j)
            })
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        nearest_scales[i] = dists
            .first()
            .map(|d| d.0)
            .unwrap_or(MIN_SCALE)
            .clamp(MIN_SCALE, MAX_SCALE);
        let neighbors: Vec<[f32; 3]> = dists
            .iter()
            .take(k)
            .map(|(_, j)| [positions[j * 3], positions[j * 3 + 1], positions[j * 3 + 2]])
            .collect();
        let mean = neighbors.iter().fold([0.0f32; 3], |acc, p| {
            [acc[0] + p[0], acc[1] + p[1], acc[2] + p[2]]
        });
        let mean = [
            mean[0] / neighbors.len() as f32,
            mean[1] / neighbors.len() as f32,
            mean[2] / neighbors.len() as f32,
        ];
        let mut cov = [[0.0f32; 3]; 3];
        for p in &neighbors {
            let d = [p[0] - mean[0], p[1] - mean[1], p[2] - mean[2]];
            for r in 0..3 {
                for c in 0..3 {
                    cov[r][c] += d[r] * d[c];
                }
            }
        }
        let denom = (neighbors.len() as f32 - 1.0).max(1.0);
        for r in 0..3 {
            for c in 0..3 {
                cov[r][c] /= denom;
            }
        }
        let (eigvecs, eigvals) = eigen_symmetric_3x3(cov);
        axis_scales[i * 3] = eigvals[0].max(0.0).sqrt();
        axis_scales[i * 3 + 1] = eigvals[1].max(0.0).sqrt();
        axis_scales[i * 3 + 2] = eigvals[2].max(0.0).sqrt();
        for r in 0..3 {
            for c in 0..3 {
                rotation_mats[i * 9 + r * 3 + c] = eigvecs[c][r];
            }
        }
    }
    (axis_scales, rotation_mats, nearest_scales)
}

fn eigen_symmetric_3x3(a: [[f32; 3]; 3]) -> ([[f32; 3]; 3], [f32; 3]) {
    let mut b = a;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..16 {
        let mut p = 0usize;
        let mut q = 1usize;
        let mut max = b[0][1].abs();
        if b[0][2].abs() > max {
            max = b[0][2].abs();
            p = 0;
            q = 2;
        }
        if b[1][2].abs() > max {
            p = 1;
            q = 2;
        }
        if max < 1e-12 {
            break;
        }
        let theta = 0.5 * (b[q][q] - b[p][p]).atan2(2.0 * b[p][q]);
        let c = theta.cos();
        let s = theta.sin();
        jacobi_rotate(&mut b, &mut v, p, q, c, s);
    }
    ([v[0], v[1], v[2]], [b[0][0], b[1][1], b[2][2]])
}

fn jacobi_rotate(b: &mut [[f32; 3]; 3], v: &mut [[f32; 3]; 3], p: usize, q: usize, c: f32, s: f32) {
    for i in 0..3 {
        let bip = b[i][p];
        let biq = b[i][q];
        b[i][p] = c * bip - s * biq;
        b[i][q] = s * bip + c * biq;
    }
    for i in 0..3 {
        let bpi = b[p][i];
        let bqi = b[q][i];
        b[p][i] = c * bpi - s * bqi;
        b[q][i] = s * bpi + c * bqi;
    }
    for i in 0..3 {
        let vip = v[i][p];
        let viq = v[i][q];
        v[i][p] = c * vip - s * viq;
        v[i][q] = s * vip + c * viq;
    }
}

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 32) as u32
    }

    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let u = self.next_u32() as f32 / u32::MAX as f32;
        lo + (hi - lo) * u
    }

    fn normal(&mut self, mean: f32, std: f32) -> f32 {
        let u1 = (self.next_u32() as f32 / u32::MAX as f32).max(1e-8);
        let u2 = self.next_u32() as f32 / u32::MAX as f32;
        mean + std * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}
