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
//! Training gradient utilities (norms, stats, cached-raster mapping).

use super::raster_gaussian::CachedRasterGrad;
use super::training::SceneGrads;

pub const GRAD_STATS_STRIDE: usize = 2;

#[derive(Clone, Debug, Default)]
pub struct GradStats {
    /// Per-splat variance numerator accumulator (index 0) and sample count (index 1).
    pub per_splat: Vec<f32>,
}

impl GradStats {
    pub fn new(count: usize) -> Self {
        Self {
            per_splat: vec![0.0; count * GRAD_STATS_STRIDE],
        }
    }

    pub fn accumulate_contribution(&mut self, splat: usize, grad: &CachedRasterGrad) {
        let norm_sq = grad.contribution_norm_sq();
        let base = splat * GRAD_STATS_STRIDE;
        self.per_splat[base] += norm_sq;
        self.per_splat[base + 1] += 1.0;
    }

    pub fn variance(&self, splat: usize) -> f32 {
        let base = splat * GRAD_STATS_STRIDE;
        let sum_sq = self.per_splat[base];
        let n = self.per_splat[base + 1];
        if n <= 1.0 {
            return 0.0;
        }
        let mean = sum_sq / n;
        (sum_sq - mean * mean / n).max(0.0) / (n - 1.0)
    }
}

pub fn color_alpha_grad_to_raster_grad(color_alpha_grad: &[f32], splat: usize) -> CachedRasterGrad {
    let base = splat * 4;
    CachedRasterGrad {
        color_alpha: [
            color_alpha_grad[base],
            color_alpha_grad[base + 1],
            color_alpha_grad[base + 2],
            color_alpha_grad[base + 3],
        ],
        ..Default::default()
    }
}

pub fn compute_grad_norms(scene_grads: &SceneGrads, count: usize) -> Vec<f32> {
    let mut norms = vec![0.0f32; count];
    for splat in 0..count {
        let mut sum = 0.0f32;
        for axis in 0..3 {
            let p = scene_grads.positions[splat * 3 + axis];
            let s = scene_grads.scales[splat * 3 + axis];
            sum += p * p + s * s;
        }
        for axis in 0..4 {
            let r = scene_grads.rotations[splat * 4 + axis];
            sum += r * r;
        }
        sum += scene_grads.opacities[splat] * scene_grads.opacities[splat];
        for ch in 0..3 {
            let c = scene_grads.colors[splat * 3 + ch];
            sum += c * c;
        }
        norms[splat] = sum.sqrt();
    }
    norms
}

pub fn compute_packed_grad_norms(
    packed_grads: &[f32],
    count: u32,
    packed_param_count: u32,
) -> Vec<f32> {
    let mut norms = vec![0.0f32; count as usize];
    for splat in 0..count {
        let mut sum = 0.0f32;
        for param in 0..packed_param_count {
            let idx = (param * count + splat) as usize;
            if idx < packed_grads.len() {
                let g = packed_grads[idx];
                sum += g * g;
            }
        }
        norms[splat as usize] = sum.sqrt();
    }
    norms
}
