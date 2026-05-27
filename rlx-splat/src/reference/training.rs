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
//! Training forward/backward on the CPU reference raster path (linear radiance).

use super::binning::{build_tile_key_value_pairs, build_tile_ranges, sort_key_values};
use super::project::{ProjectedSplats, project_splats};
use super::raster::ray_splat_intersection_alpha;
use crate::core::{Camera, GaussianScene, OUTPUT_GAMMA};

#[derive(Clone, Debug)]
pub struct SplatHit {
    pub splat_id: u32,
    pub alpha: f32,
    pub trans_before: f32,
    pub color: [f32; 3],
}

#[derive(Clone, Debug, Default)]
pub struct PixelTrace {
    pub hits: Vec<SplatHit>,
    pub trans_final: f32,
}

#[derive(Clone, Debug)]
pub struct TrainingForward {
    pub rgba_linear: Vec<f32>,
    pub traces: Vec<PixelTrace>,
    pub projected: ProjectedSplats,
    pub sorted_values: Vec<u32>,
    pub tile_ranges: Vec<u32>,
}

/// Shared project / bin / sort state for one training frame (CPU or GPU raster).
#[derive(Clone, Debug)]
pub struct TrainingPrepare {
    pub projected: ProjectedSplats,
    pub sorted_values: Vec<u32>,
    pub tile_ranges: Vec<u32>,
}

pub fn linearize_background(background: [f32; 3]) -> [f32; 3] {
    [
        background[0].max(0.0).powf(OUTPUT_GAMMA),
        background[1].max(0.0).powf(OUTPUT_GAMMA),
        background[2].max(0.0).powf(OUTPUT_GAMMA),
    ]
}

pub fn build_training_prepare(
    scene: &GaussianScene,
    camera: &Camera,
    width: u32,
    height: u32,
    tile_size: u32,
    tile_width: u32,
    radius_scale: f32,
    alpha_cutoff: f32,
    max_list_entries: u32,
) -> TrainingPrepare {
    let projected = project_splats(scene, camera, width, height, radius_scale, alpha_cutoff);
    let tile_height = height.div_ceil(tile_size);
    let tile_count = tile_width * tile_height;
    let (keys, values, generated) = build_tile_key_value_pairs(
        &projected,
        tile_width,
        tile_height,
        tile_size,
        max_list_entries,
    );
    let sorted_count = generated.min(max_list_entries);
    let (_, sorted_values) = sort_key_values(&keys, &values, sorted_count);
    let tile_ranges = build_tile_ranges(&keys, sorted_count, tile_count);
    TrainingPrepare {
        projected,
        sorted_values,
        tile_ranges,
    }
}

/// Per-pixel hit traces for training backward (CPU; pairs with GPU linear raster).
pub fn capture_training_traces(
    prep: &TrainingPrepare,
    camera: &Camera,
    width: u32,
    height: u32,
    tile_size: u32,
    tile_width: u32,
    alpha_cutoff: f32,
    max_splat_steps: u32,
    transmittance_threshold: f32,
) -> Vec<PixelTrace> {
    let projected = &prep.projected;
    let sorted_values = &prep.sorted_values;
    let tile_ranges = &prep.tile_ranges;
    let pixel_count = (width * height) as usize;
    let mut traces = vec![PixelTrace::default(); pixel_count];

    for py in 0..height {
        let tile_y = py / tile_size;
        for px in 0..width {
            let pix = (py * width + px) as usize;
            let tile = tile_y * tile_width + px / tile_size;
            let range_base = tile as usize * 2;
            let start = tile_ranges[range_base];
            let end = tile_ranges[range_base + 1];
            if start == 0xFFFF_FFFF || end <= start {
                traces[pix].trans_final = 1.0;
                continue;
            }
            let ray = camera.screen_to_world_ray([px as f32 + 0.5, py as f32 + 0.5], width, height);
            let mut trans = 1.0f32;
            let mut steps = 0u32;
            for splat_id in &sorted_values[start as usize..end as usize] {
                if steps >= max_splat_steps {
                    break;
                }
                let splat_id = *splat_id as usize;
                if projected.valid[splat_id] == 0 {
                    continue;
                }
                let alpha = ray_splat_intersection_alpha(projected, splat_id, ray, alpha_cutoff);
                if alpha < alpha_cutoff {
                    continue;
                }
                let rgb_base = splat_id * 4;
                let color = [
                    projected.color_alpha[rgb_base],
                    projected.color_alpha[rgb_base + 1],
                    projected.color_alpha[rgb_base + 2],
                ];
                traces[pix].hits.push(SplatHit {
                    splat_id: splat_id as u32,
                    alpha,
                    trans_before: trans,
                    color,
                });
                trans *= 1.0 - alpha;
                steps += 1;
                if trans < transmittance_threshold {
                    break;
                }
            }
            traces[pix].trans_final = trans;
        }
    }
    traces
}

/// CPU linear raster + traces (reference).
pub fn raster_training_linear_cpu(
    prep: &TrainingPrepare,
    camera: &Camera,
    background: [f32; 3],
    width: u32,
    height: u32,
    tile_size: u32,
    tile_width: u32,
    alpha_cutoff: f32,
    max_splat_steps: u32,
    transmittance_threshold: f32,
) -> (Vec<f32>, Vec<PixelTrace>) {
    let projected = &prep.projected;
    let sorted_values = &prep.sorted_values;
    let tile_ranges = &prep.tile_ranges;
    let pixel_count = (width * height) as usize;
    let mut rgba_linear = vec![0.0f32; pixel_count * 4];
    let mut traces = vec![PixelTrace::default(); pixel_count];
    let bg_linear = linearize_background(background);

    for py in 0..height {
        let tile_y = py / tile_size;
        for px in 0..width {
            let pix = (py * width + px) as usize;
            let out_base = pix * 4;
            let tile = tile_y * tile_width + px / tile_size;
            let range_base = tile as usize * 2;
            let start = tile_ranges[range_base];
            let end = tile_ranges[range_base + 1];
            if start == 0xFFFF_FFFF || end <= start {
                rgba_linear[out_base] = bg_linear[0];
                rgba_linear[out_base + 1] = bg_linear[1];
                rgba_linear[out_base + 2] = bg_linear[2];
                rgba_linear[out_base + 3] = 1.0;
                traces[pix].trans_final = 1.0;
                continue;
            }
            let ray = camera.screen_to_world_ray([px as f32 + 0.5, py as f32 + 0.5], width, height);
            let mut accum = [0.0f32; 3];
            let mut trans = 1.0f32;
            let mut steps = 0u32;
            for splat_id in &sorted_values[start as usize..end as usize] {
                if steps >= max_splat_steps {
                    break;
                }
                let splat_id = *splat_id as usize;
                if projected.valid[splat_id] == 0 {
                    continue;
                }
                let alpha = ray_splat_intersection_alpha(projected, splat_id, ray, alpha_cutoff);
                if alpha < alpha_cutoff {
                    continue;
                }
                let rgb_base = splat_id * 4;
                let color = [
                    projected.color_alpha[rgb_base],
                    projected.color_alpha[rgb_base + 1],
                    projected.color_alpha[rgb_base + 2],
                ];
                traces[pix].hits.push(SplatHit {
                    splat_id: splat_id as u32,
                    alpha,
                    trans_before: trans,
                    color,
                });
                accum[0] += trans * alpha * color[0];
                accum[1] += trans * alpha * color[1];
                accum[2] += trans * alpha * color[2];
                trans *= 1.0 - alpha;
                steps += 1;
                if trans < transmittance_threshold {
                    break;
                }
            }
            rgba_linear[out_base] = accum[0] + trans * bg_linear[0];
            rgba_linear[out_base + 1] = accum[1] + trans * bg_linear[1];
            rgba_linear[out_base + 2] = accum[2] + trans * bg_linear[2];
            rgba_linear[out_base + 3] = 1.0 - trans;
            traces[pix].trans_final = trans;
        }
    }

    (rgba_linear, traces)
}

pub fn training_forward_from_parts(
    prep: TrainingPrepare,
    rgba_linear: Vec<f32>,
    traces: Vec<PixelTrace>,
) -> TrainingForward {
    TrainingForward {
        rgba_linear,
        traces,
        projected: prep.projected,
        sorted_values: prep.sorted_values,
        tile_ranges: prep.tile_ranges,
    }
}

pub fn render_training_forward(
    scene: &GaussianScene,
    camera: &Camera,
    background: [f32; 3],
    width: u32,
    height: u32,
    tile_size: u32,
    tile_width: u32,
    radius_scale: f32,
    alpha_cutoff: f32,
    max_splat_steps: u32,
    transmittance_threshold: f32,
    max_list_entries: u32,
) -> TrainingForward {
    let prep = build_training_prepare(
        scene,
        camera,
        width,
        height,
        tile_size,
        tile_width,
        radius_scale,
        alpha_cutoff,
        max_list_entries,
    );
    let (rgba_linear, traces) = raster_training_linear_cpu(
        &prep,
        camera,
        background,
        width,
        height,
        tile_size,
        tile_width,
        alpha_cutoff,
        max_splat_steps,
        transmittance_threshold,
    );
    training_forward_from_parts(prep, rgba_linear, traces)
}

/// Per-splat RGBA (linear color + opacity) gradients from raster backward.
pub fn rasterize_backward(
    forward: &TrainingForward,
    pixel_rgb_grad: &[f32],
    background: [f32; 3],
    width: u32,
    height: u32,
    loss_grad_clip: f32,
) -> Vec<f32> {
    let count = forward.projected.color_alpha.len() / 4;
    let mut color_alpha_grad = vec![0.0f32; count * 4];
    let bg_linear = linearize_background(background);
    let clip = loss_grad_clip.max(1e-8);

    for py in 0..height {
        for px in 0..width {
            let pix = (py * width + px) as usize;
            let grad_base = pix * 3;
            let mut d_ld_rgb = [
                pixel_rgb_grad[grad_base],
                pixel_rgb_grad[grad_base + 1],
                pixel_rgb_grad[grad_base + 2],
            ];
            for channel in 0..3 {
                d_ld_rgb[channel] = d_ld_rgb[channel].clamp(-clip, clip);
            }
            let trace = &forward.traces[pix];
            if trace.hits.is_empty() {
                continue;
            }
            let mut d_ld_trans = d_ld_rgb[0] * bg_linear[0]
                + d_ld_rgb[1] * bg_linear[1]
                + d_ld_rgb[2] * bg_linear[2];
            for hit in trace.hits.iter().rev() {
                let splat = hit.splat_id as usize;
                let base = splat * 4;
                let t = hit.trans_before;
                let alpha = hit.alpha;
                let color = hit.color;
                let one_minus_alpha = (1.0 - alpha).max(1e-8);

                for ch in 0..3 {
                    color_alpha_grad[base + ch] += d_ld_rgb[ch] * t * alpha;
                }
                let mut d_ld_alpha =
                    t * (d_ld_rgb[0] * color[0] + d_ld_rgb[1] * color[1] + d_ld_rgb[2] * color[2]);
                d_ld_alpha += d_ld_trans * (-t / one_minus_alpha);
                color_alpha_grad[base + 3] += d_ld_alpha.clamp(-clip, clip);

                d_ld_trans = d_ld_trans * one_minus_alpha
                    + d_ld_rgb[0] * alpha * color[0]
                    + d_ld_rgb[1] * alpha * color[1]
                    + d_ld_rgb[2] * alpha * color[2];
            }
        }
    }
    color_alpha_grad
}

/// Backprop projected color/opacity grads into scene parameter grads.
pub fn backprop_scene_grads(
    scene: &GaussianScene,
    camera: &Camera,
    forward: &TrainingForward,
    pixel_rgb_grad: &[f32],
    background: [f32; 3],
    width: u32,
    height: u32,
    radius_scale: f32,
    alpha_cutoff: f32,
    max_anisotropy: f32,
    sh_band: u32,
    loss_grad_clip: f32,
) -> SceneGrads {
    let color_alpha_grad = rasterize_backward(
        forward,
        pixel_rgb_grad,
        background,
        width,
        height,
        loss_grad_clip,
    );
    backprop_scene_grads_with_color_alpha_grad(
        scene,
        camera,
        forward,
        pixel_rgb_grad,
        &color_alpha_grad,
        background,
        width,
        height,
        radius_scale,
        alpha_cutoff,
        max_anisotropy,
        sh_band,
        loss_grad_clip,
    )
}

/// Scene backprop when [`rasterize_backward`] was already run (e.g. on GPU).
pub fn backprop_scene_grads_with_color_alpha_grad(
    scene: &GaussianScene,
    camera: &Camera,
    forward: &TrainingForward,
    pixel_rgb_grad: &[f32],
    color_alpha_grad: &[f32],
    background: [f32; 3],
    width: u32,
    height: u32,
    radius_scale: f32,
    alpha_cutoff: f32,
    max_anisotropy: f32,
    sh_band: u32,
    loss_grad_clip: f32,
) -> SceneGrads {
    let count = scene.count();
    let mut grads = SceneGrads::zeroed(
        count,
        if scene.sh_coeff_count > 0 {
            scene.sh_coeff_count
        } else {
            0
        },
    );

    backprop_splat_color_grads(
        scene,
        camera,
        color_alpha_grad,
        radius_scale,
        max_anisotropy,
        sh_band,
        &mut grads,
    );

    accumulate_geometry_grads_from_traces(
        scene,
        camera,
        forward,
        pixel_rgb_grad,
        background,
        width,
        height,
        radius_scale,
        alpha_cutoff,
        loss_grad_clip,
        &mut grads,
    );

    grads
}

fn backprop_splat_color_grads(
    scene: &GaussianScene,
    camera: &Camera,
    color_alpha_grad: &[f32],
    radius_scale: f32,
    max_anisotropy: f32,
    sh_band: u32,
    grads: &mut SceneGrads,
) {
    let count = scene.count();
    for splat in 0..count {
        accumulate_one_splat_color_grad(
            scene,
            camera,
            color_alpha_grad,
            splat,
            radius_scale,
            max_anisotropy,
            sh_band,
            grads,
        );
    }
}

fn accumulate_one_splat_color_grad(
    scene: &GaussianScene,
    camera: &Camera,
    color_alpha_grad: &[f32],
    splat: usize,
    radius_scale: f32,
    max_anisotropy: f32,
    sh_band: u32,
    grads: &mut SceneGrads,
) {
    let base = splat * 4;
    let d_color = [
        color_alpha_grad[base],
        color_alpha_grad[base + 1],
        color_alpha_grad[base + 2],
    ];
    if scene.sh_coeff_count > 0 {
        for ch in 0..3 {
            grads.sh_coeffs[splat * scene.sh_coeff_count * 3 + ch] = d_color[ch];
        }
    } else {
        for ch in 0..3 {
            grads.colors[splat * 3 + ch] = d_color[ch];
        }
    }
    let d_raster = super::grads::color_alpha_grad_to_raster_grad(color_alpha_grad, splat);
    let (d_pos_r, d_scale_r, d_rot_r, d_raw) =
        super::raster_gaussian::backprop_build_raster_gaussian(
            scene,
            splat,
            camera,
            radius_scale,
            max_anisotropy,
            sh_band,
            &d_raster,
        );
    for axis in 0..3 {
        grads.positions[splat * 3 + axis] += d_pos_r[axis];
        grads.scales[splat * 3 + axis] += d_scale_r[axis];
    }
    for axis in 0..4 {
        grads.rotations[splat * 4 + axis] += d_rot_r[axis];
    }
    let alpha = scene.opacities[splat];
    grads.opacities[splat] += d_raw.max(color_alpha_grad[base + 3] * alpha * (1.0 - alpha));
}

fn accumulate_geometry_grads_from_traces(
    scene: &GaussianScene,
    camera: &Camera,
    forward: &TrainingForward,
    pixel_rgb_grad: &[f32],
    background: [f32; 3],
    width: u32,
    height: u32,
    radius_scale: f32,
    alpha_cutoff: f32,
    loss_grad_clip: f32,
    grads: &mut SceneGrads,
) {
    let bg_linear = linearize_background(background);
    let clip = loss_grad_clip.max(1e-8);
    #[cfg(feature = "parallel")]
    {
        let sh_coeff_count = if scene.sh_coeff_count > 0 {
            scene.sh_coeff_count
        } else {
            0
        };
        use rayon::prelude::*;
        let partials: Vec<_> = (0..height)
            .into_par_iter()
            .map(|py| {
                let mut local = SceneGrads::zeroed(scene.count(), sh_coeff_count);
                let mut projected = forward.projected.clone();
                accumulate_geometry_rows(
                    scene,
                    camera,
                    &forward.traces,
                    pixel_rgb_grad,
                    bg_linear,
                    width,
                    height,
                    py,
                    py + 1,
                    radius_scale,
                    alpha_cutoff,
                    clip,
                    &mut projected,
                    &mut local,
                );
                local
            })
            .collect();
        for p in partials {
            grads.merge_from(&p);
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut projected = forward.projected.clone();
        accumulate_geometry_rows(
            scene,
            camera,
            &forward.traces,
            pixel_rgb_grad,
            bg_linear,
            width,
            height,
            0,
            height,
            radius_scale,
            alpha_cutoff,
            clip,
            &mut projected,
            grads,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_geometry_rows(
    scene: &GaussianScene,
    camera: &Camera,
    traces: &[PixelTrace],
    pixel_rgb_grad: &[f32],
    bg_linear: [f32; 3],
    width: u32,
    height: u32,
    py_start: u32,
    py_end: u32,
    radius_scale: f32,
    alpha_cutoff: f32,
    clip: f32,
    projected: &mut super::project::ProjectedSplats,
    grads: &mut SceneGrads,
) {
    for py in py_start..py_end {
        for px in 0..width {
            let pix = (py * width + px) as usize;
            let trace = &traces[pix];
            if trace.hits.is_empty() {
                continue;
            }
            let ray = camera.screen_to_world_ray([px as f32 + 0.5, py as f32 + 0.5], width, height);
            let grad_base = pix * 3;
            let mut d_ld_rgb = [
                pixel_rgb_grad[grad_base],
                pixel_rgb_grad[grad_base + 1],
                pixel_rgb_grad[grad_base + 2],
            ];
            for channel in 0..3 {
                d_ld_rgb[channel] = d_ld_rgb[channel].clamp(-clip, clip);
            }
            let mut d_ld_trans = d_ld_rgb[0] * bg_linear[0]
                + d_ld_rgb[1] * bg_linear[1]
                + d_ld_rgb[2] * bg_linear[2];
            for hit in trace.hits.iter().rev() {
                let splat = hit.splat_id as usize;
                let t = hit.trans_before;
                let alpha = hit.alpha;
                let color = hit.color;
                let one_minus_alpha = (1.0 - alpha).max(1e-8);

                let mut d_ld_alpha =
                    t * (d_ld_rgb[0] * color[0] + d_ld_rgb[1] * color[1] + d_ld_rgb[2] * color[2]);
                d_ld_alpha += d_ld_trans * (-t / one_minus_alpha);
                d_ld_alpha = d_ld_alpha.clamp(-clip, clip);

                let (d_pos, d_scale, d_rot, d_opacity) =
                    super::raster::backprop_ray_hit_alpha_numeric_projected(
                        scene,
                        splat,
                        camera,
                        width,
                        height,
                        ray,
                        radius_scale,
                        alpha_cutoff,
                        d_ld_alpha,
                        projected,
                    );
                for axis in 0..3 {
                    grads.positions[splat * 3 + axis] += d_pos[axis];
                    grads.scales[splat * 3 + axis] += d_scale[axis];
                }
                for axis in 0..4 {
                    grads.rotations[splat * 4 + axis] += d_rot[axis];
                }
                grads.opacities[splat] += d_opacity;

                d_ld_trans = d_ld_trans * one_minus_alpha
                    + d_ld_rgb[0] * alpha * color[0]
                    + d_ld_rgb[1] * alpha * color[1]
                    + d_ld_rgb[2] * alpha * color[2];
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct SceneGrads {
    pub positions: Vec<f32>,
    pub scales: Vec<f32>,
    pub rotations: Vec<f32>,
    pub opacities: Vec<f32>,
    pub colors: Vec<f32>,
    pub sh_coeffs: Vec<f32>,
}

impl SceneGrads {
    pub fn zeroed(count: usize, sh_coeff_count: usize) -> Self {
        Self {
            positions: vec![0.0; count * 3],
            scales: vec![0.0; count * 3],
            rotations: vec![0.0; count * 4],
            opacities: vec![0.0; count],
            colors: vec![0.0; count * 3],
            sh_coeffs: vec![0.0; count * sh_coeff_count * 3],
        }
    }

    pub fn merge_from(&mut self, other: &Self) {
        for (a, b) in self.positions.iter_mut().zip(&other.positions) {
            *a += b;
        }
        for (a, b) in self.scales.iter_mut().zip(&other.scales) {
            *a += b;
        }
        for (a, b) in self.rotations.iter_mut().zip(&other.rotations) {
            *a += b;
        }
        for (a, b) in self.opacities.iter_mut().zip(&other.opacities) {
            *a += b;
        }
        for (a, b) in self.colors.iter_mut().zip(&other.colors) {
            *a += b;
        }
        for (a, b) in self.sh_coeffs.iter_mut().zip(&other.sh_coeffs) {
            *a += b;
        }
    }
}
