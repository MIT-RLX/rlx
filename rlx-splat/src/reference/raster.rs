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
use super::project::{project_splat_index, project_splats, ProjectedSplats, SplatProjectOverride};
use crate::core::{Camera, GaussianScene, GAUSSIAN_SUPPORT_SIGMA_RADIUS, OUTPUT_GAMMA, quat_rotate};

pub fn ray_splat_intersection_alpha(
    projected: &ProjectedSplats,
    splat_id: usize,
    ray_direction: [f32; 3],
    alpha_cutoff: f32,
) -> f32 {
    let opacity = projected.color_alpha[splat_id * 4 + 3].clamp(0.0, 1.0);
    if opacity < alpha_cutoff {
        return 0.0;
    }
    let support_sigma_radius =
        (-2.0 * (alpha_cutoff / opacity.max(alpha_cutoff)).ln()).max(0.0).sqrt();
    if support_sigma_radius <= 1e-10 {
        return 0.0;
    }
    let support_scale = GAUSSIAN_SUPPORT_SIGMA_RADIUS / support_sigma_radius;
    let ro_local = [
        projected.pos_local[splat_id * 3] * support_scale,
        projected.pos_local[splat_id * 3 + 1] * support_scale,
        projected.pos_local[splat_id * 3 + 2] * support_scale,
    ];
    let quat = [
        projected.quat[splat_id * 4],
        projected.quat[splat_id * 4 + 1],
        projected.quat[splat_id * 4 + 2],
        projected.quat[splat_id * 4 + 3],
    ];
    let inv = [
        projected.inv_scale[splat_id * 3],
        projected.inv_scale[splat_id * 3 + 1],
        projected.inv_scale[splat_id * 3 + 2],
    ];
    let ray_local = {
        let rotated = quat_rotate(ray_direction, quat);
        [
            rotated[0] * inv[0] * support_scale,
            rotated[1] * inv[1] * support_scale,
            rotated[2] * inv[2] * support_scale,
        ]
    };
    let denom = ray_local[0] * ray_local[0]
        + ray_local[1] * ray_local[1]
        + ray_local[2] * ray_local[2];
    if denom <= 1e-10 {
        return 0.0;
    }
    let t_closest = -(ray_local[0] * ro_local[0]
        + ray_local[1] * ro_local[1]
        + ray_local[2] * ro_local[2])
        / denom;
    if t_closest <= 0.0 {
        return 0.0;
    }
    let closest = [
        ro_local[0] + ray_local[0] * t_closest,
        ro_local[1] + ray_local[1] * t_closest,
        ro_local[2] + ray_local[2] * t_closest,
    ];
    let rho2 = (closest[0] * closest[0] + closest[1] * closest[1] + closest[2] * closest[2]).max(0.0);
    opacity * (-0.5 * support_sigma_radius * support_sigma_radius * rho2).exp()
}

/// Numeric backprop of intersection alpha w.r.t. scene parameters for one ray hit.
pub fn backprop_ray_hit_alpha_numeric(
    scene: &GaussianScene,
    splat: usize,
    camera: &Camera,
    width: u32,
    height: u32,
    ray: [f32; 3],
    radius_scale: f32,
    alpha_cutoff: f32,
    d_alpha: f32,
) -> ([f32; 3], [f32; 3], [f32; 4], f32) {
    if d_alpha == 0.0 {
        return ([0.0; 3], [0.0; 3], [0.0; 4], 0.0);
    }
    let eps = 1e-4f32;
    let mut d_pos = [0.0f32; 3];
    let mut d_scale = [0.0f32; 3];
    let mut d_rot = [0.0f32; 4];
    let mut d_opacity = 0.0f32;

    let alpha_at = |temp: &GaussianScene| -> f32 {
        let projected = project_splats(temp, camera, width, height, radius_scale, alpha_cutoff);
        ray_splat_intersection_alpha(&projected, splat, ray, alpha_cutoff)
    };

    let base = alpha_at(scene);
    let loss = |a: f32| d_alpha * (a - base);

    let mut temp = scene.clone();
    for axis in 0..3 {
        let idx = splat * 3 + axis;
        temp.positions[idx] += eps;
        let plus = loss(alpha_at(&temp));
        temp.positions[idx] -= 2.0 * eps;
        let minus = loss(alpha_at(&temp));
        temp.positions[idx] += eps;
        d_pos[axis] = (plus - minus) / (2.0 * eps);
    }
    for axis in 0..3 {
        let idx = splat * 3 + axis;
        temp.scales[idx] += eps;
        let plus = loss(alpha_at(&temp));
        temp.scales[idx] -= 2.0 * eps;
        let minus = loss(alpha_at(&temp));
        temp.scales[idx] += eps;
        d_scale[axis] = (plus - minus) / (2.0 * eps);
    }
    for axis in 0..4 {
        let idx = splat * 4 + axis;
        temp.rotations[idx] += eps;
        let plus = loss(alpha_at(&temp));
        temp.rotations[idx] -= 2.0 * eps;
        let minus = loss(alpha_at(&temp));
        temp.rotations[idx] += eps;
        d_rot[axis] = (plus - minus) / (2.0 * eps);
    }
    temp.opacities[splat] += eps;
    let plus = loss(alpha_at(&temp));
    temp.opacities[splat] -= 2.0 * eps;
    let minus = loss(alpha_at(&temp));
    d_opacity = (plus - minus) / (2.0 * eps);

    (d_pos, d_scale, d_rot, d_opacity)
}

/// Numeric ray-hit backprop using a cached [`ProjectedSplats`] — reprojects one splat per FD step.
pub fn backprop_ray_hit_alpha_numeric_projected(
    scene: &GaussianScene,
    splat: usize,
    camera: &Camera,
    width: u32,
    height: u32,
    ray: [f32; 3],
    radius_scale: f32,
    alpha_cutoff: f32,
    d_alpha: f32,
    projected: &mut ProjectedSplats,
) -> ([f32; 3], [f32; 3], [f32; 4], f32) {
    if d_alpha == 0.0 {
        return ([0.0; 3], [0.0; 3], [0.0; 4], 0.0);
    }
    let eps = 1e-4f32;
    let mut d_pos = [0.0f32; 3];
    let mut d_scale = [0.0f32; 3];
    let mut d_rot = [0.0f32; 4];
    let mut d_opacity = 0.0f32;

    let mut alpha_at = |ov: SplatProjectOverride| -> f32 {
        project_splat_index(
            splat,
            scene,
            camera,
            width,
            height,
            radius_scale,
            alpha_cutoff,
            &mut projected.center_radius_depth[splat * 4..splat * 4 + 4],
            &mut projected.ellipse_conic[splat * 3..splat * 3 + 3],
            &mut projected.valid[splat],
            &mut projected.pos_local[splat * 3..splat * 3 + 3],
            &mut projected.inv_scale[splat * 3..splat * 3 + 3],
            ov,
        );
        ray_splat_intersection_alpha(projected, splat, ray, alpha_cutoff)
    };

    let base = alpha_at(SplatProjectOverride::default());
    let loss = |a: f32| d_alpha * (a - base);

    for axis in 0..3 {
        let mut pos = scene.position(splat);
        pos[axis] += eps;
        let plus = loss(alpha_at(SplatProjectOverride {
            position: Some(pos),
            ..Default::default()
        }));
        pos[axis] -= 2.0 * eps;
        let minus = loss(alpha_at(SplatProjectOverride {
            position: Some(pos),
            ..Default::default()
        }));
        d_pos[axis] = (plus - minus) / (2.0 * eps);
    }
    for axis in 0..3 {
        let mut sigma = scene.scale(splat);
        sigma[axis] += eps;
        let plus = loss(alpha_at(SplatProjectOverride {
            scale_log: Some(sigma),
            ..Default::default()
        }));
        sigma[axis] -= 2.0 * eps;
        let minus = loss(alpha_at(SplatProjectOverride {
            scale_log: Some(sigma),
            ..Default::default()
        }));
        d_scale[axis] = (plus - minus) / (2.0 * eps);
    }
    for axis in 0..4 {
        let mut rot = scene.rotation(splat);
        rot[axis] += eps;
        let plus = loss(alpha_at(SplatProjectOverride {
            rotation: Some(rot),
            ..Default::default()
        }));
        rot[axis] -= 2.0 * eps;
        let minus = loss(alpha_at(SplatProjectOverride {
            rotation: Some(rot),
            ..Default::default()
        }));
        d_rot[axis] = (plus - minus) / (2.0 * eps);
    }
    let opacity = scene.opacities[splat];
    let plus = loss(alpha_at(SplatProjectOverride {
        opacity: Some(opacity + eps),
        ..Default::default()
    }));
    let minus = loss(alpha_at(SplatProjectOverride {
        opacity: Some(opacity - eps),
        ..Default::default()
    }));
    d_opacity = (plus - minus) / (2.0 * eps);

    let _ = project_splat_index(
        splat,
        scene,
        camera,
        width,
        height,
        radius_scale,
        alpha_cutoff,
        &mut projected.center_radius_depth[splat * 4..splat * 4 + 4],
        &mut projected.ellipse_conic[splat * 3..splat * 3 + 3],
        &mut projected.valid[splat],
        &mut projected.pos_local[splat * 3..splat * 3 + 3],
        &mut projected.inv_scale[splat * 3..splat * 3 + 3],
        SplatProjectOverride::default(),
    );

    (d_pos, d_scale, d_rot, d_opacity)
}

pub fn rasterize(
    projected: &ProjectedSplats,
    sorted_values: &[u32],
    tile_ranges: &[u32],
    camera: &Camera,
    width: u32,
    height: u32,
    tile_size: u32,
    tile_width: u32,
    background: [f32; 3],
    alpha_cutoff: f32,
    _max_splat_steps: u32,
    transmittance_threshold: f32,
) -> Vec<f32> {
    let mut output = vec![0.0f32; (width * height * 4) as usize];
    let bg_linear = [
        background[0].max(0.0).powf(OUTPUT_GAMMA),
        background[1].max(0.0).powf(OUTPUT_GAMMA),
        background[2].max(0.0).powf(OUTPUT_GAMMA),
    ];
    for py in 0..height {
        let tile_y = py / tile_size;
        for px in 0..width {
            let out_base = ((py * width + px) * 4) as usize;
            let tile = tile_y * tile_width + px / tile_size;
            let range_base = tile as usize * 2;
            let start = tile_ranges[range_base];
            let mut end = tile_ranges[range_base + 1];
            if end as usize > sorted_values.len() {
                end = sorted_values.len() as u32;
            }
            if start == 0xFFFF_FFFF || end <= start {
                output[out_base] = bg_linear[0];
                output[out_base + 1] = bg_linear[1];
                output[out_base + 2] = bg_linear[2];
                output[out_base + 3] = 1.0;
                continue;
            }
            let ray = camera.screen_to_world_ray([px as f32 + 0.5, py as f32 + 0.5], width, height);
            let mut accum = [0.0f32; 3];
            let mut trans = 1.0f32;
            for splat_id in &sorted_values[start as usize..end as usize] {
                let splat_id = *splat_id as usize;
                if projected.valid[splat_id] == 0 {
                    continue;
                }
                let alpha = ray_splat_intersection_alpha(projected, splat_id, ray, alpha_cutoff);
                if alpha < alpha_cutoff {
                    continue;
                }
                let rgb_base = splat_id * 4;
                accum[0] += trans * alpha * projected.color_alpha[rgb_base];
                accum[1] += trans * alpha * projected.color_alpha[rgb_base + 1];
                accum[2] += trans * alpha * projected.color_alpha[rgb_base + 2];
                trans *= 1.0 - alpha;
                if trans < transmittance_threshold {
                    break;
                }
            }
            // Match `reference_cpu.rasterize`: composite in sRGB background space, then γ=2.2.
            let composed = [
                (accum[0] + trans * background[0]).max(0.0),
                (accum[1] + trans * background[1]).max(0.0),
                (accum[2] + trans * background[2]).max(0.0),
            ];
            output[out_base] = composed[0].powf(OUTPUT_GAMMA);
            output[out_base + 1] = composed[1].powf(OUTPUT_GAMMA);
            output[out_base + 2] = composed[2].powf(OUTPUT_GAMMA);
            output[out_base + 3] = 1.0 - trans;
        }
    }
    output
}
