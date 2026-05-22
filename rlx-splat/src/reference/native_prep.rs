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
//! Shared CPU prep for native splat forward (project → bin → sort → ray grid).

use crate::core::{Camera, GaussianScene, OUTPUT_GAMMA};

use super::{
    build_tile_key_value_pairs, build_tile_ranges, project_splats, sort_key_values, RenderParams,
};

/// Uniform / constant params for the per-pixel raster kernel (all backends).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SplatRasterParams {
    pub width: u32,
    pub height: u32,
    pub tile_size: u32,
    pub tile_width: u32,
    pub alpha_cutoff: f32,
    pub transmittance_threshold: f32,
    pub bg_r: f32,
    pub bg_g: f32,
    pub bg_b: f32,
    /// `0` = unlimited splats per pixel (display); training sets a finite cap.
    pub max_splat_steps: u32,
}

/// Projected scene + tile lists + per-pixel rays for one frame.
#[derive(Clone, Debug)]
pub struct PreparedRaster {
    pub color_alpha: Vec<f32>,
    pub valid: Vec<u32>,
    pub pos_local: Vec<f32>,
    pub inv_scale: Vec<f32>,
    pub quat: Vec<f32>,
    pub sorted_values: Vec<u32>,
    pub tile_ranges: Vec<u32>,
    pub rays: Vec<f32>,
    pub params: SplatRasterParams,
}

/// Build [`GaussianScene`] from flat attribute slices.
pub fn scene_from_slices(
    positions: &[f32],
    scales: &[f32],
    rotations: &[f32],
    opacities: &[f32],
    colors: &[f32],
    sh_coeffs: &[f32],
) -> GaussianScene {
    let count = positions.len() / 3;
    let sh_coeff_count = if count == 0 {
        1
    } else {
        (sh_coeffs.len() / (count * 3)).max(1)
    };
    GaussianScene::new(
        positions.to_vec(),
        scales.to_vec(),
        rotations.to_vec(),
        opacities.to_vec(),
        colors.to_vec(),
        sh_coeffs.to_vec(),
        sh_coeff_count,
    )
}

/// Camera + background from the 23-float `meta` buffer used by RLX splat ops.
pub fn camera_and_background_from_meta(meta: &[f32]) -> (Camera, [f32; 3]) {
    let meta = &meta[..meta.len().min(23)];
    let camera = Camera::look_at(
        [meta[0], meta[1], meta[2]],
        [meta[3], meta[4], meta[5]],
        [meta[6], meta[7], meta[8]],
        meta[9],
        meta[10],
        meta[11],
    );
    let background = [meta[12], meta[13], meta[14]];
    (camera, background)
}

/// CPU project / tile bin / sort / ray grid (shared by all native GPU paths).
pub fn prepare_raster(
    scene: &GaussianScene,
    camera: &Camera,
    background: [f32; 3],
    params: &RenderParams,
) -> PreparedRaster {
    let projected = project_splats(
        scene,
        camera,
        params.width,
        params.height,
        params.radius_scale,
        params.alpha_cutoff,
    );
    let (keys, values, generated) = build_tile_key_value_pairs(
        &projected,
        params.tile_width(),
        params.tile_height(),
        params.tile_size,
        params.max_list_entries,
    );
    let sorted_count = generated.min(params.max_list_entries);
    let (ref_keys, ref_values) = sort_key_values(&keys, &values, sorted_count);
    let ref_ranges = build_tile_ranges(&ref_keys, sorted_count, params.tile_count());

    let bg_linear = [
        background[0].max(0.0).powf(OUTPUT_GAMMA),
        background[1].max(0.0).powf(OUTPUT_GAMMA),
        background[2].max(0.0).powf(OUTPUT_GAMMA),
    ];
    let w = params.width;
    let h = params.height;
    let mut rays = vec![0.0f32; (w * h * 3) as usize];
    for py in 0..h {
        for px in 0..w {
            let ray = camera.screen_to_world_ray([px as f32 + 0.5, py as f32 + 0.5], w, h);
            let i = ((py * w + px) * 3) as usize;
            rays[i..i + 3].copy_from_slice(&ray);
        }
    }

    PreparedRaster {
        color_alpha: projected.color_alpha,
        valid: projected.valid,
        pos_local: projected.pos_local,
        inv_scale: projected.inv_scale,
        quat: projected.quat,
        sorted_values: ref_values,
        tile_ranges: ref_ranges,
        rays,
        params: SplatRasterParams {
            width: w,
            height: h,
            tile_size: params.tile_size,
            tile_width: params.tile_width(),
            alpha_cutoff: params.alpha_cutoff,
            transmittance_threshold: params.transmittance_threshold,
            bg_r: bg_linear[0],
            bg_g: bg_linear[1],
            bg_b: bg_linear[2],
            max_splat_steps: 0,
        },
    }
}

/// Build [`PreparedRaster`] from training prep (tile lists + linear background).
pub fn prepared_raster_from_training(
    projected: &super::project::ProjectedSplats,
    sorted_values: &[u32],
    tile_ranges: &[u32],
    camera: &Camera,
    background_linear: [f32; 3],
    width: u32,
    height: u32,
    tile_size: u32,
    tile_width: u32,
    alpha_cutoff: f32,
    max_splat_steps: u32,
    transmittance_threshold: f32,
) -> PreparedRaster {
    let mut rays = vec![0.0f32; (width * height * 3) as usize];
    for py in 0..height {
        for px in 0..width {
            let ray = camera.screen_to_world_ray([px as f32 + 0.5, py as f32 + 0.5], width, height);
            let i = ((py * width + px) * 3) as usize;
            rays[i..i + 3].copy_from_slice(&ray);
        }
    }
    PreparedRaster {
        color_alpha: projected.color_alpha.clone(),
        valid: projected.valid.clone(),
        pos_local: projected.pos_local.clone(),
        inv_scale: projected.inv_scale.clone(),
        quat: projected.quat.clone(),
        sorted_values: sorted_values.to_vec(),
        tile_ranges: tile_ranges.to_vec(),
        rays,
        params: SplatRasterParams {
            width,
            height,
            tile_size,
            tile_width,
            alpha_cutoff,
            transmittance_threshold,
            bg_r: background_linear[0],
            bg_g: background_linear[1],
            bg_b: background_linear[2],
            max_splat_steps,
        },
    }
}

/// Convenience: slices + render settings → [`PreparedRaster`].
#[allow(clippy::too_many_arguments)]
pub fn prepare_raster_from_slices(
    positions: &[f32],
    scales: &[f32],
    rotations: &[f32],
    opacities: &[f32],
    colors: &[f32],
    sh_coeffs: &[f32],
    meta: &[f32],
    width: u32,
    height: u32,
    tile_size: u32,
    radius_scale: f32,
    alpha_cutoff: f32,
    max_splat_steps: u32,
    transmittance_threshold: f32,
    max_list_entries: u32,
) -> PreparedRaster {
    let scene = scene_from_slices(positions, scales, rotations, opacities, colors, sh_coeffs);
    let (camera, background) = camera_and_background_from_meta(meta);
    let render_params = RenderParams {
        width,
        height,
        tile_size,
        radius_scale,
        alpha_cutoff,
        max_splat_steps,
        transmittance_threshold,
        max_list_entries,
    };
    prepare_raster(&scene, &camera, background, &render_params)
}
