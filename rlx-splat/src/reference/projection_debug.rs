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
//! Projection-stage debug buffers aligned with `GaussianRenderer.debug_pipeline_data`.

use crate::core::{
    Camera, GAUSSIAN_SUPPORT_SIGMA_RADIUS, GaussianScene, ProjectionDebugBuffers,
    RASTER_CACHE_PARAM_COUNT,
};

use super::project::{ProjectedSplats, project_splats};
use super::raster_gaussian::{build_raster_gaussian, raster_gaussian_to_flat};

pub fn projection_debug_buffers(
    scene: &GaussianScene,
    camera: &Camera,
    width: u32,
    height: u32,
    radius_scale: f32,
    alpha_cutoff: f32,
    max_anisotropy: f32,
    sh_band: u32,
) -> ProjectionDebugBuffers {
    let projected = project_splats(scene, camera, width, height, radius_scale, alpha_cutoff);
    let count = scene.count();
    let mut screen_ellipse_conic = vec![0.0f32; count * 4];
    for i in 0..count {
        screen_ellipse_conic[i * 4] = projected.ellipse_conic[i * 3];
        screen_ellipse_conic[i * 4 + 1] = projected.ellipse_conic[i * 3 + 1];
        screen_ellipse_conic[i * 4 + 2] = projected.ellipse_conic[i * 3 + 2];
    }
    let mut splat_visible_area_px = vec![0.0f32; count];
    for i in 0..count {
        if projected.valid[i] != 0 {
            let r = projected.center_radius_depth[i * 4 + 2];
            splat_visible_area_px[i] = std::f32::consts::PI * r * r;
        }
    }
    let mut raster_cache = vec![0.0f32; count * RASTER_CACHE_PARAM_COUNT];
    for splat in 0..count {
        let rg = build_raster_gaussian(scene, splat, camera, radius_scale, max_anisotropy, sh_band);
        raster_cache[splat * RASTER_CACHE_PARAM_COUNT..(splat + 1) * RASTER_CACHE_PARAM_COUNT]
            .copy_from_slice(&raster_gaussian_to_flat(&rg));
    }
    ProjectionDebugBuffers {
        screen_center_radius_depth: projected.center_radius_depth,
        screen_color_alpha: projected.color_alpha,
        screen_ellipse_conic,
        splat_visible: projected.valid,
        splat_visible_area_px,
        raster_cache,
        generated_entries: 0,
        sorted_count: 0,
        keys: Vec::new(),
        values: Vec::new(),
        tile_ranges: Vec::new(),
    }
}

/// Build `ProjectedSplats` from projection debug buffers (Python `_projected_from_debug`).
pub fn projected_from_debug(
    scene: &GaussianScene,
    debug: &ProjectionDebugBuffers,
) -> ProjectedSplats {
    let count = scene.count();
    let mut ellipse_conic = vec![0.0f32; count * 3];
    for i in 0..count {
        ellipse_conic[i * 3] = debug.screen_ellipse_conic[i * 4];
        ellipse_conic[i * 3 + 1] = debug.screen_ellipse_conic[i * 4 + 1];
        ellipse_conic[i * 3 + 2] = debug.screen_ellipse_conic[i * 4 + 2];
    }
    let mut pos_local = vec![0.0f32; count * 3];
    for i in 0..count {
        let base = i * RASTER_CACHE_PARAM_COUNT;
        pos_local[i * 3] = debug.raster_cache[base];
        pos_local[i * 3 + 1] = debug.raster_cache[base + 1];
        pos_local[i * 3 + 2] = debug.raster_cache[base + 2];
    }
    let inv_scale: Vec<f32> = scene
        .scales
        .chunks(3)
        .flat_map(|s| {
            let rs = GAUSSIAN_SUPPORT_SIGMA_RADIUS;
            [
                1.0 / (s[0].exp() * rs).max(1e-6),
                1.0 / (s[1].exp() * rs).max(1e-6),
                1.0 / (s[2].exp() * rs).max(1e-6),
            ]
        })
        .collect();
    ProjectedSplats {
        center_radius_depth: debug.screen_center_radius_depth.clone(),
        ellipse_conic,
        color_alpha: debug.screen_color_alpha.clone(),
        opacity_scale: vec![1.0; count],
        valid: debug.splat_visible.clone(),
        pos_local,
        inv_scale,
        quat: scene.rotations.clone(),
    }
}
