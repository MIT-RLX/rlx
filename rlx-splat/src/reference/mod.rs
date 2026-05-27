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
//! CPU reference pipeline ported from `reference_impls/reference_cpu.py`.

mod binning;
mod grads;
pub mod native_prep;
mod packed_backward;
mod project;
mod projection_debug;
mod raster;
mod raster_analytical;
mod raster_gaussian;
mod training;
mod training_cache;
mod training_trace_gpu;

pub use crate::core::RASTER_CACHE_PARAM_COUNT;
pub use binning::{build_tile_key_value_pairs, build_tile_ranges, sort_key_values};
pub use grads::{
    GRAD_STATS_STRIDE, GradStats, color_alpha_grad_to_raster_grad, compute_grad_norms,
    compute_packed_grad_norms,
};
pub use native_prep::prepared_raster_from_training;
pub use packed_backward::{
    backward_packed_arena, backward_packed_from_training_forward, backward_packed_host_slices,
    scene_grads_to_packed,
};
pub use project::{ProjectedSplats, project_splats};
pub use projection_debug::{projected_from_debug, projection_debug_buffers};
pub use raster::{rasterize, ray_splat_intersection_alpha};
pub use raster_analytical::{
    backprop_ray_hit_alpha_analytical_scene, ray_hit_alpha_grad_projected,
};
pub use raster_gaussian::{
    CachedRasterGaussian, CachedRasterGrad, backprop_build_raster_gaussian, build_raster_gaussian,
    flat_to_raster_gaussian, raster_gaussian_to_flat,
};
pub use training::{
    SceneGrads, TrainingForward, TrainingPrepare, backprop_scene_grads,
    backprop_scene_grads_with_color_alpha_grad, build_training_prepare, capture_training_traces,
    linearize_background, raster_training_linear_cpu, rasterize_backward, render_training_forward,
    training_forward_from_parts,
};
pub use training_cache::{clear_training_forward_cache, set_training_forward_cache};
pub use training_trace_gpu::{
    TRAINING_HIT_META_FLOATS, trace_buffer_sizes, traces_from_gpu_buffers,
};

use crate::core::{Camera, GaussianScene};

#[derive(Clone, Debug)]
pub struct RenderParams {
    pub width: u32,
    pub height: u32,
    pub tile_size: u32,
    pub radius_scale: f32,
    pub alpha_cutoff: f32,
    pub max_splat_steps: u32,
    pub transmittance_threshold: f32,
    pub max_list_entries: u32,
}

impl Default for RenderParams {
    fn default() -> Self {
        Self {
            width: 64,
            height: 64,
            tile_size: crate::core::DEFAULT_TILE_SIZE,
            radius_scale: crate::core::DEFAULT_RADIUS_SCALE,
            alpha_cutoff: crate::core::ALPHA_CUTOFF_DEFAULT,
            max_splat_steps: crate::core::DEFAULT_MAX_SPLAT_STEPS,
            transmittance_threshold: crate::core::DEFAULT_TRANSMITTANCE_THRESHOLD,
            max_list_entries: 64 * 64 * crate::core::DEFAULT_LIST_CAPACITY_MULTIPLIER,
        }
    }
}

impl RenderParams {
    pub fn tile_width(&self) -> u32 {
        self.width.div_ceil(self.tile_size)
    }

    pub fn tile_height(&self) -> u32 {
        self.height.div_ceil(self.tile_size)
    }

    pub fn tile_count(&self) -> u32 {
        self.tile_width() * self.tile_height()
    }
}

pub fn render_reference(
    scene: &GaussianScene,
    camera: &Camera,
    background: [f32; 3],
    params: &RenderParams,
) -> Vec<f32> {
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
    rasterize(
        &projected,
        &ref_values,
        &ref_ranges,
        camera,
        params.width,
        params.height,
        params.tile_size,
        params.tile_width(),
        background,
        params.alpha_cutoff,
        params.max_splat_steps,
        params.transmittance_threshold,
    )
}
