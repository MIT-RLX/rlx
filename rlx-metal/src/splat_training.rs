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
//! Full-GPU training step on Metal: traced linear forward + GPU raster backward,
//! parallel CPU geometry backprop (single-splat reproject), GPU Adam.

#![cfg(all(feature = "native-splat", target_os = "macos"))]

use crate::device::metal_device;
use crate::kernels::kernels;
use rlx_splat::backends::metal_training::{
    GpuTrainingTraceBuffers, SplatRasterBwdParams, dispatch_training_backward,
    raster_linear_traced_to_vec, read_color_alpha_grad,
};
use rlx_splat::core::{Camera, GaussianScene};
use rlx_splat::reference::{
    SceneGrads, TrainingForward, backprop_scene_grads_with_color_alpha_grad,
    build_training_prepare, linearize_background, prepared_raster_from_training,
    scene_grads_to_packed, traces_from_gpu_buffers, training_forward_from_parts,
};

/// GPU trace buffers from forward — reuse in backward (no second raster forward).
pub struct MetalTrainingTraceCache {
    pub buffers: GpuTrainingTraceBuffers,
    pub width: u32,
    pub height: u32,
    pub max_splat_steps: u32,
}

/// Traced linear forward on GPU; keep [`MetalTrainingTraceCache`] for backward.
#[allow(clippy::too_many_arguments)]
pub fn training_forward_metal_traced(
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
) -> (TrainingForward, MetalTrainingTraceCache) {
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
    let bg_linear = linearize_background(background);
    let prepared = prepared_raster_from_training(
        &prep.projected,
        &prep.sorted_values,
        &prep.tile_ranges,
        camera,
        bg_linear,
        width,
        height,
        tile_size,
        tile_width,
        alpha_cutoff,
        max_splat_steps,
        transmittance_threshold,
    );
    let dev = metal_device().expect("Metal device required");
    let k = kernels();
    let traces_buf = GpuTrainingTraceBuffers::new(&dev.device, width, height, max_splat_steps);
    let rgba_linear = raster_linear_traced_to_vec(
        &dev.device,
        &dev.queue,
        &k.gaussian_splat_rasterize_linear_traced,
        &prepared,
        &traces_buf,
    );
    let (counts, ids, meta) = traces_buf.readback(width, height);
    let trace_vecs = traces_from_gpu_buffers(&counts, &ids, &meta, width, height, max_splat_steps);
    let forward = training_forward_from_parts(prep, rgba_linear, trace_vecs);
    let cache = MetalTrainingTraceCache {
        buffers: traces_buf,
        width,
        height,
        max_splat_steps,
    };
    (forward, cache)
}

/// Legacy alias (forward only, no trace cache — backward will re-raster).
pub fn render_training_forward_metal(
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
    training_forward_metal_traced(
        scene,
        camera,
        background,
        width,
        height,
        tile_size,
        tile_width,
        radius_scale,
        alpha_cutoff,
        max_splat_steps,
        transmittance_threshold,
        max_list_entries,
    )
    .0
}

/// GPU raster backward only → per-splat `color_alpha` gradient vector.
#[allow(clippy::too_many_arguments)]
pub fn training_raster_backward_metal_ca_grad(
    scene: &GaussianScene,
    trace_cache: &MetalTrainingTraceCache,
    pixel_rgb_grad: &[f32],
    background: [f32; 3],
    width: u32,
    height: u32,
    loss_grad_clip: f32,
) -> Vec<f32> {
    let bg_linear = linearize_background(background);
    let dev = metal_device().expect("Metal device required");
    let k = kernels();
    let count = scene.count();
    let ca_grad_buf = dev.device.new_buffer(
        (count * 4 * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let bwd = SplatRasterBwdParams {
        width,
        height,
        max_splat_steps: trace_cache.max_splat_steps,
        loss_grad_clip,
        bg_r: bg_linear[0],
        bg_g: bg_linear[1],
        bg_b: bg_linear[2],
        cam_px: 0.0,
        cam_py: 0.0,
        cam_pz: 0.0,
        radius_scale: 1.0,
        alpha_cutoff: 1.0 / 255.0,
    };
    dispatch_training_backward(
        &dev.device,
        &dev.queue,
        &k.gaussian_splat_rasterize_backward_linear,
        pixel_rgb_grad,
        &trace_cache.buffers,
        &ca_grad_buf,
        &bwd,
        width,
        height,
    );
    read_color_alpha_grad(&ca_grad_buf, count)
}

/// GPU raster backward using cached traces; CPU geometry backprop.
#[allow(clippy::too_many_arguments)]
pub fn training_backward_metal_cached(
    scene: &GaussianScene,
    camera: &Camera,
    forward: &TrainingForward,
    trace_cache: &MetalTrainingTraceCache,
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
    let bg_linear = linearize_background(background);
    let dev = metal_device().expect("Metal device required");
    let k = kernels();
    let count = scene.count();
    let ca_grad_buf = dev.device.new_buffer(
        (count * 4 * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let bwd = SplatRasterBwdParams {
        width,
        height,
        max_splat_steps: trace_cache.max_splat_steps,
        loss_grad_clip,
        bg_r: bg_linear[0],
        bg_g: bg_linear[1],
        bg_b: bg_linear[2],
        cam_px: camera.position[0],
        cam_py: camera.position[1],
        cam_pz: camera.position[2],
        radius_scale,
        alpha_cutoff,
    };
    dispatch_training_backward(
        &dev.device,
        &dev.queue,
        &k.gaussian_splat_rasterize_backward_linear,
        pixel_rgb_grad,
        &trace_cache.buffers,
        &ca_grad_buf,
        &bwd,
        width,
        height,
    );
    let color_alpha_grad = read_color_alpha_grad(&ca_grad_buf, count);
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

#[allow(clippy::too_many_arguments)]
pub fn training_backward_metal_packed_cached(
    scene: &GaussianScene,
    camera: &Camera,
    forward: &TrainingForward,
    trace_cache: &MetalTrainingTraceCache,
    pixel_rgb_grad: &[f32],
    background: [f32; 3],
    width: u32,
    height: u32,
    radius_scale: f32,
    alpha_cutoff: f32,
    max_anisotropy: f32,
    sh_band: u32,
    loss_grad_clip: f32,
) -> Vec<f32> {
    let grads = training_backward_metal_cached(
        scene,
        camera,
        forward,
        trace_cache,
        pixel_rgb_grad,
        background,
        width,
        height,
        radius_scale,
        alpha_cutoff,
        max_anisotropy,
        sh_band,
        loss_grad_clip,
    );
    scene_grads_to_packed(&grads, scene.sh_coeff_count.max(1))
}
