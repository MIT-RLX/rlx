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
//! Native CUDA Gaussian splat forward (RLX-owned kernel).

#![cfg(feature = "native-splat")]

use crate::kernels::{dispatch_grid_2d, gaussian_splat_rasterize_kernel};
use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig};
use slang_splat_ref::RenderParams;
use slang_splat_ref::native_prep::{
    PreparedRaster, camera_and_background_from_meta, prepare_raster, scene_from_slices,
};
use std::sync::Arc;

fn dispatch_prepared(
    stream: &Arc<CudaStream>,
    arena: &mut CudaSlice<f32>,
    prep: &PreparedRaster,
    dst_off: u32,
) {
    let d_ca = stream
        .memcpy_stod(&prep.color_alpha)
        .expect("splat native: stod color_alpha");
    let d_valid = stream
        .memcpy_stod(&prep.valid)
        .expect("splat native: stod valid");
    let d_pl = stream
        .memcpy_stod(&prep.pos_local)
        .expect("splat native: stod pos_local");
    let d_inv = stream
        .memcpy_stod(&prep.inv_scale)
        .expect("splat native: stod inv_scale");
    let d_quat = stream
        .memcpy_stod(&prep.quat)
        .expect("splat native: stod quat");
    let d_sorted = stream
        .memcpy_stod(&prep.sorted_values)
        .expect("splat native: stod sorted");
    let d_ranges = stream
        .memcpy_stod(&prep.tile_ranges)
        .expect("splat native: stod ranges");
    let d_rays = stream
        .memcpy_stod(&prep.rays)
        .expect("splat native: stod rays");
    let params = prep.params;

    let kernel = gaussian_splat_rasterize_kernel(stream.context());
    let (grid, block) = dispatch_grid_2d(params.width, params.height, 8, 8);
    let cfg = LaunchConfig {
        grid_dim: grid,
        block_dim: block,
        shared_mem_bytes: 0,
    };
    let mut launcher = stream.launch_builder(&kernel.function);
    launcher
        .arg(arena)
        .arg(&dst_off)
        .arg(&d_ca)
        .arg(&d_valid)
        .arg(&d_pl)
        .arg(&d_inv)
        .arg(&d_quat)
        .arg(&d_sorted)
        .arg(&d_ranges)
        .arg(&d_rays)
        .arg(&params);
    unsafe {
        launcher
            .launch(cfg)
            .expect("rlx-cuda: gaussian_splat_rasterize launch failed");
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run_gaussian_splat_render_native(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    arena_size_bytes: usize,
    positions_off: usize,
    positions_len: usize,
    scales_off: usize,
    scales_len: usize,
    rotations_off: usize,
    rotations_len: usize,
    opacities_off: usize,
    opacities_len: usize,
    colors_off: usize,
    colors_len: usize,
    sh_coeffs_off: usize,
    sh_coeffs_len: usize,
    meta_off: usize,
    dst_off: usize,
    width: u32,
    height: u32,
    tile_size: u32,
    radius_scale: f32,
    alpha_cutoff: f32,
    max_splat_steps: u32,
    transmittance_threshold: f32,
    max_list_entries: u32,
) {
    let n_f32 = arena_size_bytes / 4;
    stream
        .synchronize()
        .expect("rlx-cuda splat native: pre-sync failed");
    let mut host = vec![0f32; n_f32];
    stream
        .memcpy_dtoh(&buffer.slice(..), &mut host)
        .expect("rlx-cuda splat native: dtoh failed");

    let sl = |off: usize, len: usize| -> &[f32] { &host[off..off + len] };
    let meta_len = 23.min(host.len().saturating_sub(meta_off));
    let scene = scene_from_slices(
        sl(positions_off, positions_len),
        sl(scales_off, scales_len),
        sl(rotations_off, rotations_len),
        sl(opacities_off, opacities_len),
        sl(colors_off, colors_len),
        sl(sh_coeffs_off, sh_coeffs_len),
    );
    let (camera, background) = camera_and_background_from_meta(sl(meta_off, meta_len));
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
    let prep = prepare_raster(&scene, &camera, background, &render_params);
    dispatch_prepared(stream, buffer, &prep, dst_off as u32);
    stream
        .synchronize()
        .expect("rlx-cuda splat native: post-sync failed");
}
