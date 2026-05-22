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
//! Native HIP Gaussian splat forward (shared CUDA source).

#![cfg(feature = "native-splat")]

use crate::device::RocmContext;
use crate::hip::HipBuffer;
use crate::kernels::gaussian_splat_rasterize_kernel;
use slang_splat_ref::native_prep::{
    camera_and_background_from_meta, prepare_raster, scene_from_slices, PreparedRaster,
};
use slang_splat_ref::RenderParams;
use std::sync::Arc;

fn dispatch_prepared(
    ctx: &Arc<RocmContext>,
    stream: u64,
    arena_ptr: *mut f32,
    prep: &PreparedRaster,
    dst_off: u32,
) {
    let rt = &ctx.runtime;
    let mut d_ca = HipBuffer::<f32>::alloc_zeros(rt, prep.color_alpha.len())
        .expect("splat native: alloc ca");
    d_ca.copy_from_host(&prep.color_alpha).expect("splat native: h2d ca");
    let mut d_valid = HipBuffer::<u32>::alloc_zeros(rt, prep.valid.len())
        .expect("splat native: alloc valid");
    d_valid.copy_from_host(&prep.valid).expect("splat native: h2d valid");
    let mut d_pl = HipBuffer::<f32>::alloc_zeros(rt, prep.pos_local.len())
        .expect("splat native: alloc pl");
    d_pl.copy_from_host(&prep.pos_local).expect("splat native: h2d pl");
    let mut d_inv = HipBuffer::<f32>::alloc_zeros(rt, prep.inv_scale.len())
        .expect("splat native: alloc inv");
    d_inv.copy_from_host(&prep.inv_scale).expect("splat native: h2d inv");
    let mut d_quat = HipBuffer::<f32>::alloc_zeros(rt, prep.quat.len())
        .expect("splat native: alloc quat");
    d_quat.copy_from_host(&prep.quat).expect("splat native: h2d quat");
    let mut d_sorted = HipBuffer::<u32>::alloc_zeros(rt, prep.sorted_values.len())
        .expect("splat native: alloc sorted");
    d_sorted
        .copy_from_host(&prep.sorted_values)
        .expect("splat native: h2d sorted");
    let mut d_ranges = HipBuffer::<u32>::alloc_zeros(rt, prep.tile_ranges.len())
        .expect("splat native: alloc ranges");
    d_ranges
        .copy_from_host(&prep.tile_ranges)
        .expect("splat native: h2d ranges");
    let mut d_rays = HipBuffer::<f32>::alloc_zeros(rt, prep.rays.len())
        .expect("splat native: alloc rays");
    d_rays.copy_from_host(&prep.rays).expect("splat native: h2d rays");
    let params = prep.params;

    let kernel = gaussian_splat_rasterize_kernel(ctx);
    let (grid, block) = crate::kernels::dispatch_grid_2d(params.width, params.height, 8, 8);
    crate::launch_kernel!(
        kernel,
        stream,
        grid,
        block,
        [
            &mut arena_ptr,
            &dst_off,
            &mut d_ca.ptr,
            &mut d_valid.ptr,
            &mut d_pl.ptr,
            &mut d_inv.ptr,
            &mut d_quat.ptr,
            &mut d_sorted.ptr,
            &mut d_ranges.ptr,
            &mut d_rays.ptr,
            &params,
        ]
    );
}

#[allow(clippy::too_many_arguments)]
pub fn run_gaussian_splat_render_native(
    ctx: &Arc<RocmContext>,
    buffer: &HipBuffer<f32>,
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
    let rt = &ctx.runtime;
    let n_f32 = arena_size_bytes / 4;
    let mut host = vec![0f32; n_f32];
    unsafe {
        let _ = (rt.hip_stream_sync)(ctx.default_stream);
        let _ = (rt.hip_memcpy_dtoh)(host.as_mut_ptr() as *mut _, buffer.ptr, n_f32 * 4);
    }

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
    dispatch_prepared(ctx, ctx.default_stream, buffer.ptr as *mut f32, &prep, dst_off as u32);
    unsafe {
        let _ = (rt.hip_stream_sync)(ctx.default_stream);
    }
}
