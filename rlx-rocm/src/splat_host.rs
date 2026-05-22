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
//! Host-side [`Op::GaussianSplatRender`] / backward for ROCm arenas (D2H → CPU → H2D).

use crate::device::RocmContext;
use crate::hip::HipBuffer;

#[allow(clippy::too_many_arguments)]
pub fn run_gaussian_splat_render(
    ctx: &RocmContext,
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
    dst_len: usize,
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
    unsafe {
        rlx_cpu::splat::execute_gaussian_splat_render(
            positions_off,
            positions_len,
            scales_off,
            scales_len,
            rotations_off,
            rotations_len,
            opacities_off,
            opacities_len,
            colors_off,
            colors_len,
            sh_coeffs_off,
            sh_coeffs_len,
            meta_off,
            dst_off,
            dst_len,
            width,
            height,
            tile_size,
            radius_scale,
            alpha_cutoff,
            max_splat_steps,
            transmittance_threshold,
            max_list_entries,
            host.as_mut_ptr() as *mut u8,
        );
    }
    unsafe {
        let _ = (rt.hip_memcpy_htod)(buffer.ptr, host.as_ptr() as *const _, n_f32 * 4);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run_gaussian_splat_render_backward(
    ctx: &RocmContext,
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
    d_loss_off: usize,
    d_loss_len: usize,
    packed_off: usize,
    packed_len: usize,
    width: u32,
    height: u32,
    tile_size: u32,
    radius_scale: f32,
    alpha_cutoff: f32,
    max_splat_steps: u32,
    transmittance_threshold: f32,
    max_list_entries: u32,
    loss_grad_clip: f32,
    sh_band: u32,
    max_anisotropy: f32,
) {
    let rt = &ctx.runtime;
    let n_f32 = arena_size_bytes / 4;
    let mut host = vec![0f32; n_f32];
    unsafe {
        let _ = (rt.hip_stream_sync)(ctx.default_stream);
        let _ = (rt.hip_memcpy_dtoh)(host.as_mut_ptr() as *mut _, buffer.ptr, n_f32 * 4);
    }
    unsafe {
        rlx_cpu::splat::execute_gaussian_splat_render_backward(
            positions_off,
            positions_len,
            scales_off,
            scales_len,
            rotations_off,
            rotations_len,
            opacities_off,
            opacities_len,
            colors_off,
            colors_len,
            sh_coeffs_off,
            sh_coeffs_len,
            meta_off,
            d_loss_off,
            d_loss_len,
            packed_off,
            packed_len,
            width,
            height,
            tile_size,
            radius_scale,
            alpha_cutoff,
            max_splat_steps,
            transmittance_threshold,
            max_list_entries,
            loss_grad_clip,
            sh_band,
            max_anisotropy,
            host.as_mut_ptr() as *mut u8,
        );
    }
    unsafe {
        let _ = (rt.hip_memcpy_htod)(buffer.ptr, host.as_ptr() as *const _, n_f32 * 4);
    }
}
