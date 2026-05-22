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
//! Native Metal Gaussian splat — CPU prepare + MSL tile raster via `rlx-splat`.

#![cfg(all(feature = "native-splat", target_os = "macos"))]

use crate::device::metal_device;
use crate::kernels::kernels;
use rlx_splat::prep_layout::unpack_prepared;
use rlx_splat::reference::native_prep::prepare_raster_from_slices;

#[allow(clippy::too_many_arguments, unsafe_op_in_unsafe_fn)]
fn dispatch_prep(
    prep: &rlx_splat::reference::native_prep::PreparedRaster,
    arena_buffer: &metal::Buffer,
    dst_byte_off: u64,
) {
    let dev = metal_device().expect("Metal device required");
    let k = kernels();
    rlx_splat::backends::metal::dispatch_prepared_raster(
        &dev.device,
        &dev.queue,
        &k.gaussian_splat_rasterize,
        prep,
        arena_buffer,
        dst_byte_off,
    );
}

/// Monolithic forward: CPU prepare + native MSL raster into arena RGBA.
#[allow(clippy::too_many_arguments, unsafe_op_in_unsafe_fn)]
pub unsafe fn execute_gaussian_splat_render_native(
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
    _dst_len: usize,
    width: u32,
    height: u32,
    tile_size: u32,
    radius_scale: f32,
    alpha_cutoff: f32,
    max_splat_steps: u32,
    transmittance_threshold: f32,
    max_list_entries: u32,
    arena_base: *mut u8,
    arena_buffer: &metal::Buffer,
) {
    let sl = |off: usize, len: usize| {
        std::slice::from_raw_parts((arena_base as *const u8).add(off) as *const f32, len)
    };
    let prep = prepare_raster_from_slices(
        sl(positions_off, positions_len),
        sl(scales_off, scales_len),
        sl(rotations_off, rotations_len),
        sl(opacities_off, opacities_len),
        sl(colors_off, colors_len),
        sl(sh_coeffs_off, sh_coeffs_len),
        sl(meta_off, 23),
        width,
        height,
        tile_size,
        radius_scale,
        alpha_cutoff,
        max_splat_steps,
        transmittance_threshold,
        max_list_entries,
    );
    dispatch_prep(&prep, arena_buffer, dst_off as u64);
}

/// Decomposed rasterize: unpack arena prepare buffer + native MSL raster.
#[allow(clippy::too_many_arguments, unsafe_op_in_unsafe_fn)]
pub unsafe fn execute_gaussian_splat_rasterize_native(
    prep_off: usize,
    prep_len: usize,
    _meta_off: usize,
    _meta_len: usize,
    dst_off: usize,
    _dst_len: usize,
    count: usize,
    width: u32,
    height: u32,
    tile_size: u32,
    alpha_cutoff: f32,
    _max_splat_steps: u32,
    transmittance_threshold: f32,
    max_list_entries: u32,
    arena_base: *mut u8,
    arena_buffer: &metal::Buffer,
) {
    let packed = std::slice::from_raw_parts(
        (arena_base as *const u8).add(prep_off) as *const f32,
        prep_len,
    );
    let mut prep = unpack_prepared(
        packed,
        count.max(1),
        max_list_entries,
        width,
        height,
        tile_size,
    );
    prep.params.alpha_cutoff = alpha_cutoff;
    prep.params.transmittance_threshold = transmittance_threshold;
    dispatch_prep(&prep, arena_buffer, dst_off as u64);
}

#[allow(clippy::too_many_arguments)]
pub fn render_forward_host_slices(
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
) -> Vec<f32> {
    rlx_cpu::splat::render_host_slices(
        positions,
        scales,
        rotations,
        opacities,
        colors,
        sh_coeffs,
        meta,
        width,
        height,
        tile_size,
        radius_scale,
        alpha_cutoff,
        max_splat_steps,
        transmittance_threshold,
        max_list_entries,
    )
}
