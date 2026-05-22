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
//! CPU arena execution for [`rlx_ir::Op::GaussianSplat*`] (registered into `rlx-cpu`).

#![cfg(feature = "execute")]

use crate::core::Camera;
use crate::prep_layout::{pack_prepared, tile_count, unpack_prepared, SPLAT_RASTER_PARAMS_FLOATS};
use crate::reference::native_prep::{camera_and_background_from_meta, prepare_raster_from_slices};
use crate::reference::ProjectedSplats;
use crate::reference::{backward_packed_arena, rasterize, render_reference, RenderParams};
use rlx_cpu::splat::{
    ArenaPrepareArgs, ArenaRasterizeArgs, ArenaRenderArgs, ArenaRenderBwdArgs, HostBackwardArgs,
    HostRenderArgs,
};

pub fn render_host_slices(
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
    let count = positions.len() / 3;
    let out_len = (width as usize) * (height as usize) * 4;
    if count == 0 {
        return vec![0.0; out_len];
    }
    let sh_coeff_count = (sh_coeffs.len() / (count * 3)).max(1);
    let scene = crate::core::GaussianScene::new(
        positions.to_vec(),
        scales.to_vec(),
        rotations.to_vec(),
        opacities.to_vec(),
        colors.to_vec(),
        sh_coeffs.to_vec(),
        sh_coeff_count,
    );
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
    let params = RenderParams {
        width,
        height,
        tile_size,
        radius_scale,
        alpha_cutoff,
        max_splat_steps,
        transmittance_threshold,
        max_list_entries,
    };
    render_reference(&scene, &camera, background, &params)
}

/// Host-side backward (packed scene gradients).
#[allow(clippy::too_many_arguments)]
pub fn backward_host_slices(
    positions: &[f32],
    scales: &[f32],
    rotations: &[f32],
    opacities: &[f32],
    colors: &[f32],
    sh_coeffs: &[f32],
    meta: &[f32],
    d_loss_rgba: &[f32],
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
) -> Vec<f32> {
    crate::reference::backward_packed_host_slices(
        positions,
        scales,
        rotations,
        opacities,
        colors,
        sh_coeffs,
        meta,
        d_loss_rgba,
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
    )
}

pub fn render_host_slices_args(a: HostRenderArgs) -> Vec<f32> {
    render_host_slices(
        &a.positions,
        &a.scales,
        &a.rotations,
        &a.opacities,
        &a.colors,
        &a.sh_coeffs,
        &a.meta,
        a.width,
        a.height,
        a.tile_size,
        a.radius_scale,
        a.alpha_cutoff,
        a.max_splat_steps,
        a.transmittance_threshold,
        a.max_list_entries,
    )
}

pub fn backward_host_slices_args(a: HostBackwardArgs) -> Vec<f32> {
    backward_host_slices(
        &a.positions,
        &a.scales,
        &a.rotations,
        &a.opacities,
        &a.colors,
        &a.sh_coeffs,
        &a.meta,
        &a.d_loss_rgba,
        a.width,
        a.height,
        a.tile_size,
        a.radius_scale,
        a.alpha_cutoff,
        a.max_splat_steps,
        a.transmittance_threshold,
        a.max_list_entries,
        a.loss_grad_clip,
        a.sh_band,
        a.max_anisotropy,
    )
}

#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn execute_gaussian_splat_render(a: ArenaRenderArgs) {
    execute_gaussian_splat_render_inner(
        a.positions_off,
        a.positions_len,
        a.scales_off,
        a.scales_len,
        a.rotations_off,
        a.rotations_len,
        a.opacities_off,
        a.opacities_len,
        a.colors_off,
        a.colors_len,
        a.sh_coeffs_off,
        a.sh_coeffs_len,
        a.meta_off,
        a.dst_off,
        a.dst_len,
        a.width,
        a.height,
        a.tile_size,
        a.radius_scale,
        a.alpha_cutoff,
        a.max_splat_steps,
        a.transmittance_threshold,
        a.max_list_entries,
        a.base,
    );
}

#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn execute_gaussian_splat_render_inner(
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
    base: *mut u8,
) {
    let sl =
        |off: usize, len: usize| -> &[f32] { std::slice::from_raw_parts((base as *const u8).add(off) as *const f32, len) };
    let sl_mut = |off: usize, len: usize| -> &mut [f32] {
        std::slice::from_raw_parts_mut((base as *mut u8).add(off) as *mut f32, len)
    };

    let image = render_host_slices(
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
    let out = sl_mut(dst_off, dst_len);
    assert_eq!(out.len(), image.len());
    out.copy_from_slice(&image);
}

#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn execute_gaussian_splat_render_backward(a: ArenaRenderBwdArgs) {
    execute_gaussian_splat_render_backward_inner(
        a.positions_off,
        a.positions_len,
        a.scales_off,
        a.scales_len,
        a.rotations_off,
        a.rotations_len,
        a.opacities_off,
        a.opacities_len,
        a.colors_off,
        a.colors_len,
        a.sh_coeffs_off,
        a.sh_coeffs_len,
        a.meta_off,
        a.d_loss_off,
        a.d_loss_len,
        a.packed_off,
        a.packed_len,
        a.width,
        a.height,
        a.tile_size,
        a.radius_scale,
        a.alpha_cutoff,
        a.max_splat_steps,
        a.transmittance_threshold,
        a.max_list_entries,
        a.loss_grad_clip,
        a.sh_band,
        a.max_anisotropy,
        a.base,
    );
}

#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn execute_gaussian_splat_render_backward_inner(
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
    base: *mut u8,
) {
    backward_packed_arena(
        base,
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
    );
}

#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn execute_gaussian_splat_prepare(a: ArenaPrepareArgs) {
    let b = a.base;
    let sl = |off: usize, len: usize| -> &[f32] {
        std::slice::from_raw_parts((b as *const u8).add(off) as *const f32, len)
    };
    let count = a.positions_len / 3;
    let prep = prepare_raster_from_slices(
        sl(a.positions_off, a.positions_len),
        sl(a.scales_off, a.scales_len),
        sl(a.rotations_off, a.rotations_len),
        sl(a.opacities_off, a.opacities_len),
        sl(a.colors_off, a.colors_len),
        sl(a.sh_coeffs_off, a.sh_coeffs_len),
        sl(a.meta_off, a.meta_len.min(23)),
        a.width,
        a.height,
        a.tile_size,
        a.radius_scale,
        a.alpha_cutoff,
        a.max_splat_steps,
        a.transmittance_threshold,
        a.max_list_entries,
    );
    let out = std::slice::from_raw_parts_mut((b as *mut u8).add(a.prep_off) as *mut f32, a.prep_len);
    pack_prepared(out, &prep, a.max_list_entries);
}

#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn execute_gaussian_splat_rasterize(a: ArenaRasterizeArgs) {
    let b = a.base;
    let sl = |off: usize, len: usize| -> &[f32] {
        std::slice::from_raw_parts((b as *const u8).add(off) as *const f32, len)
    };
    let sl_mut = |off: usize, len: usize| -> &mut [f32] {
        std::slice::from_raw_parts_mut((b as *mut u8).add(off) as *mut f32, len)
    };
    let prep = unpack_prepared(
        sl(a.prep_off, a.prep_len),
        a.count.max(1),
        a.max_list_entries,
        a.width,
        a.height,
        a.tile_size,
    );
    let (camera, background) = camera_and_background_from_meta(sl(a.meta_off, a.meta_len.min(23)));
    let projected = ProjectedSplats {
        color_alpha: prep.color_alpha,
        valid: prep.valid,
        pos_local: prep.pos_local,
        inv_scale: prep.inv_scale,
        quat: prep.quat,
        center_radius_depth: vec![],
        ellipse_conic: vec![],
        opacity_scale: vec![],
    };
    let image = rasterize(
        &projected,
        &prep.sorted_values,
        &prep.tile_ranges,
        &camera,
        a.width,
        a.height,
        a.tile_size,
        prep.params.tile_width,
        background,
        a.alpha_cutoff,
        a.max_splat_steps,
        a.transmittance_threshold,
    );
    let out = sl_mut(a.dst_off, a.dst_len);
    assert_eq!(out.len(), image.len());
    out.copy_from_slice(&image);
}

