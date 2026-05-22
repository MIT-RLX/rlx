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
//! CPU dispatch hooks for [`rlx_ir::Op::GaussianSplatRender`] — bodies registered from `rlx-splat`.

use std::sync::OnceLock;

type RenderExec = Box<dyn Fn(ArenaRenderArgs) + Send + Sync>;
type RenderBwdExec = Box<dyn Fn(ArenaRenderBwdArgs) + Send + Sync>;
type PrepareExec = Box<dyn Fn(ArenaPrepareArgs) + Send + Sync>;
type RasterizeExec = Box<dyn Fn(ArenaRasterizeArgs) + Send + Sync>;
type HostRenderExec = Box<dyn Fn(HostRenderArgs) -> Vec<f32> + Send + Sync>;
type HostBackwardExec = Box<dyn Fn(HostBackwardArgs) -> Vec<f32> + Send + Sync>;

static RENDER: OnceLock<RenderExec> = OnceLock::new();
static RENDER_BWD: OnceLock<RenderBwdExec> = OnceLock::new();
static PREPARE: OnceLock<PrepareExec> = OnceLock::new();
static RASTERIZE: OnceLock<RasterizeExec> = OnceLock::new();
static HOST_RENDER: OnceLock<HostRenderExec> = OnceLock::new();
static HOST_BACKWARD: OnceLock<HostBackwardExec> = OnceLock::new();

/// Arena arguments for forward splat.
#[allow(clippy::struct_excessive_bools)]
pub struct ArenaRenderArgs {
    pub positions_off: usize,
    pub positions_len: usize,
    pub scales_off: usize,
    pub scales_len: usize,
    pub rotations_off: usize,
    pub rotations_len: usize,
    pub opacities_off: usize,
    pub opacities_len: usize,
    pub colors_off: usize,
    pub colors_len: usize,
    pub sh_coeffs_off: usize,
    pub sh_coeffs_len: usize,
    pub meta_off: usize,
    pub dst_off: usize,
    pub dst_len: usize,
    pub width: u32,
    pub height: u32,
    pub tile_size: u32,
    pub radius_scale: f32,
    pub alpha_cutoff: f32,
    pub max_splat_steps: u32,
    pub transmittance_threshold: f32,
    pub max_list_entries: u32,
    pub base: *mut u8,
}

/// Arena arguments for [`Op::GaussianSplatPrepare`].
pub struct ArenaPrepareArgs {
    pub positions_off: usize,
    pub positions_len: usize,
    pub scales_off: usize,
    pub scales_len: usize,
    pub rotations_off: usize,
    pub rotations_len: usize,
    pub opacities_off: usize,
    pub opacities_len: usize,
    pub colors_off: usize,
    pub colors_len: usize,
    pub sh_coeffs_off: usize,
    pub sh_coeffs_len: usize,
    pub meta_off: usize,
    pub meta_len: usize,
    pub prep_off: usize,
    pub prep_len: usize,
    pub width: u32,
    pub height: u32,
    pub tile_size: u32,
    pub radius_scale: f32,
    pub alpha_cutoff: f32,
    pub max_splat_steps: u32,
    pub transmittance_threshold: f32,
    pub max_list_entries: u32,
    pub base: *mut u8,
}

/// Arena arguments for [`Op::GaussianSplatRasterize`].
pub struct ArenaRasterizeArgs {
    pub prep_off: usize,
    pub prep_len: usize,
    pub meta_off: usize,
    pub meta_len: usize,
    pub dst_off: usize,
    pub dst_len: usize,
    pub count: usize,
    pub width: u32,
    pub height: u32,
    pub tile_size: u32,
    pub alpha_cutoff: f32,
    pub max_splat_steps: u32,
    pub transmittance_threshold: f32,
    pub max_list_entries: u32,
    pub base: *mut u8,
}

/// Arena arguments for backward splat.
pub struct ArenaRenderBwdArgs {
    pub positions_off: usize,
    pub positions_len: usize,
    pub scales_off: usize,
    pub scales_len: usize,
    pub rotations_off: usize,
    pub rotations_len: usize,
    pub opacities_off: usize,
    pub opacities_len: usize,
    pub colors_off: usize,
    pub colors_len: usize,
    pub sh_coeffs_off: usize,
    pub sh_coeffs_len: usize,
    pub meta_off: usize,
    pub d_loss_off: usize,
    pub d_loss_len: usize,
    pub packed_off: usize,
    pub packed_len: usize,
    pub width: u32,
    pub height: u32,
    pub tile_size: u32,
    pub radius_scale: f32,
    pub alpha_cutoff: f32,
    pub max_splat_steps: u32,
    pub transmittance_threshold: f32,
    pub max_list_entries: u32,
    pub loss_grad_clip: f32,
    pub sh_band: u32,
    pub max_anisotropy: f32,
    pub base: *mut u8,
}

/// Host-buffer forward splat.
pub struct HostRenderArgs {
    pub positions: Vec<f32>,
    pub scales: Vec<f32>,
    pub rotations: Vec<f32>,
    pub opacities: Vec<f32>,
    pub colors: Vec<f32>,
    pub sh_coeffs: Vec<f32>,
    pub meta: Vec<f32>,
    pub width: u32,
    pub height: u32,
    pub tile_size: u32,
    pub radius_scale: f32,
    pub alpha_cutoff: f32,
    pub max_splat_steps: u32,
    pub transmittance_threshold: f32,
    pub max_list_entries: u32,
}

/// Host-buffer backward splat.
pub struct HostBackwardArgs {
    pub positions: Vec<f32>,
    pub scales: Vec<f32>,
    pub rotations: Vec<f32>,
    pub opacities: Vec<f32>,
    pub colors: Vec<f32>,
    pub sh_coeffs: Vec<f32>,
    pub meta: Vec<f32>,
    pub d_loss_rgba: Vec<f32>,
    pub width: u32,
    pub height: u32,
    pub tile_size: u32,
    pub radius_scale: f32,
    pub alpha_cutoff: f32,
    pub max_splat_steps: u32,
    pub transmittance_threshold: f32,
    pub max_list_entries: u32,
    pub loss_grad_clip: f32,
    pub sh_band: u32,
    pub max_anisotropy: f32,
}

/// Register arena + host splat executors (`rlx_splat::register()`).
pub fn register_splat_executors(
    render: RenderExec,
    backward: RenderBwdExec,
    prepare: PrepareExec,
    rasterize: RasterizeExec,
    host_render: HostRenderExec,
    host_backward: HostBackwardExec,
) {
    let _ = RENDER.set(render);
    let _ = RENDER_BWD.set(backward);
    let _ = PREPARE.set(prepare);
    let _ = RASTERIZE.set(rasterize);
    let _ = HOST_RENDER.set(host_render);
    let _ = HOST_BACKWARD.set(host_backward);
}

#[allow(clippy::too_many_arguments)]
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
    HOST_RENDER
        .get()
        .expect("call `rlx_splat::register()` before host splat render")(
        HostRenderArgs {
            positions: positions.to_vec(),
            scales: scales.to_vec(),
            rotations: rotations.to_vec(),
            opacities: opacities.to_vec(),
            colors: colors.to_vec(),
            sh_coeffs: sh_coeffs.to_vec(),
            meta: meta.to_vec(),
            width,
            height,
            tile_size,
            radius_scale,
            alpha_cutoff,
            max_splat_steps,
            transmittance_threshold,
            max_list_entries,
        },
    )
}

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
    HOST_BACKWARD
        .get()
        .expect("call `rlx_splat::register()` before host splat backward")(HostBackwardArgs {
        positions: positions.to_vec(),
        scales: scales.to_vec(),
        rotations: rotations.to_vec(),
        opacities: opacities.to_vec(),
        colors: colors.to_vec(),
        sh_coeffs: sh_coeffs.to_vec(),
        meta: meta.to_vec(),
        d_loss_rgba: d_loss_rgba.to_vec(),
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
    })
}

/// Execute [`Op::GaussianSplatPrepare`].
#[allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
pub unsafe fn execute_gaussian_splat_prepare(
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
    meta_len: usize,
    prep_off: usize,
    prep_len: usize,
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
    PREPARE.get().expect("call `rlx_splat::register()` before GaussianSplatPrepare")(ArenaPrepareArgs {
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
        meta_len,
        prep_off,
        prep_len,
        width,
        height,
        tile_size,
        radius_scale,
        alpha_cutoff,
        max_splat_steps,
        transmittance_threshold,
        max_list_entries,
        base,
    });
}

/// Execute [`Op::GaussianSplatRasterize`].
#[allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
pub unsafe fn execute_gaussian_splat_rasterize(
    prep_off: usize,
    prep_len: usize,
    meta_off: usize,
    meta_len: usize,
    dst_off: usize,
    dst_len: usize,
    count: usize,
    width: u32,
    height: u32,
    tile_size: u32,
    alpha_cutoff: f32,
    max_splat_steps: u32,
    transmittance_threshold: f32,
    max_list_entries: u32,
    base: *mut u8,
) {
    RASTERIZE
        .get()
        .expect("call `rlx_splat::register()` before GaussianSplatRasterize")(ArenaRasterizeArgs {
        prep_off,
        prep_len,
        meta_off,
        meta_len,
        dst_off,
        dst_len,
        count,
        width,
        height,
        tile_size,
        alpha_cutoff,
        max_splat_steps,
        transmittance_threshold,
        max_list_entries,
        base,
    });
}

/// Execute [`Op::GaussianSplatRender`] against the arena `base` pointer.
#[allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
pub unsafe fn execute_gaussian_splat_render(
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
    RENDER
        .get()
        .expect("call `rlx_splat::register()` before GaussianSplatRender")(ArenaRenderArgs {
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
        base,
    });
}

/// Execute [`Op::GaussianSplatRenderBackward`].
#[allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
pub unsafe fn execute_gaussian_splat_render_backward(
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
    RENDER_BWD
        .get()
        .expect("call `rlx_splat::register()` before GaussianSplatRenderBackward")(
        ArenaRenderBwdArgs {
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
            base,
        },
    );
}
