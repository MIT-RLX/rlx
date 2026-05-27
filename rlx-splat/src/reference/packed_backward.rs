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
//! Packed RLX backward output: `[positions|scales|rotations|opacities|colors|sh]`.

use crate::core::Camera;

use super::{
    SceneGrads, TrainingForward, backprop_scene_grads_with_color_alpha_grad, rasterize_backward,
    render_training_forward,
};

fn pixel_rgb_grad_from_d_loss_rgba(d_loss_rgba: &[f32], pixel_count: usize) -> Vec<f32> {
    let mut pixel_rgb_grad = vec![0.0f32; pixel_count * 3];
    for pix in 0..pixel_count {
        let base4 = pix * 4;
        let base3 = pix * 3;
        pixel_rgb_grad[base3] = d_loss_rgba[base4];
        pixel_rgb_grad[base3 + 1] = d_loss_rgba[base4 + 1];
        pixel_rgb_grad[base3 + 2] = d_loss_rgba[base4 + 2];
    }
    pixel_rgb_grad
}

/// Training backward using a cached [`TrainingForward`] (no second project/raster forward).
#[allow(clippy::too_many_arguments)]
pub fn backward_packed_from_training_forward(
    scene: &crate::core::GaussianScene,
    camera: &Camera,
    forward: &TrainingForward,
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
    let sh_coeff_count = scene.sh_coeff_count.max(1);
    if scene.count() == 0 {
        return Vec::new();
    }
    let color_alpha_grad = rasterize_backward(
        forward,
        pixel_rgb_grad,
        background,
        width,
        height,
        loss_grad_clip,
    );
    let grads = backprop_scene_grads_with_color_alpha_grad(
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
    );
    scene_grads_to_packed(&grads, sh_coeff_count)
}

fn f32_at<'a>(base: *const u8, byte_off: usize, len: usize) -> &'a [f32] {
    unsafe { std::slice::from_raw_parts(base.add(byte_off) as *const f32, len) }
}

fn f32_at_mut<'a>(base: *mut u8, byte_off: usize, len: usize) -> &'a mut [f32] {
    unsafe { std::slice::from_raw_parts_mut(base.add(byte_off) as *mut f32, len) }
}

/// Concatenate [`SceneGrads`] into RLX [`unpack_gaussian_splat_packed_grads`] layout.
pub fn scene_grads_to_packed(grads: &SceneGrads, sh_coeff_count: usize) -> Vec<f32> {
    let _sh_coeff_count = sh_coeff_count.max(1);
    let mut out = Vec::with_capacity(
        grads.positions.len()
            + grads.scales.len()
            + grads.rotations.len()
            + grads.opacities.len()
            + grads.colors.len()
            + grads.sh_coeffs.len(),
    );
    out.extend_from_slice(&grads.positions);
    out.extend_from_slice(&grads.scales);
    out.extend_from_slice(&grads.rotations);
    out.extend_from_slice(&grads.opacities);
    out.extend_from_slice(&grads.colors);
    out.extend_from_slice(&grads.sh_coeffs);
    out
}

/// Write packed scene gradients into `packed_off` (RLX arena byte offsets; `*_len` = f32 count).
#[allow(clippy::too_many_arguments)]
pub unsafe fn backward_packed_arena(
    base: *mut u8,
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
    let base_ro = base as *const u8;
    let count = f32_at(base_ro, positions_off, positions_len).len() / 3;
    if count == 0 {
        f32_at_mut(base, packed_off, packed_len).fill(0.0);
        return;
    }
    let sh_coeff_count = (sh_coeffs_len / (count * 3)).max(1);
    let scene = crate::core::GaussianScene::new(
        f32_at(base_ro, positions_off, positions_len).to_vec(),
        f32_at(base_ro, scales_off, scales_len).to_vec(),
        f32_at(base_ro, rotations_off, rotations_len).to_vec(),
        f32_at(base_ro, opacities_off, opacities_len).to_vec(),
        f32_at(base_ro, colors_off, colors_len).to_vec(),
        f32_at(base_ro, sh_coeffs_off, sh_coeffs_len).to_vec(),
        sh_coeff_count,
    );
    let meta = f32_at(base_ro, meta_off, 23);
    let camera = Camera::look_at(
        [meta[0], meta[1], meta[2]],
        [meta[3], meta[4], meta[5]],
        [meta[6], meta[7], meta[8]],
        meta[9],
        meta[10],
        meta[11],
    );
    let background = [meta[12], meta[13], meta[14]];
    let pixel_count = (width * height) as usize;
    let d_loss = f32_at(base_ro, d_loss_off, d_loss_len);
    let pixel_rgb_grad = pixel_rgb_grad_from_d_loss_rgba(d_loss, pixel_count);
    let packed = if let Some(forward) = super::training_cache::cached_training_forward() {
        backward_packed_from_training_forward(
            &scene,
            &camera,
            forward,
            &pixel_rgb_grad,
            background,
            width,
            height,
            radius_scale,
            alpha_cutoff,
            max_anisotropy,
            sh_band,
            loss_grad_clip,
        )
    } else {
        let tile_width = width.div_ceil(tile_size);
        let forward = render_training_forward(
            &scene,
            &camera,
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
        );
        backward_packed_from_training_forward(
            &scene,
            &camera,
            &forward,
            &pixel_rgb_grad,
            background,
            width,
            height,
            radius_scale,
            alpha_cutoff,
            max_anisotropy,
            sh_band,
            loss_grad_clip,
        )
    };
    let out = f32_at_mut(base, packed_off, packed_len);
    assert_eq!(out.len(), packed.len());
    out.copy_from_slice(&packed);
}

/// Host-slice backward (no arena); returns packed gradient vector.
#[allow(clippy::too_many_arguments)]
pub fn backward_packed_host_slices(
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
    let count = positions.len() / 3;
    if count == 0 {
        return Vec::new();
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
    let camera = Camera::look_at(
        [meta[0], meta[1], meta[2]],
        [meta[3], meta[4], meta[5]],
        [meta[6], meta[7], meta[8]],
        meta[9],
        meta[10],
        meta[11],
    );
    let background = [meta[12], meta[13], meta[14]];
    let pixel_count = (width * height) as usize;
    let pixel_rgb_grad = pixel_rgb_grad_from_d_loss_rgba(d_loss_rgba, pixel_count);
    if let Some(forward) = super::training_cache::cached_training_forward() {
        return backward_packed_from_training_forward(
            &scene,
            &camera,
            forward,
            &pixel_rgb_grad,
            background,
            width,
            height,
            radius_scale,
            alpha_cutoff,
            max_anisotropy,
            sh_band,
            loss_grad_clip,
        );
    }
    let tile_width = width.div_ceil(tile_size);
    let forward = render_training_forward(
        &scene,
        &camera,
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
    );
    backward_packed_from_training_forward(
        &scene,
        &camera,
        &forward,
        &pixel_rgb_grad,
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
