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
//! [`Op::GaussianSplatRender`] / backward for the wgpu backend.

use crate::buffer::Arena;

#[allow(clippy::too_many_arguments)]
pub fn run_gaussian_splat_render(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    positions_byte_off: usize,
    positions_len: usize,
    scales_byte_off: usize,
    scales_len: usize,
    rotations_byte_off: usize,
    rotations_len: usize,
    opacities_byte_off: usize,
    opacities_len: usize,
    colors_byte_off: usize,
    colors_len: usize,
    sh_coeffs_byte_off: usize,
    sh_coeffs_len: usize,
    meta_byte_off: usize,
    dst_byte_off: usize,
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
    #[cfg(feature = "native-splat")]
    {
        return crate::splat_native::run_gaussian_splat_render_native(
            arena,
            device,
            queue,
            positions_byte_off,
            positions_len,
            scales_byte_off,
            scales_len,
            rotations_byte_off,
            rotations_len,
            opacities_byte_off,
            opacities_len,
            colors_byte_off,
            colors_len,
            sh_coeffs_byte_off,
            sh_coeffs_len,
            meta_byte_off,
            dst_byte_off,
            dst_len,
            width,
            height,
            tile_size,
            radius_scale,
            alpha_cutoff,
            max_splat_steps,
            transmittance_threshold,
            max_list_entries,
        );
    }
    #[cfg(not(feature = "native-splat"))]
    {
        let f32_bytes = |n: usize| n * 4;
        let positions = arena.read_bytes_range(
            device,
            queue,
            positions_byte_off,
            f32_bytes(positions_len),
        );
        let scales = arena.read_bytes_range(device, queue, scales_byte_off, f32_bytes(scales_len));
        let rotations = arena.read_bytes_range(
            device,
            queue,
            rotations_byte_off,
            f32_bytes(rotations_len),
        );
        let opacities = arena.read_bytes_range(
            device,
            queue,
            opacities_byte_off,
            f32_bytes(opacities_len),
        );
        let colors = arena.read_bytes_range(device, queue, colors_byte_off, f32_bytes(colors_len));
        let sh_coeffs = arena.read_bytes_range(
            device,
            queue,
            sh_coeffs_byte_off,
            f32_bytes(sh_coeffs_len),
        );
        let meta = arena.read_bytes_range(device, queue, meta_byte_off, f32_bytes(23));

        let image = rlx_cpu::splat::render_host_slices(
            bytemuck::cast_slice(&positions),
            bytemuck::cast_slice(&scales),
            bytemuck::cast_slice(&rotations),
            bytemuck::cast_slice(&opacities),
            bytemuck::cast_slice(&colors),
            bytemuck::cast_slice(&sh_coeffs),
            bytemuck::cast_slice(&meta),
            width,
            height,
            tile_size,
            radius_scale,
            alpha_cutoff,
            max_splat_steps,
            transmittance_threshold,
            max_list_entries,
        );

        assert_eq!(image.len(), dst_len);
        let out_bytes: Vec<u8> = image.iter().flat_map(|v| v.to_le_bytes()).collect();
        arena.write_bytes_range(queue, dst_byte_off, &out_bytes);
    }
}

fn splat_backward_byte_span(
    positions_byte_off: usize,
    positions_len: usize,
    scales_byte_off: usize,
    scales_len: usize,
    rotations_byte_off: usize,
    rotations_len: usize,
    opacities_byte_off: usize,
    opacities_len: usize,
    colors_byte_off: usize,
    colors_len: usize,
    sh_coeffs_byte_off: usize,
    sh_coeffs_len: usize,
    meta_byte_off: usize,
    d_loss_byte_off: usize,
    d_loss_len: usize,
    packed_byte_off: usize,
    packed_len: usize,
) -> (usize, usize) {
    let regions = [
        (positions_byte_off, positions_len * 4),
        (scales_byte_off, scales_len * 4),
        (rotations_byte_off, rotations_len * 4),
        (opacities_byte_off, opacities_len * 4),
        (colors_byte_off, colors_len * 4),
        (sh_coeffs_byte_off, sh_coeffs_len * 4),
        (meta_byte_off, 23 * 4),
        (d_loss_byte_off, d_loss_len * 4),
        (packed_byte_off, packed_len * 4),
    ];
    let start = regions.iter().map(|(o, _)| *o).min().unwrap_or(0);
    let end = regions
        .iter()
        .map(|(o, n)| o.saturating_add(*n))
        .max()
        .unwrap_or(start);
    (start, end.saturating_sub(start))
}

/// Backward: D2H tensor span → CPU reference backward → write packed grads.
#[allow(clippy::too_many_arguments)]
pub fn run_gaussian_splat_render_backward(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    positions_byte_off: usize,
    positions_len: usize,
    scales_byte_off: usize,
    scales_len: usize,
    rotations_byte_off: usize,
    rotations_len: usize,
    opacities_byte_off: usize,
    opacities_len: usize,
    colors_byte_off: usize,
    colors_len: usize,
    sh_coeffs_byte_off: usize,
    sh_coeffs_len: usize,
    meta_byte_off: usize,
    d_loss_byte_off: usize,
    d_loss_len: usize,
    packed_byte_off: usize,
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
    let (span_off, span_len) = splat_backward_byte_span(
        positions_byte_off,
        positions_len,
        scales_byte_off,
        scales_len,
        rotations_byte_off,
        rotations_len,
        opacities_byte_off,
        opacities_len,
        colors_byte_off,
        colors_len,
        sh_coeffs_byte_off,
        sh_coeffs_len,
        meta_byte_off,
        d_loss_byte_off,
        d_loss_len,
        packed_byte_off,
        packed_len,
    );
    let mut host = arena.read_bytes_range(device, queue, span_off, span_len);
    let base = host.as_mut_ptr();
    let rel = |byte_off: usize| byte_off - span_off;
    unsafe {
        rlx_cpu::splat::execute_gaussian_splat_render_backward(
            rel(positions_byte_off),
            positions_len,
            rel(scales_byte_off),
            scales_len,
            rel(rotations_byte_off),
            rotations_len,
            rel(opacities_byte_off),
            opacities_len,
            rel(colors_byte_off),
            colors_len,
            rel(sh_coeffs_byte_off),
            sh_coeffs_len,
            rel(meta_byte_off),
            rel(d_loss_byte_off),
            d_loss_len,
            rel(packed_byte_off),
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
        );
    }
    let packed_start = rel(packed_byte_off);
    let packed_bytes = &host[packed_start..packed_start + packed_len * 4];
    arena.write_bytes_range(queue, packed_byte_off, packed_bytes);
}

#[allow(clippy::too_many_arguments)]
pub fn run_gaussian_splat_prepare(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    positions_byte_off: usize,
    positions_len: usize,
    scales_byte_off: usize,
    scales_len: usize,
    rotations_byte_off: usize,
    rotations_len: usize,
    opacities_byte_off: usize,
    opacities_len: usize,
    colors_byte_off: usize,
    colors_len: usize,
    sh_coeffs_byte_off: usize,
    sh_coeffs_len: usize,
    meta_byte_off: usize,
    meta_len: usize,
    prep_byte_off: usize,
    prep_len: usize,
    width: u32,
    height: u32,
    tile_size: u32,
    radius_scale: f32,
    alpha_cutoff: f32,
    max_splat_steps: u32,
    transmittance_threshold: f32,
    max_list_entries: u32,
) {
    let mut host = arena.read_bytes_range(device, queue, 0, arena.size);
    let base = host.as_mut_ptr();
    unsafe {
        rlx_cpu::splat::execute_gaussian_splat_prepare(
            positions_byte_off,
            positions_len,
            scales_byte_off,
            scales_len,
            rotations_byte_off,
            rotations_len,
            opacities_byte_off,
            opacities_len,
            colors_byte_off,
            colors_len,
            sh_coeffs_byte_off,
            sh_coeffs_len,
            meta_byte_off,
            meta_len,
            prep_byte_off,
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
        );
    }
    arena.write_bytes_range(queue, 0, &host);
}

#[allow(clippy::too_many_arguments)]
pub fn run_gaussian_splat_rasterize(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    prep_byte_off: usize,
    prep_len: usize,
    meta_byte_off: usize,
    meta_len: usize,
    dst_byte_off: usize,
    dst_len: usize,
    count: usize,
    width: u32,
    height: u32,
    tile_size: u32,
    alpha_cutoff: f32,
    max_splat_steps: u32,
    transmittance_threshold: f32,
    max_list_entries: u32,
) {
    let mut host = arena.read_bytes_range(device, queue, 0, arena.size);
    let base = host.as_mut_ptr();
    unsafe {
        rlx_cpu::splat::execute_gaussian_splat_rasterize(
            prep_byte_off,
            prep_len,
            meta_byte_off,
            meta_len,
            dst_byte_off,
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
        );
    }
    arena.write_bytes_range(queue, 0, &host);
}
