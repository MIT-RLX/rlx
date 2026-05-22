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
//! Fused Metal training: persistent buffers, single command buffer, optional readouts.

#![cfg(all(feature = "native-splat", target_os = "macos"))]

use crate::device::metal_device;
use crate::kernels::kernels;
use metal::{Buffer, CommandQueue, Device, MTLResourceOptions};
use rlx_splat::backends::metal_training::{GpuTrainingTraceBuffers, SplatRasterBwdParams};
use rlx_splat::reference::native_prep::SplatRasterParams;
use rlx_splat::core::{Camera, GaussianScene};
use rlx_splat::reference::{
    build_training_prepare, linearize_background, prepared_raster_from_training, TrainingPrepare,
};

const BIN_SORT_THREADS: u64 = 256;

#[repr(C)]
#[derive(Clone, Copy)]
struct SplatLossParams {
    width: u32,
    height: u32,
    inv_pixel_count: f32,
    mse_grad_scale: f32,
    ssim_weight: f32,
    ssim_c2: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SplatProjectParams {
    splat_count: u32,
    radius_scale: f32,
    alpha_cutoff: f32,
    cam_px: f32,
    cam_py: f32,
    cam_pz: f32,
    focal_x: f32,
    focal_y: f32,
    principal_x: f32,
    principal_y: f32,
    view_forward_x: f32,
    view_forward_y: f32,
    view_forward_z: f32,
    view_right_x: f32,
    view_right_y: f32,
    view_right_z: f32,
    view_up_x: f32,
    view_up_y: f32,
    view_up_z: f32,
    tile_size: u32,
    tile_width: u32,
    tile_height: u32,
    image_width: u32,
    image_height: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SplatBinParams {
    tile_width: u32,
    tile_height: u32,
    tile_size: u32,
    max_list_entries: u32,
    tile_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PackGradParams {
    splat_count: u32,
    packed_param_count: u32,
    coeff_count: u32,
    opacity_param: u32,
    use_sh: u32,
}

fn project_params(
    camera: &Camera,
    width: u32,
    height: u32,
    tile_size: u32,
    tile_width: u32,
    tile_height: u32,
    count: u32,
    radius_scale: f32,
    alpha_cutoff: f32,
) -> SplatProjectParams {
    let (fx, fy) = camera.focal_pixels_xy(width, height);
    let (cx, cy) = camera.principal_point(width, height);
    let basis = camera.basis();
    SplatProjectParams {
        splat_count: count,
        radius_scale,
        alpha_cutoff,
        cam_px: camera.position[0],
        cam_py: camera.position[1],
        cam_pz: camera.position[2],
        focal_x: fx,
        focal_y: fy,
        principal_x: cx,
        principal_y: cy,
        view_forward_x: basis[2][0],
        view_forward_y: basis[2][1],
        view_forward_z: basis[2][2],
        view_right_x: basis[0][0],
        view_right_y: basis[0][1],
        view_right_z: basis[0][2],
        view_up_x: basis[1][0],
        view_up_y: basis[1][1],
        view_up_z: basis[1][2],
        tile_size,
        tile_width,
        tile_height,
        image_width: width,
        image_height: height,
    }
}

pub struct MetalFusedTraining {
    device: Device,
    queue: CommandQueue,
    count: usize,
    width: u32,
    height: u32,
    max_splat_steps: u32,
    positions: Buffer,
    scales: Buffer,
    rotations: Buffer,
    opacities: Buffer,
    color_alpha: Buffer,
    pos_local: Buffer,
    inv_scale: Buffer,
    valid: Buffer,
    quat: Buffer,
    sorted_values: Buffer,
    tile_ranges: Buffer,
    rays: Buffer,
    rgba: Buffer,
    pixel_rgb_grad: Buffer,
    target_rgba: Buffer,
    loss_atomic: Buffer,
    ssim_sum_atomic: Buffer,
    ssim_stats: Buffer,
    traces: GpuTrainingTraceBuffers,
    color_alpha_grad: Buffer,
    grad_positions: Buffer,
    grad_scales: Buffer,
    grad_rotations: Buffer,
    grad_opacities: Buffer,
    grad_colors: Buffer,
    grad_pos_local: Buffer,
    grad_inv_scale: Buffer,
    grad_quat: Buffer,
    grad_opacity_hit: Buffer,
    center_radius_depth: Buffer,
    ellipse_conic: Buffer,
    depth_sorted_ids: Buffer,
    emit_count_buf: Buffer,
    bin_keys: Buffer,
    bin_values: Buffer,
    bin_sorted_keys: Buffer,
    bin_histogram: Buffer,
    bin_cursor: Buffer,
    bin_counter: Buffer,
    packed_params: Buffer,
    packed_grads: Buffer,
    moments_flat: Buffer,
    raster_params: SplatRasterParams,
    tile_count: u32,
    max_list_entries: u32,
    packed_param_count: u32,
}

pub struct FusedStepResult {
    pub loss: f32,
    pub mse: f32,
    pub ssim: f32,
    pub rgba_linear: Option<Vec<f32>>,
    pub bin_list_count: u32,
    pub emit_visible_count: u32,
}

/// Tile list source for fused training forward.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SplatBinMode {
    /// `gaussian_splat_emit_tile_keys` radius AABB (legacy GPU path).
    GpuRadiusAabb,
    /// `rlx_splat::reference` conic ellipse binning uploaded before encode (matches CPU/Python).
    #[default]
    RlxConicPrepare,
    /// GPU project screen ellipse + MSL scanline conic emit (experimental; set `RLX_FUSED_GPU_CONIC_SCANLINE=1`).
    GpuConicScanline,
}

/// Resolve bin mode from `RLX_FUSED_GPU_CONIC_SCANLINE` (production default: CPU conic prep).
pub fn splat_bin_mode_from_env() -> SplatBinMode {
    if std::env::var_os("RLX_FUSED_GPU_CONIC_SCANLINE").is_some() {
        SplatBinMode::GpuConicScanline
    } else {
        SplatBinMode::RlxConicPrepare
    }
}

fn bin_list_dispatch_groups(max_list_entries: u32) -> u64 {
    (max_list_entries as u64 + BIN_SORT_THREADS - 1) / BIN_SORT_THREADS
}

fn tile_height_for(height: u32, tile_size: u32) -> u32 {
    (height + tile_size - 1) / tile_size
}

impl MetalFusedTraining {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        scene: &GaussianScene,
        camera: &Camera,
        background: [f32; 3],
        target_linear: &[f32],
        width: u32,
        height: u32,
        tile_size: u32,
        tile_width: u32,
        radius_scale: f32,
        alpha_cutoff: f32,
        max_splat_steps: u32,
        transmittance_threshold: f32,
        max_list_entries: u32,
        packed_param_count: u32,
        packed_params_host: &[f32],
        moments_host: &[f32],
    ) -> Self {
        let dev = metal_device().expect("Metal device");
        let count = scene.count();
        let bg = linearize_background(background);
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
        let prepared = prepared_raster_from_training(
            &prep.projected,
            &prep.sorted_values,
            &prep.tile_ranges,
            camera,
            bg,
            width,
            height,
            tile_size,
            tile_width,
            alpha_cutoff,
            max_splat_steps,
            transmittance_threshold,
        );
        let mk_f = |data: &[f32]| -> Buffer {
            dev.device.new_buffer_with_data(
                data.as_ptr() as *const _,
                (data.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        let mk_u = |data: &[u32]| -> Buffer {
            dev.device.new_buffer_with_data(
                data.as_ptr() as *const _,
                (data.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        let mk_z = |n: usize| -> Buffer {
            dev.device.new_buffer(
                (n * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        let n_pix = (width * height * 4) as usize;
        let n_grad = count * 4;
        let tile_height = (height + tile_size - 1) / tile_size;
        let tile_count = tile_width * tile_height;
        Self {
            device: dev.device.clone(),
            queue: dev.queue.clone(),
            count,
            width,
            height,
            max_splat_steps,
            positions: mk_f(&scene.positions),
            scales: mk_f(&scene.scales),
            rotations: mk_f(&scene.rotations),
            opacities: mk_f(&scene.opacities),
            color_alpha: mk_f(&prepared.color_alpha),
            pos_local: mk_f(&prepared.pos_local),
            inv_scale: mk_f(&prepared.inv_scale),
            valid: mk_u(&prepared.valid),
            quat: mk_f(&prepared.quat),
            sorted_values: mk_u(&prepared.sorted_values),
            tile_ranges: mk_u(&prepared.tile_ranges),
            rays: mk_f(&prepared.rays),
            rgba: mk_z(n_pix),
            pixel_rgb_grad: mk_z((width * height * 3) as usize),
            target_rgba: mk_f(target_linear),
            loss_atomic: mk_z(1),
            ssim_sum_atomic: mk_z(1),
            ssim_stats: mk_z((width * height * 15) as usize),
            traces: GpuTrainingTraceBuffers::new(&dev.device, width, height, max_splat_steps),
            color_alpha_grad: mk_z(n_grad),
            grad_positions: mk_z(count * 3),
            grad_scales: mk_z(count * 3),
            grad_rotations: mk_z(count * 4),
            grad_opacities: mk_z(count),
            grad_colors: mk_z(count * 3),
            grad_pos_local: mk_z(count * 3),
            grad_inv_scale: mk_z(count * 3),
            grad_quat: mk_z(count * 4),
            grad_opacity_hit: mk_z(count),
            center_radius_depth: mk_z(count * 4),
            ellipse_conic: mk_z(count * 3),
            depth_sorted_ids: mk_z(count),
            emit_count_buf: mk_z(1),
            bin_keys: mk_z(max_list_entries as usize),
            bin_values: mk_z(max_list_entries as usize),
            bin_sorted_keys: mk_z(max_list_entries as usize),
            bin_histogram: mk_z(tile_count as usize),
            bin_cursor: mk_z(tile_count as usize),
            bin_counter: mk_z(1),
            packed_params: mk_f(packed_params_host),
            packed_grads: mk_z(packed_params_host.len()),
            moments_flat: mk_f(moments_host),
            raster_params: prepared.params,
            tile_count,
            max_list_entries,
            packed_param_count,
        }
    }

    pub fn upload_packed_params(&self, params: &[f32]) {
        Self::write_f32(&self.packed_params, params);
    }

    pub fn read_packed_params(&self, len: usize) -> Vec<f32> {
        unsafe {
            std::slice::from_raw_parts(self.packed_params.contents() as *const f32, len).to_vec()
        }
    }

    pub fn read_packed_grads(&self, len: usize) -> Vec<f32> {
        unsafe {
            std::slice::from_raw_parts(self.packed_grads.contents() as *const f32, len).to_vec()
        }
    }

    pub fn read_rgba_linear(&self) -> Vec<f32> {
        let n = (self.width * self.height * 4) as usize;
        unsafe { std::slice::from_raw_parts(self.rgba.contents() as *const f32, n).to_vec() }
    }

    pub fn read_pixel_rgb_grad(&self) -> Vec<f32> {
        let n = (self.width * self.height * 3) as usize;
        unsafe { std::slice::from_raw_parts(self.pixel_rgb_grad.contents() as *const f32, n).to_vec() }
    }

    pub fn read_bin_list_count(&self) -> u32 {
        unsafe { *(self.bin_counter.contents() as *const u32) }
    }

    pub fn count_valid_splats(&self) -> u32 {
        let valid = unsafe { std::slice::from_raw_parts(self.valid.contents() as *const u32, self.count) };
        valid.iter().filter(|&&v| v != 0).count() as u32
    }

    pub fn read_moments_into(&self, moments: &mut [[f32; 2]]) {
        unsafe {
            let ptr = self.moments_flat.contents() as *const f32;
            for (i, m) in moments.iter_mut().enumerate() {
                m[0] = *ptr.add(i * 2);
                m[1] = *ptr.add(i * 2 + 1);
            }
        }
    }

    pub fn upload_moments(&self, moments: &[[f32; 2]]) {
        let mut flat = vec![0.0f32; moments.len() * 2];
        for (i, m) in moments.iter().enumerate() {
            flat[i * 2] = m[0];
            flat[i * 2 + 1] = m[1];
        }
        Self::write_f32(&self.moments_flat, &flat);
    }

    pub fn upload_scene(&self, scene: &GaussianScene) {
        Self::write_f32(&self.positions, &scene.positions);
        Self::write_f32(&self.scales, &scene.scales);
        Self::write_f32(&self.rotations, &scene.rotations);
        Self::write_f32(&self.opacities, &scene.opacities);
        // Raster backward uses `quat` buffer (w,x,y,z) — keep in sync with rotations.
        Self::write_f32(&self.quat, &scene.rotations);
    }

    pub fn upload_color_alpha(&self, color_alpha: &[f32]) {
        Self::write_f32(&self.color_alpha, color_alpha);
        let n = color_alpha.len() / 4;
        if n > 0 && n <= self.count {
            let mut opacities = vec![0.0f32; n];
            for i in 0..n {
                opacities[i] = color_alpha[i * 4 + 3];
            }
            Self::write_f32(&self.opacities, &opacities);
        }
    }

    pub fn upload_target(&self, target: &[f32]) {
        Self::write_f32(&self.target_rgba, target);
    }

    /// Upload CPU conic binning + raster prep (no redundant `build_training_prepare` if caller already has `prep`).
    pub fn apply_cpu_training_prepare(
        &mut self,
        prep: &TrainingPrepare,
        camera: &Camera,
        background: [f32; 3],
        tile_size: u32,
        tile_width: u32,
        alpha_cutoff: f32,
        transmittance_threshold: f32,
    ) {
        let bg = linearize_background(background);
        let prepared = prepared_raster_from_training(
            &prep.projected,
            &prep.sorted_values,
            &prep.tile_ranges,
            camera,
            bg,
            self.width,
            self.height,
            tile_size,
            tile_width,
            alpha_cutoff,
            self.max_splat_steps,
            transmittance_threshold,
        );
        Self::write_u32(&self.sorted_values, &prepared.sorted_values);
        Self::write_u32(&self.tile_ranges, &prepared.tile_ranges);
        Self::write_f32(&self.rays, &prepared.rays);
        Self::write_f32(&self.quat, &prepared.quat);
        Self::write_f32(&self.pos_local, &prepared.pos_local);
        Self::write_f32(&self.inv_scale, &prepared.inv_scale);
        Self::write_u32(&self.valid, &prepared.valid);
        Self::write_f32(&self.color_alpha, &prepared.color_alpha);
        self.raster_params = prepared.params;
    }

    pub fn refresh_prep_cpu(
        &mut self,
        scene: &GaussianScene,
        camera: &Camera,
        background: [f32; 3],
        tile_size: u32,
        tile_width: u32,
        radius_scale: f32,
        alpha_cutoff: f32,
        max_list_entries: u32,
        transmittance_threshold: f32,
    ) {
        let prep = build_training_prepare(
            scene,
            camera,
            self.width,
            self.height,
            tile_size,
            tile_width,
            radius_scale,
            alpha_cutoff,
            max_list_entries,
        );
        self.apply_cpu_training_prepare(
            &prep,
            camera,
            background,
            tile_size,
            tile_width,
            alpha_cutoff,
            transmittance_threshold,
        );
    }

    fn write_f32(buf: &Buffer, data: &[f32]) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                buf.contents() as *mut f32,
                data.len().min(buf.length() as usize / 4),
            );
        }
    }

    fn write_u32(buf: &Buffer, data: &[u32]) {
        unsafe {
            let n = data.len().min(buf.length() as usize / 4);
            std::ptr::copy_nonoverlapping(data.as_ptr(), buf.contents() as *mut u32, n);
        }
    }

    fn zero_buffer(buf: &Buffer) {
        unsafe {
            std::ptr::write_bytes(buf.contents(), 0, buf.length() as usize);
        }
    }

    /// Depth-sorted visible splat indices into `depth_sorted_ids` (no heap alloc).
    pub fn fill_depth_sorted_emit_order(&self) -> u32 {
        let n = self.count;
        if n == 0 {
            return 0;
        }
        let crd =
            unsafe { std::slice::from_raw_parts(self.center_radius_depth.contents() as *const f32, n * 4) };
        let valid = unsafe { std::slice::from_raw_parts(self.valid.contents() as *const u32, n) };
        let out = unsafe {
            std::slice::from_raw_parts_mut(self.depth_sorted_ids.contents() as *mut u32, n)
        };
        let mut vis = 0usize;
        for i in 0..n {
            if valid[i] != 0 {
                out[vis] = i as u32;
                vis += 1;
            }
        }
        out[..vis].sort_unstable_by(|&a, &b| {
            crd[a as usize * 4 + 3]
                .partial_cmp(&crd[b as usize * 4 + 3])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        vis as u32
    }

    /// Depth-sorted visible splat indices (matches CPU `build_tile_key_value_pairs` order).
    pub fn build_depth_sorted_emit_order(&self) -> (Vec<u32>, u32) {
        let n = self.fill_depth_sorted_emit_order() as usize;
        if n == 0 {
            return (Vec::new(), 0);
        }
        let order = unsafe {
            std::slice::from_raw_parts(self.depth_sorted_ids.contents() as *const u32, n).to_vec()
        };
        (order, n as u32)
    }

    fn splat_thread_groups(&self, thread_width: u64) -> (u64, u64) {
        let tg = thread_width.max(1);
        let groups = (self.count as u64 + tg - 1) / tg;
        (groups, tg)
    }

    fn dispatch_project_training(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        pr: &SplatProjectParams,
    ) {
        let k = kernels();
        let (groups, tg) = self.splat_thread_groups(
            k.gaussian_splat_project_training.thread_execution_width(),
        );
        enc.set_compute_pipeline_state(&k.gaussian_splat_project_training);
        enc.set_buffer(0, Some(&self.positions), 0);
        enc.set_buffer(1, Some(&self.scales), 0);
        enc.set_buffer(2, Some(&self.rotations), 0);
        enc.set_buffer(3, Some(&self.opacities), 0);
        enc.set_buffer(4, Some(&self.color_alpha), 0);
        enc.set_buffer(5, Some(&self.pos_local), 0);
        enc.set_buffer(6, Some(&self.inv_scale), 0);
        enc.set_buffer(7, Some(&self.valid), 0);
        enc.set_buffer(8, Some(&self.color_alpha), 0);
        enc.set_buffer(9, Some(&self.center_radius_depth), 0);
        enc.set_bytes(
            10,
            std::mem::size_of::<SplatProjectParams>() as u64,
            pr as *const _ as *const _,
        );
        enc.dispatch_thread_groups(
            metal::MTLSize::new(groups, 1, 1),
            metal::MTLSize::new(tg, 1, 1),
        );
    }

    fn dispatch_gpu_conic_project_training_only(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        pr: &SplatProjectParams,
    ) {
        self.dispatch_project_training(enc, pr);
    }

    fn dispatch_gpu_conic_screen_ellipse_only(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        pr: &SplatProjectParams,
    ) {
        let k = kernels();
        let tg = k.gaussian_splat_project_screen_ellipse.thread_execution_width().max(1);
        let groups = (self.count as u64 + tg as u64 - 1) / tg as u64;
        enc.set_compute_pipeline_state(&k.gaussian_splat_project_screen_ellipse);
        enc.set_buffer(0, Some(&self.positions), 0);
        enc.set_buffer(1, Some(&self.scales), 0);
        enc.set_buffer(2, Some(&self.rotations), 0);
        enc.set_buffer(3, Some(&self.opacities), 0);
        enc.set_buffer(4, Some(&self.center_radius_depth), 0);
        enc.set_buffer(5, Some(&self.ellipse_conic), 0);
        enc.set_buffer(6, Some(&self.valid), 0);
        enc.set_bytes(7, std::mem::size_of::<SplatProjectParams>() as u64, pr as *const _ as *const _);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(groups, 1, 1),
            metal::MTLSize::new(tg, 1, 1),
        );
    }

    fn dispatch_bin_sort_pipeline(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
    ) {
        let k = kernels();
        let list_dispatch = bin_list_dispatch_groups(self.max_list_entries);
        let tile_count = self.tile_count;

        enc.set_compute_pipeline_state(&k.gaussian_splat_bin_histogram);
        enc.set_buffer(0, Some(&self.bin_keys), 0);
        enc.set_buffer(1, Some(&self.bin_histogram), 0);
        enc.set_buffer(2, Some(&self.bin_counter), 0);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(list_dispatch, 1, 1),
            metal::MTLSize::new(BIN_SORT_THREADS, 1, 1),
        );

        enc.set_compute_pipeline_state(&k.gaussian_splat_bin_copy_counts);
        enc.set_buffer(0, Some(&self.bin_histogram), 0);
        enc.set_buffer(1, Some(&self.bin_cursor), 0);
        enc.set_bytes(2, 4, &tile_count as *const u32 as *const _);
        enc.dispatch_thread_groups(metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(1, 1, 1));

        enc.set_compute_pipeline_state(&k.gaussian_splat_bin_prefix_sum);
        enc.set_buffer(0, Some(&self.bin_histogram), 0);
        enc.set_bytes(1, 4, &tile_count as *const u32 as *const _);
        enc.dispatch_thread_groups(metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(1, 1, 1));

        enc.set_compute_pipeline_state(&k.gaussian_splat_bin_scatter);
        enc.set_buffer(0, Some(&self.bin_keys), 0);
        enc.set_buffer(1, Some(&self.bin_values), 0);
        enc.set_buffer(2, Some(&self.bin_sorted_keys), 0);
        enc.set_buffer(3, Some(&self.sorted_values), 0);
        enc.set_buffer(4, Some(&self.bin_histogram), 0);
        enc.set_buffer(5, Some(&self.bin_counter), 0);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(list_dispatch, 1, 1),
            metal::MTLSize::new(BIN_SORT_THREADS, 1, 1),
        );

        enc.set_compute_pipeline_state(&k.gaussian_splat_build_tile_ranges);
        enc.set_buffer(0, Some(&self.bin_histogram), 0);
        enc.set_buffer(1, Some(&self.bin_cursor), 0);
        enc.set_buffer(2, Some(&self.tile_ranges), 0);
        enc.set_bytes(3, 4, &tile_count as *const u32 as *const _);
        enc.dispatch_thread_groups(metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(1, 1, 1));
    }

    fn dispatch_emit_radius_aabb(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        bp: &SplatBinParams,
    ) {
        let k = kernels();
        let (groups, tg) = self.splat_thread_groups(
            k.gaussian_splat_emit_tile_keys.thread_execution_width(),
        );
        enc.set_compute_pipeline_state(&k.gaussian_splat_emit_tile_keys);
        enc.set_buffer(0, Some(&self.center_radius_depth), 0);
        enc.set_buffer(1, Some(&self.valid), 0);
        enc.set_buffer(2, Some(&self.bin_keys), 0);
        enc.set_buffer(3, Some(&self.bin_values), 0);
        enc.set_buffer(4, Some(&self.bin_counter), 0);
        enc.set_bytes(5, std::mem::size_of::<SplatBinParams>() as u64, bp as *const _ as *const _);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(groups, 1, 1),
            metal::MTLSize::new(tg, 1, 1),
        );
    }

    fn dispatch_gpu_conic_emit(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        bp: &SplatBinParams,
        emit_count: u32,
    ) {
        let k = kernels();
        let tg = k.gaussian_splat_emit_tile_keys_conic.thread_execution_width().max(1);
        unsafe {
            *(self.emit_count_buf.contents() as *mut u32) = emit_count;
        }
        let emit_groups = (emit_count as u64 + tg as u64 - 1) / tg as u64;
        enc.set_compute_pipeline_state(&k.gaussian_splat_emit_tile_keys_conic);
        enc.set_buffer(0, Some(&self.center_radius_depth), 0);
        enc.set_buffer(1, Some(&self.ellipse_conic), 0);
        enc.set_buffer(2, Some(&self.valid), 0);
        enc.set_buffer(3, Some(&self.depth_sorted_ids), 0);
        enc.set_buffer(4, Some(&self.bin_keys), 0);
        enc.set_buffer(5, Some(&self.bin_values), 0);
        enc.set_buffer(6, Some(&self.bin_counter), 0);
        enc.set_bytes(7, std::mem::size_of::<SplatBinParams>() as u64, bp as *const _ as *const _);
        enc.set_buffer(8, Some(&self.emit_count_buf), 0);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(emit_groups.max(1), 1, 1),
            metal::MTLSize::new(tg, 1, 1),
        );
    }

    /// GPU project + screen ellipse; in-place host depth sort for conic emit order.
    fn run_gpu_conic_bin_prep(&self, pr: &SplatProjectParams) -> u32 {
        let cmd_prep = self.queue.new_command_buffer();
        let enc_prep = cmd_prep.new_compute_command_encoder();
        self.dispatch_gpu_conic_project_training_only(&enc_prep, pr);
        self.dispatch_gpu_conic_screen_ellipse_only(&enc_prep, pr);
        enc_prep.end_encoding();
        cmd_prep.commit();
        cmd_prep.wait_until_completed();
        self.fill_depth_sorted_emit_order()
    }

    fn zero_step_grads_and_bins(&self) {
        Self::zero_buffer(&self.loss_atomic);
        Self::zero_buffer(&self.color_alpha_grad);
        Self::zero_buffer(&self.bin_counter);
        Self::zero_buffer(&self.bin_keys);
        Self::zero_buffer(&self.bin_histogram);
        Self::zero_buffer(&self.bin_cursor);
        for b in [
            &self.grad_positions,
            &self.grad_scales,
            &self.grad_rotations,
            &self.grad_opacities,
            &self.grad_colors,
            &self.grad_pos_local,
            &self.grad_inv_scale,
            &self.grad_quat,
            &self.grad_opacity_hit,
        ] {
            Self::zero_buffer(b);
        }
        self.traces.zero();
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_step(
        &self,
        camera: &Camera,
        tile_size: u32,
        tile_width: u32,
        radius_scale: f32,
        alpha_cutoff: f32,
        loss_grad_clip: f32,
        ssim_weight: f32,
        ssim_c2: f32,
        background: [f32; 3],
        want_image: bool,
        adam: Option<&crate::splat_adam::FusedAdamHostArgs>,
        bin_mode: SplatBinMode,
    ) -> FusedStepResult {
        let k = kernels();
        let bg = linearize_background(background);
        let cmd = self.queue.new_command_buffer();

        let tile_height = tile_height_for(self.height, tile_size);
        let pr = project_params(
            camera,
            self.width,
            self.height,
            tile_size,
            tile_width,
            tile_height,
            self.count as u32,
            radius_scale,
            alpha_cutoff,
        );
        let bp = SplatBinParams {
            tile_width,
            tile_height,
            tile_size,
            max_list_entries: self.max_list_entries,
            tile_count: self.tile_count,
        };
        self.zero_step_grads_and_bins();

        let (emit_count, emit_visible_count) = match bin_mode {
            SplatBinMode::GpuConicScanline => {
                let count = self.run_gpu_conic_bin_prep(&pr);
                (count, count)
            }
            _ => (0, 0),
        };

        let enc = cmd.new_compute_command_encoder();

        match bin_mode {
            SplatBinMode::RlxConicPrepare => {}
            SplatBinMode::GpuConicScanline => {
                self.dispatch_gpu_conic_emit(&enc, &bp, emit_count);
                self.dispatch_bin_sort_pipeline(&enc);
            }
            SplatBinMode::GpuRadiusAabb => {
                self.dispatch_project_training(&enc, &pr);
                self.dispatch_emit_radius_aabb(&enc, &bp);
                self.dispatch_bin_sort_pipeline(&enc);
            }
        }

        // Forward + traces
        enc.set_compute_pipeline_state(&k.gaussian_splat_rasterize_linear_traced);
        enc.set_buffer(0, Some(&self.rgba), 0);
        enc.set_buffer(1, Some(&self.color_alpha), 0);
        enc.set_buffer(2, Some(&self.valid), 0);
        enc.set_buffer(3, Some(&self.pos_local), 0);
        enc.set_buffer(4, Some(&self.inv_scale), 0);
        enc.set_buffer(5, Some(&self.quat), 0);
        enc.set_buffer(6, Some(&self.sorted_values), 0);
        enc.set_buffer(7, Some(&self.tile_ranges), 0);
        enc.set_buffer(8, Some(&self.rays), 0);
        enc.set_bytes(
            9,
            std::mem::size_of::<SplatRasterParams>() as u64,
            &self.raster_params as *const _ as *const _,
        );
        enc.set_buffer(10, Some(&self.traces.hit_counts), 0);
        enc.set_buffer(11, Some(&self.traces.hit_splat_ids), 0);
        enc.set_buffer(12, Some(&self.traces.hit_meta), 0);
        enc.dispatch_threads(
            metal::MTLSize::new(self.width as u64, self.height as u64, 1),
            metal::MTLSize::new(8.min(self.width as u64), 8.min(self.height as u64), 1),
        );

        // Photometric loss + pixel grad on GPU (MSE-only or MSE+SSIM blend).
        let inv_pc = 1.0 / (self.width * self.height).max(1) as f32;
        let loss_p = SplatLossParams {
            width: self.width,
            height: self.height,
            inv_pixel_count: inv_pc,
            mse_grad_scale: 2.0 * (1.0 / 3.0) * inv_pc,
            ssim_weight: ssim_weight.clamp(0.0, 1.0),
            ssim_c2: ssim_c2.max(1e-8),
        };
        Self::zero_buffer(&self.pixel_rgb_grad);
        Self::zero_buffer(&self.loss_atomic);
        Self::zero_buffer(&self.ssim_sum_atomic);
        let loss_tg = metal::MTLSize::new(
            8.min(self.width as u64),
            8.min(self.height as u64),
            1,
        );
        let loss_threads = metal::MTLSize::new(self.width as u64, self.height as u64, 1);
        if loss_p.ssim_weight > 0.0 {
            enc.set_compute_pipeline_state(&k.gaussian_splat_ssim_stats);
            enc.set_buffer(0, Some(&self.rgba), 0);
            enc.set_buffer(1, Some(&self.target_rgba), 0);
            enc.set_buffer(2, Some(&self.ssim_stats), 0);
            enc.set_bytes(
                3,
                std::mem::size_of::<SplatLossParams>() as u64,
                &loss_p as *const _ as *const _,
            );
            enc.dispatch_threads(loss_threads, loss_tg);

            enc.set_compute_pipeline_state(&k.gaussian_splat_blended_loss_grad);
            enc.set_buffer(0, Some(&self.rgba), 0);
            enc.set_buffer(1, Some(&self.target_rgba), 0);
            enc.set_buffer(2, Some(&self.ssim_stats), 0);
            enc.set_buffer(3, Some(&self.pixel_rgb_grad), 0);
            enc.set_buffer(4, Some(&self.loss_atomic), 0);
            enc.set_buffer(5, Some(&self.ssim_sum_atomic), 0);
            enc.set_bytes(
                6,
                std::mem::size_of::<SplatLossParams>() as u64,
                &loss_p as *const _ as *const _,
            );
            enc.dispatch_threads(loss_threads, loss_tg);
        } else {
            enc.set_compute_pipeline_state(&k.gaussian_splat_mse_loss_grad);
            enc.set_buffer(0, Some(&self.rgba), 0);
            enc.set_buffer(1, Some(&self.target_rgba), 0);
            enc.set_buffer(2, Some(&self.pixel_rgb_grad), 0);
            enc.set_buffer(3, Some(&self.loss_atomic), 0);
            enc.set_bytes(
                4,
                std::mem::size_of::<SplatLossParams>() as u64,
                &loss_p as *const _ as *const _,
            );
            enc.dispatch_threads(loss_threads, loss_tg);
        }

        // Raster backward
        let bwd = SplatRasterBwdParams {
            width: self.width,
            height: self.height,
            max_splat_steps: self.max_splat_steps,
            loss_grad_clip,
            bg_r: bg[0],
            bg_g: bg[1],
            bg_b: bg[2],
            cam_px: camera.position[0],
            cam_py: camera.position[1],
            cam_pz: camera.position[2],
            radius_scale,
            alpha_cutoff,
        };
        enc.set_compute_pipeline_state(&k.gaussian_splat_rasterize_backward_linear);
        enc.set_buffer(0, Some(&self.color_alpha_grad), 0);
        enc.set_buffer(1, Some(&self.pixel_rgb_grad), 0);
        enc.set_buffer(2, Some(&self.traces.hit_counts), 0);
        enc.set_buffer(3, Some(&self.traces.hit_splat_ids), 0);
        enc.set_buffer(4, Some(&self.traces.hit_meta), 0);
        enc.set_bytes(
            5,
            std::mem::size_of::<SplatRasterBwdParams>() as u64,
            &bwd as *const _ as *const _,
        );
        enc.dispatch_threads(
            metal::MTLSize::new(self.width as u64, self.height as u64, 1),
            metal::MTLSize::new(8.min(self.width as u64), 8.min(self.height as u64), 1),
        );

        // Splat color backward
        enc.set_compute_pipeline_state(&k.gaussian_splat_splat_color_backward);
        enc.set_buffer(0, Some(&self.grad_colors), 0);
        enc.set_buffer(1, Some(&self.grad_opacities), 0);
        enc.set_buffer(2, Some(&self.color_alpha_grad), 0);
        enc.set_buffer(3, Some(&self.opacities), 0);
        let tg2 = k.gaussian_splat_splat_color_backward.thread_execution_width().max(1);
        let g2 = (self.count as u64 + tg2 as u64 - 1) / tg2 as u64;
        enc.dispatch_thread_groups(metal::MTLSize::new(g2, 1, 1), metal::MTLSize::new(tg2, 1, 1));

        // Geometry backward (projected)
        enc.set_compute_pipeline_state(&k.gaussian_splat_geometry_backward);
        enc.set_buffer(0, Some(&self.grad_pos_local), 0);
        enc.set_buffer(1, Some(&self.grad_inv_scale), 0);
        enc.set_buffer(2, Some(&self.grad_quat), 0);
        enc.set_buffer(3, Some(&self.grad_opacity_hit), 0);
        enc.set_buffer(4, Some(&self.pixel_rgb_grad), 0);
        enc.set_buffer(5, Some(&self.traces.hit_counts), 0);
        enc.set_buffer(6, Some(&self.traces.hit_splat_ids), 0);
        enc.set_buffer(7, Some(&self.traces.hit_meta), 0);
        enc.set_buffer(8, Some(&self.pos_local), 0);
        enc.set_buffer(9, Some(&self.inv_scale), 0);
        enc.set_buffer(10, Some(&self.quat), 0);
        enc.set_buffer(11, Some(&self.color_alpha), 0);
        enc.set_buffer(12, Some(&self.rays), 0);
        enc.set_bytes(
            13,
            std::mem::size_of::<SplatRasterBwdParams>() as u64,
            &bwd as *const _ as *const _,
        );
        enc.dispatch_threads(
            metal::MTLSize::new(self.width as u64, self.height as u64, 1),
            metal::MTLSize::new(8.min(self.width as u64), 8.min(self.height as u64), 1),
        );

        // Projected → scene grads
        enc.set_compute_pipeline_state(&k.gaussian_splat_scene_grad_projection);
        enc.set_buffer(0, Some(&self.grad_positions), 0);
        enc.set_buffer(1, Some(&self.grad_scales), 0);
        enc.set_buffer(2, Some(&self.grad_rotations), 0);
        enc.set_buffer(3, Some(&self.grad_opacities), 0);
        enc.set_buffer(4, Some(&self.grad_pos_local), 0);
        enc.set_buffer(5, Some(&self.grad_inv_scale), 0);
        enc.set_buffer(6, Some(&self.grad_quat), 0);
        enc.set_buffer(7, Some(&self.grad_opacity_hit), 0);
        enc.set_buffer(8, Some(&self.positions), 0);
        enc.set_buffer(9, Some(&self.scales), 0);
        enc.set_buffer(10, Some(&self.rotations), 0);
        enc.set_bytes(
            11,
            std::mem::size_of::<SplatProjectParams>() as u64,
            &pr as *const _ as *const _,
        );
        enc.dispatch_thread_groups(
            metal::MTLSize::new(g2, 1, 1),
            metal::MTLSize::new(tg2, 1, 1),
        );

        if let Some(adam_host) = adam {
            let coeff_count = (self.packed_param_count - 11).max(1) / 3;
            let pg = PackGradParams {
                splat_count: self.count as u32,
                packed_param_count: self.packed_param_count,
                coeff_count,
                opacity_param: self.packed_param_count - 1,
                use_sh: 0,
            };
            enc.set_compute_pipeline_state(&k.gaussian_splat_pack_grads);
            enc.set_buffer(0, Some(&self.packed_grads), 0);
            enc.set_buffer(1, Some(&self.grad_positions), 0);
            enc.set_buffer(2, Some(&self.grad_scales), 0);
            enc.set_buffer(3, Some(&self.grad_rotations), 0);
            enc.set_buffer(4, Some(&self.grad_opacities), 0);
            enc.set_buffer(5, Some(&self.grad_colors), 0);
            enc.set_buffer(6, Some(&self.opacities), 0);
            enc.set_bytes(7, std::mem::size_of::<PackGradParams>() as u64, &pg as *const _ as *const _);
            enc.dispatch_thread_groups(
                metal::MTLSize::new(g2, 1, 1),
                metal::MTLSize::new(tg2, 1, 1),
            );
            let (settings_buf, hyper_buf) = crate::splat_adam::build_adam_gpu_buffers(adam_host);
            crate::splat_adam::adam_encode_step(
                &enc,
                &k.gaussian_splat_adam_step,
                &self.packed_params,
                &self.packed_grads,
                &self.moments_flat,
                &settings_buf,
                &hyper_buf,
                self.packed_params.length() as usize,
            );
        }

        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let mse = unsafe { *(self.loss_atomic.contents() as *const f32) };
        let (loss, ssim) = if loss_p.ssim_weight > 0.0 {
            let ssim_sum = unsafe { *(self.ssim_sum_atomic.contents() as *const f32) };
            let denom = (self.width * self.height * 3).max(1) as f32;
            let mean_ssim = ssim_sum / denom;
            let ssim_loss = 1.0 - mean_ssim;
            let w = loss_p.ssim_weight;
            let total = (1.0 - w) * mse + w * ssim_loss;
            (total, mean_ssim)
        } else {
            (mse, 0.0)
        };
        let rgba_linear = if want_image {
            let n = (self.width * self.height * 4) as usize;
            Some(unsafe {
                std::slice::from_raw_parts(self.rgba.contents() as *const f32, n).to_vec()
            })
        } else {
            None
        };
        FusedStepResult {
            loss,
            mse,
            ssim,
            rgba_linear,
            bin_list_count: self.read_bin_list_count(),
            emit_visible_count,
        }
    }

    pub fn read_scene_grads(&self, sh_coeff_count: usize) -> rlx_splat::reference::SceneGrads {
        let mut grads =
            rlx_splat::reference::SceneGrads::zeroed(self.count, sh_coeff_count);
        unsafe {
            std::slice::from_raw_parts_mut(grads.positions.as_mut_ptr(), grads.positions.len())
                .copy_from_slice(std::slice::from_raw_parts(
                    self.grad_positions.contents() as *const f32,
                    grads.positions.len(),
                ));
            std::slice::from_raw_parts_mut(grads.scales.as_mut_ptr(), grads.scales.len())
                .copy_from_slice(std::slice::from_raw_parts(
                    self.grad_scales.contents() as *const f32,
                    grads.scales.len(),
                ));
            std::slice::from_raw_parts_mut(grads.rotations.as_mut_ptr(), grads.rotations.len())
                .copy_from_slice(std::slice::from_raw_parts(
                    self.grad_rotations.contents() as *const f32,
                    grads.rotations.len(),
                ));
            std::slice::from_raw_parts_mut(grads.opacities.as_mut_ptr(), grads.opacities.len())
                .copy_from_slice(std::slice::from_raw_parts(
                    self.grad_opacities.contents() as *const f32,
                    grads.opacities.len(),
                ));
            if sh_coeff_count == 0 {
                std::slice::from_raw_parts_mut(grads.colors.as_mut_ptr(), grads.colors.len())
                    .copy_from_slice(std::slice::from_raw_parts(
                        self.grad_colors.contents() as *const f32,
                        grads.colors.len(),
                    ));
            }
        }
        grads
    }
}
