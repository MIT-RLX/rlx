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
//! Metal training forward (traced) + raster backward.

use crate::reference::TRAINING_HIT_META_FLOATS;
use crate::reference::native_prep::{PreparedRaster, SplatRasterParams};
use metal::{CommandQueue, ComputePipelineState, Device, MTLResourceOptions};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SplatRasterBwdParams {
    pub width: u32,
    pub height: u32,
    pub max_splat_steps: u32,
    pub loss_grad_clip: f32,
    pub bg_r: f32,
    pub bg_g: f32,
    pub bg_b: f32,
    pub cam_px: f32,
    pub cam_py: f32,
    pub cam_pz: f32,
    pub radius_scale: f32,
    pub alpha_cutoff: f32,
}

pub struct GpuTrainingTraceBuffers {
    pub hit_counts: metal::Buffer,
    pub hit_splat_ids: metal::Buffer,
    pub hit_meta: metal::Buffer,
    pub max_splat_steps: u32,
}

impl GpuTrainingTraceBuffers {
    pub fn new(device: &Device, width: u32, height: u32, max_splat_steps: u32) -> Self {
        let pixels = (width * height) as usize;
        let cap = max_splat_steps.max(1) as usize;
        Self {
            hit_counts: device
                .new_buffer((pixels * 4) as u64, MTLResourceOptions::StorageModeShared),
            hit_splat_ids: device.new_buffer(
                (pixels * cap * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            hit_meta: device.new_buffer(
                (pixels * cap * TRAINING_HIT_META_FLOATS * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            max_splat_steps,
        }
    }

    pub fn zero(&self) {
        unsafe {
            let z = |b: &metal::Buffer, bytes: usize| {
                std::ptr::write_bytes(b.contents(), 0, bytes);
            };
            let pixels = self.hit_counts.length() as usize / 4;
            let cap = self.max_splat_steps.max(1) as usize;
            z(&self.hit_counts, pixels * 4);
            z(&self.hit_splat_ids, pixels * cap * 4);
            z(&self.hit_meta, pixels * cap * TRAINING_HIT_META_FLOATS * 4);
        }
    }

    pub fn readback(&self, width: u32, height: u32) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
        let pixels = (width * height) as usize;
        let cap = self.max_splat_steps.max(1) as usize;
        unsafe {
            let counts =
                std::slice::from_raw_parts(self.hit_counts.contents() as *const u32, pixels)
                    .to_vec();
            let ids = std::slice::from_raw_parts(
                self.hit_splat_ids.contents() as *const u32,
                pixels * cap,
            )
            .to_vec();
            let meta = std::slice::from_raw_parts(
                self.hit_meta.contents() as *const f32,
                pixels * cap * TRAINING_HIT_META_FLOATS,
            )
            .to_vec();
            (counts, ids, meta)
        }
    }
}

fn shared_buffer_f32(device: &Device, data: &[f32]) -> metal::Buffer {
    device.new_buffer_with_data(
        data.as_ptr() as *const _,
        (data.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn shared_buffer_u32(device: &Device, data: &[u32]) -> metal::Buffer {
    device.new_buffer_with_data(
        data.as_ptr() as *const _,
        (data.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

/// Linear forward + GPU trace buffers.
pub fn dispatch_training_forward_traced(
    device: &Device,
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    prep: &PreparedRaster,
    dst: &metal::Buffer,
    traces: &GpuTrainingTraceBuffers,
) {
    traces.zero();
    let buf_ca = shared_buffer_f32(device, &prep.color_alpha);
    let buf_valid = shared_buffer_u32(device, &prep.valid);
    let buf_pl = shared_buffer_f32(device, &prep.pos_local);
    let buf_inv = shared_buffer_f32(device, &prep.inv_scale);
    let buf_quat = shared_buffer_f32(device, &prep.quat);
    let buf_sorted = shared_buffer_u32(device, &prep.sorted_values);
    let buf_ranges = shared_buffer_u32(device, &prep.tile_ranges);
    let buf_rays = shared_buffer_f32(device, &prep.rays);

    let cmd_buf = queue.new_command_buffer();
    let enc = cmd_buf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(dst), 0);
    enc.set_buffer(1, Some(&buf_ca), 0);
    enc.set_buffer(2, Some(&buf_valid), 0);
    enc.set_buffer(3, Some(&buf_pl), 0);
    enc.set_buffer(4, Some(&buf_inv), 0);
    enc.set_buffer(5, Some(&buf_quat), 0);
    enc.set_buffer(6, Some(&buf_sorted), 0);
    enc.set_buffer(7, Some(&buf_ranges), 0);
    enc.set_buffer(8, Some(&buf_rays), 0);
    enc.set_bytes(
        9,
        std::mem::size_of::<SplatRasterParams>() as u64,
        &prep.params as *const SplatRasterParams as *const _,
    );
    enc.set_buffer(10, Some(&traces.hit_counts), 0);
    enc.set_buffer(11, Some(&traces.hit_splat_ids), 0);
    enc.set_buffer(12, Some(&traces.hit_meta), 0);
    let w = prep.params.width;
    let h = prep.params.height;
    enc.dispatch_threads(
        metal::MTLSize {
            width: w as u64,
            height: h as u64,
            depth: 1,
        },
        metal::MTLSize {
            width: 8.min(w as u64),
            height: 8.min(h as u64),
            depth: 1,
        },
    );
    enc.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();
}

pub fn raster_linear_traced_to_vec(
    device: &Device,
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    prep: &PreparedRaster,
    traces: &GpuTrainingTraceBuffers,
) -> Vec<f32> {
    let n = (prep.params.width * prep.params.height * 4) as usize;
    let dst = device.new_buffer((n * 4) as u64, MTLResourceOptions::StorageModeShared);
    dispatch_training_forward_traced(device, queue, pipeline, prep, &dst, traces);
    unsafe { std::slice::from_raw_parts(dst.contents() as *const f32, n) }.to_vec()
}

/// GPU raster backward → `color_alpha_grad` (length `count * 4`).
pub fn dispatch_training_backward(
    device: &Device,
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    pixel_rgb_grad: &[f32],
    traces: &GpuTrainingTraceBuffers,
    color_alpha_grad: &metal::Buffer,
    bwd: &SplatRasterBwdParams,
    width: u32,
    height: u32,
) {
    unsafe {
        std::ptr::write_bytes(
            color_alpha_grad.contents(),
            0,
            color_alpha_grad.length() as usize,
        );
    }
    let grad_buf = shared_buffer_f32(device, pixel_rgb_grad);
    let cmd_buf = queue.new_command_buffer();
    let enc = cmd_buf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(color_alpha_grad), 0);
    enc.set_buffer(1, Some(&grad_buf), 0);
    enc.set_buffer(2, Some(&traces.hit_counts), 0);
    enc.set_buffer(3, Some(&traces.hit_splat_ids), 0);
    enc.set_buffer(4, Some(&traces.hit_meta), 0);
    enc.set_bytes(
        5,
        std::mem::size_of::<SplatRasterBwdParams>() as u64,
        bwd as *const SplatRasterBwdParams as *const _,
    );
    enc.dispatch_threads(
        metal::MTLSize {
            width: width as u64,
            height: height as u64,
            depth: 1,
        },
        metal::MTLSize {
            width: 8.min(width as u64),
            height: 8.min(height as u64),
            depth: 1,
        },
    );
    enc.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();
}

pub fn read_color_alpha_grad(buffer: &metal::Buffer, count: usize) -> Vec<f32> {
    let n = count * 4;
    unsafe { std::slice::from_raw_parts(buffer.contents() as *const f32, n) }.to_vec()
}
