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
//! Native Metal tile raster for prepared splat data (`shaders/splat.msl`).

use crate::reference::native_prep::{PreparedRaster, SplatRasterParams};
use metal::{CommandQueue, ComputePipelineState, Device, MTLResourceOptions};

fn dispatch_prepared_raster_impl(
    device: &Device,
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    prep: &PreparedRaster,
    dst: &metal::Buffer,
    dst_byte_off: u64,
) {
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
    enc.set_buffer(0, Some(dst), dst_byte_off);
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
    let w = prep.params.width;
    let h = prep.params.height;
    let grid = metal::MTLSize {
        width: w as u64,
        height: h as u64,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 8.min(w as u64),
        height: 8.min(h as u64),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
    enc.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();
}

/// Dispatch display `gaussian_splat_rasterize` into an arena RGBA buffer.
pub fn dispatch_prepared_raster(
    device: &Device,
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    prep: &PreparedRaster,
    arena_buffer: &metal::Buffer,
    dst_byte_off: u64,
) {
    dispatch_prepared_raster_impl(device, queue, pipeline, prep, arena_buffer, dst_byte_off);
}

/// Dispatch training `gaussian_splat_rasterize_linear` and read back linear RGBA.
pub fn raster_linear_to_vec(
    device: &Device,
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    prep: &PreparedRaster,
) -> Vec<f32> {
    let n = (prep.params.width * prep.params.height * 4) as usize;
    let dst = device.new_buffer((n * 4) as u64, MTLResourceOptions::StorageModeShared);
    dispatch_prepared_raster_impl(device, queue, pipeline, prep, &dst, 0);
    let ptr = dst.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
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
