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
//! Native wgpu Gaussian splat forward (RLX-owned WGSL).

#![cfg(feature = "native-splat")]

use crate::buffer::Arena;
use bytemuck::{Pod, Zeroable};
use slang_splat_ref::native_prep::{prepare_raster_from_slices, PreparedRaster, SplatRasterParams};
use std::sync::OnceLock;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WgpuSplatRasterParams {
    width: u32,
    height: u32,
    tile_size: u32,
    tile_width: u32,
    alpha_cutoff: f32,
    transmittance_threshold: f32,
    bg_r: f32,
    bg_g: f32,
    bg_b: f32,
    dst_base: u32,
}

impl WgpuSplatRasterParams {
    fn from_prep(p: SplatRasterParams, dst_base: u32) -> Self {
        Self {
            width: p.width,
            height: p.height,
            tile_size: p.tile_size,
            tile_width: p.tile_width,
            alpha_cutoff: p.alpha_cutoff,
            transmittance_threshold: p.transmittance_threshold,
            bg_r: p.bg_r,
            bg_g: p.bg_g,
            bg_b: p.bg_b,
            dst_base,
        }
    }
}

struct SplatRasterKernel {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

fn splat_raster_kernel(device: &wgpu::Device) -> &'static SplatRasterKernel {
    static K: OnceLock<SplatRasterKernel> = OnceLock::new();
    K.get_or_init(|| {
        let wgsl = include_str!("kernels/gaussian_splat_rasterize.wgsl");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rlx-wgpu gaussian_splat_rasterize"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let mut entries = Vec::with_capacity(10);
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        for b in 1..=8u32 {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: b,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 9,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rlx-wgpu gaussian_splat_rasterize"),
            entries: &entries,
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rlx-wgpu gaussian_splat_rasterize"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rlx-wgpu gaussian_splat_rasterize"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("gaussian_splat_rasterize"),
            compilation_options: Default::default(),
            cache: None,
        });
        SplatRasterKernel { pipeline, bgl }
    })
}

fn upload<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    data: &[T],
) -> wgpu::Buffer {
    let bytes = bytemuck::cast_slice(data);
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf, 0, bytes);
    buf
}

fn dispatch_prepared(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    arena: &wgpu::Buffer,
    dst_byte_off: u64,
    prep: &PreparedRaster,
) {
    let k = splat_raster_kernel(device);
    let buf_ca = upload(device, queue, "splat ca", &prep.color_alpha);
    let buf_valid = upload(device, queue, "splat valid", &prep.valid);
    let buf_pl = upload(device, queue, "splat pl", &prep.pos_local);
    let buf_inv = upload(device, queue, "splat inv", &prep.inv_scale);
    let buf_quat = upload(device, queue, "splat quat", &prep.quat);
    let buf_sorted = upload(device, queue, "splat sorted", &prep.sorted_values);
    let buf_ranges = upload(device, queue, "splat ranges", &prep.tile_ranges);
    let buf_rays = upload(device, queue, "splat rays", &prep.rays);
    let dst_base = (dst_byte_off / 4) as u32;
    let params = WgpuSplatRasterParams::from_prep(prep.params, dst_base);
    let buf_params = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("splat params"),
        size: std::mem::size_of::<WgpuSplatRasterParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf_params, 0, bytemuck::bytes_of(&params));

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rlx-wgpu gaussian_splat_rasterize"),
        layout: &k.bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: arena.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_ca.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_valid.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buf_pl.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: buf_inv.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: buf_quat.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: buf_sorted.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: buf_ranges.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: buf_rays.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: buf_params.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rlx-wgpu gaussian_splat_rasterize"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rlx-wgpu gaussian_splat_rasterize"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&k.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let w = prep.params.width;
        let h = prep.params.height;
        pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
    }
    queue.submit(Some(encoder.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
}

#[allow(clippy::too_many_arguments)]
pub fn run_gaussian_splat_render_native(
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
    let f32_bytes = |n: usize| n * 4;
    let positions = arena.read_bytes_range(device, queue, positions_byte_off, f32_bytes(positions_len));
    let scales = arena.read_bytes_range(device, queue, scales_byte_off, f32_bytes(scales_len));
    let rotations = arena.read_bytes_range(device, queue, rotations_byte_off, f32_bytes(rotations_len));
    let opacities = arena.read_bytes_range(device, queue, opacities_byte_off, f32_bytes(opacities_len));
    let colors = arena.read_bytes_range(device, queue, colors_byte_off, f32_bytes(colors_len));
    let sh_coeffs = arena.read_bytes_range(device, queue, sh_coeffs_byte_off, f32_bytes(sh_coeffs_len));
    let meta = arena.read_bytes_range(device, queue, meta_byte_off, f32_bytes(23));

    let prep = prepare_raster_from_slices(
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

    dispatch_prepared(device, queue, &arena.buffer, dst_byte_off as u64, &prep);
    assert_eq!(dst_len, (width * height * 4) as usize);
}
