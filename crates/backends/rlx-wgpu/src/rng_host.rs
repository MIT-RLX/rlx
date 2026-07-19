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

//! Host-side RNG fill for wgpu arenas (fill on host → H2D).
//!
//! Thin adapter over [`rlx_gpu_host::run_rng_normal`] /
//! [`rlx_gpu_host::run_rng_uniform`].

use crate::buffer::Arena;
use crate::host_stage::WgpuArena;

pub fn run_rng_normal(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    dst_byte_off: usize,
    len: usize,
    mean: f32,
    scale: f32,
    key: u64,
    op_seed: Option<f32>,
    opts: rlx_ir::RngOptions,
) {
    let mut a = WgpuArena {
        arena,
        device,
        queue,
        size_bytes: 0,
    };
    rlx_gpu_host::run_rng_normal(&mut a, dst_byte_off, len, mean, scale, key, op_seed, opts);
}

pub fn run_rng_uniform(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    dst_byte_off: usize,
    len: usize,
    low: f32,
    high: f32,
    key: u64,
    op_seed: Option<f32>,
    opts: rlx_ir::RngOptions,
) {
    let mut a = WgpuArena {
        arena,
        device,
        queue,
        size_bytes: 0,
    };
    rlx_gpu_host::run_rng_uniform(&mut a, dst_byte_off, len, low, high, key, op_seed, opts);
}
