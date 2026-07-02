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

use crate::buffer::Arena;

pub fn run_rng_normal(
    arena: &Arena,
    queue: &wgpu::Queue,
    dst_byte_off: usize,
    len: usize,
    mean: f32,
    scale: f32,
    key: u64,
    op_seed: Option<f32>,
    opts: rlx_ir::RngOptions,
) {
    if len == 0 {
        return;
    }
    assert_eq!(
        dst_byte_off % 4,
        0,
        "rng_host: dst_byte_off must be f32-aligned"
    );
    let mut host = vec![0f32; len];
    rlx_ir::fill_normal_like(&mut host, mean, scale, opts, key, op_seed);
    arena.write_bytes_range(queue, dst_byte_off, bytemuck::cast_slice(&host));
}

pub fn run_rng_uniform(
    arena: &Arena,
    queue: &wgpu::Queue,
    dst_byte_off: usize,
    len: usize,
    low: f32,
    high: f32,
    key: u64,
    op_seed: Option<f32>,
    opts: rlx_ir::RngOptions,
) {
    if len == 0 {
        return;
    }
    assert_eq!(
        dst_byte_off % 4,
        0,
        "rng_host: dst_byte_off must be f32-aligned"
    );
    let mut host = vec![0f32; len];
    rlx_ir::fill_uniform_like(&mut host, low, high, opts, key, op_seed);
    arena.write_bytes_range(queue, dst_byte_off, bytemuck::cast_slice(&host));
}
