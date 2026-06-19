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
//! Host-side training backward ops for wgpu arenas (readback → CPU → writeback).

use crate::buffer::Arena;

#[inline]
fn run_on_arena(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    f: impl FnOnce(*mut u8),
) {
    let mut host = arena.read_bytes_range(device, queue, 0, arena.size);
    f(host.as_mut_ptr());
    arena.write_bytes_range(queue, 0, &host);
}

pub fn run_rms_norm_backward_input(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    x: usize,
    gamma: usize,
    beta: usize,
    dy: usize,
    dx: usize,
    rows: u32,
    h: u32,
    eps: f32,
) {
    run_on_arena(arena, device, queue, |base| unsafe {
        rlx_cpu::thunk::execute_rms_norm_backward_input_f32(
            x, gamma, beta, dy, dx, rows, h, eps, base,
        );
    });
}

pub fn run_rms_norm_backward_gamma(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    x: usize,
    gamma: usize,
    beta: usize,
    dy: usize,
    dgamma: usize,
    rows: u32,
    h: u32,
    eps: f32,
) {
    run_on_arena(arena, device, queue, |base| unsafe {
        rlx_cpu::thunk::execute_rms_norm_backward_gamma_f32(
            x, gamma, beta, dy, dgamma, rows, h, eps, base,
        );
    });
}

pub fn run_rms_norm_backward_beta(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    x: usize,
    gamma: usize,
    beta: usize,
    dy: usize,
    dbeta: usize,
    rows: u32,
    h: u32,
    eps: f32,
) {
    run_on_arena(arena, device, queue, |base| unsafe {
        rlx_cpu::thunk::execute_rms_norm_backward_beta_f32(
            x, gamma, beta, dy, dbeta, rows, h, eps, base,
        );
    });
}

pub fn run_rope_backward(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    dy: usize,
    cos: usize,
    sin: usize,
    dx: usize,
    batch: u32,
    seq: u32,
    hidden: u32,
    head_dim: u32,
    n_rot: u32,
    cos_len: u32,
) {
    run_on_arena(arena, device, queue, |base| unsafe {
        rlx_cpu::thunk::execute_rope_backward_f32(
            dy, cos, sin, dx, batch, seq, hidden, head_dim, n_rot, cos_len, base,
        );
    });
}

pub fn run_cumsum_backward(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    dy: usize,
    dx: usize,
    rows: u32,
    cols: u32,
    exclusive: bool,
) {
    run_on_arena(arena, device, queue, |base| unsafe {
        rlx_cpu::thunk::execute_cumsum_backward_f32(dy, dx, rows, cols, exclusive, base);
    });
}

pub fn run_gather_backward(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    dy: usize,
    indices: usize,
    dst: usize,
    outer: u32,
    axis_dim: u32,
    num_idx: u32,
    trailing: u32,
) {
    run_on_arena(arena, device, queue, |base| unsafe {
        rlx_cpu::thunk::execute_gather_backward_f32(
            dy, indices, dst, outer, axis_dim, num_idx, trailing, base,
        );
    });
}
