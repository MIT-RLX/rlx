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

//! Host-side NCHW vision / RNN ops for wgpu arenas.
//!
//! Thin adapters over [`rlx_gpu_host`]. Native WGSL kernels remain a perf
//! follow-up; this closes the correctness gap for SAM/U-Net-style graphs.

use crate::buffer::Arena;
use crate::host_stage::WgpuArena;

fn full<'a>(arena: &'a Arena, device: &'a wgpu::Device, queue: &'a wgpu::Queue) -> WgpuArena<'a> {
    WgpuArena {
        arena,
        device,
        queue,
        size_bytes: arena.size,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run_group_norm(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: usize,
    gamma: usize,
    beta: usize,
    dst: usize,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    num_groups: usize,
    eps: f32,
) {
    let mut a = full(arena, device, queue);
    rlx_gpu_host::run_group_norm_nchw(&mut a, src, gamma, beta, dst, n, c, h, w, num_groups, eps);
}

#[allow(clippy::too_many_arguments)]
pub fn run_layer_norm2d(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: usize,
    gamma: usize,
    beta: usize,
    dst: usize,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    eps: f32,
) {
    let mut a = full(arena, device, queue);
    rlx_gpu_host::run_layer_norm2d_nchw(&mut a, src, gamma, beta, dst, n, c, h, w, eps);
}

pub fn run_reverse(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: usize,
    dst: usize,
    dims: &[u32],
    rev_mask: &[bool],
    elem_bytes: usize,
) {
    let mut a = full(arena, device, queue);
    rlx_gpu_host::run_reverse(&mut a, src, dst, dims, rev_mask, elem_bytes);
}

#[allow(clippy::too_many_arguments)]
pub fn run_gru(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    x: usize,
    w_ih: usize,
    w_hh: usize,
    b_ih: usize,
    b_hh: usize,
    h0: usize,
    dst: usize,
    batch: usize,
    seq: usize,
    input_size: usize,
    hidden: usize,
    num_layers: usize,
    bidirectional: bool,
    carry: bool,
) {
    let mut a = full(arena, device, queue);
    rlx_gpu_host::run_gru(
        &mut a,
        x,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        h0,
        dst,
        batch,
        seq,
        input_size,
        hidden,
        num_layers,
        bidirectional,
        carry,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn run_rnn(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    x: usize,
    w_ih: usize,
    w_hh: usize,
    bias: usize,
    h0: usize,
    dst: usize,
    batch: usize,
    seq: usize,
    input_size: usize,
    hidden: usize,
    num_layers: usize,
    bidirectional: bool,
    carry: bool,
    relu: bool,
) {
    let mut a = full(arena, device, queue);
    rlx_gpu_host::run_rnn(
        &mut a,
        x,
        w_ih,
        w_hh,
        bias,
        h0,
        dst,
        batch,
        seq,
        input_size,
        hidden,
        num_layers,
        bidirectional,
        carry,
        relu,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn run_argreduce(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: usize,
    dst: usize,
    outer: usize,
    reduced: usize,
    inner: usize,
    is_max: bool,
) {
    let mut a = full(arena, device, queue);
    rlx_gpu_host::run_argreduce(&mut a, src, dst, outer, reduced, inner, is_max);
}

#[allow(clippy::too_many_arguments)]
pub fn run_axial_rope2d(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: usize,
    dst: usize,
    batch: usize,
    seq: usize,
    hidden: usize,
    end_x: usize,
    end_y: usize,
    head_dim: usize,
    num_heads: usize,
    theta: f32,
    repeat_factor: usize,
) {
    let mut a = full(arena, device, queue);
    rlx_gpu_host::run_axial_rope2d(
        &mut a,
        src,
        dst,
        batch,
        seq,
        hidden,
        end_x,
        end_y,
        head_dim,
        num_heads,
        theta,
        repeat_factor,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn run_resize_nearest_2x(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: usize,
    dst: usize,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
) {
    let mut a = full(arena, device, queue);
    rlx_gpu_host::run_resize_nearest_2x(&mut a, src, dst, n, c, h, w);
}
