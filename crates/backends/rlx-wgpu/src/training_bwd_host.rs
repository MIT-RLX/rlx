// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host-side training backward ops for wgpu arenas.
//!
//! Thin adapters over [`rlx_gpu_host`] (whole-arena mirror).

use crate::buffer::Arena;
use crate::host_stage::WgpuArena;

fn arena<'a>(arena: &'a Arena, device: &'a wgpu::Device, queue: &'a wgpu::Queue) -> WgpuArena<'a> {
    WgpuArena {
        arena,
        device,
        queue,
        size_bytes: arena.size,
    }
}

pub fn run_rms_norm_backward_input(
    arena_buf: &Arena,
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
    let mut a = arena(arena_buf, device, queue);
    rlx_gpu_host::run_rms_norm_backward_input(&mut a, x, gamma, beta, dy, dx, rows, h, eps);
}

pub fn run_rms_norm_backward_gamma(
    arena_buf: &Arena,
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
    let mut a = arena(arena_buf, device, queue);
    rlx_gpu_host::run_rms_norm_backward_gamma(&mut a, x, gamma, beta, dy, dgamma, rows, h, eps);
}

pub fn run_rms_norm_backward_beta(
    arena_buf: &Arena,
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
    let mut a = arena(arena_buf, device, queue);
    rlx_gpu_host::run_rms_norm_backward_beta(&mut a, x, gamma, beta, dy, dbeta, rows, h, eps);
}

pub fn run_rope_backward(
    arena_buf: &Arena,
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
    let mut a = arena(arena_buf, device, queue);
    rlx_gpu_host::run_rope_backward(
        &mut a, dy, cos, sin, dx, batch, seq, hidden, head_dim, n_rot, cos_len,
    );
}

pub fn run_cumsum_backward(
    arena_buf: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    dy: usize,
    dx: usize,
    rows: u32,
    cols: u32,
    exclusive: bool,
) {
    let mut a = arena(arena_buf, device, queue);
    rlx_gpu_host::run_cumsum_backward(&mut a, dy, dx, rows, cols, exclusive);
}

pub fn run_gather_backward(
    arena_buf: &Arena,
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
    let mut a = arena(arena_buf, device, queue);
    rlx_gpu_host::run_gather_backward(&mut a, dy, indices, dst, outer, axis_dim, num_idx, trailing);
}

#[allow(clippy::too_many_arguments)]
pub fn run_maxpool2d_backward(
    arena_buf: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    x_f32_off: usize,
    dy_f32_off: usize,
    dx_f32_off: usize,
    n: u32,
    c: u32,
    h: u32,
    w: u32,
    h_out: u32,
    w_out: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
) {
    let mut a = arena(arena_buf, device, queue);
    rlx_gpu_host::run_maxpool2d_backward(
        &mut a, x_f32_off, dy_f32_off, dx_f32_off, n, c, h, w, h_out, w_out, kh, kw, sh, sw, ph, pw,
    );
}
