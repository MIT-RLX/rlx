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

//! Host-side training backward / conv ops for CUDA device arenas.
//!
//! Thin adapters over [`rlx_gpu_host`].

use crate::host_stage::CudaArena;
use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

pub fn run_rms_norm_backward_input(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    arena_size_bytes: usize,
    x: usize,
    gamma: usize,
    beta: usize,
    dy: usize,
    dx: usize,
    rows: u32,
    h: u32,
    eps: f32,
) {
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: arena_size_bytes,
    };
    rlx_gpu_host::run_rms_norm_backward_input(&mut arena, x, gamma, beta, dy, dx, rows, h, eps);
}

pub fn run_rms_norm_backward_gamma(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    arena_size_bytes: usize,
    x: usize,
    gamma: usize,
    beta: usize,
    dy: usize,
    dgamma: usize,
    rows: u32,
    h: u32,
    eps: f32,
) {
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: arena_size_bytes,
    };
    rlx_gpu_host::run_rms_norm_backward_gamma(&mut arena, x, gamma, beta, dy, dgamma, rows, h, eps);
}

pub fn run_rms_norm_backward_beta(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    arena_size_bytes: usize,
    x: usize,
    gamma: usize,
    beta: usize,
    dy: usize,
    dbeta: usize,
    rows: u32,
    h: u32,
    eps: f32,
) {
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: arena_size_bytes,
    };
    rlx_gpu_host::run_rms_norm_backward_beta(&mut arena, x, gamma, beta, dy, dbeta, rows, h, eps);
}

pub fn run_rope_backward(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    arena_size_bytes: usize,
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
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: arena_size_bytes,
    };
    rlx_gpu_host::run_rope_backward(
        &mut arena, dy, cos, sin, dx, batch, seq, hidden, head_dim, n_rot, cos_len,
    );
}

pub fn run_cumsum_backward(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    arena_size_bytes: usize,
    dy: usize,
    dx: usize,
    rows: u32,
    cols: u32,
    exclusive: bool,
) {
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: arena_size_bytes,
    };
    rlx_gpu_host::run_cumsum_backward(&mut arena, dy, dx, rows, cols, exclusive);
}

pub fn run_gather_backward(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    arena_size_bytes: usize,
    dy: usize,
    indices: usize,
    dst: usize,
    outer: u32,
    axis_dim: u32,
    num_idx: u32,
    trailing: u32,
) {
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: arena_size_bytes,
    };
    rlx_gpu_host::run_gather_backward(
        &mut arena, dy, indices, dst, outer, axis_dim, num_idx, trailing,
    );
}

pub fn run_maxpool2d_backward(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    x_off: usize,
    dy_off: usize,
    dx_off: usize,
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
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: 0,
    };
    rlx_gpu_host::run_maxpool2d_backward(
        &mut arena, x_off, dy_off, dx_off, n, c, h, w, h_out, w_out, kh, kw, sh, sw, ph, pw,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn run_conv2d_forward(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    arena_size_bytes: usize,
    in_off: u32,
    w_off: u32,
    out_off: u32,
    n: u32,
    c_in: u32,
    c_out: u32,
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
    dh: u32,
    dw: u32,
    groups: u32,
) {
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: arena_size_bytes,
    };
    rlx_gpu_host::run_conv2d_forward(
        &mut arena, in_off, w_off, out_off, n, c_in, c_out, h, w, h_out, w_out, kh, kw, sh, sw, ph,
        pw, dh, dw, groups,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn run_conv2d_backward_input(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    dy_off: usize,
    w_off: usize,
    dx_off: usize,
    n: u32,
    c_in: u32,
    h: u32,
    w_in: u32,
    c_out: u32,
    h_out: u32,
    w_out: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    dh: u32,
    dw: u32,
    groups: u32,
) {
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: 0,
    };
    rlx_gpu_host::run_conv2d_backward_input(
        &mut arena, dy_off, w_off, dx_off, n, c_in, h, w_in, c_out, h_out, w_out, kh, kw, sh, sw,
        ph, pw, dh, dw, groups,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn run_conv2d_backward_weight(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    x_off: usize,
    dy_off: usize,
    dw_off: usize,
    n: u32,
    c_in: u32,
    h: u32,
    w: u32,
    c_out: u32,
    h_out: u32,
    w_out: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    dh: u32,
    dw_dil: u32,
    groups: u32,
) {
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: 0,
    };
    rlx_gpu_host::run_conv2d_backward_weight(
        &mut arena, x_off, dy_off, dw_off, n, c_in, h, w, c_out, h_out, w_out, kh, kw, sh, sw, ph,
        pw, dh, dw_dil, groups,
    );
}
