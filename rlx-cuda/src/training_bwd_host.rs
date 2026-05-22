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
//! Host-side training backward ops for CUDA device arenas (D2H → CPU → H2D).

use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

fn run_on_arena(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    arena_size_bytes: usize,
    f: impl FnOnce(*mut u8),
) {
    let n_f32 = arena_size_bytes / 4;
    stream.synchronize().expect("rlx-cuda: training_bwd pre-sync failed");
    let mut host = vec![0f32; n_f32];
    stream
        .memcpy_dtoh(&buffer.slice(..), &mut host)
        .expect("rlx-cuda: training_bwd arena dtoh failed");
    unsafe {
        f(host.as_mut_ptr() as *mut u8);
    }
    stream
        .memcpy_htod(&host, &mut buffer.slice_mut(..))
        .expect("rlx-cuda: training_bwd arena htod failed");
}

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
    run_on_arena(stream, buffer, arena_size_bytes, |base| unsafe {
        rlx_cpu::thunk::execute_rms_norm_backward_input_f32(
            x, gamma, beta, dy, dx, rows, h, eps, base,
        );
    });
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
    run_on_arena(stream, buffer, arena_size_bytes, |base| unsafe {
        rlx_cpu::thunk::execute_rms_norm_backward_gamma_f32(
            x, gamma, beta, dy, dgamma, rows, h, eps, base,
        );
    });
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
    run_on_arena(stream, buffer, arena_size_bytes, |base| unsafe {
        rlx_cpu::thunk::execute_rms_norm_backward_beta_f32(
            x, gamma, beta, dy, dbeta, rows, h, eps, base,
        );
    });
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
    run_on_arena(stream, buffer, arena_size_bytes, |base| unsafe {
        rlx_cpu::thunk::execute_rope_backward_f32(
            dy, cos, sin, dx, batch, seq, hidden, head_dim, n_rot, cos_len, base,
        );
    });
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
    run_on_arena(stream, buffer, arena_size_bytes, |base| unsafe {
        rlx_cpu::thunk::execute_cumsum_backward_f32(dy, dx, rows, cols, exclusive, base);
    });
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
    run_on_arena(stream, buffer, arena_size_bytes, |base| unsafe {
        rlx_cpu::thunk::execute_gather_backward_f32(
            dy, indices, dst, outer, axis_dim, num_idx, trailing, base,
        );
    });
}
