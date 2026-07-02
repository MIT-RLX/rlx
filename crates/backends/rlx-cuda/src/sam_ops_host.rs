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
//! Legacy host-side SAM conv/norm ops (D2H → CPU → H2D).
//! Superseded by native kernels in `kernels/layer_norm2d.cu` and
//! `kernels/conv_transpose2d.cu`; kept for manual debugging.

use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

fn run_on_arena(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    arena_size_bytes: usize,
    f: impl FnOnce(*mut u8),
) {
    let n_f32 = arena_size_bytes / 4;
    stream
        .synchronize()
        .expect("rlx-cuda: sam_ops pre-sync failed");
    let mut host = vec![0f32; n_f32];
    stream
        .memcpy_dtoh(&buffer.slice(..), &mut host)
        .expect("rlx-cuda: sam_ops arena dtoh failed");
    f(host.as_mut_ptr() as *mut u8);
    stream
        .memcpy_htod(&host, &mut buffer.slice_mut(..))
        .expect("rlx-cuda: sam_ops arena htod failed");
}

pub fn run_layer_norm2d(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    arena_size_bytes: usize,
    src: usize,
    g: usize,
    b: usize,
    dst: usize,
    n: u32,
    c: u32,
    h: u32,
    w: u32,
    eps: f32,
) {
    run_on_arena(stream, buffer, arena_size_bytes, |base| unsafe {
        rlx_cpu::thunk::execute_layer_norm2d_nchw_f32(
            src, g, b, dst, n as usize, c as usize, h as usize, w as usize, eps, base,
        );
    });
}

pub fn run_conv_transpose2d(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    arena_size_bytes: usize,
    src: usize,
    weight: usize,
    dst: usize,
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
    run_on_arena(stream, buffer, arena_size_bytes, |base| unsafe {
        rlx_cpu::thunk::execute_conv_transpose2d_nchw_f32(
            src,
            weight,
            dst,
            n as usize,
            c_in as usize,
            h as usize,
            w_in as usize,
            c_out as usize,
            h_out as usize,
            w_out as usize,
            kh as usize,
            kw as usize,
            sh as usize,
            sw as usize,
            ph as usize,
            pw as usize,
            dh as usize,
            dw as usize,
            groups as usize,
            base,
        );
    });
}
