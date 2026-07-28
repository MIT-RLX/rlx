// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host-side `Op::ConvTranspose2d` for wgpu arenas.
//!
//! Thin adapter over [`rlx_gpu_host::run_conv_transpose2d_nchw`].

use crate::buffer::Arena;
use crate::host_stage::WgpuArena;

#[allow(clippy::too_many_arguments)]
pub fn run_conv_transpose2d(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: usize,
    weight: usize,
    dst: usize,
    n: usize,
    c_in: usize,
    h: usize,
    w_in: usize,
    c_out: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
    groups: usize,
) {
    let mut a = WgpuArena {
        arena,
        device,
        queue,
        size_bytes: arena.size,
    };
    rlx_gpu_host::run_conv_transpose2d_nchw(
        &mut a, src, weight, dst, n, c_in, h, w_in, c_out, h_out, w_out, kh, kw, sh, sw, ph, pw,
        dh, dw, groups,
    );
}
