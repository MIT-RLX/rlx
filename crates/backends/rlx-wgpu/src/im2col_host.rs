// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host-side `Op::Im2Col` for wgpu arenas.

use crate::buffer::Arena;
use crate::host_stage::WgpuArena;

#[allow(clippy::too_many_arguments)]
pub fn run_im2col(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    x_byte_off: usize,
    col_byte_off: usize,
    n: u32,
    c_in: u32,
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
    dw_dil: u32,
) {
    let mut a = WgpuArena {
        arena,
        device,
        queue,
        size_bytes: 0,
    };
    rlx_gpu_host::run_im2col(
        &mut a,
        x_byte_off,
        col_byte_off,
        n,
        c_in,
        h,
        w,
        h_out,
        w_out,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw_dil,
    );
}
