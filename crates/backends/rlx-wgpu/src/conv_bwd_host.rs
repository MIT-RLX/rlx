// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Host `Conv2dBackward*` for wgpu (D2H → CPU → H2D).
//!
//! Thin adapters over [`rlx_gpu_host`] compact-scratch staging. wgpu has no
//! native conv-backward kernel; reading `x`/`dy` once and computing on the CPU
//! sidesteps autodiff decomposition under memory pressure.

use crate::buffer::Arena;
use crate::host_stage::WgpuArena;

#[allow(clippy::too_many_arguments)]
pub fn run_conv2d_backward_weight(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    x_byte_off: u32,
    dy_byte_off: u32,
    dw_byte_off: u32,
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
    let mut a = WgpuArena {
        arena,
        device,
        queue,
        size_bytes: 0,
    };
    rlx_gpu_host::run_conv2d_backward_weight(
        &mut a,
        x_byte_off as usize / 4,
        dy_byte_off as usize / 4,
        dw_byte_off as usize / 4,
        n,
        c_in,
        h,
        w,
        c_out,
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
        groups,
    );
}

/// Host `Op::Conv2dBackwardInput` (D2H → CPU → H2D).
#[allow(clippy::too_many_arguments)]
pub fn run_conv2d_backward_input(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    dy_byte_off: u32,
    w_byte_off: u32,
    dx_byte_off: u32,
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
    dw_dil: u32,
    groups: u32,
) {
    let mut a = WgpuArena {
        arena,
        device,
        queue,
        size_bytes: 0,
    };
    rlx_gpu_host::run_conv2d_backward_input(
        &mut a,
        dy_byte_off as usize / 4,
        w_byte_off as usize / 4,
        dx_byte_off as usize / 4,
        n,
        c_in,
        h,
        w_in,
        c_out,
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
        groups,
    );
}
