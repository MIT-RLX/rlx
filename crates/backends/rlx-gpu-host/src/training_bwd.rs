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

//! Compact-scratch / whole-arena host-fallbacks for pool and conv training paths.
//!
//! Norm / rope / cumsum / gather backwards live in `lib.rs` (whole-arena). These
//! ops stage only the touched tensors (CUDA-style compact scratch) so multi-GiB
//! arenas are not mirrored.

use crate::{DeviceArena, with_whole_arena};

/// Host-side `Op::MaxPool2dBackward`. Offsets are **f32 elements**.
#[allow(clippy::too_many_arguments)]
pub fn run_maxpool2d_backward<A: DeviceArena>(
    a: &mut A,
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
    let x_len = (n as usize) * (c as usize) * (h as usize) * (w as usize);
    let dy_len = (n as usize) * (c as usize) * (h_out as usize) * (w_out as usize);
    a.sync();
    let mut x_host = vec![0f32; x_len];
    let mut dy_host = vec![0f32; dy_len];
    let mut dx_host = vec![0f32; x_len];
    a.dtoh(x_f32_off * 4, bytemuck::cast_slice_mut(&mut x_host));
    a.dtoh(dy_f32_off * 4, bytemuck::cast_slice_mut(&mut dy_host));
    rlx_cpu::training_bwd::maxpool2d_backward_nchw(
        &x_host,
        &dy_host,
        &mut dx_host,
        n as usize,
        c as usize,
        h as usize,
        w as usize,
        h_out as usize,
        w_out as usize,
        kh as usize,
        kw as usize,
        sh as usize,
        sw as usize,
        ph as usize,
        pw as usize,
    );
    a.htod(dx_f32_off * 4, bytemuck::cast_slice(&dx_host));
}

/// Host-side `Op::Conv2d` forward (full-arena mirror). Offsets are **f32 elements**.
#[allow(clippy::too_many_arguments)]
pub fn run_conv2d_forward<A: DeviceArena>(
    a: &mut A,
    in_f32_off: u32,
    w_f32_off: u32,
    out_f32_off: u32,
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
    with_whole_arena(a, |base| unsafe {
        rlx_cpu::thunk::execute_conv2d_forward_f32(
            (in_f32_off as usize) * 4,
            (w_f32_off as usize) * 4,
            (out_f32_off as usize) * 4,
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
            dw,
            groups,
            base.as_mut_ptr(),
        );
    });
}

/// Host-side `Op::Conv2dBackwardInput` via compact scratch. Offsets are **f32 elements**.
#[allow(clippy::too_many_arguments)]
pub fn run_conv2d_backward_input<A: DeviceArena>(
    a: &mut A,
    dy_f32_off: usize,
    w_f32_off: usize,
    dx_f32_off: usize,
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
    let n = n as usize;
    let c_in = c_in as usize;
    let h = h as usize;
    let w_in = w_in as usize;
    let c_out = c_out as usize;
    let h_out = h_out as usize;
    let w_out = w_out as usize;
    let groups = groups.max(1) as usize;
    let c_in_per_g = c_in / groups;
    let kh = kh as usize;
    let kw = kw as usize;
    let dy_len = n * c_out * h_out * w_out;
    let w_len = c_out * c_in_per_g * kh * kw;
    let dx_len = n * c_in * h * w_in;
    let scratch_len = dy_len + w_len + dx_len;

    a.sync();
    let mut scratch = vec![0f32; scratch_len];
    a.dtoh(
        dy_f32_off * 4,
        bytemuck::cast_slice_mut(&mut scratch[..dy_len]),
    );
    a.dtoh(
        w_f32_off * 4,
        bytemuck::cast_slice_mut(&mut scratch[dy_len..dy_len + w_len]),
    );
    let dx_base = (dy_len + w_len) * 4;
    unsafe {
        rlx_cpu::conv_bwd::execute_conv2d_backward_input_f32(
            scratch.as_mut_ptr() as *mut u8,
            0,
            dy_len * 4,
            dx_base,
            n as u32,
            c_in as u32,
            h as u32,
            w_in as u32,
            c_out as u32,
            h_out as u32,
            w_out as u32,
            kh as u32,
            kw as u32,
            sh,
            sw,
            ph,
            pw,
            dh,
            dw,
            groups as u32,
        );
    }
    a.htod(
        dx_f32_off * 4,
        bytemuck::cast_slice(&scratch[dy_len + w_len..dy_len + w_len + dx_len]),
    );
}

/// Host-side `Op::Conv2dBackwardWeight` via compact scratch. Offsets are **f32 elements**.
#[allow(clippy::too_many_arguments)]
pub fn run_conv2d_backward_weight<A: DeviceArena>(
    a: &mut A,
    x_f32_off: usize,
    dy_f32_off: usize,
    dw_f32_off: usize,
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
    let n = n as usize;
    let c_in = c_in as usize;
    let h = h as usize;
    let w = w as usize;
    let c_out = c_out as usize;
    let h_out = h_out as usize;
    let w_out = w_out as usize;
    let groups = groups.max(1) as usize;
    let c_in_per_g = c_in / groups;
    let kh = kh as usize;
    let kw = kw as usize;
    let x_len = n * c_in * h * w;
    let dy_len = n * c_out * h_out * w_out;
    let dw_len = c_out * c_in_per_g * kh * kw;
    let scratch_len = x_len + dy_len + dw_len;

    a.sync();
    let mut scratch = vec![0f32; scratch_len];
    a.dtoh(
        x_f32_off * 4,
        bytemuck::cast_slice_mut(&mut scratch[..x_len]),
    );
    a.dtoh(
        dy_f32_off * 4,
        bytemuck::cast_slice_mut(&mut scratch[x_len..x_len + dy_len]),
    );
    let dw_base = (x_len + dy_len) * 4;
    unsafe {
        rlx_cpu::conv_bwd::execute_conv2d_backward_weight_f32(
            scratch.as_mut_ptr() as *mut u8,
            0,
            x_len * 4,
            dw_base,
            n as u32,
            c_in as u32,
            h as u32,
            w as u32,
            c_out as u32,
            h_out as u32,
            w_out as u32,
            kh as u32,
            kw as u32,
            sh,
            sw,
            ph,
            pw,
            dh,
            dw_dil,
            groups as u32,
        );
    }
    a.htod(
        dw_f32_off * 4,
        bytemuck::cast_slice(&scratch[x_len + dy_len..x_len + dy_len + dw_len]),
    );
}
