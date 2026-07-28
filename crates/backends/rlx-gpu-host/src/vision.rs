// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Whole-arena host-fallbacks for NCHW vision / RNN ops without native GPU kernels.

use crate::{DeviceArena, with_whole_arena};

/// Host-side NCHW GroupNorm. Byte offsets.
#[allow(clippy::too_many_arguments)]
pub fn run_group_norm_nchw<A: DeviceArena>(
    a: &mut A,
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
    with_whole_arena(a, |host| unsafe {
        rlx_cpu::thunk::execute_group_norm_nchw_f32(
            src,
            gamma,
            beta,
            dst,
            n,
            c,
            h,
            w,
            num_groups,
            eps,
            host.as_mut_ptr(),
        );
    });
}

/// Host-side NCHW LayerNorm2d (channel-wise). Byte offsets.
#[allow(clippy::too_many_arguments)]
pub fn run_layer_norm2d_nchw<A: DeviceArena>(
    a: &mut A,
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
    with_whole_arena(a, |host| unsafe {
        rlx_cpu::thunk::execute_layer_norm2d_nchw_f32(
            src,
            gamma,
            beta,
            dst,
            n,
            c,
            h,
            w,
            eps,
            host.as_mut_ptr(),
        );
    });
}

/// Host-side GRU. Byte offsets.
#[allow(clippy::too_many_arguments)]
pub fn run_gru<A: DeviceArena>(
    a: &mut A,
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
    with_whole_arena(a, |host| unsafe {
        rlx_cpu::thunk::execute_gru_f32(
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
            host.as_mut_ptr(),
        );
    });
}

/// Host-side Elman RNN. Byte offsets.
#[allow(clippy::too_many_arguments)]
pub fn run_rnn<A: DeviceArena>(
    a: &mut A,
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
    with_whole_arena(a, |host| unsafe {
        rlx_cpu::thunk::execute_rnn_f32(
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
            host.as_mut_ptr(),
        );
    });
}

/// Host-side Mamba-2 SSD scan. Byte offsets.
#[allow(clippy::too_many_arguments)]
pub fn run_mamba2<A: DeviceArena>(
    a: &mut A,
    x: usize,
    dt: usize,
    a_off: usize,
    b: usize,
    c: usize,
    dst: usize,
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    state_size: usize,
) {
    with_whole_arena(a, |host| unsafe {
        rlx_cpu::thunk::execute_mamba2_f32(
            x,
            dt,
            a_off,
            b,
            c,
            dst,
            batch,
            seq,
            heads,
            head_dim,
            state_size,
            host.as_mut_ptr(),
        );
    });
}

/// Host-side nearest 2× upsample on NCHW. Byte offsets.
#[allow(clippy::too_many_arguments)]
pub fn run_resize_nearest_2x<A: DeviceArena>(
    a: &mut A,
    src: usize,
    dst: usize,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
) {
    with_whole_arena(a, |host| unsafe {
        rlx_cpu::thunk::execute_resize_nearest_2x_f32(src, dst, n, c, h, w, host.as_mut_ptr());
    });
}

/// Host-side `Op::ConvTranspose2d` (NCHW). Byte offsets.
#[allow(clippy::too_many_arguments)]
pub fn run_conv_transpose2d_nchw<A: DeviceArena>(
    a: &mut A,
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
    with_whole_arena(a, |host| unsafe {
        rlx_cpu::thunk::execute_conv_transpose2d_nchw_f32(
            src,
            weight,
            dst,
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
            dw,
            groups,
            host.as_mut_ptr(),
        );
    });
}

/// Host-side `Op::ConvTranspose3d` (NCDHW). Byte offsets.
#[allow(clippy::too_many_arguments)]
pub fn run_conv_transpose3d_ncdhw<A: DeviceArena>(
    a: &mut A,
    src: usize,
    weight: usize,
    dst: usize,
    n: usize,
    c_in: usize,
    d: usize,
    h: usize,
    w_in: usize,
    c_out: usize,
    d_out: usize,
    h_out: usize,
    w_out: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    sd: usize,
    sh: usize,
    sw: usize,
    pd: usize,
    ph: usize,
    pw: usize,
    dd: usize,
    dh: usize,
    dw: usize,
    groups: usize,
) {
    with_whole_arena(a, |host| unsafe {
        rlx_cpu::thunk::execute_conv_transpose3d_ncdhw_f32(
            src,
            weight,
            dst,
            n as u32,
            c_in as u32,
            d as u32,
            h as u32,
            w_in as u32,
            c_out as u32,
            d_out as u32,
            h_out as u32,
            w_out as u32,
            kd as u32,
            kh as u32,
            kw as u32,
            sd as u32,
            sh as u32,
            sw as u32,
            pd as u32,
            ph as u32,
            pw as u32,
            dd as u32,
            dh as u32,
            dw as u32,
            groups as u32,
            host.as_mut_ptr(),
        );
    });
}
