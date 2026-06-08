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
    stream
        .synchronize()
        .expect("rlx-cuda: training_bwd pre-sync failed");
    let mut host = vec![0f32; n_f32];
    stream
        .memcpy_dtoh(&buffer.slice(..), &mut host)
        .expect("rlx-cuda: training_bwd arena dtoh failed");
    f(host.as_mut_ptr() as *mut u8);
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
    let x_len = (n as usize) * (c as usize) * (h as usize) * (w as usize);
    let dy_len = (n as usize) * (c as usize) * (h_out as usize) * (w_out as usize);
    stream
        .synchronize()
        .expect("rlx-cuda: maxpool2d_bwd pre-sync failed");
    let mut x_host = vec![0f32; x_len];
    let mut dy_host = vec![0f32; dy_len];
    let mut dx_host = vec![0f32; x_len];
    stream
        .memcpy_dtoh(&buffer.slice(x_off..x_off + x_len), &mut x_host)
        .expect("rlx-cuda: maxpool2d_bwd x dtoh failed");
    stream
        .memcpy_dtoh(&buffer.slice(dy_off..dy_off + dy_len), &mut dy_host)
        .expect("rlx-cuda: maxpool2d_bwd dy dtoh failed");
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
    stream
        .memcpy_htod(&dx_host, &mut buffer.slice_mut(dx_off..dx_off + x_len))
        .expect("rlx-cuda: maxpool2d_bwd dx htod failed");
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
    run_on_arena(stream, buffer, arena_size_bytes, |base| unsafe {
        rlx_cpu::thunk::execute_conv2d_forward_f32(
            (in_off as usize) * 4,
            (w_off as usize) * 4,
            (out_off as usize) * 4,
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
            base,
        );
    });
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
    stream
        .synchronize()
        .expect("rlx-cuda: conv2d_bwd_input pre-sync failed");
    let mut scratch = vec![0f32; scratch_len];
    stream
        .memcpy_dtoh(
            &buffer.slice(dy_off..dy_off + dy_len),
            &mut scratch[..dy_len],
        )
        .expect("rlx-cuda: conv2d_bwd_input dy dtoh failed");
    stream
        .memcpy_dtoh(
            &buffer.slice(w_off..w_off + w_len),
            &mut scratch[dy_len..dy_len + w_len],
        )
        .expect("rlx-cuda: conv2d_bwd_input w dtoh failed");
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
    stream
        .memcpy_htod(
            &scratch[dy_len + w_len..],
            &mut buffer.slice_mut(dx_off..dx_off + dx_len),
        )
        .expect("rlx-cuda: conv2d_bwd_input dx htod failed");
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
    stream
        .synchronize()
        .expect("rlx-cuda: conv2d_bwd_weight pre-sync failed");
    let mut scratch = vec![0f32; scratch_len];
    stream
        .memcpy_dtoh(&buffer.slice(x_off..x_off + x_len), &mut scratch[..x_len])
        .expect("rlx-cuda: conv2d_bwd_weight x dtoh failed");
    stream
        .memcpy_dtoh(
            &buffer.slice(dy_off..dy_off + dy_len),
            &mut scratch[x_len..x_len + dy_len],
        )
        .expect("rlx-cuda: conv2d_bwd_weight dy dtoh failed");
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
    stream
        .memcpy_htod(
            &scratch[x_len + dy_len..],
            &mut buffer.slice_mut(dw_off..dw_off + dw_len),
        )
        .expect("rlx-cuda: conv2d_bwd_weight dw htod failed");
}
