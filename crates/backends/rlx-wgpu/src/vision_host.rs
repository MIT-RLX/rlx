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
//! Host-side NCHW vision ops for wgpu arenas (readback → CPU → writeback).
//! wgpu has no native kernel for these; they are a handful of ops in vision
//! backbones (SAM/U-Net decoders), so the CPU reference round-trip mirrors
//! the established `conv_transpose2d_host` pattern. Native WGSL kernels are a
//! perf follow-up; this closes the correctness gap.

use crate::buffer::Arena;

/// NCHW GroupNorm. Inputs `src`, `gamma`, `beta` (byte offsets); output `dst`.
#[allow(clippy::too_many_arguments)]
pub fn run_group_norm(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
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
    let mut host = arena.read_bytes_range(device, queue, 0, arena.size);
    unsafe {
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
    }
    arena.write_bytes_range(queue, 0, &host);
}

/// NCHW LayerNorm2d (channel-wise; SAM / candle semantics).
#[allow(clippy::too_many_arguments)]
pub fn run_layer_norm2d(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
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
    let mut host = arena.read_bytes_range(device, queue, 0, arena.size);
    unsafe {
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
    }
    arena.write_bytes_range(queue, 0, &host);
}

/// Batch-general reverse/flip along `rev_mask` axes. `dims` is the row-major
/// input shape; output shape is identical.
pub fn run_reverse(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: usize,
    dst: usize,
    dims: &[u32],
    rev_mask: &[bool],
    elem_bytes: usize,
) {
    let mut host = arena.read_bytes_range(device, queue, 0, arena.size);
    unsafe {
        rlx_cpu::thunk::execute_reverse(src, dst, dims, rev_mask, elem_bytes, host.as_mut_ptr());
    }
    arena.write_bytes_range(queue, 0, &host);
}

/// GRU host fallback (multi-layer / bidir / carry / hidden > 256).
#[allow(clippy::too_many_arguments)]
pub fn run_gru(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
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
    let mut host = arena.read_bytes_range(device, queue, 0, arena.size);
    unsafe {
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
    }
    arena.write_bytes_range(queue, 0, &host);
}

/// Elman RNN host fallback.
#[allow(clippy::too_many_arguments)]
pub fn run_rnn(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
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
    let mut host = arena.read_bytes_range(device, queue, 0, arena.size);
    unsafe {
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
    }
    arena.write_bytes_range(queue, 0, &host);
}

/// ArgMax/ArgMin (f32-encoded indices) over the middle `reduced` axis.
#[allow(clippy::too_many_arguments)]
pub fn run_argreduce(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: usize,
    dst: usize,
    outer: usize,
    reduced: usize,
    inner: usize,
    is_max: bool,
) {
    let mut host = arena.read_bytes_range(device, queue, 0, arena.size);
    unsafe {
        rlx_cpu::thunk::execute_argreduce_f32(
            src,
            dst,
            outer,
            reduced,
            inner,
            is_max,
            host.as_mut_ptr(),
        );
    }
    arena.write_bytes_range(queue, 0, &host);
}

/// Nearest 2× upsample on NCHW. Input `src`; output `dst` is `[n, c, 2h, 2w]`.
#[allow(clippy::too_many_arguments)]
pub fn run_resize_nearest_2x(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: usize,
    dst: usize,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
) {
    let mut host = arena.read_bytes_range(device, queue, 0, arena.size);
    unsafe {
        rlx_cpu::thunk::execute_resize_nearest_2x_f32(src, dst, n, c, h, w, host.as_mut_ptr());
    }
    arena.write_bytes_range(queue, 0, &host);
}
