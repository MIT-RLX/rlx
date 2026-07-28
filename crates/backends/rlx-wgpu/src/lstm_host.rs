// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host-side `Op::Lstm` for wgpu arenas (readback → CPU → writeback).

use crate::buffer::Arena;
use crate::host_stage::WgpuArena;

pub fn run_lstm(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    x_byte_off: usize,
    w_ih_byte_off: usize,
    w_hh_byte_off: usize,
    bias_byte_off: usize,
    h0_byte_off: usize,
    c0_byte_off: usize,
    dst_byte_off: usize,
    batch: usize,
    seq: usize,
    input_size: usize,
    hidden: usize,
    num_layers: usize,
    bidirectional: bool,
    carry: bool,
) {
    let mut a = WgpuArena {
        arena,
        device,
        queue,
        size_bytes: arena.size,
    };
    rlx_gpu_host::run_lstm(
        &mut a,
        x_byte_off,
        w_ih_byte_off,
        w_hh_byte_off,
        bias_byte_off,
        h0_byte_off,
        c0_byte_off,
        dst_byte_off,
        batch,
        seq,
        input_size,
        hidden,
        num_layers,
        bidirectional,
        carry,
    );
}
