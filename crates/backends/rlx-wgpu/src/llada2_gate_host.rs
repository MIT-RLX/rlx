// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host-side `Op::Custom("llada2.group_limited_gate")` for wgpu arenas.

use crate::buffer::Arena;
use crate::host_stage::WgpuArena;

pub fn run_llada2_group_limited_gate(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    sig_byte_off: usize,
    route_byte_off: usize,
    out_byte_off: usize,
    n_elems: usize,
    attrs: &[u8],
) {
    debug_assert_eq!(sig_byte_off % 4, 0);
    debug_assert_eq!(route_byte_off % 4, 0);
    debug_assert_eq!(out_byte_off % 4, 0);
    let mut a = WgpuArena {
        arena,
        device,
        queue,
        size_bytes: arena.size,
    };
    rlx_gpu_host::run_llada2_group_limited_gate(
        &mut a,
        sig_byte_off / 4,
        route_byte_off / 4,
        out_byte_off / 4,
        n_elems,
        attrs,
    );
}
