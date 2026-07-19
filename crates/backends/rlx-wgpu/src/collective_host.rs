// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Host-side `Op::Custom("collective.*")` for wgpu arenas.
//!
//! Thin adapter over [`rlx_gpu_host::run_collective_bytes`].

use crate::buffer::Arena;
use crate::host_stage::WgpuArena;

pub use rlx_gpu_host::COLLECTIVE_OPS;

#[allow(clippy::too_many_arguments)]
pub fn run_collective(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    name: &str,
    in_byte_off: usize,
    in_bytes: usize,
    out_byte_off: usize,
    out_bytes: usize,
    attrs: &[u8],
) {
    let mut a = WgpuArena {
        arena,
        device,
        queue,
        size_bytes: 0,
    };
    rlx_gpu_host::run_collective_bytes(
        &mut a,
        name,
        in_byte_off,
        in_bytes,
        out_byte_off,
        out_bytes,
        attrs,
    );
}
