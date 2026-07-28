// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host-side `Op::Custom("collective.*")` for ROCm arenas.
//!
//! Thin adapter over [`rlx_gpu_host::run_collective_f32`].

use crate::device::RocmContext;
use crate::hip::HipBuffer;
use crate::host_stage::RocmArena;

pub use rlx_gpu_host::COLLECTIVE_OPS;

#[allow(clippy::too_many_arguments)]
pub fn run_collective(
    ctx: &RocmContext,
    buffer: &HipBuffer<f32>,
    _arena_size_bytes: usize,
    name: &str,
    in_f32_off: usize,
    in_len: usize,
    out_f32_off: usize,
    out_len: usize,
    attrs: &[u8],
) {
    let mut arena = RocmArena {
        ctx,
        buffer,
        size_bytes: 0,
    };
    rlx_gpu_host::run_collective_f32(
        &mut arena,
        name,
        in_f32_off,
        in_len,
        out_f32_off,
        out_len,
        attrs,
    );
}
