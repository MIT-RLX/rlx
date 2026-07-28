// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host-side RNG fill for ROCm arenas (fill on host → H2D).
//!
//! Thin adapter over [`rlx_gpu_host::run_rng_normal`] /
//! [`rlx_gpu_host::run_rng_uniform`].

use crate::device::RocmContext;
use crate::hip::HipBuffer;
use crate::host_stage::RocmArena;

pub fn run_rng_normal(
    ctx: &RocmContext,
    buffer: &HipBuffer<f32>,
    dst_byte_off: usize,
    len: usize,
    mean: f32,
    scale: f32,
    key: u64,
    op_seed: Option<f32>,
    opts: rlx_ir::RngOptions,
) {
    let mut arena = RocmArena {
        ctx,
        buffer,
        size_bytes: 0,
    };
    rlx_gpu_host::run_rng_normal(
        &mut arena,
        dst_byte_off,
        len,
        mean,
        scale,
        key,
        op_seed,
        opts,
    );
}

pub fn run_rng_uniform(
    ctx: &RocmContext,
    buffer: &HipBuffer<f32>,
    dst_byte_off: usize,
    len: usize,
    low: f32,
    high: f32,
    key: u64,
    op_seed: Option<f32>,
    opts: rlx_ir::RngOptions,
) {
    let mut arena = RocmArena {
        ctx,
        buffer,
        size_bytes: 0,
    };
    rlx_gpu_host::run_rng_uniform(&mut arena, dst_byte_off, len, low, high, key, op_seed, opts);
}
