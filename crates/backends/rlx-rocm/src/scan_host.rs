// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host-side `Op::Scan` / HostOp / indexing for ROCm arenas.
//!
//! Thin adapters over [`rlx_gpu_host`] (`run_scan` / `run_host_op` / `run_indexing`).

use crate::device::RocmContext;
use crate::hip::HipBuffer;
use crate::host_stage::RocmArena;
use rlx_cpu::thunk::{HostOpDesc, INDEXING_CONTIGUOUS_SPAN_CAP, IndexingThunk, ScanHostDesc};

pub fn run_scan(
    ctx: &RocmContext,
    buffer: &HipBuffer<f32>,
    arena_size_bytes: usize,
    desc: &ScanHostDesc,
) {
    let mut arena = RocmArena {
        ctx,
        buffer,
        size_bytes: arena_size_bytes,
    };
    rlx_gpu_host::run_scan(&mut arena, desc);
}

pub fn run_host_op(
    ctx: &RocmContext,
    buffer: &HipBuffer<f32>,
    arena_size_bytes: usize,
    desc: &HostOpDesc,
) {
    let mut arena = RocmArena {
        ctx,
        buffer,
        size_bytes: arena_size_bytes,
    };
    rlx_gpu_host::run_host_op(&mut arena, desc);
}

pub fn run_indexing(
    ctx: &RocmContext,
    buffer: &HipBuffer<f32>,
    arena_size_bytes: usize,
    thunk: &IndexingThunk,
) {
    let mut arena = RocmArena {
        ctx,
        buffer,
        size_bytes: arena_size_bytes,
    };
    rlx_gpu_host::run_indexing(
        &mut arena,
        arena_size_bytes,
        thunk,
        INDEXING_CONTIGUOUS_SPAN_CAP,
    );
}
