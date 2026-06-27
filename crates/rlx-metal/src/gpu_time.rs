// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! True GPU-busy time from `MTLCommandBuffer` `GPUStartTime`/`GPUEndTime`.
//!
//! The per-thunk profiler (`thunk_profile`) wraps each `encode_commit` in a
//! wall-clock `Instant`, which folds CPU encode + `commit` + `wait` latency
//! (~150–500 µs/commit) into every sample — so with hundreds of thunks the
//! reported time is dominated by sync, not GPU work, and over-attributes to
//! high-count thunks. `GPUStartTime`/`GPUEndTime` are the GPU scheduler's own
//! timestamps for when the command buffer actually ran on the device,
//! independent of how it was submitted. Reading the delta after
//! `wait_until_completed` gives the real GPU-busy time — the number that
//! decides whether a decode step is GPU-compute-bound or CPU-orchestration-
//! bound.
//!
//! Enable via `RLX_METAL_GPU_TIME=1` (per-step whole-buffer busy print) or
//! implicitly whenever `RLX_METAL_THUNK_PROFILE=1` (per-thunk busy column).
//!
//! Limitation: a graph that splits into multiple command buffers mid-run
//! (deferred host ops — `GatedDeltaNet`/`SelectiveScan`/`Sample` sync points)
//! only reports the final segment's busy time. The GGUF-Llama/Orpheus decode
//! graph has no in-graph host ops (sampling is host-side, post-readback), so
//! its single command buffer is captured in full.

use objc::{msg_send, sel, sel_impl};
use std::cell::Cell;
use std::time::Duration;

thread_local! {
    static LAST_BUSY_NS: Cell<u128> = const { Cell::new(0) };
}

/// True GPU-busy seconds for a finished command buffer
/// (`GPUEndTime - GPUStartTime`). Call only after `wait_until_completed`;
/// before completion the timestamps are 0. Clamped to ≥0.
pub fn busy_seconds(cmd_buf: &metal::CommandBufferRef) -> f64 {
    unsafe {
        let start: f64 = msg_send![cmd_buf, GPUStartTime];
        let end: f64 = msg_send![cmd_buf, GPUEndTime];
        (end - start).max(0.0)
    }
}

/// Stash the last command buffer's GPU-busy time so a caller that no longer
/// holds the buffer (the thunk profiler / the normal `encode_and_run` path,
/// which let `encode_commit` consume and drop it) can retrieve it via
/// [`take_last`].
pub fn set_last(cmd_buf: &metal::CommandBufferRef) {
    let ns = (busy_seconds(cmd_buf) * 1e9) as u128;
    LAST_BUSY_NS.with(|c| c.set(ns));
}

/// Take (and clear) the last stashed GPU-busy duration.
pub fn take_last() -> Duration {
    LAST_BUSY_NS.with(|c| Duration::from_nanos(c.replace(0) as u64))
}

/// Whether GPU-busy timestamps should be captured this run.
pub fn enabled() -> bool {
    rlx_ir::env::flag("RLX_METAL_GPU_TIME") || crate::thunk_profile::enabled()
}
