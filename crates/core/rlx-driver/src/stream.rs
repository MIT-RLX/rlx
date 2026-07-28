// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Command-stream abstraction.
//!
//! Every GPU-shaped backend has the same shape: enqueue work, submit, wait.
//!   - Metal: `MTLCommandBuffer` + `commit` + `waitUntilCompleted`
//!   - CUDA: `cudaStream_t` + `cudaStreamSynchronize`
//!   - ROCm: `hipStream_t` + `hipStreamSynchronize`
//!   - wgpu: `CommandEncoder.finish()` → `Queue.submit()` → `Device.poll(Wait)`
//!   - WASM (single-threaded): no-op (work runs synchronously)
//!
//! Hoisting this into one trait means:
//!   - the runtime can drive *any* backend via the same submit-and-wait API
//!   - new backends only need a thin command-stream impl
//!   - test infrastructure works against the trait, not per-backend types

/// Per-backend command stream.
///
/// Implementations are free to be no-ops on synchronous backends (host CPU,
/// WASM): `submit` runs work eagerly, `wait` returns immediately.
pub trait CommandStream {
    /// Submit any pending work to the device (non-blocking).
    fn submit(&mut self);

    /// Block until all submitted work has completed.
    fn wait(&mut self);

    /// Convenience: `submit` followed by `wait`. Backends may override
    /// for a fused fast path (e.g. Metal's `commit` + `waitUntilCompleted`).
    fn submit_and_wait(&mut self) {
        self.submit();
        self.wait();
    }
}

/// Default implementation for synchronous backends — work has already
/// happened by the time `submit` is called.
pub struct SyncStream;

impl CommandStream for SyncStream {
    fn submit(&mut self) {}
    fn wait(&mut self) {}
}
