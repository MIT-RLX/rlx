// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU host-fallback for Riemannian / SPD-manifold ops on the Metal backend.
//!
//! Evaluation lives in [`rlx_gpu_host`]; this module re-exports the predicate
//! and eval entry points. On Apple Silicon the arena is shared-storage
//! `MTLBuffer`, so callers read/write f32 spans via `Buffer::contents()` and
//! pass them to [`eval`].

pub use rlx_gpu_host::{eval_spd as eval, is_spd_host};
