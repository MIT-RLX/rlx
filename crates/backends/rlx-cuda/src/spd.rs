// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU host-fallback for the core Riemannian / SPD-manifold ops on CUDA.
//!
//! Evaluation lives in [`rlx_gpu_host`]; this module re-exports the predicate
//! and eval entry points used by compile/runtime.

pub use rlx_gpu_host::{eval_spd as eval, is_spd_host};
