// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! CPU host-fallback for Riemannian / SPD-manifold ops on the Vulkan backend.
//!
//! Evaluation lives in [`rlx_gpu_host`]; this module re-exports the predicate
//! and eval entry points used by compile/runtime.

pub use rlx_gpu_host::{eval_spd as eval, is_spd_host};
