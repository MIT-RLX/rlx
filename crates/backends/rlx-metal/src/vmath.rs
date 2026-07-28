// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Vector transcendentals — same API as [`rlx_cpu::vmath`].
//!
//! Host path: Accelerate vForce on Apple, libm elsewhere.
//! Device path: Metal `exp_inplace` / `tanh_inplace` / `rec_inplace`
//! (and `*_h` half variants).

pub use rlx_cpu::vmath::*;

/// Metal compute kernel name for in-place `vvexpf` (f32).
pub const KERNEL_EXP: &str = "exp_inplace";
/// Metal compute kernel name for in-place `vvtanhf` (f32).
pub const KERNEL_TANH: &str = "tanh_inplace";
/// Metal compute kernel name for in-place `vvrecf` (f32).
pub const KERNEL_REC: &str = "rec_inplace";
