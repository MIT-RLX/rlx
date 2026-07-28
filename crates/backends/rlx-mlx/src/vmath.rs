// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Vector transcendentals — same API as [`rlx_cpu::vmath`].
//!
//! Host path for staging. Device path: MLX `Exp` / `Tanh` unaries; reciprocal
//! is `1/x` via reciprocal or divide.

pub use rlx_cpu::vmath::*;
