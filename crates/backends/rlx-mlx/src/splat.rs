// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Gaussian splat forward/backward for MLX lowering (CPU reference on host buffers).

#![cfg(rlx_mlx_host)]

pub use rlx_cpu::splat::{backward_host_slices, render_host_slices};
