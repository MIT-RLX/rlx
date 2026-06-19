// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Runtime tuning for the MLX backend.

use crate::ffi;

/// Default `mlx::compile` multi-output slot limit when env / explicit override unset.
pub const DEFAULT_COMPILE_OUTPUT_CAP: usize = 1024;

/// Primary env var (also read by `rlx-runtime::compile_output_cap`).
pub const COMPILE_OUTPUT_CAP_ENV: &str = "RLX_COMPILE_OUTPUT_CAP";

/// Legacy alias env var.
pub const COMPILE_OUTPUT_CAP_ENV_MLX: &str = "RLX_MLX_COMPILE_OUTPUT_CAP";

/// Current compile output cap (explicit override, else env, else
/// [`DEFAULT_COMPILE_OUTPUT_CAP`]).
pub fn compile_output_cap() -> usize {
    unsafe { ffi::rlx_mlx_compile_output_cap() }
}

/// Override the compile output cap for this process. Pass `0` to clear the override
/// (same as [`reset_compile_output_cap`]).
pub fn set_compile_output_cap(cap: usize) {
    unsafe {
        ffi::rlx_mlx_set_compile_output_cap(cap);
    }
}

/// Clear an explicit cap override; subsequent reads use env / default again.
pub fn reset_compile_output_cap() {
    unsafe {
        ffi::rlx_mlx_reset_compile_output_cap();
    }
}
