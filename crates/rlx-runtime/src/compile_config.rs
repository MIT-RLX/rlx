// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Compile-time limits shared across backends (training backward output slots, …).

use std::sync::atomic::{AtomicUsize, Ordering};

/// Primary env var: max graph outputs for compiled/traced backends.
pub const COMPILE_OUTPUT_CAP_ENV: &str = "RLX_COMPILE_OUTPUT_CAP";

/// Legacy alias kept for MLX-only scripts (`RLX_MLX_COMPILE_OUTPUT_CAP`).
pub const COMPILE_OUTPUT_CAP_ENV_MLX: &str = "RLX_MLX_COMPILE_OUTPUT_CAP";

/// Default when env / explicit override unset.
pub const DEFAULT_COMPILE_OUTPUT_CAP: usize = 1024;

static EXPLICIT_COMPILE_OUTPUT_CAP: AtomicUsize = AtomicUsize::new(0);

fn cap_from_env() -> usize {
    for key in [COMPILE_OUTPUT_CAP_ENV, COMPILE_OUTPUT_CAP_ENV_MLX] {
        if let Ok(v) = std::env::var(key) {
            if let Ok(n) = v.parse::<usize>() {
                if n > 0 {
                    return n;
                }
            }
        }
    }
    DEFAULT_COMPILE_OUTPUT_CAP
}

/// Current compile output cap (explicit override, else env, else [`DEFAULT_COMPILE_OUTPUT_CAP`]).
pub fn compile_output_cap() -> usize {
    let explicit = EXPLICIT_COMPILE_OUTPUT_CAP.load(Ordering::Relaxed);
    if explicit != 0 {
        return explicit;
    }
    cap_from_env()
}

/// Override compile output cap for this process. Pass `0` to clear (same as [`reset_compile_output_cap`]).
pub fn set_compile_output_cap(cap: usize) {
    EXPLICIT_COMPILE_OUTPUT_CAP.store(cap, Ordering::Relaxed);
    sync_mlx_compile_output_cap(cap);
}

/// Clear explicit override; subsequent reads use env / default again.
pub fn reset_compile_output_cap() {
    EXPLICIT_COMPILE_OUTPUT_CAP.store(0, Ordering::Relaxed);
    sync_mlx_compile_output_cap(0);
}

/// True when this device enforces [`compile_output_cap`] at compile/run time today.
pub fn device_has_compile_output_cap(device: rlx_driver::Device) -> bool {
    use rlx_driver::Device;
    matches!(device, Device::Mlx)
}

#[cfg(all(feature = "mlx", rlx_mlx_host))]
fn sync_mlx_compile_output_cap(cap: usize) {
    if cap == 0 {
        rlx_mlx::reset_compile_output_cap();
    } else {
        rlx_mlx::set_compile_output_cap(cap);
    }
}

#[cfg(not(all(feature = "mlx", rlx_mlx_host)))]
fn sync_mlx_compile_output_cap(_cap: usize) {}
