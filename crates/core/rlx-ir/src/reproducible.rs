// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Reproducible runs**: one switch that makes every choice the runtime
//! would otherwise take from the host — its device, its core count, its
//! calibrated cost model, its memory budget — a fixed one, so a run's numbers
//! depend only on the graph, the inputs and the build.
//!
//! Off (the default), backends pick for speed: `fastest_device_for` may put a
//! graph on Metal on one machine and the CPU on another, and the CPU backend
//! sends a small GEMM to its NEON kernel or to BLAS by a calibrated estimate,
//! and splits a large one into per-thread BLAS calls by core count — each a
//! different summation order. On (`RLX_REPRODUCIBLE=1` or
//! [`set_reproducible`]), automatic device selection takes the CPU and the
//! CPU backend takes one kernel per op whatever the host.
//!
//! What it does not fix: the arithmetic of the kernel itself. Two hosts whose
//! BLAS differs (Accelerate, OpenBLAS) or whose SIMD differs still round
//! differently; the CPU backend's portable mode removes that. An explicitly
//! requested device is always honored.

use core::sync::atomic::{AtomicU8, Ordering};

/// 0: follow `RLX_REPRODUCIBLE`; 1: off; 2: on.
static OVERRIDE: AtomicU8 = AtomicU8::new(0);

/// Whether reproducible runs are on: [`set_reproducible`], else
/// `RLX_REPRODUCIBLE`.
#[inline]
pub fn reproducible() -> bool {
    match OVERRIDE.load(Ordering::Relaxed) {
        0 => crate::env::flag("RLX_REPRODUCIBLE"),
        1 => false,
        _ => true,
    }
}

/// Turn reproducible runs on or off for the process, over the environment.
/// Set it before compiling: a compiled graph keeps the choices it was
/// compiled with.
pub fn set_reproducible(on: bool) {
    OVERRIDE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
}
