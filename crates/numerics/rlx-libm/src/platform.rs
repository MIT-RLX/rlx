// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Which platform's reference the ports reproduce where the two disagree.
//!
//! SciPy, NumPy and torch round differently on macOS and on Linux x86-64: the
//! Apple builds contract `a*b + c` into FMAs, merge a `sin` and a `cos` into
//! `__sincos_stret`, call Apple's libm and have no 80-bit `long double`. The
//! ports here (`rlx-pocketfft`, `rlx-signal`, `rlx-special`, `rlx-rng`, …)
//! reproduce the reference of the platform they run on. A run that must round
//! the same everywhere (a browser run compared with a native one) calls
//! [`use_linux_reference`] first: every port then takes the Linux x86-64 arm,
//! the one wasm always takes.

use core::sync::atomic::{AtomicBool, Ordering};

static LINUX: AtomicBool = AtomicBool::new(false);

/// Whether the ports reproduce the Apple builds of their references: on
/// Apple targets, unless [`use_linux_reference`] was called.
#[inline]
pub fn apple_reference() -> bool {
    cfg!(target_vendor = "apple") && !LINUX.load(Ordering::Relaxed)
}

/// [`apple_reference`] on macOS (the arms keyed on `target_os = "macos"`).
#[inline]
pub fn macos_reference() -> bool {
    cfg!(target_os = "macos") && !LINUX.load(Ordering::Relaxed)
}

/// From now on, in this process, every port reproduces the Linux x86-64
/// reference. Call it before computing anything that should match.
pub fn use_linux_reference() {
    LINUX.store(true, Ordering::Relaxed);
}
