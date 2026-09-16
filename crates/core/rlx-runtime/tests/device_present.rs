// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Every backend this build contains must be instantiable, or say why not.**
//!
//! `tests/common/mod.rs::skip_unless_available` gives an individual test the
//! `RLX_REQUIRE_DEVICE` discipline — but only 29 of the ~110 device-gated files
//! in this directory call it. The other ~81 gate on `is_available` directly and
//! `return`, which prints `ok`. Migrating them all would pinpoint *which* test
//! did not run; this gets the outcome that matters — a suite that never touched
//! a compiled-in backend cannot come back green — without touching 81 files.
//!
//! The two are complementary, not redundant. Keep migrating; this is the floor.
//!
//! Unset (a laptop, CI without hardware) it is a no-op. A backend the build
//! does not contain is absent, not broken, and is not asserted on.

use rlx_runtime::Device;

mod common;

const BACKENDS: &[(&str, Device)] = &[
    ("cpu", Device::Cpu),
    ("metal", Device::Metal),
    ("mlx", Device::Mlx),
    ("wgpu", Device::Gpu),
    ("vulkan", Device::Vulkan),
    ("cuda", Device::Cuda),
    ("rocm", Device::Rocm),
];

#[test]
fn every_compiled_in_backend_can_be_instantiated() {
    let mut present: Vec<&str> = Vec::new();
    let mut absent: Vec<&str> = Vec::new();

    for (name, dev) in BACKENDS {
        if !rlx_runtime::feature_compiled(*dev) {
            absent.push(name);
            continue;
        }
        // Asserts under RLX_REQUIRE_DEVICE=1; a loud skip otherwise.
        if common::skip_unless_available(*dev, name) {
            continue;
        }
        present.push(name);
    }

    eprintln!(
        "compiled in and usable: {}\nnot built into this binary: {}",
        if present.is_empty() {
            "(none)".to_string()
        } else {
            present.join(", ")
        },
        if absent.is_empty() {
            "(none)".to_string()
        } else {
            absent.join(", ")
        }
    );

    // CPU is unconditional — if even that is missing, the build is broken in a
    // way no parity test would explain clearly.
    assert!(
        present.contains(&"cpu") || !rlx_runtime::feature_compiled(Device::Cpu),
        "the cpu backend is compiled in but could not be instantiated"
    );
}
