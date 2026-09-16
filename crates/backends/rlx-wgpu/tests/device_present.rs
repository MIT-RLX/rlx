// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **A run of this suite that never touched the GPU must not report `ok`.**
//!
//! Every other test in this crate skips when the device is missing, which is
//! correct — a parity test has nothing to say without hardware. But a skipped
//! test prints `ok`, so the summary line for a suite that executed nothing on
//! the device is indistinguishable from one that passed on it. This repo has
//! watched a card fall off the PCIe bus and take a whole green suite with it.
//!
//! Migrating ~180 individual skips onto `rlx_ir::env::skip_unless_device` would
//! pinpoint *which* test did not run. This gets the outcome that actually
//! matters for a fraction of the churn: under `RLX_REQUIRE_DEVICE=1` — which
//! `rig.sh` sets on every remote runtime — the **binary** fails if the device
//! this crate exists to drive is not there. The rest of the suite can go on
//! skipping quietly; the suite as a whole cannot come back green.
//!
//! Unset (a developer laptop, CI without hardware) this is a no-op.

#[test]
fn the_device_this_crate_targets_is_actually_present() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    eprintln!("wgpu: device present — the suite's skips are genuine skips");
}
