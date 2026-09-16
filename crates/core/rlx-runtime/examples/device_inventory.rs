// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What this host can actually dispatch to, and what it merely has.
//!
//! The distinction matters: an AMD XDNA NPU with no userspace runtime is
//! present in `/sys/class/accel` and completely undriveable. `is_available`
//! is honest-gated so selection never picks it; `detected_unavailable_devices`
//! is where it shows up instead, with the reason.
fn main() {
    println!("=== dispatchable ===");
    for d in rlx_runtime::available_devices() {
        println!("  {d:?}");
    }
    println!("=== present but NOT dispatchable ===");
    let unavailable = rlx_runtime::detected_unavailable_devices();
    if unavailable.is_empty() {
        println!("  (none)");
    }
    for (d, why) in unavailable {
        println!("  {d:?}: {why}");
    }
}
