// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **`RLX_REQUIRE_DEVICE` must reach every device-gated test. No exceptions.**
//!
//! The flag turns "no GPU" from a silent skip into a failure — but only for
//! tests that route their skip through `skip_unless_available` /
//! `rlx_ir::env::skip_unless_device`. The alternative reports `ok`:
//!
//! ```ignore
//! if !is_available(device) {
//!     eprintln!("skip ... (unavailable)");
//!     return;                            // → the summary line says `ok`
//! }
//! ```
//!
//! This started as a ratchet — a pinned count that could only go down — because
//! 186 files were on the wrong side of it and migrating them in one pass was
//! not safe. They have since all been migrated, so the pin is **0** and this is
//! now an absolute rule rather than a backlog: a raw `if !is_available(..)`
//! skip anywhere in the test tree fails here.
//!
//! `tests/device_present.rs` (and its per-backend twins) remain as a floor at
//! suite granularity — they fail if a compiled-in backend cannot be
//! instantiated at all, whatever the individual tests do. The two are
//! complementary: the floor catches a dead card, this catches the skip that
//! would have hidden it.
//!
//! Left unmeasured, coverage sat at 29-of-110 without anyone noticing.

use std::path::{Path, PathBuf};

/// **Zero. Do not raise this.**
///
/// Raising it means a test was written with a skip that reports `ok` on a rig
/// with no device — the thing this whole mechanism exists to stop. Route the
/// skip through the helper instead.
///
/// The count went 186 → 128 → 45 → 0: ~700 call sites across 155 files in
/// rlx-metal, wgpu, vulkan, mlx, cuda, rocm, coreml, tpu and rlx-runtime's own
/// tests. Every rewrite was compile-checked; metal, wgpu, vulkan, mlx and
/// rlx-runtime were also run in full.
///
/// Two corrections along the way, both found by the gate arguing with itself:
///
/// * it **scanned a hardcoded list of backend directories** and missed seven,
///   including rlx-tpu's three unmigrated files. Fixing that pushed the count
///   47 → 50 and the ratchet refused the raise, so the files got migrated.
/// * it counted things that are **not skips** — `assert!(is_available())`,
///   `assert!(!is_available())` asserting a device is *absent*, and
///   `.find(|d| !is_available(*d))` searching *for* an unavailable one. See
///   [`gates_on_availability`]; a skip is now specifically a negated call in an
///   `if` condition.
///
/// Six hand-rolled copies of the skip decision also turned up — as
/// `device_missing()` in four backend crates and a shadowing
/// `fn skip_unless_available` in `activation_batch_parity.rs` — and **one of
/// them had lost the `RLX_REQUIRE_DEVICE` assert entirely**, so it read like a
/// guarded file while being exempt. All six now call the shared helper.
const PINNED_UNMIGRATED: usize = 0;

/// Every test directory to scan: this crate's, plus `tests/` under every crate
/// in `crates/backends/`.
///
/// **Discovered, not listed.** The first version of this hardcoded eight
/// backend paths and silently missed seven others — rlx-tpu alone had three
/// unmigrated files sitting outside the count. A hand-maintained list of the
/// places to look for drift is itself a thing that drifts, which is the lesson
/// this whole gate exists to encode. A new backend is now covered the day its
/// `tests/` directory appears.
fn test_dirs(root: &Path) -> Vec<PathBuf> {
    let mut dirs = vec![root.join("tests")];
    let backends = root.join("../../backends");
    if let Ok(entries) = std::fs::read_dir(&backends) {
        let mut found: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path().join("tests"))
            .filter(|p| p.is_dir())
            .collect();
        found.sort();
        dirs.extend(found);
    }
    dirs
}

/// Does this file decide to **skip** based on device availability?
///
/// Three things that are *not* skips have to be excluded, or the backlog fills
/// with work that must never be done:
///
/// * `assert!(rlx_oneapi::is_available())` — a positive assertion that the
///   backend is always selectable. The opposite of a skip.
/// * `assert!(!rlx_runtime::is_available(Device::Egpu))` (`egpu_seam.rs`) and
///   `assert!(!rlx_tpu::is_available(), ..)` (`rlx-tpu/basic.rs`) — deliberate
///   NEGATIVE assertions that a device is absent. Turning one into a skip would
///   delete the test.
/// * `.find(|d| !is_available(*d))` (`graph_devices_parity.rs`) — *searches
///   for* an unavailable device so the fallback path runs on every host.
///   Inverting it would invert the test's premise.
///
/// So a skip is: a **negated** availability call **in an `if` condition**.
/// Assertions and closure predicates do not qualify. That is narrow enough to
/// exclude all three above and wide enough to catch every real skip left.
fn gates_on_availability(src: &str) -> bool {
    src.lines().any(|line| {
        let t = line.trim_start();
        if !(t.starts_with("if ") || t.starts_with("} else if ") || t.starts_with("&& ")) {
            return false;
        }
        ["is_available(", "is_supported("].iter().any(|probe| {
            let mut from = 0usize;
            while let Some(hit) = line[from..].find(probe) {
                let at = from + hit;
                let head = line[..at]
                    .trim_end_matches(|c: char| c.is_alphanumeric() || c == '_' || c == ':');
                if head.trim_end().ends_with('!') {
                    return true;
                }
                from = at + probe.len();
            }
            false
        })
    })
}

/// Does it route that decision through something `RLX_REQUIRE_DEVICE` sees?
///
/// Matching the *name* alone is not enough, and this gate was fooled by exactly
/// that: `activation_batch_parity.rs` defined its own local
/// `fn skip_unless_available` — same name, same signature, no
/// `RLX_REQUIRE_DEVICE` assert — so a file exempt from the flag counted as
/// migrated. A local definition now disqualifies the file; only a call into the
/// shared helper counts.
fn uses_the_helper(src: &str) -> bool {
    if src.contains("fn skip_unless_available") || src.contains("fn skip_unless_device") {
        return false;
    }
    src.contains("skip_unless_available") || src.contains("skip_unless_device")
}

fn scan(dir: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        // A backend crate absent from this checkout is not a finding.
        return Vec::new();
    };
    let mut out = Vec::new();
    for e in entries.flatten() {
        let p = e.path();
        if p.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        // This file quotes every pattern it looks for — in the doc comment and
        // in `uses_the_helper`'s own body — so it matches itself. Scanning a
        // scanner is a category error, not a finding.
        if p.file_name().and_then(|s| s.to_str()) == Some("require_device_coverage.rs") {
            continue;
        }
        let Ok(src) = std::fs::read_to_string(&p) else {
            continue;
        };
        if gates_on_availability(&src) && !uses_the_helper(&src) {
            let name = p
                .strip_prefix(dir)
                .unwrap_or(&p)
                .to_string_lossy()
                .into_owned();
            let crate_dir = dir
                .parent()
                .and_then(|d| d.file_name())
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            out.push(format!("{crate_dir}/tests/{name}"));
        }
    }
    out
}

#[test]
fn every_device_gated_test_routes_through_the_helper() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dirs = test_dirs(&root);
    let mut unmigrated: Vec<String> = dirs.iter().flat_map(|d| scan(d)).collect();
    unmigrated.sort();
    eprintln!(
        "scanned {} test directory/ies; {} file(s) gate on device availability \
         without the shared helper",
        dirs.len(),
        unmigrated.len()
    );

    assert_eq!(
        unmigrated.len(),
        PINNED_UNMIGRATED,
        "these test file(s) gate on device availability with a raw \
         `if !is_available(..) {{ return }}`, which reports `ok` on a rig with no \
         device:\n  {}\n\nRoute the skip through `common::skip_unless(dev)` (label \
         derived), `common::skip_unless_available(dev, label)`, or — in a backend \
         crate, which cannot depend on rlx-runtime — \
         `rlx_ir::env::skip_unless_device(label, compiled, available)`.\n\nIf the \
         call is NOT a skip (a negative assertion that a device is absent, or a \
         search FOR an unavailable device), it should not be in an `if` condition \
         reachable by `gates_on_availability` — check that function before \
         changing this number.",
        unmigrated.join("\n  ")
    );
}

/// **A test that drives a GPU device must serialize against the others.**
///
/// The adapters in this tree are not safe to init/teardown from several test
/// threads at once. `common::GpuTestGuard` is the mechanism; the failure when a
/// file skips it is not a crash but *wrong numbers* — `elementwise_backend_parity`
/// returned `worst_rel=1.0` on Vulkan about one run in six, and
/// `vulkan_parity::fma_decomposed` did the same, both clean at
/// `--test-threads=1`. A flake that rare reads as "someone else's problem" for a
/// long time.
///
/// So the same ratchet as [`every_device_gated_test_routes_through_the_helper`]:
/// a file that constructs a `Session` on a non-CPU device and never takes the
/// guard fails here. It is pinned at 0 — the sweep that fixed the two flakes
/// covered every file in this directory.
///
/// CPU-only files are exempt: there is no device to contend for.
#[test]
fn gpu_tests_serialize_device_access() {
    const PINNED_UNGUARDED: usize = 0;

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut unguarded: Vec<String> = Vec::new();

    let Ok(entries) = std::fs::read_dir(root.join("tests")) else {
        panic!("tests/ unreadable");
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
        // This file and the coverage/floor gates construct no GPU session.
        if name == "require_device_coverage.rs" {
            continue;
        }
        let Ok(src) = std::fs::read_to_string(&p) else {
            continue;
        };
        if !src.contains("Session::new") {
            continue;
        }
        // Does it name a GPU device at all?
        let gpu = [
            "Device::Metal",
            "Device::Gpu",
            "Device::Cuda",
            "Device::Rocm",
            "Device::Mlx",
            "Device::Vulkan",
            "Device::Ane",
        ]
        .iter()
        .any(|d| src.contains(d));
        if !gpu {
            continue;
        }
        if src.contains("GpuTestGuard") || src.contains("serialize_gpu") {
            continue;
        }
        unguarded.push(name.to_string());
    }
    unguarded.sort();

    assert_eq!(
        unguarded.len(),
        PINNED_UNGUARDED,
        "these test file(s) build a `Session` on a GPU device without taking \
         `common::GpuTestGuard`, so they race every other GPU test in their \
         binary:\n  {}\n\nAdd `let _gpu = common::serialize_gpu();` as the first \
         statement of each `#[test]`. The guard is re-entrant, so a helper that \
         also acquires costs nothing.",
        unguarded.join("\n  ")
    );
}
