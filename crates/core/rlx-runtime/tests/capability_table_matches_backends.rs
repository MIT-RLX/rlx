// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **`advisory_capabilities` must agree with what the backends actually do.**
//!
//! There are two answers to "can this device keep a KV cache resident?": the
//! `ExecutableCapabilities` a compiled executable reports, and the table in
//! `device_policy::advisory_capabilities`, which answers from a `Device` enum
//! before anything is compiled. Planning code and `device_report` read the
//! table; runtime fast paths read the executable. Nothing kept them in sync.
//!
//! They had already drifted in both directions:
//!
//! * ROCm implements `register_kv_row_feed` / `feed_kv_row` and the table said
//!   `kv_resident`, but its wrapper's `capabilities()` did not — so a caller
//!   gating on the executable silently took the slow path on a backend that
//!   supported the fast one.
//! * wgpu gained residency and the table still said it had none — the mirror
//!   image, and the one a planner would believe.
//!
//! Neither shows up as a test failure anywhere else: both directions are a
//! missed optimisation, not a wrong number. Hence a structural test.
//!
//! Only devices this build can actually instantiate are checked; the table
//! still covers devices no host has, and those entries stay unverified.
//!
//! That "unverified" was doing too much work: the skips were a bare `continue`
//! on `is_available`, so a rig where CUDA is **compiled in and broken** — the
//! case that most needs reporting — looked identical to a Mac that has no CUDA
//! at all, and CUDA's entry stayed unverified either way. They now go through
//! `skip_unless_available`, so `RLX_REQUIRE_DEVICE=1` (which `rig.sh` sets)
//! turns the first case into a failure while leaving the second a skip.

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::{Device, Session, advisory_capabilities};

mod common;
use common::skip_unless_available;

const DEVICES: &[(&str, Device)] = &[
    ("cpu", Device::Cpu),
    ("metal", Device::Metal),
    ("mlx", Device::Mlx),
    ("wgpu", Device::Gpu),
    ("cuda", Device::Cuda),
    ("rocm", Device::Rocm),
    ("vulkan", Device::Vulkan),
];

/// The smallest graph every backend can compile.
fn trivial() -> Graph {
    let mut g = Graph::new("caps");
    let x = g.input("x", Shape::new(&[4], DType::F32));
    let y = g.relu(x);
    g.set_outputs(vec![y]);
    g
}

#[test]
fn the_advisory_table_reports_what_each_backend_actually_supports() {
    let _gpu = common::serialize_gpu();
    let mut checked = 0usize;
    let mut drift: Vec<String> = Vec::new();

    for (name, dev) in DEVICES {
        if skip_unless_available(*dev, name) {
            continue;
        }
        let actual = Session::new(*dev).compile(trivial()).capabilities();
        let advisory = advisory_capabilities(*dev);
        checked += 1;

        let (a, b) = (actual.enabled_names(), advisory.enabled_names());
        if a != b {
            let missing: Vec<_> = a.iter().filter(|c| !b.contains(c)).collect();
            let extra: Vec<_> = b.iter().filter(|c| !a.contains(c)).collect();
            drift.push(format!(
                "  {name}: table is missing {missing:?} and wrongly claims \
                 {extra:?}\n    executable: {a:?}\n    table:      {b:?}"
            ));
        }
    }

    assert!(
        drift.is_empty(),
        "advisory_capabilities disagrees with the compiled executables:\n{}\n\n\
         Both directions are bugs: a capability the table omits is a fast path \
         planners never take, and one it invents is a fast path callers take \
         and lose.",
        drift.join("\n")
    );
    assert!(checked > 0, "no device was available to check");
    eprintln!("  {checked} device(s) agree with the advisory table");
}

/// **Flags that can be probed must match the probe, in both directions.**
///
/// The test above pins the table to the executable; it cannot catch the case
/// where BOTH are wrong the same way — which is what CPU did, implementing
/// `bind_handle` / `read_handle` that return real data while reporting
/// `persistent_handles: false` on both sides.
///
/// Only the flags with an observable probe are checked. `typed_io`,
/// `active_extent` and `moe` are setters returning `()`, so there is nothing to
/// observe without a graph built to expose each one; `clone` is checked one way
/// only, since `clone_box`'s default is a panic rather than a `false`.
#[test]
fn probeable_flags_agree_with_what_the_methods_return() {
    let _gpu = common::serialize_gpu();
    for (name, dev) in DEVICES {
        if skip_unless_available(*dev, name) {
            continue;
        }
        let mut c = Session::new(*dev).compile(trivial());
        let caps = c.capabilities();

        let binds = c.bind_handle("x", &[0.0f32; 4]);
        assert_eq!(
            binds, caps.persistent_handles,
            "{name}: bind_handle returned {binds} but persistent_handles is {}",
            caps.persistent_handles
        );
        if binds {
            assert!(
                c.read_handle("x").is_some(),
                "{name}: bind_handle succeeded but read_handle returns None — \
                 the handle is write-only, which is not what the flag promises"
            );
        }

        let gpu = c.bind_gpu_handle("x", &[0.0f32; 4]);
        assert_eq!(
            gpu, caps.gpu_handles,
            "{name}: bind_gpu_handle returned {gpu} but gpu_handles is {}",
            caps.gpu_handles
        );

        if caps.clone {
            let _ = c.clone();
        }
        eprintln!("  {name}: handle flags match their methods");
    }
}

/// **A backend that advertises `kv_resident` must have a working row feed.**
///
/// `register_kv_row_feed` returning `true` is the contract the flag stands for;
/// the wrappers return a bare `true` without consulting anything, so the flag
/// and the method can disagree without either looking wrong in isolation.
#[test]
fn kv_resident_implies_a_row_feed_that_registers() {
    let _gpu = common::serialize_gpu();
    for (name, dev) in DEVICES {
        if skip_unless_available(*dev, name) {
            continue;
        }
        let mut c = Session::new(*dev).compile(trivial());
        if !c.capabilities().kv_resident {
            continue;
        }
        // `bind_gpu_handle` is what makes an input resident; residency without
        // it has nothing to feed into.
        assert!(
            c.capabilities().gpu_handles,
            "{name}: claims kv_resident without gpu_handles — there is no \
             resident buffer for a row feed to write into"
        );
        assert!(
            c.bind_gpu_handle("x", &[0.0f32; 4]),
            "{name}: claims gpu_handles but bind_gpu_handle refused"
        );
        assert!(
            c.register_kv_row_feed("x", 0),
            "{name}: claims kv_resident but register_kv_row_feed refused"
        );
        eprintln!("  {name}: kv_resident is backed by a real row feed");
    }
}
