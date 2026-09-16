// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A queried target must behave differently from a static one — specifically,
//! it must be able to say "I don't know" where a static table can say "no".

use rlx_ir::kernel_schedule::{Feature, Target};
use rlx_wgpu::kernel_schedule_port::{COOP_TILE, target_for_device, target_without_coop_matrix};

#[test]
fn a_queried_target_reports_that_its_list_may_be_partial() {
    // The load-bearing distinction. wgpu exposes a boolean where Vulkan has a
    // property table, so a queried set is never known to be complete — and a
    // `false` from `has()` must therefore mean "not known to be supported",
    // not "known to be unsupported".
    let t = target_without_coop_matrix(32 * 1024);
    assert!(
        t.capabilities_may_be_incomplete(),
        "a wgpu-derived target cannot claim a complete capability list"
    );
    assert!(!t.has(Feature::CoopMatrix {
        m: COOP_TILE,
        n: COOP_TILE,
        k: COOP_TILE
    }));
}

#[test]
fn a_static_target_does_not_claim_incompleteness() {
    // The contrast: an ISA table IS complete for the shapes it lists, so a
    // `false` there is a real negative.
    for t in [Target::CUDA_SM86, Target::METAL_APPLE, Target::PORTABLE] {
        assert!(
            !t.capabilities_may_be_incomplete(),
            "{} is a compile-time table and must not be marked partial",
            t.name
        );
    }
}

#[test]
fn a_live_device_target_uses_the_adapters_own_storage_limit() {
    // Skips rather than fails without an adapter: this asserts a property of
    // the query, and no device means nothing was queried.
    let Some(d) = rlx_wgpu::device::wgpu_device() else {
        eprintln!("no wgpu adapter — skipped");
        return;
    };
    let t = target_for_device(&d.device);
    assert_eq!(
        t.max_shared_bytes,
        d.device.limits().max_compute_workgroup_storage_size as usize,
        "the budget must come from the adapter, not a constant"
    );
    assert!(t.max_warps > 0);
    assert!(t.capabilities_may_be_incomplete());
    eprintln!(
        "wgpu target: {} B workgroup storage, coop8x8={}, list partial={}",
        t.max_shared_bytes,
        t.has(Feature::CoopMatrix {
            m: COOP_TILE,
            n: COOP_TILE,
            k: COOP_TILE
        }),
        t.capabilities_may_be_incomplete()
    );
}
