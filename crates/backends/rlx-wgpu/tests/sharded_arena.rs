// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sharded activation arenas (> wgpu max_buffer_size / ~4 GiB).

use rlx_ir::NodeId;
use rlx_opt::memory::{BufferSlot, MemoryPlan};
use rlx_wgpu::buffer::{Arena, effective_shard_cap};
use std::collections::HashMap;

/// Shard cap and stage reserve forced small, so striping is exercised at
/// kilobyte scale.
///
/// These tests used to size themselves off the adapter's real
/// `max_buffer_size`. On a discrete Vulkan adapter that is 4 GiB, so a
/// `shard_cap * 2` plan snapped to three 4 GiB buffers and died with
/// `wgpu error: Out of Memory`; they passed on Apple only because its smaller
/// limit happened to fit in unified memory. The striping LOGIC is what is under
/// test, and it does not care how big a shard is — so pin a small one and the
/// same assertions run everywhere.
const TEST_SHARD_MIB: usize = 4;
const TEST_STAGE_MIB: usize = 1;

/// Install the small-shard environment. Both knobs clamp toward smaller, so this
/// can never ask an adapter for more than it supports.
fn force_small_shards() {
    // SAFETY: tests in this binary run single-threaded w.r.t. this setup, and
    // both vars are read on each `from_plan` call rather than cached.
    unsafe {
        std::env::set_var("RLX_WGPU_SHARD_CAP_MIB", TEST_SHARD_MIB.to_string());
        std::env::set_var("RLX_WGPU_SHARD_STAGE_MIB", TEST_STAGE_MIB.to_string());
        // The striping warning is expected here; keep the output readable.
        std::env::set_var("RLX_WGPU_QUIET_SHARD", "1");
        // Striping is refused by default now — it produces wrong results
        // silently. These tests are about the striping machinery itself, so
        // they opt back in.
        std::env::set_var("RLX_WGPU_ALLOW_SHARD", "1");
    }
}

fn fake_plan(arena_size: usize, slots: &[(u32, usize, usize)]) -> MemoryPlan {
    let mut assignments = HashMap::new();
    for &(id, offset, size) in slots {
        assignments.insert(NodeId(id), BufferSlot { offset, size });
    }
    MemoryPlan {
        arena_size,
        assignments,
        schedule: Vec::new(),
    }
}

#[test]
fn sharded_from_plan_stripes_and_reserves_stage() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    force_small_shards();
    let wgpu = rlx_wgpu::device::wgpu_device().expect("wgpu device");
    let device = &wgpu.device;
    let shard_cap = effective_shard_cap(device);
    let usable = shard_cap
        .saturating_sub(rlx_wgpu::buffer::shard_stage_reserve())
        .max(256);
    // Two large slots that cannot share one stripe's usable region.
    let slot = usable - 4096;
    let plan = fake_plan(
        shard_cap * 2,
        &[
            (0, 0, slot),
            (1, shard_cap, slot),
            (2, shard_cap + slot, 4096),
        ],
    );
    let arena = Arena::from_plan(device, &plan);
    assert!(
        arena.is_sharded(),
        "expected sharded arena (logical {} > shard cap {})",
        arena.size,
        shard_cap
    );
    assert!(!arena.extra_shards.is_empty());
    assert_eq!(arena.shard_size, shard_cap);

    for (&id, &off) in &arena.offsets {
        let len = arena.len_of(id);
        let local = off % shard_cap;
        assert!(
            local + len <= usable,
            "node {id:?} @ {off}+{len} invades stage reserve (usable={usable})"
        );
    }

    let id = NodeId(2);
    let payload = vec![1.0f32, 2.0, 3.0, 4.0];
    arena.write_f32(&wgpu.queue, id, &payload);
    let got = arena.read_f32(device, &wgpu.queue, id);
    assert_eq!(&got[..4], &payload[..]);
}

#[test]
fn bind_spec_stays_inside_one_shard() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    force_small_shards();
    let wgpu = rlx_wgpu::device::wgpu_device().expect("wgpu device");
    let device = &wgpu.device;
    let shard_cap = effective_shard_cap(device);
    let usable = shard_cap
        .saturating_sub(rlx_wgpu::buffer::shard_stage_reserve())
        .max(256);
    let slot = usable - 4096;
    let plan = fake_plan(shard_cap * 2, &[(0, 0, slot), (1, shard_cap, 4096)]);
    let arena = Arena::from_plan(device, &plan);
    assert!(arena.is_sharded());

    let spec = arena.bind_spec_for_nodes(device, &[NodeId(0)]);
    assert_eq!(spec.local_base, 0);
    assert_eq!(spec.rebase, 0);

    let off1 = arena.offset(NodeId(1));
    let spec2 = arena.bind_spec_for_nodes(device, &[NodeId(1)]);
    assert_eq!(spec2.rebase, (off1 / shard_cap * shard_cap) as u64);
    assert_eq!(spec2.local_base, 0);
}
