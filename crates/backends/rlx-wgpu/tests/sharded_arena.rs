// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sharded activation arenas (> wgpu max_buffer_size / ~4 GiB).

use rlx_ir::NodeId;
use rlx_opt::memory::{BufferSlot, MemoryPlan};
use rlx_wgpu::buffer::{Arena, SHARD_STAGE_RESERVE};
use std::collections::HashMap;

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
    if !rlx_wgpu::is_available() {
        return;
    }
    let wgpu = rlx_wgpu::device::wgpu_device().expect("wgpu device");
    let device = &wgpu.device;
    let max_buf = device.limits().max_buffer_size as usize;
    let shard_cap = (max_buf / 256) * 256;
    let usable = shard_cap.saturating_sub(SHARD_STAGE_RESERVE).max(256);
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
        "expected sharded arena (logical {} > max_buffer_size {})",
        arena.size,
        max_buf
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
    if !rlx_wgpu::is_available() {
        return;
    }
    let wgpu = rlx_wgpu::device::wgpu_device().expect("wgpu device");
    let device = &wgpu.device;
    let max_buf = device.limits().max_buffer_size as usize;
    let shard_cap = (max_buf / 256) * 256;
    let usable = shard_cap.saturating_sub(SHARD_STAGE_RESERVE).max(256);
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
