// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A dropped Metal executable gives its memory back. Compiling to MPSGraph
//! made autoreleased objects that retain the arena; on a thread with no
//! autorelease pool (any test or worker thread) they lived until the thread
//! exited, so every compile-and-drop kept its whole arena.

#![cfg(all(feature = "metal", target_vendor = "apple"))]

mod common;

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::{Device, Session};

const K: usize = 4096;
const N: usize = 8192; // 128 MiB of f32 weight

unsafe extern "C" {
    fn mach_task_self() -> u32;
    fn task_info(task: u32, flavor: i32, info: *mut i32, count: *mut u32) -> i32;
}

/// This task's `phys_footprint` (`TASK_VM_INFO`, the 19th u64).
fn footprint() -> u64 {
    let mut info = [0i32; 128];
    let mut count = info.len() as u32;
    // SAFETY: a buffer of `count` naturals for TASK_VM_INFO (22).
    unsafe { task_info(mach_task_self(), 22, info.as_mut_ptr(), &mut count) };
    // The buffer is `i32`s, so the u64 is only 4-byte aligned.
    unsafe { (info.as_ptr() as *const u64).add(18).read_unaligned() }
}

#[test]
fn compiling_and_dropping_does_not_grow_the_footprint() {
    if common::skip_unless(Device::Metal) {
        return;
    }
    let _gpu = common::GpuTestGuard::acquire(Device::Metal);
    let w: Vec<f32> = (0..K * N).map(|i| (i % 7) as f32 * 0.01).collect();
    let x = vec![0.5f32; 4 * K];
    let cycle = || {
        let mut g = Graph::new("drop");
        let xi = g.input("x", Shape::new(&[4, K], DType::F32));
        let wi = g.param("w", Shape::new(&[K, N], DType::F32));
        let y = g.mm(xi, wi);
        let y = g.relu(y);
        g.set_outputs(vec![y]);
        let mut c = Session::new(Device::Metal).compile(g);
        c.set_param("w", &w);
        let y = c.run(&[("x", &x)]).remove(0);
        assert!(y.iter().all(|v| v.is_finite()));
    };
    cycle();
    let settled = footprint();
    for _ in 0..6 {
        cycle();
    }
    let grown = footprint().saturating_sub(settled);
    let weight = (K * N * 4) as u64;
    assert!(
        grown < 2 * weight,
        "6 more compile/drop cycles grew the footprint by {} MiB (weight {} MiB)",
        grown >> 20,
        weight >> 20
    );
}
