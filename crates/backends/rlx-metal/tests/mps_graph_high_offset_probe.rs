// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Probe: does an MPSGraph bind into a buffer work past 4 GiB? Metal keeps
//! executables off MPSGraph once their weight buffer reaches 4 GiB
//! (`MPS_BIND_CLIFF_BYTES`); this runs `y = 2·x` with `x` and `y` bound at
//! offsets below and above 4 GiB, aligned to a page and not, and prints which
//! give the right answer. Ignored by default: it maps a 4.1 GiB buffer (only
//! the touched pages become resident).
//!
//! `cargo test -p rlx-metal --test mps_graph_high_offset_probe -- --ignored --nocapture`

#![cfg(target_os = "macos")]

use rlx_metal::mps_graph::MpsGraph;

const F32: u32 = 0x10000000 | 32;

#[test]
#[ignore]
fn mpsgraph_binds_past_four_gib() {
    let dev = rlx_metal::device::metal_device().expect("Metal device");
    let n = 4096usize;
    let bytes = n * 4;
    let len = (1usize << 32) + (64 << 20);
    let buf = dev.alloc_shared(len);
    let g = MpsGraph::new();
    let x = g.placeholder(&[n], F32, "x");
    let two = g.constant_scalar(2.0);
    let y = g.mul(&x, &two);
    let page = 16384usize;
    let mut failures = Vec::new();
    for (what, base) in [
        ("1 GiB, page-aligned", 1usize << 30),
        ("1 GiB + 128 B", (1 << 30) + 128),
        ("4 GiB + 1 page", (1 << 32) + page),
        ("4 GiB + 128 B", (1 << 32) + 128),
        ("4 GiB - 8 KiB (straddles)", (1 << 32) - 8192),
    ] {
        let (xo, yo) = (base, base + 4 * bytes);
        let p = buf.contents() as *mut f32;
        unsafe {
            for i in 0..n {
                *p.add(xo / 4 + i) = i as f32;
                *p.add(yo / 4 + i) = -1.0;
            }
        }
        g.run_jit(
            &dev.queue,
            &[&x],
            &[&buf],
            &[xo],
            &[vec![n]],
            &[F32],
            &[&y],
            &[&buf],
            &[yo],
            &[vec![n]],
            &[F32],
        );
        let ok = (0..n).all(|i| unsafe { *p.add(yo / 4 + i) } == 2.0 * i as f32);
        eprintln!("[probe] {what}: {}", if ok { "ok" } else { "WRONG" });
        if !ok {
            failures.push(what);
        }
    }
    assert!(
        failures.is_empty(),
        "wrong past these offsets: {failures:?}"
    );
}
