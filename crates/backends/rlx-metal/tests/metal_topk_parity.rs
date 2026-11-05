// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::TopK` on Metal vs CPU, at row lengths past the old kernel's 1024-entry
//! scratch (it wrote nothing for longer rows, so every index read as 0) and
//! with repeated values, where ties go to the lower index.

#![cfg(target_os = "macos")]

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

fn topk(outer: usize, n: usize, k: usize) -> Graph {
    let mut g = Graph::new("topk");
    let x = g.input("x", Shape::new(&[outer, n], DType::F32));
    let y = g.add_node(Op::TopK { k }, vec![x], Shape::new(&[outer, k], DType::F32));
    g.set_outputs(vec![y]);
    g
}

#[test]
fn metal_topk_matches_cpu_on_long_rows_and_ties() {
    if rlx_ir::env::skip_unless_device(
        "metal",
        cfg!(target_os = "macos"),
        rlx_runtime::is_available(Device::Metal),
    ) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    for (outer, n, k) in [
        (1, 4096, 50),
        (3, 4096, 1),
        (2, 155_776, 8),
        (4, 128, 8),
        (2, 1025, 1025),
    ] {
        // Values on a coarse grid: plenty of exact ties.
        let x: Vec<f32> = (0..outer * n)
            .map(|i| (((i * 2_654_435_761usize) >> 7) % 61) as f32 * 0.5 - 15.0)
            .collect();
        let g = topk(outer, n, k);
        let metal = Session::new(Device::Metal)
            .compile(g.clone())
            .run(&[("x", &x)])
            .remove(0);
        let cpu = Session::new(Device::Cpu)
            .compile(g)
            .run(&[("x", &x)])
            .remove(0);
        assert_eq!(metal, cpu, "outer {outer}, n {n}, k {k}");
    }
}
