// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::TopK` on wgpu vs CPU: long rows (a sampler's vocabulary), large k,
//! many short rows (MoE routing, a 2-D dispatch) and repeated values, where
//! ties go to the lower index. The kernel used to run one thread per row and
//! took ~0.5 s for k = 50 over 4096 values.

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
fn wgpu_topk_matches_cpu() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    for (outer, n, k) in [
        (1, 4096, 50),
        (3, 4096, 1),
        (2, 155_776, 8),
        (4, 128, 8),
        (2, 1025, 1025),
        (70_000, 8, 2),
    ] {
        let x: Vec<f32> = (0..outer * n)
            .map(|i| (((i * 2_654_435_761usize) >> 7) % 61) as f32 * 0.5 - 15.0)
            .collect();
        let gpu = Session::new(Device::Gpu)
            .compile(topk(outer, n, k))
            .run(&[("x", &x)])
            .remove(0);
        let cpu = Session::new(Device::Cpu)
            .compile(topk(outer, n, k))
            .run(&[("x", &x)])
            .remove(0);
        assert_eq!(gpu, cpu, "outer {outer}, n {n}, k {k}");
    }
}
