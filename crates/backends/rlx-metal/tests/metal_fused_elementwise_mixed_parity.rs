// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `RLX_METAL_FUSED_ELEMENTWISE_MIXED=1`: the elementwise regions of a graph
//! that also has a matmul become generated kernels. The DAG below has the
//! shape of an optimizer update — a value read twice, a scalar step size
//! computed from an input (joins the region, broadcast), and an intermediate
//! that is also a graph output (ends its own region). Metal must match CPU.

#![cfg(target_os = "macos")]

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::{Device, Session};

fn build(m: usize, k: usize, n: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("fused_ew_mixed");
    let x = g.input("x", Shape::new(&[m, k], f));
    let w = g.input("w", Shape::new(&[k, n], f));
    let v = g.input("v", Shape::new(&[m, n], f));
    let lr = g.input("lr", Shape::new(&[1], f));
    let full = Shape::new(&[m, n], f);
    let one = Shape::new(&[1], f);
    let y = g.add_node(Op::MatMul, vec![x, w], full.clone());
    let bin = |g: &mut Graph, op, a: NodeId, b: NodeId, s: &Shape| {
        g.add_node(Op::Binary(op), vec![a, b], s.clone())
    };
    // v' = lr²·v + y·y (an output, and read again below)
    let lr2 = g.add_node(Op::Binary(BinaryOp::Mul), vec![lr, lr], one.clone());
    let vh = bin(&mut g, BinaryOp::Mul, v, lr2, &full);
    let yy = bin(&mut g, BinaryOp::Mul, y, y, &full);
    let v2 = bin(&mut g, BinaryOp::Add, vh, yy, &full);
    // p' = y − lr²·y / (sqrt(v') + y·y)
    let sq = g.add_node(Op::Activation(Activation::Sqrt), vec![v2], full.clone());
    let den = bin(&mut g, BinaryOp::Add, sq, yy, &full);
    let step = bin(&mut g, BinaryOp::Mul, y, lr2, &full);
    let q = bin(&mut g, BinaryOp::Div, step, den, &full);
    let p2 = bin(&mut g, BinaryOp::Sub, y, q, &full);
    g.set_outputs(vec![p2, v2]);
    g
}

#[test]
fn mixed_graph_elementwise_regions_match_cpu() {
    if rlx_ir::env::skip_unless_device(
        "metal",
        cfg!(target_os = "macos"),
        rlx_runtime::is_available(Device::Metal),
    ) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    rlx_ir::env::set("RLX_METAL_FUSED_ELEMENTWISE_MIXED", "1");
    let (m, k, n) = (48usize, 32usize, 40usize);
    let val = |len: usize, a: usize| -> Vec<f32> {
        (0..len)
            .map(|i| (((i * a + 7) % 23) as f32 - 11.0) * 0.05)
            .collect()
    };
    let (x, w) = (val(m * k, 31), val(k * n, 17));
    let v: Vec<f32> = val(m * n, 13).iter().map(|t| t.abs()).collect();
    let lr = [0.7f32];
    let feed: [(&str, &[f32]); 4] = [("x", &x), ("w", &w), ("v", &v), ("lr", &lr)];

    let g = build(m, k, n);
    let metal = Session::new(Device::Metal).compile(g.clone()).run(&feed);
    let cpu = Session::new(Device::Cpu).compile(g).run(&feed);
    for (o, (a, b)) in metal.iter().zip(&cpu).enumerate() {
        assert_eq!(a.len(), m * n);
        for (j, (p, q)) in a.iter().zip(b).enumerate() {
            assert!(
                (p - q).abs() <= 1e-4 * (1.0 + q.abs()),
                "output {o}[{j}]: metal {p} vs cpu {q}"
            );
        }
    }
}
