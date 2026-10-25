// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(all(feature = "metal", feature = "cpu"))]

//! A pure view (a `Narrow` at 0) has no arena slot: the memory planner
//! aliases it to its root. Metal's arena gave it no offset either, so every
//! thunk reading it got `usize::MAX` and reading it back panicked ("node not
//! in arena"). A stateful decode step's mask, `iota[0..cap] <= position`,
//! read the iota that way and came out all ones on Metal.
//!
//! Only the weight-split layout (params placed after the activation plan)
//! left a param's view without a slot. Real models reach it with half a
//! gigabyte of weights; `RLX_METAL_WEIGHT_SPLIT_MIN` lets a one-matmul graph.

mod common;

use rlx_ir::op::CmpOp;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

fn mask(cap: usize, rows: usize) -> Graph {
    let f = |d: &[usize]| Shape::new(d, DType::F32);
    let mut g = Graph::new("mask");
    let pos = g.input("position", f(&[1]));
    let iota = g.param("iota", f(&[rows]));
    let head = g.add_node(
        Op::Narrow {
            axis: 0,
            start: 0,
            len: cap,
        },
        vec![iota],
        f(&[cap]),
    );
    let seen = g.add_node(
        Op::Compare(CmpOp::Le),
        vec![head, pos],
        Shape::new(&[cap], DType::Bool),
    );
    let m = g.add_node(Op::Cast { to: DType::F32 }, vec![seen], f(&[cap]));
    // A weight big enough to go to the weights buffer, so the split happens.
    let x = g.input("x", f(&[1, 256]));
    let w = g.param("w", f(&[256, 256]));
    let y = g.matmul(x, w, f(&[1, 256]));
    g.set_outputs(vec![m, y]);
    g
}

#[test]
fn a_view_of_a_param_reads_its_root_on_metal() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless(Device::Metal) {
        return;
    }
    rlx_ir::env::set("RLX_METAL_WEIGHT_SPLIT_MIN", "1");
    let (cap, rows) = (33, 129);
    let iota: Vec<f32> = (0..rows).map(|i| i as f32).collect();
    let mut out = Vec::new();
    for dev in [Device::Cpu, Device::Metal] {
        let mut c = Session::new(dev).compile(mask(cap, rows));
        c.set_param("iota", &iota);
        c.set_param("w", &vec![0.5; 256 * 256]);
        out.push(c.run(&[("position", &[2.0]), ("x", &[1.0; 256])]));
    }
    let want_mask: Vec<f32> = (0..cap).map(|i| if i <= 2 { 1.0 } else { 0.0 }).collect();
    assert_eq!(out[0][0], want_mask, "cpu mask");
    assert_eq!(out[1][0], want_mask, "metal mask");
}
