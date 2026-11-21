// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Broadcast binaries on ROCm against the CPU. An `Expand` read only by
//! `Binary`s is not materialised: the Binary reads the Expand's input
//! through strides. Here the Expands come long before their Binaries, with
//! other work between, so a planner that let the input die at the Expand
//! would hand its slot to that work and the Binary would read garbage.

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape};
use rlx_runtime::{Device, Session};

fn data(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 9) as f32 / 4_194_304.0 - 1.0
        })
        .collect()
}

fn build() -> (Graph, Vec<(&'static str, Vec<f32>)>) {
    let mut g = Graph::new("bcast");
    let full = [2usize, 8, 6, 5];
    let shape = |d: &[usize]| Shape::new(d, DType::F32);
    let x = g.input("x", shape(&full));
    let bias = g.input("bias", shape(&[1, 8, 1, 1]));
    let row = g.input("row", shape(&[5]));
    let scale = g.input("scale", shape(&[2, 1, 6, 1]));
    // Small operands computed then expanded early.
    let bias2 = g.mul(bias, bias);
    let eb = g.add_node(
        Op::Expand {
            target_shape: full.iter().map(|&d| d as i64).collect(),
        },
        vec![bias2],
        shape(&full),
    );
    let er = g.add_node(
        Op::Expand {
            target_shape: full.iter().map(|&d| d as i64).collect(),
        },
        vec![row],
        shape(&full),
    );
    let es = g.add_node(
        Op::Expand {
            target_shape: full.iter().map(|&d| d as i64).collect(),
        },
        vec![scale],
        shape(&full),
    );
    // Unrelated full-size work in between, to take any freed slot.
    let mut y: NodeId = x;
    for _ in 0..6 {
        let t = g.mul(y, y);
        y = g.add(t, x);
        y = g.relu(y);
    }
    let a = g.binary(BinaryOp::Add, y, eb, shape(&full));
    let b = g.binary(BinaryOp::Mul, er, a, shape(&full));
    let c = g.binary(BinaryOp::Sub, b, es, shape(&full));
    let d = g.binary(BinaryOp::Max, c, er, shape(&full));
    g.set_outputs(vec![d]);
    let feeds = vec![
        ("x", data(full.iter().product(), 3)),
        ("bias", data(8, 5)),
        ("row", data(5, 7)),
        ("scale", data(12, 11)),
    ];
    (g, feeds)
}

fn run(device: Device) -> Vec<f32> {
    let (g, feeds) = build();
    let feeds: Vec<(&str, &[f32])> = feeds.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    Session::new(device).compile(g).run(&feeds).remove(0)
}

#[test]
fn broadcast_binaries_match_cpu() {
    let want = run(Device::Cpu);
    let got = run(Device::Rocm);
    let err = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("max |diff| {err:.2e}");
    assert!(err <= 1e-5, "max |diff| {err}");
}
