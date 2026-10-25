// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **`Op::DynamicUpdateSlice` means the same thing on every backend.**
//!
//! It writes `update` into `operand` at a `start` read when the graph runs
//! (a decode step's cache row at its `position`). CPU and Metal write in
//! place; the rest run `rlx_fusion::lower_dynamic_update_slice`, a
//! `ScatterNd`. Each is checked against the tensor computed here, with a
//! start inside, at the end, and past either end (clamped, as JAX clamps).

use rlx_ir::op::Op;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

mod common;

const F: DType = DType::F32;

/// `operand[batch, cap, width]` with `rows` rows written at `start`.
fn dus_graph(batch: usize, cap: usize, rows: usize, width: usize, start: DType) -> Graph {
    let mut g = Graph::new("dus");
    let operand = g.input("operand", Shape::new(&[batch, cap, width], F));
    let update = g.input("update", Shape::new(&[batch, rows, width], F));
    let at = g.input("start", Shape::new(&[1], start));
    let out = g.add_node(
        Op::DynamicUpdateSlice { axis: 1 },
        vec![operand, update, at],
        Shape::new(&[batch, cap, width], F),
    );
    g.set_outputs(vec![out]);
    g
}

/// What the op must return: `start` clamped to `[0, cap - rows]`.
fn reference(
    operand: &[f32],
    update: &[f32],
    batch: usize,
    cap: usize,
    rows: usize,
    width: usize,
    start: i64,
) -> Vec<f32> {
    let at = start.clamp(0, (cap - rows) as i64) as usize;
    let mut out = operand.to_vec();
    for b in 0..batch {
        let d = (b * cap + at) * width;
        let s = b * rows * width;
        out[d..d + rows * width].copy_from_slice(&update[s..s + rows * width]);
    }
    out
}

fn devices() -> Vec<(&'static str, Device)> {
    let mut v = vec![("cpu", Device::Cpu)];
    for (name, d) in [
        ("metal", Device::Metal),
        ("mlx", Device::Mlx),
        ("wgpu", Device::Gpu),
        ("cuda", Device::Cuda),
        ("rocm", Device::Rocm),
        ("vulkan", Device::Vulkan),
    ] {
        if common::skip_unless(d) {
            continue;
        }
        v.push((name, d));
    }
    v
}

#[test]
fn rows_land_at_the_runtime_start_on_every_backend() {
    let _gpu = common::serialize_gpu();
    const CAP: usize = 8;
    const ROWS: usize = 2;
    const W: usize = 4;
    let operand: Vec<f32> = (0..CAP * W).map(|i| i as f32).collect();
    let update: Vec<f32> = (0..ROWS * W).map(|i| 1000.0 + i as f32).collect();
    for (name, dev) in devices() {
        // One compile, every start: the start is data, not an attribute.
        let mut c = Session::new(dev).compile(dus_graph(1, CAP, ROWS, W, F));
        for start in [0i64, 3, 6, 7, -2] {
            let got = c
                .run(&[
                    ("operand", &operand),
                    ("update", &update),
                    ("start", &[start as f32]),
                ])
                .remove(0);
            let want = reference(&operand, &update, 1, CAP, ROWS, W, start);
            assert_eq!(got, want, "{name} start={start}");
        }
    }
}

/// An integer start, and more than one slice before the axis (a batched
/// cache: the native kernels' stride, and the `ScatterNd` lowering's index
/// per sequence).
#[test]
fn an_integer_start_and_a_batch_of_slices_on_every_backend() {
    let _gpu = common::serialize_gpu();
    const B: usize = 2;
    const CAP: usize = 8;
    const ROWS: usize = 3;
    const W: usize = 4;
    let operand: Vec<f32> = (0..B * CAP * W).map(|i| i as f32).collect();
    let update: Vec<f32> = (0..B * ROWS * W).map(|i| -1.0 - i as f32).collect();
    let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    for (name, dev) in devices() {
        let mut c = Session::new(dev).compile(dus_graph(B, CAP, ROWS, W, DType::I32));
        for start in [0i32, 4, 9] {
            let outs = c.run_typed(&[
                ("operand", &bytes(&operand), F),
                ("update", &bytes(&update), F),
                ("start", &start.to_le_bytes(), DType::I32),
            ]);
            let got: Vec<f32> = outs[0]
                .0
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let want = reference(&operand, &update, B, CAP, ROWS, W, i64::from(start));
            assert_eq!(got, want, "{name} start={start}");
        }
    }
}
