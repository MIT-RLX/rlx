// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! End-to-end check that the `RLX_DEBUG_NANS` output-boundary scan runs on a
//! real Metal execution. `sqrt` of a negative input produces a NaN on-device;
//! we confirm it reaches the host output (so the scanner has something to
//! report). Run with `RLX_DEBUG_NANS=1 … -- --nocapture` to see the
//! `rlx nan-check [metal]` diagnostic printed by the backend.

#![cfg(target_os = "macos")]

use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

#[test]
fn metal_produces_and_reaches_nan_output() {
    let mut g = Graph::new("nan_metal");
    let x = g.input("x", Shape::new(&[4], DType::F32));
    let y = g.add_node(
        Op::Activation(Activation::Sqrt),
        vec![x],
        Shape::new(&[4], DType::F32),
    );
    g.set_outputs(vec![y]);

    let mut m = Session::new(Device::Metal).compile(g);
    // sqrt(-1) and sqrt(-9) are NaN; sqrt(4)=2, sqrt(16)=4.
    let out = m
        .run(&[("x", [-1.0f32, 4.0, -9.0, 16.0].as_slice())])
        .remove(0);

    assert!(
        out.iter().any(|v| v.is_nan()),
        "Metal sqrt of a negative should yield NaN at the output; got {out:?}"
    );
}
