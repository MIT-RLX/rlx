// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `StopGradient` is a forward identity, but the memory planner does not alias
//! it to its parent the way it aliases `Reshape`, so it owns an arena slot. The
//! Vulkan lowering used to treat it as a no-op view and never wrote that slot:
//! every reader saw stale bytes. Found porting CLEF, where the VQ codebook loss
//! `‖sg(z) − q‖²` read as 0 and a real T5's softmax max-shift `x − sg(max x)`
//! overflowed to inf. Runs only when a Vulkan device is present.

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_vulkan::backend::VulkanExecutable;

mod common;

#[test]
fn stop_gradient_output_slot_is_written() {
    if rlx_ir::env::skip_unless_device("vulkan", true, rlx_vulkan::is_available()) {
        return;
    }
    let _gpu = common::serialize_gpu();
    let n = 1000;
    let s = Shape::new(&[n], DType::F32);
    let mut g = Graph::new("stop_gradient_slot");
    let x = g.input("x", s.clone());
    let a = g.add_node(Op::Binary(BinaryOp::Mul), vec![x, x], s.clone());
    let sg = g.add_node(Op::StopGradient, vec![a], s.clone());
    // `a` stays live past `sg`, so `sg` cannot reuse its slot.
    let y = g.add_node(Op::Binary(BinaryOp::Add), vec![sg, a], s.clone());
    g.set_outputs(vec![y, sg]);

    let xs: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 3.0).collect();
    let mut exe = VulkanExecutable::compile(g);
    let out = exe.run(&[("x", &xs)]);
    for (i, &xv) in xs.iter().enumerate() {
        let a = xv * xv;
        assert_eq!(out[1][i], a, "sg[{i}]");
        assert_eq!(out[0][i], a + a, "sg + a [{i}]");
    }
}
