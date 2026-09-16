// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Regression: a graph containing a **zero-element** tensor must still compile.
//!
//! `rlx_compile::memory` only pushes a buffer when its slot size is non-zero, so a tensor with
//! a zero-length dimension gets no assignment at all. Every arena lookup in this backend then
//! panicked — `rlx-wgpu arena: no offset for node NodeId(n) (not in arena or weight buffer)`
//! — on a node the planner had deliberately skipped. LuxTTS's flow decoder builds an
//! `Expand [0,1,512]`, so the whole model was unrunnable on wgpu.
//!
//! `compile_static_inner` now gives each such node an explicit empty slot, and
//! `arena_span_bytes` skips zero-length ids so they never anchor a bind window.

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_wgpu::backend::WgpuExecutable;

/// `x + x` alongside a zero-element `Expand`, both returned.
fn graph_with_empty_tensor(empty_dims: &[usize]) -> Graph {
    let mut g = Graph::new("empty_slot");
    let x = g.input("x", Shape::new(&[1, 1, 4], DType::F32));
    let y = g.binary(BinaryOp::Add, x, x, Shape::new(&[1, 1, 4], DType::F32));
    let tgt: Vec<i64> = empty_dims.iter().map(|&d| d as i64).collect();
    let empty = g.add_node(
        Op::Expand { target_shape: tgt },
        vec![x],
        Shape::new(empty_dims, DType::F32),
    );
    g.set_outputs(vec![y, empty]);
    g
}

#[test]
fn zero_element_expand_compiles_and_leaves_real_output_intact() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    // The LuxTTS shape class: a leading zero-length batch dimension.
    let mut exe = WgpuExecutable::compile(graph_with_empty_tensor(&[0, 1, 4]));
    let out = exe.run(&[("x", &[1.0, 2.0, 3.0, 4.0])]);
    assert_eq!(
        out[0],
        vec![2.0, 4.0, 6.0, 8.0],
        "the empty sibling must not disturb the real output"
    );
    assert!(
        out[1].is_empty(),
        "a zero-element tensor must read back empty, got {:?}",
        out[1]
    );
}

#[test]
fn zero_length_inner_dimension_also_compiles() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    // The zero can sit on any axis — the planner's `size > 0` test is on the product.
    let mut exe = WgpuExecutable::compile(graph_with_empty_tensor(&[1, 0, 4]));
    let out = exe.run(&[("x", &[1.0, 2.0, 3.0, 4.0])]);
    assert_eq!(out[0], vec![2.0, 4.0, 6.0, 8.0]);
    assert!(out[1].is_empty());
}
