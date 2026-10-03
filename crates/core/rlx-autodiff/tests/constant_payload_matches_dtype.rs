// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Every `Op::Constant` the backward pass creates must carry exactly
//! `num_elements × dtype.size_bytes()` bytes.
//!
//! Two sites built zero payloads at a hardcoded 4 bytes per element while taking
//! the shape — dtype included — from a graph node. On an autocast graph that node
//! is bf16, so the constant declared 2 B/elem and held 4. Nothing checked it, and
//! the consequence landed far away: `cpu_low_precision::promote_to_f32` widened
//! those f32 bytes a *second* time as if they were bf16, and MLX then refused the
//! leaf ("shape [1,41,2,1] wants 82 elements, got 164"), which is why MLX could
//! not run an `rlx-kev` bf16 training graph.
//!
//! Zero is zero in every float format, so only the length was ever wrong — which
//! is exactly the kind of bug a length invariant catches and a value check does
//! not.

use rlx_autodiff::{GradWithLossOptions, Wrt, grad_with_loss_wrt};
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Op, Shape};

/// `wrt` a parameter the loss does not use, so `zero_missing_wrt` fires.
fn graph_with_unused_param(dtype: DType) -> Graph {
    let mut g = Graph::new("unused");
    let x = g.input("x", Shape::new(&[2, 4], dtype));
    let w = g.param("w", Shape::new(&[4, 3], dtype));
    let unused = g.param("unused", Shape::new(&[1, 41, 2, 1], dtype));
    let y = g.matmul(x, w, Shape::new(&[2, 3], dtype));
    let flat = g.reshape_(y, vec![-1]);
    let loss = g.mean(flat, vec![0], false);
    g.set_outputs(vec![loss]);
    let _ = unused;
    g
}

fn check(g: &Graph, label: &str) {
    let bwd = grad_with_loss_wrt(
        g,
        &[Wrt::Leaf("w".into()), Wrt::Leaf("unused".into())],
        GradWithLossOptions::TRAINING.with_aux(false),
    );
    let mut bad = Vec::new();
    for n in bwd.nodes() {
        if let Op::Constant { data } = &n.op {
            let numel = n.shape.num_elements().unwrap_or(0);
            let want = numel * n.shape.dtype().size_bytes();
            if data.len() != want {
                bad.push(format!(
                    "{:?} dtype {:?} dims {:?}: wants {want} bytes, has {}",
                    n.id,
                    n.shape.dtype(),
                    n.shape.dims(),
                    data.len()
                ));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "{label}: constant payload disagrees with its declared dtype:\n  {}",
        bad.join("\n  ")
    );
}

#[test]
fn zero_missing_wrt_sizes_by_dtype() {
    for (label, dt) in [
        ("f32", DType::F32),
        ("bf16", DType::BF16),
        ("f16", DType::F16),
    ] {
        check(&graph_with_unused_param(dt), label);
    }
}
