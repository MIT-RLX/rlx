// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Gradients must survive the fusion pipeline.
//!
//! A backward graph is compiled like any other graph, so the fusion passes run
//! over it — including `FuseSharedInputMatMul`, which groups matmuls sharing an
//! input and concatenates their weights. That pass hoists the later weights to
//! the fusion site via `Rewriter::ensure_mapped`; when `copy_node` then copied
//! those already-mapped nodes a second time, the graph ended up with two
//! `Op::Param` nodes per weight. Parameter binding is by name and reaches one
//! node, so the duplicate read zeros.
//!
//! A **forward** graph never showed it: the fused-away matmul was the weight's
//! only reader, so the duplicate was dead code. A backward graph reads each
//! weight twice — once in the mirrored forward, once in `dX = dY · Wᵀ` — so the
//! duplicate was live, and every gradient term flowing through it became
//! exactly zero. Found via a Qwen3.5 gated-delta-net block, where it silently
//! erased all cross-position gradient.
//!
//! The check is a gradient computed with fusion on versus off: they must agree.

use rlx::CompileOptions;
use rlx_autodiff::{GradWithLossOptions, Wrt, grad_with_loss_wrt};
use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Shape};

const ROWS: usize = 2;
const D_IN: usize = 3;
const D_OUT: usize = 4;

/// `y = (x·w1) + (x·w2)·(x·w3)` — three matmuls sharing `x`, so
/// `FuseSharedInputMatMul` groups all three and concatenates their weights.
fn shared_input_graph() -> Graph {
    let f = DType::F32;
    let out = Shape::new(&[ROWS, D_OUT], f);
    let mut g = Graph::new("shared_input_grad");
    let x = g.input("x", Shape::new(&[ROWS, D_IN], f));
    let w1 = g.param("w1", Shape::new(&[D_IN, D_OUT], f));
    let a = g.matmul(x, w1, out.clone());
    // Declared after the first matmul so the hoist has to pull them backwards.
    let w2 = g.param("w2", Shape::new(&[D_IN, D_OUT], f));
    let b = g.matmul(x, w2, out.clone());
    let w3 = g.param("w3", Shape::new(&[D_IN, D_OUT], f));
    let c = g.matmul(x, w3, out.clone());
    let bc = g.binary(BinaryOp::Mul, b, c, out.clone());
    let y = g.binary(BinaryOp::Add, a, bc, out);
    g.set_outputs(vec![y]);
    g
}

fn weights(seed: usize) -> Vec<f32> {
    (0..D_IN * D_OUT)
        .map(|i| 0.1 * (((i * 7 + seed * 5) % 11) as f32 - 5.0))
        .collect()
}

fn run(opts: &CompileOptions) -> Vec<Vec<f32>> {
    let bwd = grad_with_loss_wrt(
        &shared_input_graph(),
        &[
            Wrt::Leaf("x".into()),
            Wrt::Leaf("w1".into()),
            Wrt::Leaf("w2".into()),
            Wrt::Leaf("w3".into()),
        ],
        GradWithLossOptions::STRICT.with_aux(false),
    );
    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile_with(bwd, opts);
    compiled.set_param("w1", &weights(0));
    compiled.set_param("w2", &weights(1));
    compiled.set_param("w3", &weights(2));

    let x: Vec<f32> = (0..ROWS * D_IN).map(|i| 0.2 * (i as f32) - 0.5).collect();
    let cot: Vec<f32> = (0..ROWS * D_OUT)
        .map(|i| 0.5 + 0.25 * ((i % 5) as f32))
        .collect();
    compiled.run(&[("x", &x[..]), ("d_output", &cot[..])])
}

#[test]
fn gradients_are_unchanged_by_shared_input_matmul_fusion() {
    let fused = run(&CompileOptions::default());
    let unfused = run(&{
        let mut o = CompileOptions::default();
        o.fusion_opts.skip_fusion = true;
        o
    });

    assert_eq!(fused.len(), unfused.len(), "output count changed");
    let labels = ["y", "dx", "dw1", "dw2", "dw3"];
    for (i, (a, b)) in fused.iter().zip(&unfused).enumerate() {
        let label = labels.get(i).copied().unwrap_or("?");
        assert_eq!(a.len(), b.len(), "{label}: length changed");
        for (j, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x - y).abs() < 1e-5,
                "{label}[{j}]: fused {x} vs unfused {y} — a fusion pass changed the gradient"
            );
        }
        // A gradient of exactly zero everywhere is the failure mode this test
        // exists for; it must not pass by both sides being zero.
        let magnitude = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(magnitude > 1e-4, "{label} is ~zero (max {magnitude})");
    }
}
