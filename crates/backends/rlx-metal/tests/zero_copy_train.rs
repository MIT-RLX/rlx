// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Zero-copy GPU-resident training step (`MetalExecutable::optimizer_step_resident`):
//! the weight lives in the unified-memory arena (resident), the backward graph
//! writes its gradient into the arena, and the optimizer updates the weight
//! IN PLACE on the arena — no GPU→host→GPU roundtrip. Drives a tiny regression
//! (`loss = Σ (w − target)²`, hand-built `[loss, grad]` graph) to convergence.

#![cfg(target_os = "macos")]

use rlx_ir::op::{BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_metal::backend::MetalExecutable;

#[test]
fn zero_copy_resident_optimizer_step_converges() {
    if rlx_metal::device::metal_device().is_none() {
        return;
    }
    const N: usize = 8;
    let target: Vec<f32> = (0..N).map(|i| i as f32 * 0.3 - 1.0).collect();

    // Hand-built forward + gradient graph (avoids an autodiff dep; the point is
    // the resident zero-copy STEP, not autodiff):
    //   loss   = Σ (w − target)²
    //   grad_w = 2·(w − target)
    // Outputs = [loss, grad_w] — grad at output slot 1, as the step expects.
    let mut g = Graph::new("resident_train");
    let w = g.input("w", Shape::new(&[N], DType::F32));
    let tgt = g.add_node(
        Op::Constant {
            data: target.iter().flat_map(|v| v.to_le_bytes()).collect(),
        },
        vec![],
        Shape::new(&[N], DType::F32),
    );
    let diff = g.add_node(
        Op::Binary(BinaryOp::Sub),
        vec![w, tgt],
        Shape::new(&[N], DType::F32),
    );
    let sq = g.add_node(
        Op::Binary(BinaryOp::Mul),
        vec![diff, diff],
        Shape::new(&[N], DType::F32),
    );
    let loss = g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![0],
            keep_dim: false,
        },
        vec![sq],
        Shape::from_dims(&[], DType::F32),
    );
    let two = g.add_node(
        Op::Constant {
            data: 2.0f32.to_le_bytes().to_vec(),
        },
        vec![],
        Shape::new(&[1], DType::F32),
    );
    let grad_w = g.add_node(
        Op::Binary(BinaryOp::Mul),
        vec![diff, two],
        Shape::new(&[N], DType::F32),
    );
    g.set_outputs(vec![loss, grad_w]);

    let mut exe = MetalExecutable::compile(g);

    // Weight starts at zero and lives RESIDENT in the arena (no host re-upload).
    assert!(
        exe.bind_gpu_handle("w", &[0f32; N]),
        "weight should bind resident"
    );

    let trainable = vec![("w".to_string(), vec![N])];
    let lr = 0.2f32;
    let mut first = f32::NAN;
    let mut last = f32::NAN;
    for step in 0..60 {
        // w is resident → not fed; the backward writes grad_w into the arena.
        let outs = exe.run(&[]);
        let loss_val = outs[0][0];
        if step == 0 {
            first = loss_val;
        }
        last = loss_val;
        // Zero-copy SGD: reads grad + writes w, both aliasing the arena in place.
        exe.optimizer_step_resident(&trainable, |_name, _shape, p, grad| {
            for i in 0..p.len() {
                p[i] -= lr * grad[i];
            }
        });
    }

    assert!(first.is_finite() && last.is_finite());
    assert!(
        last < first * 0.001,
        "resident training should converge: loss {first} → {last}"
    );
    // The resident weight (never round-tripped through host) converged to target.
    let w_final = exe.read_gpu_handle("w").expect("read resident weight");
    let err = w_final
        .iter()
        .zip(&target)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(err < 1e-2, "w should converge to target (max err {err})");
}
