// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `split_vjp` must reproduce the monolithic backward exactly.
//!
//! The split runs the forward once and replays the gradient half per cotangent,
//! instead of recomputing the forward inside every backward. That is only worth
//! anything if the answer is unchanged, so the check is direct: the same
//! gradients, under several different cotangents, from both paths.
//!
//! Several cotangents matter. Running one would not catch the failure mode the
//! split invites — a saved activation that is stale, or accidentally recomputed
//! from the *current* cotangent rather than held fixed.

use rlx_autodiff::{GradWithLossOptions, Wrt, grad_with_loss_wrt, split_vjp};
use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, Shape};

const ROWS: usize = 3;
const D_IN: usize = 4;
const D_HID: usize = 5;

/// `y = tanh(x·w1) · w2 + x·w3` — nonlinear, several matmuls sharing `x`, and a
/// residual path, so the backward genuinely needs saved activations.
fn forward() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("split_case");
    let x = g.input("x", Shape::new(&[ROWS, D_IN], f));
    let w1 = g.param("w1", Shape::new(&[D_IN, D_HID], f));
    let a = g.matmul(x, w1, Shape::new(&[ROWS, D_HID], f));
    let t = g.activation(Activation::Tanh, a, Shape::new(&[ROWS, D_HID], f));
    let w2 = g.param("w2", Shape::new(&[D_HID, D_IN], f));
    let b = g.matmul(t, w2, Shape::new(&[ROWS, D_IN], f));
    let w3 = g.param("w3", Shape::new(&[D_IN, D_IN], f));
    let c = g.matmul(x, w3, Shape::new(&[ROWS, D_IN], f));
    let y = g.binary(BinaryOp::Add, b, c, Shape::new(&[ROWS, D_IN], f));
    g.set_outputs(vec![y]);
    g
}

fn weights(seed: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.15 * (((i * 7 + seed * 5) % 11) as f32 - 5.0))
        .collect()
}

fn bind(compiled: &mut rlx::CompiledGraph) {
    compiled.set_param("w1", &weights(0, D_IN * D_HID));
    compiled.set_param("w2", &weights(1, D_HID * D_IN));
    compiled.set_param("w3", &weights(2, D_IN * D_IN));
}

fn x_values() -> Vec<f32> {
    (0..ROWS * D_IN).map(|i| 0.2 * (i as f32) - 0.6).collect()
}

/// Three visibly different cotangents.
fn cotangents() -> Vec<Vec<f32>> {
    let n = ROWS * D_IN;
    vec![
        (0..n).map(|i| 0.5 + 0.25 * ((i % 5) as f32)).collect(),
        (0..n)
            .map(|i| if i % 3 == 0 { 1.0 } else { -0.5 })
            .collect(),
        (0..n).map(|i| 1.0 / (1.0 + i as f32)).collect(),
    ]
}

fn backward_graph() -> Graph {
    grad_with_loss_wrt(
        &forward(),
        &[
            Wrt::Leaf("x".into()),
            Wrt::Leaf("w1".into()),
            Wrt::Leaf("w2".into()),
            Wrt::Leaf("w3".into()),
        ],
        GradWithLossOptions::STRICT.with_aux(false),
    )
}

#[test]
fn split_reproduces_the_monolithic_backward() {
    let bwd = backward_graph();
    let n_outputs = bwd.outputs.len();
    let split = split_vjp(&bwd).expect("split");
    assert!(
        !split.saved.is_empty(),
        "this graph should need saved activations"
    );

    let x = x_values();

    // Monolithic: forward recomputed inside every backward.
    let mut mono = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    bind(&mut mono);
    let expected: Vec<Vec<Vec<f32>>> = cotangents()
        .iter()
        .map(|cot| mono.run(&[("x", &x[..]), ("d_output", &cot[..])]))
        .collect();

    // Split: forward once, then replay per cotangent.
    let mut save = rlx::Session::new(rlx::Device::Cpu).compile(split.save);
    bind(&mut save);
    let saved_values = save.run(&[("x", &x[..])]);

    let mut replay = rlx::Session::new(rlx::Device::Cpu).compile(split.replay);
    bind(&mut replay);
    for s in &split.saved {
        replay.set_param(&s.name, &saved_values[s.save_output]);
    }

    for (case, cot) in cotangents().iter().enumerate() {
        let got = replay.run(&[("d_output", &cot[..])]);
        assert_eq!(
            got.len(),
            split.replay_output_indices.len(),
            "case {case}: replay output count"
        );

        // Reassemble the original output order from both halves.
        let mut merged: Vec<Option<&Vec<f32>>> = vec![None; n_outputs];
        for (slot, &idx) in split.save_output_indices.iter().enumerate() {
            merged[idx] = Some(&saved_values[slot]);
        }
        for (slot, &idx) in split.replay_output_indices.iter().enumerate() {
            merged[idx] = Some(&got[slot]);
        }

        for (idx, slot) in merged.iter().enumerate() {
            let actual = slot.unwrap_or_else(|| panic!("output {idx} produced by neither half"));
            let want = &expected[case][idx];
            assert_eq!(actual.len(), want.len(), "case {case} output {idx}: length");
            for (j, (a, b)) in actual.iter().zip(want).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "case {case} output {idx}[{j}]: split {a} vs monolithic {b}"
                );
            }
        }
        // Guard against a vacuous pass: the gradients must be non-trivial.
        let magnitude = got.iter().flatten().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(magnitude > 1e-3, "case {case}: replay produced ~zero");
    }
}

/// The point of the split: `replay` must not contain the forward.
#[test]
fn replay_drops_the_forward_recompute() {
    let bwd = backward_graph();
    let split = split_vjp(&bwd).expect("split");
    assert!(
        split.replay.len() < bwd.len(),
        "replay ({}) should be smaller than the monolithic backward ({})",
        split.replay.len(),
        bwd.len()
    );
    // `tanh` belongs to the forward; recomputing it in replay would mean the
    // cut failed to separate the halves.
    let tanh_in_replay = split
        .replay
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, rlx_ir::Op::Activation(Activation::Tanh)))
        .count();
    assert_eq!(
        tanh_in_replay, 0,
        "replay still recomputes the forward activation"
    );
}
