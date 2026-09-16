// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Wrt::Output` — gradients with respect to an **intermediate** activation.
//!
//! `grad_with_loss` re-resolves `wrt` by name after `prepare_graph_for_ad`
//! renumbers, but only `Op::Input`/`Op::Param` leaves carry a name. An
//! intermediate has no stable handle, so a raw `NodeId` captured before
//! prepare goes stale. `Wrt::Output(i)` fixes that by designating the target
//! through `graph.outputs`, which passes remap in place.
//!
//! This is the VJP-at-a-cut-point primitive: seed `d_output` with an arbitrary
//! cotangent on a *non-scalar* `outputs[0]` and read `∂outputs[0]/∂tap`.

use rlx_autodiff::{GradWithLossOptions, Wrt, grad_with_loss_wrt, prepare_graph_for_ad};
use rlx_ir::op::{Activation, BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, Shape};

/// `h = sum(x)·w` is an intermediate built *after* a multi-axis reduce, so
/// prepare's reduce legalization renumbers past it. Tapped as `outputs[1]`,
/// `dy/dh` still lands on the right node.
#[test]
fn wrt_output_survives_prepare_renumbering() {
    let f = DType::F32;
    let mut g = Graph::new("tap_after_reduce");
    let x = g.input("x", Shape::new(&[2, 2, 2], f)); // 8 elements
    // Multi-axis reduce: legalized into per-axis reduces + reshape by prepare,
    // renumbering every node built after it.
    let s = g.reduce(
        x,
        ReduceOp::Sum,
        vec![0, 1, 2],
        false,
        Shape::from_dims(&[], f),
    );
    let w = g.param("w", Shape::new(&[1], f));
    let h = g.binary(BinaryOp::Mul, s, w, Shape::new(&[1], f)); // intermediate tap
    let y = g.binary(BinaryOp::Mul, h, h, Shape::new(&[1], f)); // y = h²
    g.set_outputs(vec![y, h]);

    // The hazard this test exists for: after prepare, the tap is no longer the
    // node id the caller captured. Without the output-position remap, a raw
    // NodeId would differentiate some *other* node.
    let prepared = prepare_graph_for_ad(g.clone());
    assert_ne!(
        prepared.outputs[1], h,
        "expected prepare to renumber past the tap — test no longer covers the hazard"
    );

    let bwd = grad_with_loss_wrt(
        &g,
        &[Wrt::Output(1)],
        GradWithLossOptions::STRICT.with_aux(false),
    );
    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    compiled.set_param("w", &[1.0]);
    let outs = compiled.run(&[("x", &[1.0f32; 8][..]), ("d_output", &[1.0f32])]);

    // emit_aux = false ⇒ outputs are [y, dy/dh], with the `h` mirror dropped.
    assert_eq!(outs.len(), 2, "expect [y, dy/dh]");
    // x = ones ⇒ sum(x) = 8, w = 1 ⇒ h = 8, y = h² = 64, dy/dh = 2h = 16.
    assert!((outs[0][0] - 64.0).abs() < 1e-4, "y: got {}", outs[0][0]);
    assert!(
        (outs[1][0] - 16.0).abs() < 1e-4,
        "dy/dh: got {}",
        outs[1][0]
    );
}

/// `emit_aux = true` keeps the tap's *value* alongside its gradient.
#[test]
fn wrt_output_can_also_emit_the_tap_value() {
    let f = DType::F32;
    let mut g = Graph::new("tap_value");
    let x = g.input("x", Shape::new(&[2, 2, 2], f));
    let s = g.reduce(
        x,
        ReduceOp::Sum,
        vec![0, 1, 2],
        false,
        Shape::from_dims(&[], f),
    );
    let w = g.param("w", Shape::new(&[1], f));
    let h = g.binary(BinaryOp::Mul, s, w, Shape::new(&[1], f));
    let y = g.binary(BinaryOp::Mul, h, h, Shape::new(&[1], f));
    g.set_outputs(vec![y, h]);

    let bwd = grad_with_loss_wrt(&g, &[Wrt::Output(1)], GradWithLossOptions::STRICT);
    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    compiled.set_param("w", &[1.0]);
    let outs = compiled.run(&[("x", &[1.0f32; 8][..]), ("d_output", &[1.0f32])]);

    assert_eq!(outs.len(), 3, "expect [y, h, dy/dh]");
    assert!((outs[0][0] - 64.0).abs() < 1e-4, "y: got {}", outs[0][0]);
    assert!((outs[1][0] - 8.0).abs() < 1e-4, "h: got {}", outs[1][0]);
    assert!(
        (outs[2][0] - 16.0).abs() < 1e-4,
        "dy/dh: got {}",
        outs[2][0]
    );
}

const ROWS: usize = 2;
const D_IN: usize = 3;
const D_HID: usize = 4;

/// Deterministic small weights — kept O(0.1) so `tanh` stays off its flat
/// tails and central differences are well-conditioned in f32.
fn weight(n: usize, stride: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.1 * (((i * stride) % 7) as f32 - 3.0))
        .collect()
}

/// Build `y = tanh(h · w1) · w2` with `h = x + zero` — an intermediate that is
/// numerically the identity on `x`, so `∂y/∂h` and `∂y/∂x` must agree and both
/// admit a finite-difference oracle on `x`.
fn mlp_graph() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("mlp_tap");
    let x = g.input("x", Shape::new(&[ROWS, D_IN], f));
    let zero = g.param("zero", Shape::new(&[ROWS, D_IN], f));
    let h = g.binary(BinaryOp::Add, x, zero, Shape::new(&[ROWS, D_IN], f));
    let w1 = g.param("w1", Shape::new(&[D_IN, D_HID], f));
    let a = g.matmul(h, w1, Shape::new(&[ROWS, D_HID], f));
    let t = g.activation(Activation::Tanh, a, Shape::new(&[ROWS, D_HID], f));
    let w2 = g.param("w2", Shape::new(&[D_HID, D_IN], f));
    let y = g.matmul(t, w2, Shape::new(&[ROWS, D_IN], f));
    g.set_outputs(vec![y, h]);
    g
}

fn set_mlp_params(compiled: &mut rlx::CompiledGraph) {
    compiled.set_param("zero", &[0.0f32; ROWS * D_IN]);
    compiled.set_param("w1", &weight(D_IN * D_HID, 3));
    compiled.set_param("w2", &weight(D_HID * D_IN, 5));
}

/// The lens's actual primitive: a **non-scalar** `outputs[0]` seeded with an
/// arbitrary cotangent `v`, differentiated w.r.t. an intermediate. Checked
/// against central finite differences of `Σ v·y`.
#[test]
fn vjp_at_intermediate_matches_finite_differences() {
    let n = ROWS * D_IN;
    let x0: Vec<f32> = (0..n).map(|i| 0.2 * (i as f32) - 0.5).collect();
    // A non-uniform cotangent — a uniform one would hide index-order bugs.
    let v: Vec<f32> = (0..n).map(|i| 1.0 + 0.37 * (i as f32)).collect();

    let g = mlp_graph();
    let bwd = grad_with_loss_wrt(
        &g,
        &[Wrt::Output(1), Wrt::Leaf("x".into())],
        GradWithLossOptions::STRICT.with_aux(false),
    );
    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    set_mlp_params(&mut compiled);
    let outs = compiled.run(&[("x", &x0[..]), ("d_output", &v[..])]);
    assert_eq!(outs.len(), 3, "expect [y, dy/dh, dy/dx]");
    let d_h = &outs[1];
    let d_x = &outs[2];

    // h = x + 0, so the intermediate tap and the leaf must agree exactly.
    for i in 0..n {
        assert!(
            (d_h[i] - d_x[i]).abs() < 1e-6,
            "tap vs leaf disagree at {i}: {} vs {}",
            d_h[i],
            d_x[i]
        );
    }

    // Finite-difference oracle on Σ v·y.
    let mut fwd = rlx::Session::new(rlx::Device::Cpu).compile(mlp_graph());
    set_mlp_params(&mut fwd);
    let weighted_sum = |fwd: &mut rlx::CompiledGraph, x: &[f32]| -> f64 {
        let y = &fwd.run(&[("x", x)])[0];
        y.iter()
            .zip(&v)
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum()
    };

    let eps = 1e-3f32;
    for i in 0..n {
        let mut xp = x0.clone();
        let mut xm = x0.clone();
        xp[i] += eps;
        xm[i] -= eps;
        let fd = (weighted_sum(&mut fwd, &xp) - weighted_sum(&mut fwd, &xm)) / (2.0 * eps as f64);
        assert!(
            (fd - d_h[i] as f64).abs() < 1e-3,
            "dh[{i}]: autodiff {} vs finite-difference {fd}",
            d_h[i]
        );
    }
}

/// An out-of-range tap index is a panic, not a silently wrong gradient.
#[test]
#[should_panic(expected = "out of range")]
fn wrt_output_out_of_range_panics() {
    let g = mlp_graph(); // 2 outputs
    let _ = grad_with_loss_wrt(&g, &[Wrt::Output(7)], GradWithLossOptions::STRICT);
}
