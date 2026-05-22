// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Numerical parity for `rlx_opt::autodiff::grad_with_loss`.
//!
//! Builds a small forward graph (BERT FFN block: `MatMul → Add(bias)
//! → Activation(Gelu) → MatMul → Reduce(Sum)`), runs the gradient
//! graph on CPU, and compares each parameter gradient against
//! second-order central finite differences computed by re-running the
//! same forward graph at perturbed parameter values.
//!
//! This pins down the autodiff phases 1–9 with an actual number — not
//! just "the gradient walk completes without panicking" but "the
//! gradient values match the FD approximation within a relative
//! tolerance set by the FD truncation error". Catches sign flips,
//! transpose mistakes, broadcast bugs, and missing-VJP-rule fallbacks
//! that no basic test would detect.

#![cfg(feature = "cpu")]

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};

/// Forward: `loss = sum( gelu(x @ W1 + b1) @ W2 )`.
/// Returns (graph, x_id, [W1, b1, W2]).
fn build_forward(m: usize, k: usize, n: usize, p: usize) -> (Graph, NodeId, Vec<NodeId>) {
    let mut g = Graph::new("ffn_loss");
    let x_shape = Shape::new(&[m, k], DType::F32);
    let w1_shape = Shape::new(&[k, n], DType::F32);
    let b1_shape = Shape::new(&[n], DType::F32);
    let w2_shape = Shape::new(&[n, p], DType::F32);
    let mid_shape = Shape::new(&[m, n], DType::F32);
    let out_shape = Shape::new(&[m, p], DType::F32);
    let scalar = Shape::new(&[1], DType::F32);

    let x = g.input("x", x_shape);
    let w1 = g.param("w1", w1_shape);
    let b1 = g.param("b1", b1_shape);
    let w2 = g.param("w2", w2_shape);

    let xw1 = g.matmul(x, w1, mid_shape.clone());
    // b1 expand from [n] to [m, n] (CPU Binary kernel handles trailing
    // broadcast, but expand keeps the test path identical to what the
    // autodiff legalizer would do).
    let b1_e = g.add_node(
        Op::Expand {
            target_shape: vec![m as i64, n as i64],
        },
        vec![b1],
        mid_shape.clone(),
    );
    let pre = g.binary(BinaryOp::Add, xw1, b1_e, mid_shape.clone());
    let act = g.activation(Activation::Gelu, pre, mid_shape);
    let y = g.matmul(act, w2, out_shape);
    let loss = g.add_node(
        Op::Reduce {
            op: rlx_ir::op::ReduceOp::Sum,
            axes: vec![0, 1],
            keep_dim: false,
        },
        vec![y],
        scalar,
    );
    g.set_outputs(vec![loss]);

    (g, x, vec![w1, b1, w2])
}

/// Run the forward graph once and return the scalar loss.
fn forward_loss(
    m: usize,
    k: usize,
    n: usize,
    p: usize,
    x_data: &[f32],
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
) -> f32 {
    let (g, _x_id, _params) = build_forward(m, k, n, p);
    let session = Session::new(Device::Cpu);
    let mut compiled = session.compile(g);
    compiled.set_param("w1", w1);
    compiled.set_param("b1", b1);
    compiled.set_param("w2", w2);
    let outs = compiled.run(&[("x", x_data)]);
    outs[0][0]
}

/// Compute central-difference gradient of `forward_loss` w.r.t. one
/// scalar entry of `param`. Two forward evaluations per entry — slow
/// but unbeatable as a correctness baseline.
fn fd_grad_one(
    m: usize,
    k: usize,
    n: usize,
    p: usize,
    x_data: &[f32],
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    which: usize,
    idx: usize,
    eps: f32,
) -> f32 {
    let perturb = |pos: bool| -> f32 {
        let (mut w1m, mut b1m, mut w2m) = (w1.to_vec(), b1.to_vec(), w2.to_vec());
        let target = match which {
            0 => &mut w1m,
            1 => &mut b1m,
            2 => &mut w2m,
            _ => unreachable!(),
        };
        target[idx] += if pos { eps } else { -eps };
        forward_loss(m, k, n, p, x_data, &w1m, &b1m, &w2m)
    };
    let plus = perturb(true);
    let minus = perturb(false);
    (plus - minus) / (2.0 * eps)
}

#[test]
fn cpu_grad_matches_finite_differences_on_ffn_block() {
    // Tiny dims so the FD sweep completes quickly: m=2, k=3, n=4, p=2.
    // Parameter sizes: W1 = 12, b1 = 4, W2 = 8 → 24 FD evaluations
    // total at 2 forward passes each = 48 forwards. Negligible.
    let (m, k, n, p) = (2, 3, 4, 2);

    let x_data: Vec<f32> = (0..m * k).map(|i| 0.1 + 0.07 * (i as f32)).collect();
    let w1: Vec<f32> = (0..k * n).map(|i| -0.2 + 0.05 * (i as f32)).collect();
    let b1: Vec<f32> = (0..n).map(|i| 0.01 * (i as f32 + 1.0)).collect();
    let w2: Vec<f32> = (0..n * p).map(|i| 0.05 + 0.03 * (i as f32)).collect();

    // 1) Build forward + grad graph, compile, run.
    let (g, _x_id, params) = build_forward(m, k, n, p);
    let bwd_g = grad_with_loss(&g, &params);

    let session = Session::new(Device::Cpu);
    let mut bwd = session.compile(bwd_g);
    bwd.set_param("w1", &w1);
    bwd.set_param("b1", &b1);
    bwd.set_param("w2", &w2);

    let d_output = vec![1.0f32];
    let outs = bwd.run(&[("x", &x_data), ("d_output", &d_output)]);

    // outs[0] = loss (scalar [1]), outs[1..] = grads for params in
    // declaration order: W1, b1, W2.
    assert_eq!(
        outs.len(),
        1 + params.len(),
        "expected loss + {} grads, got {}",
        params.len(),
        outs.len()
    );

    let loss_autodiff = outs[0][0];
    let loss_forward = forward_loss(m, k, n, p, &x_data, &w1, &b1, &w2);
    assert!(
        (loss_autodiff - loss_forward).abs() < 1e-4,
        "autodiff loss {loss_autodiff} disagrees with forward {loss_forward}"
    );

    // 2) For each parameter entry, compare autodiff to FD.
    let names = ["w1", "b1", "w2"];
    let lengths = [k * n, n, n * p];
    let eps = 1e-3f32;
    let abs_tol = 5e-3f32;
    let rel_tol = 5e-3f32;
    for (which, &len) in lengths.iter().enumerate() {
        let ad_grads = &outs[1 + which];
        assert_eq!(
            ad_grads.len(),
            len,
            "param {} grad length: got {} want {}",
            names[which],
            ad_grads.len(),
            len
        );
        for idx in 0..len {
            let fd = fd_grad_one(m, k, n, p, &x_data, &w1, &b1, &w2, which, idx, eps);
            let ad = ad_grads[idx];
            let abs_err = (fd - ad).abs();
            let rel_err = abs_err / fd.abs().max(1e-6);
            assert!(
                abs_err < abs_tol || rel_err < rel_tol,
                "{} grad[{idx}]: autodiff {ad:e} vs FD {fd:e} \
                 (abs {abs_err:e}, rel {rel_err:e})",
                names[which]
            );
        }
    }
}
