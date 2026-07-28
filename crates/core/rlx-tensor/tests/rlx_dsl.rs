// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! End-to-end tests for the `rlx! { … }` graph DSL. Each builds a real
//! `rlx_ir::Graph`, so a shape-inference or wiring bug fails here.
#![cfg(feature = "dsl")]

use rlx_ir::op::{Activation, BinaryOp, Op};
use rlx_ir::{Dim, Graph};
use rlx_tensor::rlx;

/// Count nodes whose op matches a predicate.
fn count(g: &Graph, pred: impl Fn(&Op) -> bool) -> usize {
    g.nodes().iter().filter(|n| pred(&n.op)).count()
}

#[test]
fn mlp_matmul_bias_activation() {
    let g = rlx! {
        graph "mlp";
        input x: [4, 784];
        param w1: [784, 256];   param b1: [256];
        param w2: [256, 10];    param b2: [10];

        let h = gelu(x @ w1 + b1);
        let y = h @ w2 + b2;
        out y;
    };

    assert_eq!(g.name, "mlp");
    assert_eq!(g.outputs.len(), 1);

    // Two projections, two bias adds, one GELU.
    assert_eq!(count(&g, |op| matches!(op, Op::MatMul)), 2);
    assert_eq!(
        count(&g, |op| matches!(op, Op::Binary(BinaryOp::Add))),
        2
    );
    assert_eq!(
        count(
            &g,
            |op| matches!(op, Op::Activation(Activation::Gelu))
        ),
        1
    );

    // Output shape flows through inference: [4, 784] · [784,256] · [256,10].
    let out = g.shape(g.outputs[0]);
    assert_eq!(out.dim(0), Dim::Static(4));
    assert_eq!(out.dim(1), Dim::Static(10));
}

#[test]
fn default_output_is_last_let() {
    // No explicit `out` — the last `let` is the output.
    let g = rlx! {
        input x: [2, 4];
        param w: [4, 3];
        let y = relu(x @ w);
    };
    assert_eq!(g.name, "rlx_graph");
    assert_eq!(g.outputs.len(), 1);
    assert!(matches!(
        g.node(g.outputs[0]).op,
        Op::Activation(Activation::Relu)
    ));
}

#[test]
fn precedence_and_scalar_promotion() {
    // `a + b * c` must parse as `a + (b*c)`, and `x * 0.5` promotes the scalar.
    let g = rlx! {
        input a: [8];
        input b: [8];
        input c: [8];
        let y = a + b * c;
        let z = y * 0.5;
        out z;
    };
    // 1 mul (b*c), 1 add (a+...), 1 scalar mul (y*0.5) → 2 muls total.
    assert_eq!(
        count(&g, |op| matches!(op, Op::Binary(BinaryOp::Mul))),
        2
    );
    assert_eq!(
        count(&g, |op| matches!(op, Op::Binary(BinaryOp::Add))),
        1
    );
}

#[test]
fn dynamic_batch_via_let() {
    let g = rlx! {
        input x: [?, 128];
        param w: [128, 64];
        let y = x @ w;
    };
    let out = g.shape(g.outputs[0]);
    assert!(matches!(out.dim(0), Dim::Dynamic(_)));
    assert_eq!(out.dim(1), Dim::Static(64));
}

#[test]
fn method_escape_hatch_and_auto_borrow() {
    // A self-attention-ish block exercising `@`, the method escape hatch, and
    // auto-borrowing of bare tensor arguments (`k`, `v` → `&k`, `&v`).
    let g = rlx! {
        graph "attn";
        input x: [2, 16, 64];
        param wq: [64, 64];  param wk: [64, 64];  param wv: [64, 64];
        param wo: [64, 64];

        let q = x @ wq;
        let k = x @ wk;
        let v = x @ wv;
        let a = q.attention(k, v, 8, 8, MaskKind::Causal);
        let o = a @ wo;
        out o;
    };
    assert_eq!(g.name, "attn");
    assert_eq!(count(&g, |op| matches!(op, Op::MatMul)), 4);
    assert_eq!(count(&g, |op| matches!(op, Op::Attention { .. })), 1);
}

#[test]
fn method_arg_external_value_via_paren() {
    // A bare ident in a method arg is a binding; to pass an *external* Rust
    // value, parenthesise it. Here `(axis)` references an outer `let`.
    let axis = -1i32;
    let g = rlx! {
        input scores: [2, 4, 4];
        let p = scores.softmax((axis));
        out p;
    };
    assert_eq!(count(&g, |op| matches!(op, Op::Softmax { .. })), 1);
}

#[test]
fn method_raw_args_negative_literal() {
    // A raw method arg like the `-1` axis must pass through verbatim (it is not
    // a single `literal` token), and non-var scalars stay by-value.
    let g = rlx! {
        input scores: [2, 4, 4];
        let p = scores.softmax(-1);
        out p;
    };
    assert_eq!(count(&g, |op| matches!(op, Op::Softmax { .. })), 1);
}

#[test]
fn scalar_left_promotion() {
    // `0.5 * x` (scalar on the left) must lower through the `f64 * &Tensor`
    // impls — the branch that was previously untested.
    let g = rlx! {
        input x: [4];
        let z = 0.5 * x;
        out z;
    };
    assert_eq!(
        count(&g, |op| matches!(op, Op::Binary(BinaryOp::Mul))),
        1
    );
    assert_eq!(g.shape(g.outputs[0]).dim(0), Dim::Static(4));
}

#[test]
fn numpy_precedence_matmul_and_multiply() {
    // `x @ w * s` must parse as `(x @ w) * s` (NumPy precedence). With these
    // shapes only that grouping type-checks: the other, `x @ (w * s)`, needs
    // `w[4,3] * s[2,3]` which is a broadcast error — so a successful build IS
    // the proof.
    let g = rlx! {
        input x: [2, 4];
        param w: [4, 3];
        param s: [2, 3];
        let y = x @ w * s;
        out y;
    };
    let out = g.shape(g.outputs[0]);
    assert_eq!(out.dim(0), Dim::Static(2));
    assert_eq!(out.dim(1), Dim::Static(3));
}

#[test]
fn negative_const_literal() {
    let g = rlx! {
        input x: [2];
        const bias = -1.5 : F32;
        let y = x + bias;
        out y;
    };
    assert_eq!(count(&g, |op| matches!(op, Op::Constant { .. })), 1);
}

#[test]
fn multiple_outputs_and_const() {
    let g = rlx! {
        input x: [4, 8];
        param w: [8, 8];
        const scale = 2.0 : F32;
        let a = x @ w;
        let b = a * scale;
        out a, b;
    };
    assert_eq!(g.outputs.len(), 2);
    assert_eq!(count(&g, |op| matches!(op, Op::Constant { .. })), 1);
}
