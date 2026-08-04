// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! End-to-end tests for the `rlx! { … }` graph DSL. Each builds a real
//! `rlx_ir::Graph`, so a shape-inference or wiring bug fails here.
#![cfg(feature = "dsl")]

use rlx_ir::op::{Activation, BinaryOp, CmpOp, Op};
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
    assert_eq!(count(&g, |op| matches!(op, Op::Binary(BinaryOp::Add))), 2);
    assert_eq!(
        count(&g, |op| matches!(op, Op::Activation(Activation::Gelu))),
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
    assert_eq!(count(&g, |op| matches!(op, Op::Binary(BinaryOp::Mul))), 2);
    assert_eq!(count(&g, |op| matches!(op, Op::Binary(BinaryOp::Add))), 1);
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
    assert_eq!(count(&g, |op| matches!(op, Op::Binary(BinaryOp::Mul))), 1);
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

// ── New-feature end-to-end coverage ────────────────────────────────────────

#[test]
fn comparison_and_select_build_where_and_compare() {
    // `select(x > 0.0, x, 0.0)` is ReLU expressed through a mask — a `Compare`
    // (Gt) feeding a `Where`. The scalar `0.0` false-branch is promoted.
    let g = rlx! {
        input x: [4];
        let y = select(x > 0.0, x, 0.0);
        out y;
    };
    assert_eq!(count(&g, |op| matches!(op, Op::Compare(CmpOp::Gt))), 1);
    assert_eq!(count(&g, |op| matches!(op, Op::Where)), 1);
    assert_eq!(g.shape(g.outputs[0]).dim(0), Dim::Static(4));
}

#[test]
fn scalar_left_comparison_swaps() {
    // `0.0 < x` lowers by swapping to `x > 0.0` (a single `Gt` compare).
    let g = rlx! {
        input x: [4];
        let m = 0.0 < x;
        let y = select(m, x, x);
        out y;
    };
    assert_eq!(count(&g, |op| matches!(op, Op::Compare(CmpOp::Gt))), 1);
}

#[test]
fn binary_sugar_and_infix_ops() {
    // maximum/minimum/pow/rem reach the right `BinaryOp`s, with scalar operands
    // promoted (`minimum(x, 1.0)`, `x ** 2`, `x % 3.0`).
    let g = rlx! {
        input x: [4];
        input w: [4];
        let a = maximum(x, w);   // Max (tensor rhs)
        let b = minimum(a, 1.0); // Min (scalar rhs)
        let c = b ** 2;          // Pow
        let d = c % 3.0;         // Mod
        let e = atan2(d, w);     // Atan2
        out e;
    };
    assert_eq!(count(&g, |op| matches!(op, Op::Binary(BinaryOp::Max))), 1);
    assert_eq!(count(&g, |op| matches!(op, Op::Binary(BinaryOp::Min))), 1);
    assert_eq!(count(&g, |op| matches!(op, Op::Binary(BinaryOp::Pow))), 1);
    assert_eq!(count(&g, |op| matches!(op, Op::Binary(BinaryOp::Mod))), 1);
    assert_eq!(count(&g, |op| matches!(op, Op::Binary(BinaryOp::Atan2))), 1);
}

#[test]
fn clamp_sugar_lowers_to_min_max() {
    let g = rlx! {
        input x: [4];
        let y = clamp(x, 0.0, 6.0);
        out y;
    };
    assert_eq!(count(&g, |op| matches!(op, Op::Binary(BinaryOp::Max))), 1);
    assert_eq!(count(&g, |op| matches!(op, Op::Binary(BinaryOp::Min))), 1);
}

#[test]
fn newly_exposed_activations() {
    let g = rlx! {
        input x: [4];
        let a = softplus(x);
        let b = erf(a);
        let c = mish(b);
        out c;
    };
    assert_eq!(
        count(&g, |op| matches!(op, Op::Activation(Activation::Softplus))),
        1
    );
    assert_eq!(
        count(&g, |op| matches!(op, Op::Activation(Activation::Erf))),
        1
    );
    assert_eq!(
        count(&g, |op| matches!(op, Op::Activation(Activation::Mish))),
        1
    );
}

#[test]
fn fn_inline_reuses_body_per_call() {
    // A two-layer FFN block, instantiated twice with different weights.
    let g = rlx! {
        fn ffn(x, w1, w2) {
            let h = gelu(x @ w1);
            let o = h @ w2;
        }
        input a: [2, 8];
        param u1: [8, 16];  param u2: [16, 8];
        param v1: [8, 16];  param v2: [16, 8];
        let s = ffn(a, u1, u2);
        let y = ffn(s, v1, v2);
        out y;
    };
    // Two calls → 4 matmuls, 2 gelus.
    assert_eq!(count(&g, |op| matches!(op, Op::MatMul)), 4);
    assert_eq!(
        count(&g, |op| matches!(op, Op::Activation(Activation::Gelu))),
        2
    );
    let out = g.shape(g.outputs[0]);
    assert_eq!(out.dim(0), Dim::Static(2));
    assert_eq!(out.dim(1), Dim::Static(8));
}

#[test]
fn fn_body_escape_hatch_args_are_renamed() {
    // An attention block using the escape hatch inside the fn body: `k`/`v` are
    // fn locals, so their bare-ident method args must be renamed when inlined.
    let g = rlx! {
        fn attn(x, wq, wk, wv) {
            let q = x @ wq;
            let k = x @ wk;
            let v = x @ wv;
            let o = q.attention(k, v, 8, 8, MaskKind::Causal);
        }
        input a: [2, 16, 64];
        param wq: [64, 64];  param wk: [64, 64];  param wv: [64, 64];
        let y = attn(a, wq, wk, wv);
        out y;
    };
    assert_eq!(count(&g, |op| matches!(op, Op::MatMul)), 3);
    assert_eq!(count(&g, |op| matches!(op, Op::Attention { .. })), 1);
}

#[test]
fn repeat_unrolls_weight_tied_stack() {
    // A weight-tied residual stack, unrolled 4×.
    let g = rlx! {
        input x: [4, 16];
        param w: [16, 16];
        repeat 4 {
            let x = x + relu(x @ w);
        }
        out x;
    };
    assert_eq!(count(&g, |op| matches!(op, Op::MatMul)), 4);
    assert_eq!(
        count(&g, |op| matches!(op, Op::Activation(Activation::Relu))),
        4
    );
    assert_eq!(count(&g, |op| matches!(op, Op::Binary(BinaryOp::Add))), 4);
}

#[test]
fn scan_builds_one_compact_scan_node() {
    // `scan` is a single `Op::Scan` (with the weight as a broadcast), NOT an
    // unrolled copy — the body's matmul/relu live in the nested body graph, so
    // the OUTER graph has zero MatMul nodes regardless of the length.
    let g = rlx! {
        input h0: [1, 8];
        param w: [8, 8];
        scan h = h0 for 6 {
            let h = relu(h @ w);
        }
        out h;
    };
    assert_eq!(
        count(&g, |op| matches!(op, Op::Scan { num_bcast: 1, .. })),
        1
    );
    assert_eq!(count(&g, |op| matches!(op, Op::MatMul)), 0);
    let out = g.shape(g.outputs[0]);
    assert_eq!(out.dim(0), Dim::Static(1));
    assert_eq!(out.dim(1), Dim::Static(8));
}

#[test]
fn linear_sugar_builds_fused_matmul_bias() {
    // `linear(x, w, b)` = x·Wᵀ+b (W is HF `[out,in]`) as one fused op.
    let g = rlx! {
        input x: [1, 4, 8];
        param w: [16, 8];
        param b: [16];
        let y = linear(x, w, b);
        out y;
    };
    assert_eq!(
        count(&g, |op| matches!(
            op,
            Op::FusedMatMulBiasAct { activation: None }
        )),
        1
    );
    assert_eq!(g.shape(g.outputs[0]).dim(2), Dim::Static(16));
}

#[test]
fn activation_of_linear_folds_into_fused_op() {
    // `gelu(linear(..))` folds the activation into the fused op — no separate
    // Gelu node.
    let g = rlx! {
        input x: [1, 8];
        param w: [4, 8];  param b: [4];
        let y = gelu(linear(x, w, b));
        out y;
    };
    assert_eq!(
        count(&g, |op| matches!(
            op,
            Op::FusedMatMulBiasAct {
                activation: Some(Activation::Gelu)
            }
        )),
        1
    );
    assert_eq!(count(&g, |op| matches!(op, Op::Activation(_))), 0);
}

#[test]
fn param_family_and_indexed_repeat() {
    // A 3-layer stack with DISTINCT per-layer weights via a family + `repeat i`.
    let g = rlx! {
        input x: [1, 8];
        param w[3]: [8, 8];
        param b[3]: [8];
        repeat i in 0..3 {
            let x = relu(linear(x, w[i], b[i]));
        }
        out x;
    };
    assert_eq!(count(&g, |op| matches!(op, Op::Param { .. })), 6); // 3 w + 3 b
    assert_eq!(
        count(&g, |op| matches!(op, Op::FusedMatMulBiasAct { .. })),
        3
    );
    let names: Vec<String> = g
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Param { name } => Some(name.clone()),
            _ => None,
        })
        .collect();
    assert!(names.contains(&"w_0".to_string()));
    assert!(names.contains(&"w_2".to_string()));
}

#[test]
fn param_ir_name_override_and_template() {
    // `@ "key"` overrides the IR/set_param name; `{i}` fills the family index.
    let g = rlx! {
        input x: [1, 8];
        param w[2] @ "layer.{i}.weight" : [8, 8];
        param b[2] @ "layer.{i}.bias" : [8];
        repeat i in 0..2 {
            let x = linear(x, w[i], b[i]);
        }
        out x;
    };
    let names: Vec<String> = g
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Param { name } => Some(name.clone()),
            _ => None,
        })
        .collect();
    assert!(names.contains(&"layer.0.weight".to_string()));
    assert!(names.contains(&"layer.1.weight".to_string()));
    assert!(names.contains(&"layer.0.bias".to_string()));
}

#[test]
fn tap_exposes_intermediate_as_output() {
    let g = rlx! {
        input x: [1, 8];
        param w: [8, 8];  param b: [8];
        let h = relu(linear(x, w, b));
        let y = linear(h, w, b);
        tap h;
        out y;
    };
    assert_eq!(g.outputs.len(), 2); // y, then the tapped h
}

#[test]
fn masked_attention_wires_four_inputs() {
    // `attention_masked` reaches the 4-input Custom Op::Attention (padding mask).
    let g = rlx! {
        input x: [1, 4, 8];
        input mask: [1, 4];
        param wq: [8, 8];  param wk: [8, 8];  param wv: [8, 8];
        let q = x @ wq;  let k = x @ wk;  let v = x @ wv;
        let a = q.attention_masked(k, v, mask, 2, 4);
        out a;
    };
    let attn = g
        .nodes()
        .iter()
        .find(|n| matches!(n.op, Op::Attention { .. }))
        .expect("attention op");
    assert_eq!(attn.inputs.len(), 4); // q, k, v, mask
}

#[test]
fn embed_and_tuple_split_sugar() {
    // `embed(table, ids)` = gather; `let (a,b) = split(x, axis, n)` destructures.
    let g = rlx! {
        input ids: [1, 4];
        param table: [100, 8];
        let e = embed(table, ids);       // [1, 4, 8]
        let (a, b) = split(e, 2, 2);     // → 2 × [1, 4, 4]
        out a, b;
    };
    assert_eq!(count(&g, |op| matches!(op, Op::Gather { .. })), 1);
    assert_eq!(g.outputs.len(), 2);
    assert_eq!(g.shape(g.outputs[0]).dim(2), Dim::Static(4));
}

#[test]
fn rope_sugar_builds_rope_op() {
    let g = rlx! {
        input x: [1, 4, 8];
        param cos: [4, 8];  param sin: [4, 8];
        let y = rope(x, cos, sin, 8);
        out y;
    };
    assert_eq!(count(&g, |op| matches!(op, Op::Rope { .. })), 1);
}

#[test]
fn rlx_expr_rust_bridge_drives_a_runtime_loop() {
    // `rlx_expr!` evaluates one DSL expression over Rust `Tensor` variables, so
    // an ordinary Rust `for` over a RUNTIME count builds the stack — something a
    // compile-time-unrolled `rlx! { repeat … }` can't express.
    use rlx_tensor::{graph_with, rlx_expr, shape};
    let n: usize = 3; // a runtime value
    let (g, ()) = graph_with("stack", |s| {
        let mut h = s.input("x", shape![1, 8]);
        let ws: Vec<_> = (0..n)
            .map(|i| s.param(format!("w{i}"), shape![8, 8]))
            .collect();
        let bs: Vec<_> = (0..n)
            .map(|i| s.param(format!("b{i}"), shape![8]))
            .collect();
        for i in 0..n {
            let (w, b) = (ws[i].clone(), bs[i].clone());
            h = rlx_expr!(relu(linear(h, w, b))); // Rust vars + fused-op sugar
        }
        s.set_outputs([h.id()]);
    });
    assert_eq!(
        count(&g, |op| matches!(op, Op::FusedMatMulBiasAct { .. })),
        3
    );
    assert_eq!(count(&g, |op| matches!(op, Op::Param { .. })), 6);
}

#[test]
fn fn_named_args_reorder_to_param_order() {
    // Named args (`lin(w: …, b: …, x: …)`) reorder to the fn's parameter order.
    let g = rlx! {
        fn lin(x, w, b) { let y = linear(x, w, b); }
        input a: [1, 4];
        param wt: [8, 4];  param bt: [8];
        let o = lin(w: wt, b: bt, x: a);   // out-of-order
        out o;
    };
    assert_eq!(
        count(&g, |op| matches!(op, Op::FusedMatMulBiasAct { .. })),
        1
    );
    assert_eq!(g.shape(g.outputs[0]).dim(1), Dim::Static(8)); // a·Wᵀ → [1, 8]
}

#[test]
fn array_const_materializes_shaped_constant() {
    let g = rlx! {
        input x: [2, 2];
        const mask = [[1.0, 0.0], [0.0, 1.0]] : F32;
        let y = x * mask;
        out y;
    };
    assert_eq!(count(&g, |op| matches!(op, Op::Constant { .. })), 1);
    // The constant carries its [2, 2] shape.
    let cst = g
        .nodes()
        .iter()
        .find(|n| matches!(n.op, Op::Constant { .. }))
        .unwrap();
    assert_eq!(cst.shape.dim(0), Dim::Static(2));
    assert_eq!(cst.shape.dim(1), Dim::Static(2));
}

// ── Training-DX sugar: loss + reduce-to-scalar ─────────────────────────────

/// Number of elements in a graph output shape (static dims; dynamic → 1).
fn out_numel(g: &Graph, out: usize) -> usize {
    let s = g.shape(g.outputs[out]);
    (0..s.rank())
        .map(|i| match s.dim(i) {
            Dim::Static(n) => n,
            _ => 1,
        })
        .product()
}

#[test]
fn cross_entropy_sugar_builds_soft_ce_op() {
    // `cross_entropy(logits, targets)` → one fused SoftmaxCrossEntropy, per-row [N].
    let g = rlx! {
        input logits: [4, 10];
        input tgt: [4, 10];
        let ce = cross_entropy(logits, tgt);
        out ce;
    };
    assert_eq!(count(&g, |op| matches!(op, Op::SoftmaxCrossEntropy)), 1);
    assert_eq!(g.shape(g.outputs[0]).dim(0), Dim::Static(4)); // per-row loss
}

#[test]
fn softmax_cross_entropy_sugar_builds_logits_ce_op() {
    // `softmax_cross_entropy(logits, labels)` (integer labels) → the *_with_logits op.
    let g = rlx! {
        input logits: [4, 10];
        input labels: [4];
        let ce = softmax_cross_entropy(logits, labels);
        out ce;
    };
    assert_eq!(
        count(&g, |op| matches!(op, Op::SoftmaxCrossEntropyWithLogits)),
        1
    );
}

#[test]
fn mean_and_sum_sugar_reduce_to_scalar() {
    // 1-arg `mean(x)` / `sum(x)` collapse EVERY axis to a scalar.
    let gm = rlx! { input x: [4, 8]; let m = mean(x); out m; };
    assert_eq!(out_numel(&gm, 0), 1);
    let gs = rlx! { input x: [4, 8]; let s = sum(x); out s; };
    assert_eq!(out_numel(&gs, 0), 1);
}

// ── By-value scalar method args (`~`) ──────────────────────────────────────

#[test]
fn tilde_passes_runtime_scalar_and_const_by_value() {
    // The method escape auto-borrows bare-ident args (`k` → `&k`) — right for a
    // tensor, wrong for a *scalar* variable/const, which must reach an `f32`
    // parameter BY VALUE (otherwise `expected f32, found &f32`). Prefixing `~`
    // forces by-value: `~eps` (a runtime var) and `~EPS` (a const) both feed
    // `layer_norm`'s `eps: f32`, and the exact value flows into the op.
    use rlx_tensor::{graph_with, rlx_expr, shape};
    const EPS: f32 = 2.5e-5;
    let eps = 1e-5f32; // a runtime value; the macro can't see its type
    let (g, ()) = graph_with("ln", |s| {
        let h = s.input("h", shape![2, 8]);
        let gamma = s.param("gamma", shape![8]);
        let beta = s.param("beta", shape![8]);
        let a = rlx_expr!(h.layer_norm(gamma, beta, ~eps)); // (a) runtime var
        let b = rlx_expr!(a.layer_norm(gamma, beta, ~EPS)); // (b) const
        s.set_outputs([b.id()]);
    });
    // Both norms built, each carrying its scalar eps through by value.
    let epsilons: Vec<f32> = g
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::LayerNorm { eps, .. } => Some(*eps),
            _ => None,
        })
        .collect();
    assert_eq!(epsilons.len(), 2, "both layer_norms present");
    assert!(epsilons.contains(&1e-5), "runtime var eps passed by value");
    assert!(epsilons.contains(&2.5e-5), "const eps passed by value");
}

#[test]
fn tilde_drives_a_config_attention_block_with_runtime_dims() {
    // A whole attention block whose head count / head dim come from RUNTIME
    // config values (not literals): `~num_heads` / `~head_dim` pass the `usize`s
    // by value, while `MaskKind::Causal` is a raw enum path and `k`/`v` are
    // auto-borrowed tensors. This is the pattern for a config-driven transformer.
    use rlx_tensor::{graph_with, rlx_expr, shape};
    let num_heads: usize = 8; // config-driven, not a literal `8`
    let head_dim: usize = 8;
    let (g, ()) = graph_with("attn", |s| {
        let x = s.input("x", shape![2, 16, 64]);
        let wq = s.param("wq", shape![64, 64]);
        let wk = s.param("wk", shape![64, 64]);
        let wv = s.param("wv", shape![64, 64]);
        let q = rlx_expr!(x @ wq);
        let k = rlx_expr!(x @ wk);
        let v = rlx_expr!(x @ wv);
        let a = rlx_expr!(q.attention(k, v, ~num_heads, ~head_dim, MaskKind::Causal));
        s.set_outputs([a.id()]);
    });
    assert_eq!(count(&g, |op| matches!(op, Op::Attention { .. })), 1);
    assert_eq!(count(&g, |op| matches!(op, Op::MatMul)), 3); // q, k, v projections
}

#[test]
fn tilde_in_full_graph_dsl_references_outer_scalar() {
    // `~` also works inside `rlx! { }`, passing an outer Rust scalar by value —
    // cleaner than the `(value)` escape and, unlike a bare ident, not rejected
    // as an unknown binding.
    let eps = 1e-6f32;
    let g = rlx! {
        input h: [2, 8];
        param gamma: [8];  param beta: [8];
        let y = h.layer_norm(gamma, beta, ~eps);
        out y;
    };
    assert_eq!(count(&g, |op| matches!(op, Op::LayerNorm { .. })), 1);
    let got = g
        .nodes()
        .iter()
        .find_map(|n| match &n.op {
            Op::LayerNorm { eps, .. } => Some(*eps),
            _ => None,
        })
        .unwrap();
    assert_eq!(got, 1e-6);
}

#[test]
fn dsl_language_model_loss_is_one_line() {
    // The whole point: a next-token loss is `mean(cross_entropy(logits, tgt))`.
    let g = rlx! {
        graph "lm_loss";
        input logits: [16, 256];
        input tgt: [16, 256];
        let loss = mean(cross_entropy(logits, tgt));
        out loss;
    };
    assert_eq!(count(&g, |op| matches!(op, Op::SoftmaxCrossEntropy)), 1);
    assert_eq!(out_numel(&g, 0), 1); // scalar loss
}

// ── Config-driven structure: runtime shape dims + runtime `repeat n` ────────

#[test]
fn runtime_shape_dims_size_inputs_and_params() {
    // A shape entry may be any in-scope `usize` value — the block is written
    // once and sized from config, not from literals baked into the source.
    let bt: usize = 4;
    let d: usize = 8;
    let ff: usize = d * 2;
    let g = rlx! {
        input x: [bt, d];
        param w1: [d, ff];  param b1: [ff];
        param w2: [ff, d];  param b2: [d];
        let h = gelu(x @ w1 + b1);
        let y = h @ w2 + b2;
        out y;
    };
    let out = g.shape(g.outputs[0]);
    assert_eq!(out.dim(0), Dim::Static(4)); // bt
    assert_eq!(out.dim(1), Dim::Static(8)); // d
    assert_eq!(count(&g, |op| matches!(op, Op::MatMul)), 2);
    // Literal and `?` dynamic forms keep working alongside runtime dims.
    let gd = rlx! { input x: [?, d]; param w: [d, 3]; let y = x @ w; };
    assert!(matches!(gd.shape(gd.outputs[0]).dim(0), Dim::Dynamic(_)));
}

#[test]
fn runtime_repeat_unrolls_a_config_depth_stack() {
    // `repeat n { … }` with a RUNTIME `n` emits a Rust `for` loop, so the depth
    // is config-driven (a literal `repeat` can't express this). The carried `x`
    // threads through iterations exactly like a literal unroll's shadowing.
    let n_layer: usize = 5; // from a config / CLI, not a literal in the block
    let g = rlx! {
        input x: [4, 16];
        param w: [16, 16];
        repeat n_layer {
            let x = x + relu(x @ w);
        }
        out x;
    };
    assert_eq!(count(&g, |op| matches!(op, Op::MatMul)), 5);
    assert_eq!(
        count(&g, |op| matches!(op, Op::Activation(Activation::Relu))),
        5
    );
    // A literal `repeat` still unrolls at macro time (unchanged semantics).
    let gl = rlx! {
        input x: [4, 16];  param w: [16, 16];
        repeat 3 { let x = x + relu(x @ w); }
        out x;
    };
    assert_eq!(count(&gl, |op| matches!(op, Op::MatMul)), 3);
}

#[test]
fn runtime_repeat_default_output_and_zero_iters() {
    // With no explicit `out`, the last loop-carried binding is the output; a
    // runtime zero count leaves the input untouched (the carry is re-exposed
    // regardless of iteration count).
    let zero: usize = 0;
    let g = rlx! {
        input x: [2, 8];  param w: [8, 8];
        repeat zero { let x = relu(x @ w); }
    };
    assert_eq!(g.outputs.len(), 1);
    assert_eq!(count(&g, |op| matches!(op, Op::MatMul)), 0);
    assert_eq!(g.shape(g.outputs[0]).dim(1), Dim::Static(8));
}

#[test]
fn config_driven_transformer_block_is_one_rlx_block() {
    // The headline: an entire config-driven GPT block — RUNTIME embed/head/ffn
    // dims AND a RUNTIME layer count — written as a single `rlx! { }` block.
    // `~nh`/`~dh`/`~eps` pass the `usize`/`f32` config values by value; the
    // repeated body threads the residual stream `x` across layers.
    let (b, s): (usize, usize) = (2, 4);
    let (nh, dh): (usize, usize) = (2, 4);
    let d = nh * dh; // n_embd
    let ff = 4 * d;
    let n_layer: usize = 3;
    let eps = 1e-5f32;

    let g = rlx! {
        graph "mini_gpt";
        input x: [b, s, d];
        param wq: [d, d];  param wk: [d, d];  param wv: [d, d];  param wo: [d, d];
        param g1: [d];  param b1: [d];
        param g2: [d];  param b2: [d];
        param w1: [d, ff];  param w2: [ff, d];

        repeat n_layer {
            let hn = x.layer_norm(g1, b1, ~eps);
            let q = hn @ wq;
            let k = hn @ wk;
            let v = hn @ wv;
            let attn = q.attention(k, v, ~nh, ~dh, MaskKind::Causal);
            let x = x + attn @ wo;
            let fn2 = x.layer_norm(g2, b2, ~eps);
            let ff1 = gelu(fn2 @ w1);
            let x = x + ff1 @ w2;
        }
        out x;
    };

    assert_eq!(g.name, "mini_gpt");
    assert_eq!(count(&g, |op| matches!(op, Op::Attention { .. })), n_layer);
    assert_eq!(
        count(&g, |op| matches!(op, Op::LayerNorm { .. })),
        2 * n_layer
    );
    assert_eq!(count(&g, |op| matches!(op, Op::MatMul)), 6 * n_layer);
    assert_eq!(
        count(&g, |op| matches!(op, Op::Activation(Activation::Gelu))),
        n_layer
    );
    let out = g.shape(g.outputs[0]);
    assert_eq!(out.dim(0), Dim::Static(2));
    assert_eq!(out.dim(1), Dim::Static(4));
    assert_eq!(out.dim(2), Dim::Static(8)); // d = nh * dh
}

// ── `bind`: adopt an outer-scope constant / param tensor ───────────────────

#[test]
fn bind_adopts_outer_constant_and_param_tensors() {
    // A baked u8 quantization-index constant and a codebook param are built
    // OUTSIDE the block, then adopted by name with `bind`, so `synth_matmul`
    // reads them as bare (auto-borrowed) arguments — the "functions not data"
    // quantized-matmul pattern in one `rlx! { }` block.
    use rlx_tensor::{DType, GraphScope, shape};
    let mut ext = GraphScope::new("ext");
    let indices = ext.constant_nd(vec![0.0, 1.0, 2.0, 3.0, 1.0, 0.0], vec![3, 2], DType::U8);
    let codebook = ext.param("codebook", shape![4, 2]);
    let entry_dim: u32 = 2;
    let num_entries: u32 = 4;

    let g = rlx! {
        bind indices, codebook;
        input x: [2, 4];
        let y = x.synth_matmul(indices, codebook, ~entry_dim, ~num_entries);
        out y;
    };
    assert_eq!(count(&g, |op| matches!(op, Op::SynthMatMul { .. })), 1);
    assert_eq!(g.shape(g.outputs[0]).dim(1), Dim::Static(3)); // n = indices rows

    // The same outer tensors also reach the method through the no-new-grammar
    // `(&t)` / `~&t` reference escape, without `bind`.
    let g2 = rlx! {
        input x: [2, 4];
        let y = x.synth_matmul((&indices), ~&codebook, 2u32, 4u32);
        out y;
    };
    assert_eq!(count(&g2, |op| matches!(op, Op::SynthMatMul { .. })), 1);
}

#[test]
fn indexed_collection_bind_builds_per_layer_synth_stack() {
    // A codebook-weight-synthesis stack with PER-LAYER distinct codebook + index
    // tensors — pre-built as outer Rust `Vec<Tensor>` collections — expressed as
    // ONE `rlx! { }` block. `bind cb[], idx[];` adopts the whole collections;
    // inside `repeat i in 0..n`, `cb[i]`/`idx[i]` adopt the i-th element (the
    // repeat index flows into the outer `Vec` access). This is the
    // "functions not data" quantized transformer written declaratively.
    use rlx_tensor::{DType, GraphScope, Tensor, shape};
    let n_layer = 2usize;
    let mut ext = GraphScope::new("ext");
    // idx[l]: [n, k/entry_dim] = [4, 2] (u8);  cb[l]: [num_entries, entry_dim] = [4, 2].
    let idx: Vec<Tensor> = (0..n_layer)
        .map(|l| {
            ext.constant_nd(
                vec![0.0, 1.0, 2.0, 3.0, 1.0, 0.0, 3.0, (l % 4) as f64],
                vec![4, 2],
                DType::U8,
            )
        })
        .collect();
    let cb: Vec<Tensor> = (0..n_layer)
        .map(|l| ext.param(format!("cb.{l}"), shape![4, 2]))
        .collect();
    let entry_dim: u32 = 2;
    let num_entries: u32 = 4;

    let g = rlx! {
        graph "synth_stack";
        input x: [2, 4];
        bind cb[], idx[];                          // adopt the outer collections
        let h = x;
        repeat i in 0..2 {
            // per-layer distinct codebook/index via the repeat index
            let h = h + x.synth_matmul(idx[i], cb[i], ~entry_dim, ~num_entries);
        }
        out h;
    };

    // One synth-matmul per layer, each reading its own (idx[i], cb[i]); the two
    // codebook params are distinct nodes adopted from the outer Vec.
    assert_eq!(
        count(&g, |op| matches!(op, Op::SynthMatMul { .. })),
        n_layer
    );
    assert_eq!(count(&g, |op| matches!(op, Op::Param { .. })), n_layer);
    assert_eq!(
        count(&g, |op| matches!(op, Op::Binary(BinaryOp::Add))),
        n_layer
    );
    let out = g.shape(g.outputs[0]);
    assert_eq!(out.dim(0), Dim::Static(2));
    assert_eq!(out.dim(1), Dim::Static(4)); // n = idx rows
}

#[test]
fn struct_collection_bind_with_field_access_builds_synth_stack() {
    // The verbosity-killer: instead of ~40 separate `bind cb_wq[], idx_wq[], …;`
    // collection binds, group every per-layer tensor into ONE `Vec<LayerParams>`
    // struct and write ONE `bind layers[];`. Inside `repeat i in 0..n`, a
    // collection index carrying a `.field` — `layers[i].idx`, `layers[i].cb`,
    // `layers[i].ln_g` — adopts that struct field's Tensor (auto-borrowed in
    // method-arg position, with the repeat index substituted per iteration).
    use rlx_tensor::{DType, GraphScope, Tensor, shape};

    #[derive(Clone)]
    struct LayerParams {
        idx: Tensor,
        cb: Tensor,
        ln_g: Tensor,
        ln_b: Tensor,
    }

    let n_layer = 2usize;
    let mut ext = GraphScope::new("ext");
    // idx: [n, k/entry_dim] = [4, 2] (u8);  cb: [num_entries, entry_dim] = [4, 2].
    let layers: Vec<LayerParams> = (0..n_layer)
        .map(|l| LayerParams {
            idx: ext.constant_nd(
                vec![0.0, 1.0, 2.0, 3.0, 1.0, 0.0, 3.0, (l % 4) as f64],
                vec![4, 2],
                DType::U8,
            ),
            cb: ext.param(format!("cb.{l}"), shape![4, 2]),
            ln_g: ext.param(format!("ln_g.{l}"), shape![4]),
            ln_b: ext.param(format!("ln_b.{l}"), shape![4]),
        })
        .collect();
    let ed: u32 = 2;
    let ne: u32 = 4;
    let eps = 1e-5f32;

    let g = rlx! {
        graph "synth_struct_stack";
        input x: [2, 4];
        bind layers[];                                  // ONE bind for the whole model
        let h = x;
        repeat i in 0..2 {
            // per-layer struct fields, indexed by the repeat index
            let s = h + x.synth_matmul(layers[i].idx, layers[i].cb, ~ed, ~ne);
            let h = s.layer_norm(layers[i].ln_g, layers[i].ln_b, ~eps);
        }
        out h;
    };

    assert_eq!(
        count(&g, |op| matches!(op, Op::SynthMatMul { .. })),
        n_layer
    );
    assert_eq!(count(&g, |op| matches!(op, Op::LayerNorm { .. })), n_layer);
    // Per layer: cb + ln_g + ln_b params (idx is a constant, not a Param).
    assert_eq!(count(&g, |op| matches!(op, Op::Param { .. })), 3 * n_layer);
    let out = g.shape(g.outputs[0]);
    assert_eq!(out.dim(0), Dim::Static(2));
    assert_eq!(out.dim(1), Dim::Static(4));
}

#[test]
fn struct_field_vec_indexed_by_inner_repeat_builds_residual_vq_stack() {
    // Residual multi-stage VQ: a `Vec<Tensor>` struct FIELD (`cb`/`idx`, one entry
    // per stage) indexed by an INNER `repeat s`, inside the OUTER `repeat i` over
    // layers — `layers[i].cb[s]`. Both indices are substituted per unrolled
    // iteration (nested repeats), and the indexed field element is adopted
    // (auto-borrowed in `synth_matmul`'s method-arg position). This is the
    // `W = Σ_s codebook_s[idx_s]` graph-live stage sum in ONE `rlx! { }` block.
    use rlx_tensor::{DType, GraphScope, Tensor, shape};

    #[derive(Clone)]
    struct LayerParams {
        idx: Vec<Tensor>, // per-stage baked u8 index constants
        cb: Vec<Tensor>,  // per-stage codebook params
    }

    let n_layer = 2usize;
    let n_stage = 2usize;
    let mut ext = GraphScope::new("ext");
    let layers: Vec<LayerParams> = (0..n_layer)
        .map(|l| LayerParams {
            idx: (0..n_stage)
                .map(|s| {
                    ext.constant_nd(
                        vec![0.0, 1.0, 2.0, 3.0, 1.0, 0.0, 3.0, ((l + s) % 4) as f64],
                        vec![4, 2],
                        DType::U8,
                    )
                })
                .collect(),
            cb: (0..n_stage)
                .map(|s| ext.param(format!("cb.{l}.{s}"), shape![4, 2]))
                .collect(),
        })
        .collect();
    let ed: u32 = 2;
    let ne: u32 = 4;

    let g = rlx! {
        graph "residual_vq_stack";
        input x: [2, 4];
        bind layers[];
        let acc = x;
        repeat i in 0..2 {
            repeat s in 0..2 {
                let acc = acc + x.synth_matmul(layers[i].idx[s], layers[i].cb[s], ~ed, ~ne);
            }
        }
        out acc;
    };

    // 2 layers × 2 stages = 4 synth-matmuls, each reading its own (idx[s], cb[s]);
    // 4 accumulate adds; 4 distinct codebook params adopted from the outer Vec.
    assert_eq!(count(&g, |op| matches!(op, Op::SynthMatMul { .. })), 4);
    assert_eq!(count(&g, |op| matches!(op, Op::Binary(BinaryOp::Add))), 4);
    assert_eq!(count(&g, |op| matches!(op, Op::Param { .. })), 4);
    let out = g.shape(g.outputs[0]);
    assert_eq!(out.dim(0), Dim::Static(2));
    assert_eq!(out.dim(1), Dim::Static(4));
}

#[test]
fn struct_field_vec_indexed_in_expr_position_adopts_inner_element() {
    // The same `layers[i].cb[s]` field-element access in EXPR position (`.clone()`
    // codegen, via `AdoptIndex` with a trailing inner index) rather than a
    // method-arg — a bare `Vec<Tensor>` field element used directly in arithmetic.
    use rlx_tensor::{GraphScope, Tensor, shape};

    #[derive(Clone)]
    struct LayerParams {
        cb: Vec<Tensor>,
    }

    let n_layer = 2usize;
    let n_stage = 2usize;
    let mut ext = GraphScope::new("ext");
    let layers: Vec<LayerParams> = (0..n_layer)
        .map(|l| LayerParams {
            cb: (0..n_stage)
                .map(|s| ext.param(format!("cb.{l}.{s}"), shape![4, 2]))
                .collect(),
        })
        .collect();

    let g = rlx! {
        graph "field_vec_expr";
        input x: [4, 2];
        bind layers[];
        let acc = x;
        repeat i in 0..2 {
            repeat s in 0..2 {
                let acc = acc + layers[i].cb[s]; // AdoptIndex expr position
            }
        }
        out acc;
    };

    // 4 adds (2×2), each adopting a distinct codebook param from the outer Vec.
    assert_eq!(count(&g, |op| matches!(op, Op::Binary(BinaryOp::Add))), 4);
    assert_eq!(count(&g, |op| matches!(op, Op::Param { .. })), 4);
    let out = g.shape(g.outputs[0]);
    assert_eq!(out.dim(0), Dim::Static(4));
    assert_eq!(out.dim(1), Dim::Static(2));
}
