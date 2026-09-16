// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `jvp(hvp(f))` — forward-over-forward-over-reverse for a third derivative.
//!
//! `hvp` *is* a `jvp`, so its result already declares `tangent_<name>`. The
//! outer `jvp` then declared a second leaf of that name, and a duplicate leaf
//! name is not an error anywhere in the stack: binding resolves by name to a
//! single node, so the second declaration was never bound, read zeros, and the
//! third derivative came back exactly zero. `rlx_ir::verify_unique_leaf_names`
//! is what surfaced it.
//!
//! The check is a scalar-per-element polynomial where every order is known in
//! closed form, and it asserts the *second* order too — a fix that broke the
//! Hessian while unblocking the third derivative would otherwise pass.

#![cfg(feature = "cpu")]

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Shape};
use rlx_opt::autodiff_fwd::{hvp, jvp_with_tangent_names};
use rlx_runtime::{Device, Session};

fn f64s_to_bytes(xs: &[f64]) -> Vec<u8> {
    let mut o = Vec::with_capacity(xs.len() * 8);
    for x in xs {
        o.extend_from_slice(&x.to_le_bytes());
    }
    o
}

fn bytes_to_f64s(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// `f(x) = Σ xᵢ⁴` — ∇f = 4x³, H·v = 12x²v, and ∂(H·v)/∂x · w = 24xvw.
fn quartic(n: usize) -> Graph {
    let f64s = Shape::new(&[n], DType::F64);
    let mut g = Graph::new("quartic");
    let x = g.input("x", f64s.clone());
    let x2 = g.binary(rlx_ir::op::BinaryOp::Mul, x, x, f64s.clone());
    let x4 = g.binary(rlx_ir::op::BinaryOp::Mul, x2, x2, f64s);
    let f = g.sum(x4, vec![0], false);
    g.set_outputs(vec![f]);
    g
}

#[test]
fn jvp_over_hvp_gives_the_third_derivative() {
    let n = 4;
    let fwd = quartic(n);
    let x_id = fwd
        .nodes()
        .iter()
        .find(|node| matches!(&node.op, rlx_ir::Op::Input { name } if name == "x"))
        .expect("x")
        .id;

    let hess = hvp(&fwd, &[x_id]);
    let hess_x = hess
        .nodes()
        .iter()
        .find(|node| matches!(&node.op, rlx_ir::Op::Input { name } if name == "x"))
        .expect("x survives into the hvp graph")
        .id;

    let (third, tangent_names) = jvp_with_tangent_names(&hess, &[hess_x]);
    assert_eq!(
        tangent_names,
        vec!["tangent_x_2".to_string()],
        "the outer jvp must not reuse the name hvp already took"
    );
    assert!(
        rlx_ir::verify_unique_leaf_names(&third).is_empty(),
        "duplicate leaf names in the composed graph: {:?}",
        rlx_ir::verify_unique_leaf_names(&third)
    );

    let x = vec![1.0, 2.0, 3.0, 0.5];
    let v = vec![0.5, -0.25, 1.0, -1.5];
    let w = vec![2.0, 1.0, -0.5, 3.0];

    let mut compiled = Session::new(Device::Cpu).compile(third);
    let outs = compiled.run_typed(&[
        ("x", &f64s_to_bytes(&x), DType::F64),
        ("tangent_x", &f64s_to_bytes(&v), DType::F64),
        ("tangent_x_2", &f64s_to_bytes(&w), DType::F64),
    ]);

    // `hvp` outputs `[primal_f, grad, tangent_f, H·v]`; `jvp` returns
    // `[primals…, tangents…]`, so index 3 is H·v and index 4+3 = 7 is its
    // directional derivative.
    assert_eq!(
        outs.len(),
        8,
        "jvp of a 4-output graph must yield 8 outputs"
    );
    let hv = bytes_to_f64s(&outs[3].0);
    let third_order = bytes_to_f64s(&outs[7].0);

    for i in 0..n {
        let want_hv = 12.0 * x[i] * x[i] * v[i];
        assert!(
            (hv[i] - want_hv).abs() < 1e-9,
            "H·v[{i}]: got {}, want {want_hv}",
            hv[i]
        );
        let want = 24.0 * x[i] * v[i] * w[i];
        assert!(
            (third_order[i] - want).abs() < 1e-9,
            "third derivative[{i}]: got {}, want {want}",
            third_order[i]
        );
    }

    let magnitude = third_order.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(
        magnitude > 1e-6,
        "third derivative is ~zero ({magnitude}) — the duplicate-leaf failure \
         mode is back"
    );
}
