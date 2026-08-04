// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Verifies the `rlx!` graph DSL resolves *through the umbrella crate* — both
//! as `rlx::rlx!` and `rlx::prelude::rlx!`. This is the payoff of the
//! `macro_rules!` wrapper: `$crate` resolves transitively, so the DSL works
//! even when `rlx-tensor` is only a transitive dependency (via `rlx`), not a
//! direct one.
//!
//! Needs the `tensor` DSL and the `cpu` backend; the umbrella defaults to no
//! features, so run with `cargo test -p rlx --features cpu,tensor`.
#![cfg(all(feature = "tensor", feature = "cpu"))]

#[test]
fn rlx_dsl_via_umbrella_root() {
    let g = rlx::rlx! {
        graph "umbrella";
        input x: [2, 4];
        param w: [4, 3];
        let y = gelu(x @ w);
        out y;
    };
    assert_eq!(g.name, "umbrella");
    assert_eq!(g.outputs.len(), 1);
}

#[test]
fn rlx_dsl_via_prelude() {
    use rlx::prelude::*;
    let g = rlx! {
        input x: [?, 8];
        param w: [8, 8];
        let y = x @ w;
    };
    assert_eq!(g.outputs.len(), 1);
}

/// The DSL graph doesn't just type-check — it compiles and runs. Inputs and
/// params are fed by their auto-derived names (`x`, `w`, `b`).
#[test]
fn rlx_dsl_compiles_and_runs() {
    use rlx::runtime::{Device, Session};

    let g = rlx::rlx! {
        input x: [1, 2];
        param w: [2, 2];
        param b: [2];
        let y = x @ w + b;
        out y;
    };

    let mut compiled = Session::new(Device::Cpu).compile(g);
    // x = [1, 2]; w = identity; b = [10, 20]  ⇒  x·w + b = [11, 22].
    compiled.set_param("w", &[1.0, 0.0, 0.0, 1.0]);
    compiled.set_param("b", &[10.0, 20.0]);
    let out = compiled.run(&[("x", &[1.0, 2.0][..])]);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0], vec![11.0, 22.0]);
}

/// The new comparison + `select` sugar computes real values: ReLU expressed as
/// `select(x > 0, x, 0)` runs on CPU and matches the elementwise definition.
#[test]
fn rlx_dsl_select_relu_runs() {
    use rlx::runtime::{Device, Session};

    let g = rlx::rlx! {
        input x: [4];
        let y = select(x > 0.0, x, 0.0);
        out y;
    };
    let mut compiled = Session::new(Device::Cpu).compile(g);
    let out = compiled.run(&[("x", &[-2.0, -0.5, 1.0, 3.0][..])]);
    assert_eq!(out[0], vec![0.0, 0.0, 1.0, 3.0]);
}

/// `clamp` sugar + a scalar `maximum` run and clip as expected.
#[test]
fn rlx_dsl_clamp_runs() {
    use rlx::runtime::{Device, Session};

    let g = rlx::rlx! {
        input x: [4];
        let y = clamp(x, 0.0, 2.0);
        out y;
    };
    let mut compiled = Session::new(Device::Cpu).compile(g);
    let out = compiled.run(&[("x", &[-1.0, 0.5, 1.5, 3.0][..])]);
    assert_eq!(out[0], vec![0.0, 0.5, 1.5, 2.0]);
}

/// An array `const` materializes its values and multiplies elementwise.
#[test]
fn rlx_dsl_array_const_runs() {
    use rlx::runtime::{Device, Session};

    let g = rlx::rlx! {
        input x: [2, 2];
        const mask = [[1.0, 0.0], [0.0, 1.0]] : F32;
        let y = x * mask;
        out y;
    };
    let mut compiled = Session::new(Device::Cpu).compile(g);
    let out = compiled.run(&[("x", &[1.0, 2.0, 3.0, 4.0][..])]);
    assert_eq!(out[0], vec![1.0, 0.0, 0.0, 4.0]);
}

/// A compact `scan` (`Op::Scan`) computes the same result as the equivalent
/// unrolled `repeat` — proving the sequential-loop lowering is correct.
#[test]
fn rlx_dsl_scan_matches_unrolled() {
    use rlx::runtime::{Device, Session};

    // hₜ₊₁ = relu(hₜ @ w), 4 steps.
    let g_scan = rlx::rlx! {
        input h0: [1, 2];
        param w: [2, 2];
        scan h = h0 for 4 { let h = relu(h @ w); }
        out h;
    };
    let g_unroll = rlx::rlx! {
        input h0: [1, 2];
        param w: [2, 2];
        repeat 4 { let h0 = relu(h0 @ w); }
        out h0;
    };

    let w = [0.5, 0.0, 0.0, 0.5];
    let h0 = [1.0, 1.0];
    let mut cs = Session::new(Device::Cpu).compile(g_scan);
    let mut cu = Session::new(Device::Cpu).compile(g_unroll);
    cs.set_param("w", &w);
    cu.set_param("w", &w);
    let out_scan = cs.run(&[("h0", &h0[..])]);
    let out_unroll = cu.run(&[("h0", &h0[..])]);

    // relu(0.5·1)⁴ = 0.0625 on each of the two lanes.
    assert_eq!(out_scan[0], vec![0.0625, 0.0625]);
    assert_eq!(out_scan[0], out_unroll[0]);
}

/// A `fn` block inlined and then run: `relu(x @ w)` via a reusable subgraph.
#[test]
fn rlx_dsl_fn_inline_runs() {
    use rlx::runtime::{Device, Session};

    let g = rlx::rlx! {
        fn lin(x, w) { let o = x @ w; }
        input x: [1, 2];
        param w: [2, 2];
        let y = relu(lin(x, w));
        out y;
    };
    let mut compiled = Session::new(Device::Cpu).compile(g);
    compiled.set_param("w", &[1.0, 0.0, 0.0, -1.0]);
    // x·w = [1, -2]  →  relu = [1, 0].
    let out = compiled.run(&[("x", &[1.0, 2.0][..])]);
    assert_eq!(out[0], vec![1.0, 0.0]);
}
