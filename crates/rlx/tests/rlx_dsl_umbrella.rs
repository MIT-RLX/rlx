// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Verifies the `rlx!` graph DSL resolves *through the umbrella crate* — both
//! as `rlx::rlx!` and `rlx::prelude::rlx!`. This is the payoff of the
//! `macro_rules!` wrapper: `$crate` resolves transitively, so the DSL works
//! even when `rlx-tensor` is only a transitive dependency (via `rlx`), not a
//! direct one.

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
