// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! End-to-end: `#[rlx_model(check)]` must expand, trace the graph, run the
//! injected `model_self_check` hook, and compile — all without a panic on a
//! clean model.

use rlx_ir::DType;
use rlx_macros::rlx_model;
use rlx_runtime::trace::{TracedTensor, Tracer};

#[rlx_model(check)]
fn tiny_mlp(t: &Tracer) -> Vec<TracedTensor> {
    let x = t.input("x", &[2, 4], DType::F32);
    let w = t.param("w", &[4, 4], DType::F32);
    vec![t.matmul(x, w)]
}

// A model WITHOUT the opt-in — proves the macro still expands the plain form.
#[rlx_model]
fn tiny_plain(t: &Tracer) -> Vec<TracedTensor> {
    let x = t.input("x", &[2, 4], DType::F32);
    let w = t.param("w", &[4, 4], DType::F32);
    vec![t.matmul(x, w)]
}

#[test]
fn checked_model_traces_hooks_and_compiles() {
    // Triggers trace → injected model_self_check(&graph) → compile. The clean
    // MLP yields no error-level findings, so the default hook must not panic.
    let _compiled = tiny_mlp_compiled();
}

#[test]
fn unchecked_model_still_builds() {
    let _compiled = tiny_plain_compiled();
}
