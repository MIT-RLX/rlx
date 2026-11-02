// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The row kernels (matmul, softmax, LayerNorm, RmsNorm) on widened slots, in
//! a browser, against the plan's CPU executor. A slot past the texture limit
//! (more than 8192 rows, or a row longer than 16384) is laid out wide, so
//! these kernels now read every operand by linear index; before, they needed
//! each slot's natural `rows × last-dim` layout, and a training graph (an
//! im2col matmul of 59392 rows, a 1.5 M-long weight-gradient operand) did not
//! plan. Natural-layout cases check the fast path is unchanged.
//!
//! ```sh
//! WASM_BINDGEN_USE_BROWSER=1 NO_HEADLESS=1 \
//!   CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner \
//!   cargo test -p rlx-webgl --target wasm32-unknown-unknown --test gl_layout_parity
//! ```
#![cfg(target_arch = "wasm32")]

use rlx_ir::{DType, Graph, GraphExt, Shape};
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

fn data(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 * 0.37 + seed).sin()) * 0.5)
        .collect()
}

/// `g` on WebGL against the CPU executor; `widened`: some slot must have left
/// its natural layout (the case under test).
fn check(label: &str, g: &Graph, inputs: &[(&str, Vec<f32>)], widened: bool) {
    let plan = rlx_webgl::build_plan(g).expect("plan");
    let side = plan.max_texture_side();
    assert!(side <= 16384, "{label}: a {side}-texel side");
    let natural = plan
        .slot_dims
        .iter()
        .zip(&plan.slot_len)
        .all(|(&(r, c), &len)| r * c == len);
    assert_eq!(
        !natural || !widened,
        true,
        "{label}: expected a widened slot"
    );
    let named: Vec<(&str, &[f32])> = inputs.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    let want = rlx_webgl::run_cpu(&plan, &named)
        .expect("run_cpu")
        .remove(0);
    let gl = rlx_webgl::exec_gl::GlBackend::new().expect("webgl2");
    let got = gl.run(&plan, &named).expect("run gl").remove(0);
    assert_eq!(got.len(), want.len(), "{label}");
    for (i, (a, e)) in got.iter().zip(&want).enumerate() {
        assert!(
            (a - e).abs() <= 1e-4 * (1.0 + e.abs()),
            "{label} [{i}]: webgl {a} cpu {e}"
        );
    }
}

fn mm(m: usize, k: usize, n: usize, widened: bool) {
    let f = DType::F32;
    let mut g = Graph::new("mm");
    let a = g.input("a", Shape::new(&[m, k], f));
    let b = g.input("b", Shape::new(&[k, n], f));
    let y = g.mm(a, b);
    let z = g.activation(rlx_ir::op::Activation::Tanh, y, Shape::new(&[m, n], f));
    g.set_outputs(vec![z]);
    check(
        &format!("mm {m}x{k}x{n}"),
        &g,
        &[("a", data(m * k, 0.1)), ("b", data(k * n, 0.7))],
        widened,
    );
}

#[wasm_bindgen_test]
fn matmul_on_widened_operands() {
    mm(19, 13, 7, false); // natural
    mm(20000, 25, 40, true); // tall A and output (an im2col conv)
    mm(8, 20000, 6, true); // K past the texture side
    mm(40, 59392, 3, true); // the weight-gradient shape: a 59392-long row
}

#[wasm_bindgen_test]
fn batched_matmul_on_widened_operands() {
    let f = DType::F32;
    let (bt, m, k, n) = (3usize, 4000usize, 9usize, 5usize);
    let mut g = Graph::new("bmm");
    let a = g.input("a", Shape::new(&[bt, m, k], f));
    let b = g.input("b", Shape::new(&[bt, k, n], f));
    let y = g.matmul(a, b, Shape::new(&[bt, m, n], f));
    g.set_outputs(vec![y]);
    check(
        "bmm",
        &g,
        &[("a", data(bt * m * k, 0.3)), ("b", data(bt * k * n, 0.9))],
        true,
    );
}

#[wasm_bindgen_test]
fn softmax_and_norms_on_widened_rows() {
    let f = DType::F32;
    for (rows, cols, widened) in [(5usize, 11usize, false), (3, 20000, true), (12000, 4, true)] {
        let sh = Shape::new(&[rows, cols], f);
        let mut g = Graph::new("softmax");
        let x = g.input("x", sh.clone());
        let y = g.softmax(x, -1, sh.clone());
        g.set_outputs(vec![y]);
        check(
            &format!("softmax {rows}x{cols}"),
            &g,
            &[("x", data(rows * cols, 0.2))],
            widened,
        );

        let mut g = Graph::new("ln");
        let x = g.input("x", sh.clone());
        let ga = g.input("g", Shape::new(&[cols], f));
        let be = g.input("b", Shape::new(&[cols], f));
        let y = g.layer_norm(x, ga, be, -1, 1e-5, sh.clone());
        g.set_outputs(vec![y]);
        let ins = [
            ("x", data(rows * cols, 0.4)),
            ("g", data(cols, 1.1)),
            ("b", data(cols, 1.9)),
        ];
        check(&format!("layernorm {rows}x{cols}"), &g, &ins, widened);

        let mut g = Graph::new("rms");
        let x = g.input("x", sh.clone());
        let ga = g.input("g", Shape::new(&[cols], f));
        let be = g.input("b", Shape::new(&[cols], f));
        let y = g.rms_norm(x, ga, be, 1e-6);
        g.set_outputs(vec![y]);
        check(&format!("rmsnorm {rows}x{cols}"), &g, &ins, widened);
    }
}
