// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! WebGL2 forward/backward entry points (browser).
//!
//! Routes through [`rlx_runtime::BrowserSession`] on `Device::OpenGl`.

use crate::mlp::{self, MlpDims};
use rlx_autodiff::grad_with_loss;
use rlx_runtime::{BrowserSession, Device};
use wasm_bindgen::prelude::*;

fn to_js<E: std::fmt::Display>(e: E) -> JsValue {
    JsValue::from_str(&e.to_string())
}

fn set_params(
    compiled: &mut rlx_runtime::BrowserCompiledGraph,
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
) {
    compiled.set_param("w1", w1);
    compiled.set_param("b1", b1);
    compiled.set_param("w2", w2);
    compiled.set_param("b2", b2);
}

/// MLP forward `relu(x·W1 + b1)·W2 + b2` on WebGL2. Returns the output row.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn mlp_forward_webgl(
    x: &[f32],
    in_dim: usize,
    hidden: usize,
    out_dim: usize,
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
) -> std::result::Result<Vec<f32>, JsValue> {
    let (g, _params) = mlp::build_forward(MlpDims {
        in_dim,
        hidden,
        out_dim,
    });
    let mut compiled = BrowserSession::new(Device::OpenGl)
        .map_err(to_js)?
        .compile(g)
        .map_err(to_js)?;
    set_params(&mut compiled, w1, b1, w2, b2);
    Ok(compiled
        .run(&[("x", x)])
        .into_iter()
        .next()
        .unwrap_or_default())
}

/// MLP forward + backward (MSE loss) on WebGL2. Returns a flat vector
/// `[loss, ∂/∂w1 …, ∂/∂b1 …, ∂/∂w2 …, ∂/∂b2 …]`.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn mlp_grads_webgl(
    x: &[f32],
    target: &[f32],
    in_dim: usize,
    hidden: usize,
    out_dim: usize,
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
) -> std::result::Result<Vec<f32>, JsValue> {
    let (fwd, params) = mlp::build_loss(MlpDims {
        in_dim,
        hidden,
        out_dim,
    });
    let bwd = grad_with_loss(&fwd, &params);
    let mut compiled = BrowserSession::new(Device::OpenGl)
        .map_err(to_js)?
        .compile(bwd)
        .map_err(to_js)?;
    set_params(&mut compiled, w1, b1, w2, b2);
    let outs = compiled.run(&[
        ("x", x),
        ("target", target),
        ("d_output", &[1.0]),
        ("w1", w1),
        ("b1", b1),
        ("w2", w2),
        ("b2", b2),
    ]);

    let mut flat = Vec::new();
    for o in &outs {
        flat.extend_from_slice(o);
    }
    Ok(flat)
}
