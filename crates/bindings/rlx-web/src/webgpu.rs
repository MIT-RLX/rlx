// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPL-3.0-only.

//! WebGPU forward/backward entry points (browser, async).
//!
//! Routes through [`rlx_runtime::BrowserSession`] so device selection and
//! preflight match the unified browser runtime.

use crate::mlp::{self, MlpDims};
use rlx_autodiff::grad_with_loss;
use rlx_runtime::BrowserSession;
use wasm_bindgen::prelude::*;

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

/// MLP forward `relu(x·W1 + b1)·W2 + b2` on WebGPU. Async — resolves to the
/// output row.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub async fn mlp_forward_gpu(
    x: Vec<f32>,
    in_dim: usize,
    hidden: usize,
    out_dim: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
) -> Result<Vec<f32>, JsValue> {
    let dims = MlpDims {
        in_dim,
        hidden,
        out_dim,
    };
    let (g, _params) = mlp::build_forward(dims);
    let mut compiled = BrowserSession::new(rlx_runtime::Device::WebGpu)
        .map_err(|e| JsValue::from_str(&e.to_string()))?
        .compile(g)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    set_params(&mut compiled, &w1, &b1, &w2, &b2);
    compiled
        .run_async(&[("x", &x)])
        .await
        .map(|outs| outs.into_iter().next().unwrap_or_default())
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// MLP forward + backward (MSE loss) on WebGPU. Async — resolves to a flat
/// vector `[loss, ∂/∂w1 …, ∂/∂b1 …, ∂/∂w2 …, ∂/∂b2 …]`.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub async fn mlp_grads_gpu(
    x: Vec<f32>,
    target: Vec<f32>,
    in_dim: usize,
    hidden: usize,
    out_dim: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
) -> Result<Vec<f32>, JsValue> {
    let dims = MlpDims {
        in_dim,
        hidden,
        out_dim,
    };
    let (fwd, params) = mlp::build_loss(dims);
    let bwd = grad_with_loss(&fwd, &params);

    let mut compiled = BrowserSession::new(rlx_runtime::Device::WebGpu)
        .map_err(|e| JsValue::from_str(&e.to_string()))?
        .compile(bwd)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    set_params(&mut compiled, &w1, &b1, &w2, &b2);
    let outs = compiled
        .run_async(&[("x", &x), ("target", &target), ("d_output", &[1.0])])
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let mut flat = Vec::new();
    for o in &outs {
        flat.extend_from_slice(o);
    }
    Ok(flat)
}
