// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPL-3.0-only.

//! WebGL2 forward/backward entry points (browser).
//!
//! WebGL2 has no compute shaders, so [`rlx_webgl`] runs the same IR graphs via
//! render-to-texture fragment shaders. `readPixels` is synchronous, so unlike
//! the WebGPU path these are plain (non-async) functions. Errors surface to JS
//! as exceptions.
//!
//! The graphs are the same as the CPU/WebGPU paths ([`crate::mlp`]); backward
//! is just the autodiff gradient graph lowered to WebGL.

use crate::mlp::{self, MlpDims};
use rlx_autodiff::grad_with_loss;
use rlx_webgl::{build_plan, exec_gl::GlBackend};
use wasm_bindgen::prelude::*;

fn to_js<E: std::fmt::Display>(e: E) -> JsValue {
    JsValue::from_str(&e.to_string())
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
    let plan = build_plan(&g).map_err(to_js)?;
    let backend = GlBackend::new().map_err(to_js)?;
    let outs = backend
        .run(
            &plan,
            &[("x", x), ("w1", w1), ("b1", b1), ("w2", w2), ("b2", b2)],
        )
        .map_err(to_js)?;
    Ok(outs.into_iter().next().unwrap_or_default())
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
    let plan = build_plan(&bwd).map_err(to_js)?;
    let backend = GlBackend::new().map_err(to_js)?;
    let outs = backend
        .run(
            &plan,
            &[
                ("x", x),
                ("target", target),
                ("d_output", &[1.0]),
                ("w1", w1),
                ("b1", b1),
                ("w2", w2),
                ("b2", b2),
            ],
        )
        .map_err(to_js)?;

    let mut flat = Vec::new();
    for o in &outs {
        flat.extend_from_slice(o);
    }
    Ok(flat)
}
