// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPL-3.0-only.

//! WebGPU forward/backward entry points (browser, async).
//!
//! GPU→CPU readback cannot block the browser event loop, so these are `async`
//! (they `.await` `WgpuExecutable::run_async`). Call [`crate::init_webgpu`]
//! once before using them so the device is ready.
//!
//! The graphs are the same as the CPU path ([`crate::mlp`]); only the executor
//! differs. Backward is just the autodiff gradient graph compiled + run on the
//! GPU, so forward and backward share all machinery.

use crate::mlp::{self, MlpDims};
use rlx_autodiff::grad_with_loss;
use rlx_wgpu::backend::WgpuExecutable;
use wasm_bindgen::prelude::*;

fn set_params(exe: &mut WgpuExecutable, w1: &[f32], b1: &[f32], w2: &[f32], b2: &[f32]) {
    exe.set_param("w1", w1);
    exe.set_param("b1", b1);
    exe.set_param("w2", w2);
    exe.set_param("b2", b2);
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
) -> Vec<f32> {
    let dims = MlpDims {
        in_dim,
        hidden,
        out_dim,
    };
    let (g, _params) = mlp::build_forward(dims);

    let mut exe = WgpuExecutable::compile(g);
    set_params(&mut exe, &w1, &b1, &w2, &b2);
    exe.run_async(&[("x", &x)])
        .await
        .into_iter()
        .next()
        .unwrap_or_default()
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
) -> Vec<f32> {
    let dims = MlpDims {
        in_dim,
        hidden,
        out_dim,
    };
    let (fwd, params) = mlp::build_loss(dims);
    let bwd = grad_with_loss(&fwd, &params);

    let mut exe = WgpuExecutable::compile(bwd);
    set_params(&mut exe, &w1, &b1, &w2, &b2);
    let outs = exe
        .run_async(&[("x", &x), ("target", &target), ("d_output", &[1.0])])
        .await;

    let mut flat = Vec::new();
    for o in &outs {
        flat.extend_from_slice(o);
    }
    flat
}
