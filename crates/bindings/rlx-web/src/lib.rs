// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! WebAssembly entry point for RLX.
//!
//! Compiles to `wasm32-unknown-unknown` and exposes a small JS-callable
//! surface via `wasm-bindgen`. Both **forward** (inference) and **backward**
//! (gradients via [`rlx_autodiff`]) passes are available.
//!
//! ## Compute paths
//!
//! - **CPU backend (default, works today):** [`mlp_forward`], [`mlp_loss`],
//!   [`mlp_grads`], [`mlp_train_step`] build an IR graph and run it through
//!   [`rlx_runtime::Session`] on the CPU backend. Fully synchronous — runs on
//!   the browser main thread (`rlx-cpu` is single-threaded on wasm).
//!
//! - **WebGPU (`webgpu` feature):** async forward/backward via
//!   `rlx_wgpu::WgpuExecutable::run_async` (see [`webgpu`]); GPU→CPU readback
//!   cannot block the browser event loop, so those entry points are `async`.
//!
//! The forward and backward graphs are identical across backends — only the
//! executor differs. The backward graph comes from
//! [`rlx_autodiff::grad_with_loss`]; its outputs are `[loss, ∂loss/∂p…]` and
//! it is seeded with a `d_output` input (the cotangent, `[1.0]` for a scalar
//! loss).

use rlx_autodiff::grad_with_loss;
use rlx_ir::Graph;
use rlx_runtime::{Device, Session};
use wasm_bindgen::prelude::*;

mod api;
mod exec;
mod mlp;
mod transformer;
mod vision;
use mlp::MlpDims;
use transformer::TfConfig;

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
pub mod webgpu;

#[cfg(all(feature = "webgl", target_arch = "wasm32"))]
pub mod webgl;

/// Module entry point — installs a panic hook that routes Rust panics to the
/// browser console. Runs automatically when the wasm module is instantiated.
#[wasm_bindgen(start)]
pub fn start() {
    #[cfg(target_arch = "wasm32")]
    install_panic_hook();
}

#[cfg(target_arch = "wasm32")]
fn install_panic_hook() {
    use std::sync::Once;
    static HOOK: Once = Once::new();
    HOOK.call_once(|| {
        std::panic::set_hook(Box::new(|info| {
            web_sys::console::error_1(&JsValue::from_str(&format!("rlx-web panic: {info}")));
        }));
    });
}

/// Label of the compute backend used by the synchronous entry points.
#[wasm_bindgen]
pub fn backend() -> String {
    "cpu".to_string()
}

// ── Real transformer (decoder-only) ─────────────────────────────────────

/// Run a real decoder-only transformer (RMSNorm + RoPE + causal multi-head
/// attention + SwiGLU) on the CPU backend with deterministic synthesized
/// weights, and return the **next-token logits** (last position), length
/// `vocab`. `tokens` are token ids (as f32); `seq` = `tokens.len()`.
/// `dim` must equal `n_heads * head_dim`.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn transformer_next_logits(
    tokens: &[f32],
    vocab: usize,
    dim: usize,
    n_layers: usize,
    n_heads: usize,
    head_dim: usize,
    ffn: usize,
    seed: u32,
) -> Vec<f32> {
    let cfg = TfConfig {
        vocab,
        dim,
        n_layers,
        n_heads,
        head_dim,
        ffn,
        seq: tokens.len().max(1),
        eps: 1e-5,
        theta: 10000.0,
    };
    let all = transformer::transformer_logits(&cfg, tokens, seed as u64);
    // Last position's row = next-token distribution.
    let start = (cfg.seq - 1) * vocab;
    all.get(start..start + vocab)
        .map(|s| s.to_vec())
        .unwrap_or_default()
}

// ── Forward (inference) ─────────────────────────────────────────────────

/// Run a two-layer MLP — `relu(x·W1 + b1)·W2 + b2` — on the CPU backend and
/// return the output row.
///
/// Shapes (row-major, batch size 1): `x`=`in_dim`, `w1`=`in_dim×hidden`,
/// `b1`=`hidden`, `w2`=`hidden×out_dim`, `b2`=`out_dim`.
#[wasm_bindgen]
pub fn mlp_forward(
    x: &[f32],
    in_dim: usize,
    hidden: usize,
    out_dim: usize,
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
) -> Vec<f32> {
    let dims = MlpDims {
        in_dim,
        hidden,
        out_dim,
    };
    let (g, _params) = mlp::build_forward(dims);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    set_params(&mut compiled, w1, b1, w2, b2);
    compiled
        .run(&[("x", x)])
        .into_iter()
        .next()
        .unwrap_or_default()
}

/// Mean-of-squares loss `Σ (y − target)²` for the MLP, on the CPU backend.
/// (Used for inference + as the scalar the backward pass differentiates.)
#[wasm_bindgen]
pub fn mlp_loss(
    x: &[f32],
    target: &[f32],
    in_dim: usize,
    hidden: usize,
    out_dim: usize,
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
) -> f32 {
    let dims = MlpDims {
        in_dim,
        hidden,
        out_dim,
    };
    let (g, _params) = mlp::build_loss(dims);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    set_params(&mut compiled, w1, b1, w2, b2);
    let outs = compiled.run(&[("x", x), ("target", target)]);
    outs.first().and_then(|o| o.first().copied()).unwrap_or(0.0)
}

// ── Backward (gradients) ────────────────────────────────────────────────

/// Forward + backward in one call on the CPU backend. Returns a flat vector:
/// `[loss, ∂/∂w1 …, ∂/∂b1 …, ∂/∂w2 …, ∂/∂b2 …]` (gradients in row-major
/// parameter order, same lengths as the inputs).
#[wasm_bindgen]
pub fn mlp_grads(
    x: &[f32],
    target: &[f32],
    in_dim: usize,
    hidden: usize,
    out_dim: usize,
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
) -> Vec<f32> {
    let dims = MlpDims {
        in_dim,
        hidden,
        out_dim,
    };
    let g = grads_graph(dims);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    set_params(&mut compiled, w1, b1, w2, b2);
    let outs = compiled.run(&[("x", x), ("target", target), ("d_output", &[1.0])]);

    // outs[0] = loss; outs[1..5] = grads for [w1, b1, w2, b2].
    let mut flat = Vec::new();
    for o in &outs {
        flat.extend_from_slice(o);
    }
    flat
}

/// One SGD step: compute gradients of the loss and return the updated
/// parameters as `[w1 …, b1 …, w2 …, b2 …]` (same layout as the inputs).
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn mlp_train_step(
    x: &[f32],
    target: &[f32],
    in_dim: usize,
    hidden: usize,
    out_dim: usize,
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
    lr: f32,
) -> Vec<f32> {
    let dims = MlpDims {
        in_dim,
        hidden,
        out_dim,
    };
    let g = grads_graph(dims);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    set_params(&mut compiled, w1, b1, w2, b2);
    let outs = compiled.run(&[("x", x), ("target", target), ("d_output", &[1.0])]);

    // outs[1..5] = grads for [w1, b1, w2, b2]; apply p -= lr * grad.
    let mut updated = Vec::new();
    for (param, grad) in [w1, b1, w2, b2].iter().zip(&outs[1..]) {
        updated.extend(param.iter().zip(grad.iter()).map(|(p, gd)| p - lr * gd));
    }
    updated
}

// ── Shared helpers ──────────────────────────────────────────────────────

/// Build the autodiff backward graph for the MLP MSE loss w.r.t. all four
/// parameters. Outputs: `[loss, ∂/∂w1, ∂/∂b1, ∂/∂w2, ∂/∂b2]`.
fn grads_graph(dims: MlpDims) -> Graph {
    let (fwd, params) = mlp::build_loss(dims);
    grad_with_loss(&fwd, &params)
}

fn set_params(
    compiled: &mut rlx_runtime::CompiledGraph,
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

/// Bring up a WebGPU device in the browser (async). Returns `true` when an
/// adapter + device were obtained. See [`webgpu`] for the GPU forward/backward
/// entry points.
#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen]
pub async fn init_webgpu() -> bool {
    rlx_runtime::init_webgpu().await
}

pub use api::{
    VisionBench, VisionModelInfo, list_vision_models, parse_backend, preferred_backend,
    vision_model_info,
};

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
    }

    #[test]
    fn mlp_forward_runs_on_cpu() {
        // x·I = x; relu([1, -2]) = [1, 0]; [1, 0]·[1; 1] = 1.
        let out = mlp_forward(
            &[1.0, -2.0],
            2,
            2,
            1,
            &[1.0, 0.0, 0.0, 1.0],
            &[0.0, 0.0],
            &[1.0, 1.0],
            &[0.0],
        );
        assert_eq!(out.len(), 1);
        assert!((out[0] - 1.0).abs() < 1e-5, "got {out:?}");
    }

    #[test]
    fn mlp_grads_match_finite_differences() {
        let (in_dim, hidden, out_dim) = (2usize, 3usize, 2usize);
        let x = vec![0.5_f32, -0.3];
        let target = vec![0.2_f32, 0.7];
        // Deterministic non-degenerate weights (some pre-activations > 0 so
        // ReLU has live gradients).
        let w1: Vec<f32> = (0..in_dim * hidden)
            .map(|i| 0.1 * (i as f32 + 1.0) - 0.2)
            .collect();
        let b1 = vec![0.05_f32, -0.1, 0.2];
        let w2: Vec<f32> = (0..hidden * out_dim)
            .map(|i| 0.15 * (i as f32) - 0.1)
            .collect();
        let b2 = vec![0.0_f32, 0.1];

        let g = mlp_grads(&x, &target, in_dim, hidden, out_dim, &w1, &b1, &w2, &b2);
        // [loss, w1(6), b1(3), w2(6), b2(2)]
        assert_eq!(g.len(), 1 + 6 + 3 + 6 + 2);
        let analytic = &g[1..];

        // Finite-difference every parameter element and compare.
        let mut params = [w1.clone(), b1.clone(), w2.clone(), b2.clone()];
        let eps = 1e-3_f32;
        let mut idx = 0;
        for p in 0..4 {
            for e in 0..params[p].len() {
                let orig = params[p][e];
                params[p][e] = orig + eps;
                let lp = mlp_loss(
                    &x, &target, in_dim, hidden, out_dim, &params[0], &params[1], &params[2],
                    &params[3],
                );
                params[p][e] = orig - eps;
                let lm = mlp_loss(
                    &x, &target, in_dim, hidden, out_dim, &params[0], &params[1], &params[2],
                    &params[3],
                );
                params[p][e] = orig;

                let fd = (lp - lm) / (2.0 * eps);
                assert!(
                    approx(analytic[idx], fd, 2e-2),
                    "param {p}[{e}]: analytic={} fd={}",
                    analytic[idx],
                    fd
                );
                idx += 1;
            }
        }
    }

    #[test]
    fn train_step_decreases_loss() {
        let (in_dim, hidden, out_dim) = (3usize, 4usize, 2usize);
        let x = vec![0.4_f32, -0.2, 0.9];
        let target = vec![1.0_f32, -0.5];
        let mut w1: Vec<f32> = (0..in_dim * hidden)
            .map(|i| 0.1 * (i as f32 % 5.0) - 0.2)
            .collect();
        let mut b1 = vec![0.0_f32; hidden];
        let mut w2: Vec<f32> = (0..hidden * out_dim)
            .map(|i| 0.05 * (i as f32) - 0.1)
            .collect();
        let mut b2 = vec![0.0_f32; out_dim];

        let l0 = mlp_loss(&x, &target, in_dim, hidden, out_dim, &w1, &b1, &w2, &b2);
        for _ in 0..50 {
            let up = mlp_train_step(
                &x, &target, in_dim, hidden, out_dim, &w1, &b1, &w2, &b2, 0.05,
            );
            let (a, b) = up.split_at(in_dim * hidden);
            w1 = a.to_vec();
            let (a, b) = b.split_at(hidden);
            b1 = a.to_vec();
            let (a, b) = b.split_at(hidden * out_dim);
            w2 = a.to_vec();
            b2 = b.to_vec();
        }
        let l1 = mlp_loss(&x, &target, in_dim, hidden, out_dim, &w1, &b1, &w2, &b2);
        assert!(l1 < l0 * 0.5, "loss did not drop enough: {l0} -> {l1}");
    }
}
