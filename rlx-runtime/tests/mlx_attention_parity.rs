// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Licensed under the GNU General Public License, version 3.

//! MLX scaled dot-product attention parity vs CPU.
//!
//! Mirrors [`mps_attention_parity`] for Apple MLX. Includes a stacked
//! ViT-style predictor graph shaped like Brain-JEPA (`pred_dim=384`,
//! `depth=6`, `heads=12`, BSNH layout).
//!
//! **Note:** pretrained Brain-JEPA drift (~3% predictor out) was traced to MLX
//! lowering `Activation::GeluApprox` through exact `gelu` instead of the tanh
//! approximation (`rlx-mlx` `ops::gelu_approx`). Re-run `brainjepa`'s
//! `mlx_matches_cpu` after bumping `rlx` to confirm end-to-end.
//!
//! ```text
//! cargo test -p rlx-runtime --features cpu,mlx --test mlx_attention_parity -- --nocapture
//! MLX_ATTN_PARITY_SEQ=4096 cargo test -p rlx-runtime --features cpu,mlx \
//!     --test mlx_attention_parity cpu_vs_mlx_brainjepa_predictor_stack --release -- --nocapture
//! ```

#![cfg(all(feature = "mlx", feature = "cpu", rlx_mlx_host))]

use rlx_ir::infer::GraphExt;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{CompileOptions, Device, Session, is_available};

/// Brain-JEPA cross-backend tolerance (encoder + predictor ctx).
const TOL: f32 = 5e-3;

fn s1(d: usize) -> Shape {
    Shape::new(&[d], DType::F32)
}
fn s2(a: usize, b: usize) -> Shape {
    Shape::new(&[a, b], DType::F32)
}
fn s4(a: usize, b: usize, c: usize, d: usize) -> Shape {
    Shape::new(&[a, b, c, d], DType::F32)
}

fn det_random(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (state >> 33) as f32 / (1u64 << 31) as f32;
            (u - 0.5) * 2.0 * scale
        })
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn skip_mlx() -> bool {
    if !is_available(Device::Mlx) {
        eprintln!("[mlx_attention_parity] MLX unavailable — skipping");
        return true;
    }
    false
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn seq_len() -> usize {
    std::env::var("MLX_ATTN_PARITY_SEQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128)
}

fn weight_scale() -> f32 {
    env_f32("MLX_ATTN_PARITY_WSCALE", 0.02)
}

fn token_scale() -> f32 {
    env_f32("MLX_ATTN_PARITY_XSCALE", 0.05)
}

/// One Brain-JEPA predictor block: pre-norm attention + MLP (GeluApprox), BSNH SDPA.
fn pred_attn_block(
    g: &mut Graph,
    x: rlx_ir::NodeId,
    b: usize,
    n: usize,
    d: usize,
    nh: usize,
    dh: usize,
    hidden: usize,
    norm_eps: f32,
    layer: usize,
) -> rlx_ir::NodeId {
    let p = format!("blk{layer}");

    let ln1_w = g.param(format!("{p}.norm1.weight"), s1(d));
    let ln1_b = g.param(format!("{p}.norm1.bias"), s1(d));
    let xn = g.ln(x, ln1_w, ln1_b, norm_eps);

    let qkv_w = g.param(format!("{p}.attn.qkv.weight"), s2(d, 3 * d));
    let qkv_b = g.param(format!("{p}.attn.qkv.bias"), s1(3 * d));
    let qkv_mm = g.mm(xn, qkv_w);
    let qkv = g.add(qkv_mm, qkv_b);
    let qkv5 = g.reshape_(qkv, vec![b as i64, n as i64, 3, nh as i64, dh as i64]);

    let q5 = g.narrow_(qkv5, 2, 0, 1);
    let k5 = g.narrow_(qkv5, 2, 1, 1);
    let v5 = g.narrow_(qkv5, 2, 2, 1);
    let q = g.reshape_(q5, vec![b as i64, n as i64, nh as i64, dh as i64]);
    let k = g.reshape_(k5, vec![b as i64, n as i64, nh as i64, dh as i64]);
    let v = g.reshape_(v5, vec![b as i64, n as i64, nh as i64, dh as i64]);

    let attn = g.attention_kind(q, k, v, nh, dh, MaskKind::None, s4(b, n, nh, dh));
    let attn3 = g.reshape_(attn, vec![b as i64, n as i64, d as i64]);

    let proj_w = g.param(format!("{p}.attn.proj.weight"), s2(d, d));
    let proj_b = g.param(format!("{p}.attn.proj.bias"), s1(d));
    let proj_mm = g.mm(attn3, proj_w);
    let attn_out = g.add(proj_mm, proj_b);
    let x = g.add(x, attn_out);

    let ln2_w = g.param(format!("{p}.norm2.weight"), s1(d));
    let ln2_b = g.param(format!("{p}.norm2.bias"), s1(d));
    let hn = g.ln(x, ln2_w, ln2_b, norm_eps);

    let fc1_w = g.param(format!("{p}.mlp.fc1.weight"), s2(d, hidden));
    let fc1_b = g.param(format!("{p}.mlp.fc1.bias"), s1(hidden));
    let fc2_w = g.param(format!("{p}.mlp.fc2.weight"), s2(hidden, d));
    let fc2_b = g.param(format!("{p}.mlp.fc2.bias"), s1(d));

    let fc1_mm = g.mm(hn, fc1_w);
    let m1 = g.add(fc1_mm, fc1_b);
    let act = g.gelu_approx(m1);
    let fc2_mm = g.mm(act, fc2_w);
    let m2 = g.add(fc2_mm, fc2_b);
    g.add(x, m2)
}

/// Stacked predictor transformer + final LN + narrow pred tokens + proj to encoder dim.
fn build_brainjepa_predictor(
    b: usize,
    n: usize,
    n_ctx: usize,
    n_pred: usize,
    d: usize,
    enc_dim: usize,
    depth: usize,
    nh: usize,
    mlp_ratio: f64,
    norm_eps: f32,
) -> Graph {
    let dh = d / nh;
    let hidden = (d as f64 * mlp_ratio) as usize;
    let mut g = Graph::new("brainjepa_predictor_parity");

    let mut x = g.input("tokens", Shape::new(&[b, n, d], DType::F32));
    for layer in 0..depth {
        x = pred_attn_block(&mut g, x, b, n, d, nh, dh, hidden, norm_eps, layer);
    }

    let ln_w = g.param("predictor_norm.weight", s1(d));
    let ln_b = g.param("predictor_norm.bias", s1(d));
    x = g.ln(x, ln_w, ln_b, norm_eps);
    x = g.narrow_(x, 1, n_ctx, n_pred);

    let proj_w = g.param("predictor_proj.weight", s2(d, enc_dim));
    let proj_b = g.param("predictor_proj.bias", s1(enc_dim));
    let proj_mm = g.mm(x, proj_w);
    let out = g.add(proj_mm, proj_b);
    g.set_outputs(vec![out]);
    g
}

fn upload_block_params(
    compiled: &mut rlx_runtime::CompiledGraph,
    layer: usize,
    d: usize,
    hidden: usize,
    wscale: f32,
) {
    let p = format!("blk{layer}");
    let seed = 1000 + layer as u64;
    let bscale = wscale * 0.05;
    compiled.set_param(&format!("{p}.norm1.weight"), &vec![1.0; d]);
    compiled.set_param(&format!("{p}.norm1.bias"), &vec![0.0; d]);
    compiled.set_param(
        &format!("{p}.attn.qkv.weight"),
        &det_random(seed, d * 3 * d, wscale),
    );
    compiled.set_param(
        &format!("{p}.attn.qkv.bias"),
        &det_random(seed + 1, 3 * d, bscale),
    );
    compiled.set_param(
        &format!("{p}.attn.proj.weight"),
        &det_random(seed + 2, d * d, wscale),
    );
    compiled.set_param(
        &format!("{p}.attn.proj.bias"),
        &det_random(seed + 3, d, bscale),
    );
    compiled.set_param(&format!("{p}.norm2.weight"), &vec![1.0; d]);
    compiled.set_param(&format!("{p}.norm2.bias"), &vec![0.0; d]);
    compiled.set_param(
        &format!("{p}.mlp.fc1.weight"),
        &det_random(seed + 4, d * hidden, wscale),
    );
    compiled.set_param(
        &format!("{p}.mlp.fc1.bias"),
        &det_random(seed + 5, hidden, bscale),
    );
    compiled.set_param(
        &format!("{p}.mlp.fc2.weight"),
        &det_random(seed + 6, hidden * d, wscale),
    );
    compiled.set_param(
        &format!("{p}.mlp.fc2.bias"),
        &det_random(seed + 7, d, bscale),
    );
}

fn upload_predictor_params(
    compiled: &mut rlx_runtime::CompiledGraph,
    depth: usize,
    d: usize,
    enc_dim: usize,
    hidden: usize,
    wscale: f32,
) {
    for layer in 0..depth {
        upload_block_params(compiled, layer, d, hidden, wscale);
    }
    compiled.set_param("predictor_norm.weight", &vec![1.0; d]);
    compiled.set_param("predictor_norm.bias", &vec![0.0; d]);
    compiled.set_param(
        "predictor_proj.weight",
        &det_random(9000, d * enc_dim, wscale),
    );
    compiled.set_param(
        "predictor_proj.bias",
        &det_random(9001, enc_dim, wscale * 0.05),
    );
}

fn run_predictor_stack(
    device: Device,
    g: Graph,
    tokens: &[f32],
    depth: usize,
    d: usize,
    enc_dim: usize,
    hidden: usize,
    wscale: f32,
) -> Vec<f32> {
    let mut compiled = Session::new(device).compile_with(g, &CompileOptions::default());
    upload_predictor_params(&mut compiled, depth, d, enc_dim, hidden, wscale);
    compiled
        .run(&[("tokens", tokens)])
        .into_iter()
        .next()
        .expect("one output")
}

#[test]
fn cpu_vs_mlx_attention_bsnh_single_layer() {
    if skip_mlx() {
        return;
    }

    let (b, n, nh, dh) = (1, 8, 12, 32);
    let mut g = Graph::new("attn_bsnh");
    let q = g.input("q", Shape::new(&[b, n, nh, dh], DType::F32));
    let k = g.input("k", Shape::new(&[b, n, nh, dh], DType::F32));
    let v = g.input("v", Shape::new(&[b, n, nh, dh], DType::F32));
    let out = g.attention_kind(q, k, v, nh, dh, MaskKind::None, s4(b, n, nh, dh));
    g.set_outputs(vec![out]);

    let qd = det_random(11, b * n * nh * dh, 0.1);
    let kd = det_random(22, b * n * nh * dh, 0.1);
    let vd = det_random(33, b * n * nh * dh, 0.1);

    let cpu = Session::new(Device::Cpu)
        .compile_with(g.clone(), &CompileOptions::default())
        .run(&[("q", &qd), ("k", &kd), ("v", &vd)])
        .remove(0);
    let mlx = Session::new(Device::Mlx)
        .compile_with(g, &CompileOptions::default())
        .run(&[("q", &qd), ("k", &kd), ("v", &vd)])
        .remove(0);

    let diff = max_abs_diff(&cpu, &mlx);
    eprintln!("[mlx-attn BSNH S={n}] max_abs = {diff:e} (tol {TOL})");
    assert!(
        diff < TOL,
        "MLX SDPA single-layer BSNH diverges from CPU: max_abs {diff:e} >= {TOL}"
    );
}

#[test]
fn cpu_vs_mlx_brainjepa_predictor_stack() {
    if skip_mlx() {
        return;
    }

    // vit_base predictor hyperparams (Brain-JEPA defaults)
    let b = 1usize;
    let d = 384usize;
    let enc_dim = 768usize;
    let depth = 6usize;
    let nh = 12usize;
    let mlp_ratio = 4.0f64;
    let norm_eps = 1e-6f32;
    let hidden = (d as f64 * mlp_ratio) as usize;

    let n = seq_len();
    let n_pred = (n / 8).max(8);
    let n_ctx = n - n_pred;
    let wscale = weight_scale();
    let xscale = token_scale();

    let g = build_brainjepa_predictor(
        b, n, n_ctx, n_pred, d, enc_dim, depth, nh, mlp_ratio, norm_eps,
    );
    let tokens = det_random(42, b * n * d, xscale);

    let cpu_out = run_predictor_stack(
        Device::Cpu,
        g.clone(),
        &tokens,
        depth,
        d,
        enc_dim,
        hidden,
        wscale,
    );
    let mlx_out = run_predictor_stack(Device::Mlx, g, &tokens, depth, d, enc_dim, hidden, wscale);

    let diff = max_abs_diff(&cpu_out, &mlx_out);
    let cpu_max = cpu_out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let rel = diff / cpu_max.max(1e-6);
    eprintln!(
        "[mlx-predictor L={depth} S={n} D={d}] max_abs = {diff:e}, rel = {rel:e} (tol {TOL})"
    );
    assert!(
        diff < TOL,
        "MLX brainjepa-shaped predictor stack diverges from CPU: \
         max_abs {diff:e} >= {TOL} (set MLX_ATTN_PARITY_SEQ for long-seq stress)"
    );
}
