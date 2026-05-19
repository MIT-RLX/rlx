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

//! End-to-end parity test for `Device::Tpu` against `Device::Cpu`.
//!
//! Builds a small FFN (LayerNorm → MatMul → GELU → MatMul → residual)
//! with deterministic random weights, compiles it on both backends,
//! runs forward on the same input, and confirms the outputs agree
//! within bf16-aware tolerance.
//!
//! Why FFN-only and not the full transformer block: causal-attention
//! mask handling has historically diverged between backends (mask
//! interpretation, scaling-position). Pin that down separately once
//! we trust the FFN baseline. The FFN already exercises LayerNorm,
//! MatMul, GELU, residual-add — most of the IR-optimization surface
//! (FuseResidualLN, FuseMatMulBiasAct, MarkElementwiseRegions).
//!
//! Gated on the `tpu` feature **and** `LIBTPU_PATH` (resolves the
//! PJRT plugin). On hosts without either, the test skips cleanly.

#![cfg(feature = "tpu")]

use rlx_driver::Device;
use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{PrecisionPolicy, Session};

fn skip_without_plugin() -> bool {
    if std::env::var("LIBTPU_PATH").is_err() {
        eprintln!("[real_model_parity] LIBTPU_PATH not set — skipping");
        return true;
    }
    false
}

fn build_ffn(b: usize, s: usize, h: usize, ffn: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("real_model_parity_ffn");
    let i64v = |dims: &[usize]| -> Vec<i64> { dims.iter().map(|&d| d as i64).collect() };
    let bs = b * s;

    let x = g.input("x", Shape::new(&[b, s, h], f));
    let ln_g = g.param("ln_g", Shape::new(&[h], f));
    let ln_b = g.param("ln_b", Shape::new(&[h], f));
    let w_up = g.param("w_up", Shape::new(&[h, ffn], f));
    let w_down = g.param("w_down", Shape::new(&[ffn, h], f));

    let xn = g.layer_norm(x, ln_g, ln_b, -1, 1e-5, Shape::new(&[b, s, h], f));
    let xn_2d = g.reshape(xn, i64v(&[bs, h]), Shape::new(&[bs, h], f));
    let up = g.matmul(xn_2d, w_up, Shape::new(&[bs, ffn], f));
    let act = g.activation(Activation::Gelu, up, Shape::new(&[bs, ffn], f));
    let down = g.matmul(act, w_down, Shape::new(&[bs, h], f));
    let down_3d = g.reshape(down, i64v(&[b, s, h]), Shape::new(&[b, s, h], f));
    let out = g.binary(BinaryOp::Add, x, down_3d, Shape::new(&[b, s, h], f));
    g.set_outputs(vec![out]);
    g
}

fn det_random(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut rng = seed;
    (0..n)
        .map(|_| {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng >> 33) as f32 / u32::MAX as f32) * scale - scale * 0.5
        })
        .collect()
}

fn upload_ffn_params(exec: &mut rlx_runtime::CompiledGraph, h: usize, ffn: usize) {
    let w_up: Vec<f32> = det_random(11, h * ffn, 0.04);
    let w_down: Vec<f32> = det_random(22, ffn * h, 0.04);
    let ln_g: Vec<f32> = vec![1.0; h];
    let ln_b: Vec<f32> = vec![0.0; h];
    exec.set_param("w_up", &w_up);
    exec.set_param("w_down", &w_down);
    exec.set_param("ln_g", &ln_g);
    exec.set_param("ln_b", &ln_b);
}

#[test]
fn ffn_cpu_vs_tpu_f32() {
    if skip_without_plugin() {
        return;
    }

    let b = 1usize;
    let s = 32;
    let h = 64;
    let ffn = 256;
    let xs: Vec<f32> = det_random(7, b * s * h, 0.1);

    // Both backends forced to F32 so the comparison isolates lowering
    // / fusion differences from precision policy differences.
    let mut cpu = Session::new(Device::Cpu)
        .with_policy(PrecisionPolicy::AlwaysF32)
        .compile(build_ffn(b, s, h, ffn));
    upload_ffn_params(&mut cpu, h, ffn);
    let cpu_out = cpu.run(&[("x", &xs)]);

    let mut tpu = Session::new(Device::Tpu)
        .with_policy(PrecisionPolicy::AlwaysF32)
        .compile(build_ffn(b, s, h, ffn));
    upload_ffn_params(&mut tpu, h, ffn);
    let tpu_out = tpu.run(&[("x", &xs)]);

    let n = cpu_out[0].len();
    let mut max_err = 0.0f32;
    let mut sum_abs_err = 0.0f64;
    let mut sum_abs_ref = 0.0f64;
    for i in 0..n {
        let a = cpu_out[0][i];
        let bv = tpu_out[0][i];
        max_err = max_err.max((a - bv).abs());
        sum_abs_err += (a - bv).abs() as f64;
        sum_abs_ref += a.abs() as f64;
    }
    let rel_err = if sum_abs_ref > 0.0 {
        sum_abs_err / sum_abs_ref
    } else {
        sum_abs_err
    };
    eprintln!(
        "[real_model_parity] f32-vs-f32 ffn  max abs = {max_err:e}, \
               mean rel = {rel_err:.4}"
    );

    // Tolerances: GELU has erf which uses different approximations
    // across backends, so per-element max-abs can be a few mE-3
    // legitimately. Mean relative error stays tight.
    assert!(
        rel_err < 0.01,
        "f32 ffn: TPU and CPU diverge by {rel_err:.4} (>1%)"
    );
}

#[test]
fn ffn_tpu_bf16_default() {
    if skip_without_plugin() {
        return;
    }

    let b = 1usize;
    let s = 32;
    let h = 64;
    let ffn = 256;
    let xs: Vec<f32> = det_random(7, b * s * h, 0.1);

    let mut cpu = Session::new(Device::Cpu)
        .with_policy(PrecisionPolicy::AlwaysF32)
        .compile(build_ffn(b, s, h, ffn));
    upload_ffn_params(&mut cpu, h, ffn);
    let cpu_out = cpu.run(&[("x", &xs)]);

    // Default policy on Device::Tpu is `AutoMixedBf16`. No `with_policy`.
    let mut tpu = Session::new(Device::Tpu).compile(build_ffn(b, s, h, ffn));
    upload_ffn_params(&mut tpu, h, ffn);
    let tpu_out = tpu.run(&[("x", &xs)]);

    let n = cpu_out[0].len();
    let mut sum_abs_err = 0.0f64;
    let mut sum_abs_ref = 0.0f64;
    for i in 0..n {
        let a = cpu_out[0][i];
        let bv = tpu_out[0][i];
        sum_abs_err += (a - bv).abs() as f64;
        sum_abs_ref += a.abs() as f64;
    }
    let rel_err = if sum_abs_ref > 0.0 {
        sum_abs_err / sum_abs_ref
    } else {
        sum_abs_err
    };
    eprintln!("[real_model_parity] bf16-vs-f32 ffn mean rel = {rel_err:.4}");
    // bf16 has ~1% precision per op, and an FFN with two big matmuls
    // and a GELU stacks ~10-15% rel error before any restabilizing
    // LayerNorm kicks in. Real models normalize between every block,
    // bringing this well under 5%; this isolated FFN block doesn't,
    // so we use a 20% bound. The point of the test is "bf16 is sane,
    // not catastrophically wrong" — not bit-parity to f32.
    assert!(
        rel_err < 0.20,
        "bf16 ffn diverges from f32 reference by {rel_err:.4} (>20%)"
    );
}

// ── Multi-layer FFN: deeper proxy for real model E2E ────────────

/// Build N stacked FFN-with-residual blocks (each block: LayerNorm
/// → MatMul → GELU → MatMul → Add residual). 4 blocks gives a
/// reasonable proxy for a small BERT/LLM in terms of total ops and
/// fp accumulation depth, without needing to load real weights or
/// exercise the WeightMap path.
fn build_stacked_ffn(b: usize, s: usize, h: usize, ffn: usize, n_layers: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("stacked_ffn");
    let i64v = |dims: &[usize]| -> Vec<i64> { dims.iter().map(|&d| d as i64).collect() };
    let bs = b * s;
    let mut h_in = g.input("x", Shape::new(&[b, s, h], f));
    for layer in 0..n_layers {
        let ln_g = g.param(format!("ln{layer}_g"), Shape::new(&[h], f));
        let ln_b = g.param(format!("ln{layer}_b"), Shape::new(&[h], f));
        let w_up = g.param(format!("w_up{layer}"), Shape::new(&[h, ffn], f));
        let w_down = g.param(format!("w_down{layer}"), Shape::new(&[ffn, h], f));
        let xn = g.layer_norm(h_in, ln_g, ln_b, -1, 1e-5, Shape::new(&[b, s, h], f));
        let xn_2d = g.reshape(xn, i64v(&[bs, h]), Shape::new(&[bs, h], f));
        let up = g.matmul(xn_2d, w_up, Shape::new(&[bs, ffn], f));
        let act = g.activation(Activation::Gelu, up, Shape::new(&[bs, ffn], f));
        let down = g.matmul(act, w_down, Shape::new(&[bs, h], f));
        let down_3d = g.reshape(down, i64v(&[b, s, h]), Shape::new(&[b, s, h], f));
        h_in = g.binary(BinaryOp::Add, h_in, down_3d, Shape::new(&[b, s, h], f));
    }
    g.set_outputs(vec![h_in]);
    g
}

fn upload_stacked_ffn(
    exec: &mut rlx_runtime::CompiledGraph,
    h: usize,
    ffn: usize,
    n_layers: usize,
) {
    for layer in 0..n_layers {
        let w_up: Vec<f32> = det_random(11 + layer as u64, h * ffn, 0.04);
        let w_down: Vec<f32> = det_random(22 + layer as u64, ffn * h, 0.04);
        let ln_g: Vec<f32> = vec![1.0; h];
        let ln_b: Vec<f32> = vec![0.0; h];
        exec.set_param(&format!("w_up{layer}"), &w_up);
        exec.set_param(&format!("w_down{layer}"), &w_down);
        exec.set_param(&format!("ln{layer}_g"), &ln_g);
        exec.set_param(&format!("ln{layer}_b"), &ln_b);
    }
}

#[test]
fn stacked_ffn_4_layers_cpu_vs_tpu() {
    if skip_without_plugin() {
        return;
    }

    let b = 1usize;
    let s = 32;
    let h = 64;
    let ffn = 256;
    let n_layers = 4;
    let xs: Vec<f32> = det_random(7, b * s * h, 0.1);

    let mut cpu = Session::new(Device::Cpu)
        .with_policy(PrecisionPolicy::AlwaysF32)
        .compile(build_stacked_ffn(b, s, h, ffn, n_layers));
    upload_stacked_ffn(&mut cpu, h, ffn, n_layers);
    let cpu_out = cpu.run(&[("x", &xs)]);

    let mut tpu = Session::new(Device::Tpu)
        .with_policy(PrecisionPolicy::AlwaysF32)
        .compile(build_stacked_ffn(b, s, h, ffn, n_layers));
    upload_stacked_ffn(&mut tpu, h, ffn, n_layers);
    let tpu_out = tpu.run(&[("x", &xs)]);

    let n = cpu_out[0].len();
    let mut max_err = 0.0f32;
    let mut sum_abs_err = 0.0f64;
    let mut sum_abs_ref = 0.0f64;
    for i in 0..n {
        let a = cpu_out[0][i];
        let bv = tpu_out[0][i];
        max_err = max_err.max((a - bv).abs());
        sum_abs_err += (a - bv).abs() as f64;
        sum_abs_ref += a.abs() as f64;
    }
    let rel_err = if sum_abs_ref > 0.0 {
        sum_abs_err / sum_abs_ref
    } else {
        sum_abs_err
    };
    eprintln!(
        "[real_model_parity] f32 stacked_ffn(L={n_layers}) \
               max abs = {max_err:e}, mean rel = {rel_err:.4}"
    );
    // Tolerance scales with depth (each layer contributes ~ε of
    // relative error from fp accumulation order). At 4 layers the
    // bound stays under 1%.
    assert!(
        rel_err < 0.01,
        "stacked f32 ffn diverges by {rel_err:.4} (>1%)"
    );
}

// ── Attention isolation: pin down where causal attention diverges ──

fn build_attn_only(
    b: usize,
    s: usize,
    n_heads: usize,
    d_head: usize,
    mask: rlx_ir::op::MaskKind,
) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("attn_only");
    let q = g.input("q", Shape::new(&[b, n_heads, s, d_head], f));
    let k = g.input("k", Shape::new(&[b, n_heads, s, d_head], f));
    let v = g.input("v", Shape::new(&[b, n_heads, s, d_head], f));
    let out = g.attention_kind(
        q,
        k,
        v,
        n_heads,
        d_head,
        mask,
        Shape::new(&[b, n_heads, s, d_head], f),
    );
    g.set_outputs(vec![out]);
    g
}

fn compare_outputs(cpu: &[f32], tpu: &[f32]) -> (f32, f64) {
    assert_eq!(cpu.len(), tpu.len());
    let mut max_err = 0.0f32;
    let mut sum_abs_err = 0.0f64;
    let mut sum_abs_ref = 0.0f64;
    for i in 0..cpu.len() {
        let a = cpu[i];
        let bv = tpu[i];
        max_err = max_err.max((a - bv).abs());
        sum_abs_err += (a - bv).abs() as f64;
        sum_abs_ref += a.abs() as f64;
    }
    let rel = if sum_abs_ref > 0.0 {
        sum_abs_err / sum_abs_ref
    } else {
        sum_abs_err
    };
    (max_err, rel)
}

#[test]
fn attention_no_mask_cpu_vs_tpu() {
    if skip_without_plugin() {
        return;
    }
    let b = 1usize;
    let s = 8;
    let n_heads = 2;
    let d_head = 8;

    let qs: Vec<f32> = det_random(101, b * n_heads * s * d_head, 0.1);
    let ks: Vec<f32> = det_random(202, b * n_heads * s * d_head, 0.1);
    let vs: Vec<f32> = det_random(303, b * n_heads * s * d_head, 0.1);

    let mut cpu = Session::new(Device::Cpu)
        .with_policy(PrecisionPolicy::AlwaysF32)
        .compile(build_attn_only(
            b,
            s,
            n_heads,
            d_head,
            rlx_ir::op::MaskKind::None,
        ));
    let cpu_out = cpu.run(&[("q", &qs), ("k", &ks), ("v", &vs)]);

    let mut tpu = Session::new(Device::Tpu)
        .with_policy(PrecisionPolicy::AlwaysF32)
        .compile(build_attn_only(
            b,
            s,
            n_heads,
            d_head,
            rlx_ir::op::MaskKind::None,
        ));
    let tpu_out = tpu.run(&[("q", &qs), ("k", &ks), ("v", &vs)]);

    let (max_err, rel) = compare_outputs(&cpu_out[0], &tpu_out[0]);
    eprintln!("[attn_parity] mask=None  max abs = {max_err:e}, mean rel = {rel:.4}");
    assert!(
        rel < 0.01,
        "attention(None): TPU vs CPU diverge by {rel:.4} (>1%)"
    );
}

#[test]
fn attention_causal_cpu_vs_tpu() {
    if skip_without_plugin() {
        return;
    }
    let b = 1usize;
    let s = 8;
    let n_heads = 2;
    let d_head = 8;

    let qs: Vec<f32> = det_random(101, b * n_heads * s * d_head, 0.1);
    let ks: Vec<f32> = det_random(202, b * n_heads * s * d_head, 0.1);
    let vs: Vec<f32> = det_random(303, b * n_heads * s * d_head, 0.1);

    let mut cpu = Session::new(Device::Cpu)
        .with_policy(PrecisionPolicy::AlwaysF32)
        .compile(build_attn_only(
            b,
            s,
            n_heads,
            d_head,
            rlx_ir::op::MaskKind::Causal,
        ));
    let cpu_out = cpu.run(&[("q", &qs), ("k", &ks), ("v", &vs)]);

    let mut tpu = Session::new(Device::Tpu)
        .with_policy(PrecisionPolicy::AlwaysF32)
        .compile(build_attn_only(
            b,
            s,
            n_heads,
            d_head,
            rlx_ir::op::MaskKind::Causal,
        ));
    let tpu_out = tpu.run(&[("q", &qs), ("k", &ks), ("v", &vs)]);

    let (max_err, rel) = compare_outputs(&cpu_out[0], &tpu_out[0]);
    eprintln!("[attn_parity] mask=Causal max abs = {max_err:e}, mean rel = {rel:.4}");
    assert!(
        rel < 0.01,
        "attention(Causal): TPU vs CPU diverge by {rel:.4} (>1%)"
    );
}
