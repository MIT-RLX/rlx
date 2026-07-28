// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A real decoder-only transformer (Llama / Qwen3-style) forward pass, built
//! directly on `rlx_ir::Graph` and run through `rlx_runtime::Session` on the
//! CPU backend — so it works natively *and* in the browser (wasm).
//!
//! Each layer is: RMSNorm → causal multi-head self-attention with RoPE →
//! residual → RMSNorm → SwiGLU MLP → residual. Final RMSNorm + lm_head.
//!
//! Weights are **synthesized deterministically** from a seed (the
//! GGUF-weight-loading path is separate — see the crate README), so this is a
//! faithful *architecture* run, verifiable end-to-end: determinism, output
//! shape, and the causal property (a position's logits never depend on later
//! tokens) are checked in the native tests.

use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, GraphExt, NodeId, Shape};
use rlx_runtime::{Device, Session};

/// Transformer hyperparameters. `dim` must equal `n_heads * head_dim`.
/// Multi-head attention (`n_kv_heads == n_heads`); GQA would add a KV-repeat.
#[derive(Clone, Copy)]
pub struct TfConfig {
    pub vocab: usize,
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub seq: usize,
    pub eps: f32,
    pub theta: f32,
}

fn sh(dims: &[usize]) -> Shape {
    Shape::new(dims, DType::F32)
}

fn mkparam(
    g: &mut Graph,
    params: &mut Vec<(String, usize)>,
    name: String,
    dims: &[usize],
) -> NodeId {
    params.push((name.clone(), dims.iter().product::<usize>()));
    g.param(&name, sh(dims))
}

/// Build the forward graph. Returns the graph and the (name, element-count) of
/// every parameter, in a stable order, for weight loading.
fn build(cfg: &TfConfig) -> (Graph, Vec<(String, usize)>) {
    let (s, d, nh, dh, ff, v) = (
        cfg.seq,
        cfg.dim,
        cfg.n_heads,
        cfg.head_dim,
        cfg.ffn,
        cfg.vocab,
    );
    assert_eq!(d, nh * dh, "dim must equal n_heads * head_dim");
    let half = dh / 2;
    let eps = cfg.eps;

    let mut g = Graph::new("rlx_web_transformer");
    let mut params: Vec<(String, usize)> = Vec::new();

    // Inputs: token ids [S], and the RoPE cos/sin tables [S, head_dim/2].
    let tokens = g.input("tokens", sh(&[s]));
    let cos = g.input("rope_cos", sh(&[s, half]));
    let sin = g.input("rope_sin", sh(&[s, half]));

    let zero_beta = mkparam(&mut g, &mut params, "zero_beta".into(), &[d]);
    let embed = mkparam(&mut g, &mut params, "embed".into(), &[v, d]);

    // Token embedding: gather rows of `embed` by id → [S, D] → [1, S, D].
    let emb = g.gather(embed, tokens, 0, sh(&[s, d]));
    let mut h = g.reshape(emb, vec![1, s as i64, d as i64], sh(&[1, s, d]));

    for l in 0..cfg.n_layers {
        let p = |n: &str| format!("layers.{l}.{n}");
        let attn_norm = mkparam(&mut g, &mut params, p("attn_norm"), &[d]);
        let wq = mkparam(&mut g, &mut params, p("wq"), &[d, d]);
        let wk = mkparam(&mut g, &mut params, p("wk"), &[d, d]);
        let wv = mkparam(&mut g, &mut params, p("wv"), &[d, d]);
        let wo = mkparam(&mut g, &mut params, p("wo"), &[d, d]);
        let ffn_norm = mkparam(&mut g, &mut params, p("ffn_norm"), &[d]);
        let wgate = mkparam(&mut g, &mut params, p("wgate"), &[d, ff]);
        let wup = mkparam(&mut g, &mut params, p("wup"), &[d, ff]);
        let wdown = mkparam(&mut g, &mut params, p("wdown"), &[ff, d]);

        // ── Attention block ──
        let n1 = g.rms_norm(h, attn_norm, zero_beta, eps); // [1,S,D]
        let n1_2d = g.reshape(n1, vec![s as i64, d as i64], sh(&[s, d]));
        let q = g.matmul(n1_2d, wq, sh(&[s, d]));
        let k = g.matmul(n1_2d, wk, sh(&[s, d]));
        let v_ = g.matmul(n1_2d, wv, sh(&[s, d]));
        let q3 = g.reshape(q, vec![1, s as i64, d as i64], sh(&[1, s, d]));
        let k3 = g.reshape(k, vec![1, s as i64, d as i64], sh(&[1, s, d]));
        let v3 = g.reshape(v_, vec![1, s as i64, d as i64], sh(&[1, s, d]));
        let q_rope = g.rope(q3, cos, sin, dh);
        let k_rope = g.rope(k3, cos, sin, dh);
        let attn = g.attention_kind(q_rope, k_rope, v3, nh, dh, MaskKind::Causal, sh(&[1, s, d]));
        let attn_2d = g.reshape(attn, vec![s as i64, d as i64], sh(&[s, d]));
        let o = g.matmul(attn_2d, wo, sh(&[s, d]));
        let o3 = g.reshape(o, vec![1, s as i64, d as i64], sh(&[1, s, d]));
        h = g.add(h, o3);

        // ── SwiGLU MLP block ──
        let n2 = g.rms_norm(h, ffn_norm, zero_beta, eps);
        let n2_2d = g.reshape(n2, vec![s as i64, d as i64], sh(&[s, d]));
        let gate = g.matmul(n2_2d, wgate, sh(&[s, ff]));
        let gate = g.silu(gate);
        let up = g.matmul(n2_2d, wup, sh(&[s, ff]));
        let ff_act = g.mul(gate, up);
        let down = g.matmul(ff_act, wdown, sh(&[s, d]));
        let down3 = g.reshape(down, vec![1, s as i64, d as i64], sh(&[1, s, d]));
        h = g.add(h, down3);
    }

    let final_norm = mkparam(&mut g, &mut params, "final_norm".into(), &[d]);
    let lm_head = mkparam(&mut g, &mut params, "lm_head".into(), &[d, v]);
    let hf = g.rms_norm(h, final_norm, zero_beta, eps);
    let hf_2d = g.reshape(hf, vec![s as i64, d as i64], sh(&[s, d]));
    let logits = g.matmul(hf_2d, lm_head, sh(&[s, v]));
    g.set_outputs(vec![logits]);

    (g, params)
}

/// Standard NeoX RoPE cos/sin tables: `[seq, head_dim/2]`.
fn rope_tables(seq: usize, head_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0f32; seq * half];
    let mut sin = vec![0f32; seq * half];
    for pos in 0..seq {
        for i in 0..half {
            let freq = 1.0f32 / theta.powf(2.0 * i as f32 / head_dim as f32);
            let a = pos as f32 * freq;
            cos[pos * half + i] = a.cos();
            sin[pos * half + i] = a.sin();
        }
    }
    (cos, sin)
}

/// Deterministic splitmix64-based pseudo-random weights (small magnitude).
fn synth(seed: u64, salt: u64, n: usize, scale: f32, center: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut x = seed
                ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (i as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^= x >> 31;
            let u = (x >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
            center + (u * 2.0 - 1.0) * scale
        })
        .collect()
}

fn name_hash(name: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in name.bytes() {
        h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Load the deterministic synthesized weights for `params` into `compiled`.
fn set_synth_params(
    compiled: &mut rlx_runtime::CompiledGraph,
    params: &[(String, usize)],
    seed: u64,
) {
    for (name, n) in params {
        let data = if name == "zero_beta" {
            vec![0.0; *n]
        } else if name.ends_with("norm") {
            // RMSNorm gains: centered near 1.
            synth(seed, name_hash(name), *n, 0.02, 1.0)
        } else {
            // Linear / embedding weights: small, zero-centered.
            synth(seed, name_hash(name), *n, 0.04, 0.0)
        };
        compiled.set_param(name, &data);
    }
}

/// Run the transformer with deterministic synthesized weights; returns all
/// logits, row-major `[seq, vocab]`.
pub fn transformer_logits(cfg: &TfConfig, tokens: &[f32], seed: u64) -> Vec<f32> {
    let (g, params) = build(cfg);
    let mut compiled = Session::new(Device::Cpu).compile(g);
    set_synth_params(&mut compiled, &params, seed);
    let (cos, sin) = rope_tables(cfg.seq, cfg.head_dim, cfg.theta);
    compiled
        .run(&[("tokens", tokens), ("rope_cos", &cos), ("rope_sin", &sin)])
        .into_iter()
        .next()
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> TfConfig {
        TfConfig {
            vocab: 32,
            dim: 32,
            n_layers: 2,
            n_heads: 4,
            head_dim: 8,
            ffn: 64,
            seq: 5,
            eps: 1e-5,
            theta: 10000.0,
        }
    }

    #[test]
    fn runs_with_correct_shape_and_finite() {
        let cfg = tiny();
        let tokens: Vec<f32> = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let logits = transformer_logits(&cfg, &tokens, 42);
        assert_eq!(logits.len(), cfg.seq * cfg.vocab);
        assert!(
            logits.iter().all(|x| x.is_finite()),
            "logits must be finite"
        );
    }

    #[test]
    fn deterministic() {
        let cfg = tiny();
        let tokens: Vec<f32> = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let a = transformer_logits(&cfg, &tokens, 7);
        let b = transformer_logits(&cfg, &tokens, 7);
        assert_eq!(a, b);
    }

    /// Isolation: with no layers (embed → final RMSNorm → lm_head, all
    /// per-position) changing a later token must not touch earlier logits.
    #[test]
    fn causal_no_layers() {
        let mut cfg = tiny();
        cfg.n_layers = 0;
        let v = cfg.vocab;
        let base: Vec<f32> = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let mut changed = base.clone();
        changed[4] = 9.0;
        let a = transformer_logits(&cfg, &base, 7);
        let b = transformer_logits(&cfg, &changed, 7);
        for pos in 0..4 {
            for c in 0..v {
                let i = pos * v + c;
                assert!(
                    (a[i] - b[i]).abs() < 1e-5,
                    "no-layers leak at pos {pos}: {} vs {}",
                    a[i],
                    b[i]
                );
            }
        }
    }

    #[test]
    fn causal_attention_only() {
        // embed -> attention_kind(Causal) -> lm_head, nothing else.
        let (s, d, nh, dh, v) = (5usize, 32usize, 4usize, 8usize, 32usize);
        let mut g = Graph::new("attn_only");
        let tokens = g.input("tokens", sh(&[s]));
        let embed = g.param("embed", sh(&[v, d]));
        let lm = g.param("lm", sh(&[d, v]));
        let emb = g.gather(embed, tokens, 0, sh(&[s, d]));
        let h3 = g.reshape(emb, vec![1, s as i64, d as i64], sh(&[1, s, d]));
        let attn = g.attention_kind(h3, h3, h3, nh, dh, MaskKind::Causal, sh(&[1, s, d]));
        let a2 = g.reshape(attn, vec![s as i64, d as i64], sh(&[s, d]));
        let logits = g.matmul(a2, lm, sh(&[s, v]));
        g.set_outputs(vec![logits]);
        let mut c = Session::new(Device::Cpu).compile(g);
        c.set_param("embed", &synth(1, 1, v * d, 0.1, 0.0));
        c.set_param("lm", &synth(2, 2, d * v, 0.1, 0.0));
        let run = |toks: &[f32], c: &mut rlx_runtime::CompiledGraph| {
            c.run(&[("tokens", toks)]).into_iter().next().unwrap()
        };
        let a = run(&[3.0, 1.0, 4.0, 1.0, 5.0], &mut c);
        let b = run(&[3.0, 1.0, 4.0, 1.0, 9.0], &mut c);
        for pos in 0..4 {
            for col in 0..v {
                let i = pos * v + col;
                assert!(
                    (a[i] - b[i]).abs() < 1e-5,
                    "attn leak pos {pos}: {} vs {}",
                    a[i],
                    b[i]
                );
            }
        }
    }

    #[test]
    fn causal_attn_with_rope() {
        let (s, d, nh, dh, v) = (5usize, 32usize, 4usize, 8usize, 32usize);
        let half = dh / 2;
        let mut g = Graph::new("attn_rope");
        let tokens = g.input("tokens", sh(&[s]));
        let cos = g.input("rope_cos", sh(&[s, half]));
        let sin = g.input("rope_sin", sh(&[s, half]));
        let embed = g.param("embed", sh(&[v, d]));
        let lm = g.param("lm", sh(&[d, v]));
        let emb = g.gather(embed, tokens, 0, sh(&[s, d]));
        let h3 = g.reshape(emb, vec![1, s as i64, d as i64], sh(&[1, s, d]));
        let qr = g.rope(h3, cos, sin, dh);
        let kr = g.rope(h3, cos, sin, dh);
        let attn = g.attention_kind(qr, kr, h3, nh, dh, MaskKind::Causal, sh(&[1, s, d]));
        let a2 = g.reshape(attn, vec![s as i64, d as i64], sh(&[s, d]));
        let logits = g.matmul(a2, lm, sh(&[s, v]));
        g.set_outputs(vec![logits]);
        let mut c = Session::new(Device::Cpu).compile(g);
        c.set_param("embed", &synth(1, 1, v * d, 0.1, 0.0));
        c.set_param("lm", &synth(2, 2, d * v, 0.1, 0.0));
        let (cosv, sinv) = rope_tables(s, dh, 10000.0);
        let run = |toks: &[f32], c: &mut rlx_runtime::CompiledGraph| {
            c.run(&[("tokens", toks), ("rope_cos", &cosv), ("rope_sin", &sinv)])
                .into_iter()
                .next()
                .unwrap()
        };
        let a = run(&[3.0, 1.0, 4.0, 1.0, 5.0], &mut c);
        let b = run(&[3.0, 1.0, 4.0, 1.0, 9.0], &mut c);
        for pos in 0..4 {
            for col in 0..v {
                let i = pos * v + col;
                assert!(
                    (a[i] - b[i]).abs() < 1e-5,
                    "rope-attn leak pos {pos}: {} vs {}",
                    a[i],
                    b[i]
                );
            }
        }
    }

    #[test]
    fn causal_one_layer() {
        let mut cfg = tiny();
        cfg.n_layers = 1;
        let v = cfg.vocab;
        let base: Vec<f32> = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let mut changed = base.clone();
        changed[4] = 9.0;
        let a = transformer_logits(&cfg, &base, 7);
        let b = transformer_logits(&cfg, &changed, 7);
        for pos in 0..4 {
            for col in 0..v {
                let i = pos * v + col;
                assert!(
                    (a[i] - b[i]).abs() < 1e-4,
                    "1-layer leak pos {pos}: {} vs {}",
                    a[i],
                    b[i]
                );
            }
        }
    }

    /// Compile the 1-layer transformer with the given fusion setting and return
    /// the max earlier-position leak when only the last token changes.
    fn leak_with_fusion(skip_fusion: bool, seed: u64) -> f32 {
        let mut cfg = tiny();
        cfg.n_layers = 1;
        let (g, params) = build(&cfg);
        let mut opts = rlx_runtime::CompileOptions::default();
        opts.fusion_opts.skip_fusion = skip_fusion;
        let mut c = Session::new(Device::Cpu).compile_with(g, &opts);
        set_synth_params(&mut c, &params, seed);
        let (cos, sin) = rope_tables(cfg.seq, cfg.head_dim, cfg.theta);
        let a = c
            .run(&[
                ("tokens", &[3.0f32, 1.0, 4.0, 1.0, 5.0][..]),
                ("rope_cos", &cos),
                ("rope_sin", &sin),
            ])
            .remove(0);
        let b = c
            .run(&[
                ("tokens", &[3.0f32, 1.0, 4.0, 1.0, 9.0][..]),
                ("rope_cos", &cos),
                ("rope_sin", &sin),
            ])
            .remove(0);
        let v = cfg.vocab;
        (0..4 * v).map(|i| (a[i] - b[i]).abs()).fold(0.0, f32::max)
    }

    /// The CPU fusion pipeline must produce the same causal behavior as the
    /// unfused path. Previously the fused `FusedAttnBlock` dropped the causal
    /// mask, so the fused leak was large (~0.07) while the unfused leak was 0.
    #[test]
    fn fused_path_is_causal_like_unfused() {
        let unfused = leak_with_fusion(true, 7);
        let fused = leak_with_fusion(false, 7);
        assert!(unfused < 1e-4, "unfused path leaked: {unfused}");
        assert!(
            fused < 1e-4,
            "fused path leaked (causal mask dropped in fusion?): {fused}"
        );
    }

    /// The defining property of a causal transformer: changing a *later* token
    /// must not change the logits at earlier positions.
    #[test]
    fn causal_property() {
        let cfg = tiny();
        let v = cfg.vocab;
        let base: Vec<f32> = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let mut changed = base.clone();
        changed[4] = 9.0; // change only the LAST token
        let a = transformer_logits(&cfg, &base, 7);
        let b = transformer_logits(&cfg, &changed, 7);
        // positions 0..=3 must be identical; position 4 should differ.
        for pos in 0..4 {
            for c in 0..v {
                let i = pos * v + c;
                assert!(
                    (a[i] - b[i]).abs() < 1e-4,
                    "pos {pos} col {c} changed by a later token: {} vs {}",
                    a[i],
                    b[i]
                );
            }
        }
        let last_diff: f32 = (0..v).map(|c| (a[4 * v + c] - b[4 * v + c]).abs()).sum();
        assert!(
            last_diff > 1e-3,
            "last position should react to its own token"
        );
    }
}
