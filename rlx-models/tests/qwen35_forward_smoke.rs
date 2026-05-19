// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// (license header truncated — see workspace root.)

//! Smoke test for the qwen35 forward graph builder.
//!
//! Builds a tiny `Qwen35Weights` struct by hand (3 trunk layers
//! — `linear, linear, full_attn` with full_attention_interval=3
//! — plus 1 MTP layer), runs `build_qwen35_graph_sized` + a
//! compiled prefill, and verifies the output shape is the expected
//! `[1, 1, n_vocab]` last-token logits (with `last_logits_only=true`)
//! plus the MTP head's `[1, 1, n_vocab]` next-token logits.
//!
//! This is a *plumbing* test — it doesn't verify numerical
//! correctness (that needs a llama-cpp-rs oracle on a real file).
//! It does verify:
//!   - Every shape inference path is valid for the layer mix.
//!   - The `Op::GatedDeltaNet` kernel slots into the trunk graph
//!     and produces a tensor that the rest of the graph accepts.
//!   - The `set_param` upload + `run` pipeline works end-to-end
//!     for the qwen35-shaped IR.

use rlx_models::qwen35::MatWeight;
use rlx_models::{
    Qwen35Config, Qwen35FullAttnLayer, Qwen35LinearLayer, Qwen35MtpLayer, Qwen35Runner,
    Qwen35TrunkLayer, Qwen35Weights, build_qwen35_graph_sized,
};
use rlx_runtime::{Device, Session};

fn mat(data: Vec<f32>) -> MatWeight {
    MatWeight::F32(data)
}

fn tiny_cfg() -> Qwen35Config {
    // Pick dims that all factor cleanly:
    //   hidden = n_head * head_dim = 4 * 4 = 16
    //   n_kv_head = 2 → GQA group = 2
    //   ssm: state=4, group=2, dt_rank=2 (so n_v_heads=n_k_heads=2,
    //   key_dim=state*group=8, value_dim=state*dt_rank=8)
    Qwen35Config {
        vocab_size: 32,
        hidden_size: 16,
        intermediate_size: 32,
        num_hidden_layers: 4, // 3 trunk + 1 MTP
        nextn_predict_layers: 1,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        key_length: 4,
        value_length: 4,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        rope_dim_count: 4,
        rope_dim_sections: vec![],
        full_attention_interval: 3, // layer 2 (il=2) is full-attn
        ssm_conv_kernel: 4,
        ssm_group_count: 2,
        ssm_inner_size: 8,
        ssm_state_size: 4,
        ssm_time_step_rank: 2,
        tie_word_embeddings: true,
    }
}

fn ramp(n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| 0.001 + scale * (i as f32) * 0.01).collect()
}

fn linear_layer(cfg: &Qwen35Config) -> Qwen35LinearLayer {
    let n_embd = cfg.hidden_size;
    let n_state = cfg.ssm_state_size;
    let n_k_heads = cfg.ssm_group_count;
    let n_v_heads = cfg.ssm_time_step_rank;
    let key_dim = n_state * n_k_heads;
    let value_dim = n_state * n_v_heads;
    let conv_channels = key_dim * 2 + value_dim;
    let n_ff = cfg.intermediate_size;
    let k_conv = cfg.ssm_conv_kernel;
    Qwen35LinearLayer {
        attn_norm: vec![1.0f32; n_embd],
        attn_post_norm: vec![1.0f32; n_embd],
        attn_qkv: mat(ramp(n_embd * conv_channels, 0.01)),
        attn_gate: mat(ramp(n_embd * value_dim, 0.01)),
        ssm_conv1d: ramp(k_conv * conv_channels, 0.02),
        ssm_dt_bias: ramp(n_v_heads, 0.05),
        // Negative log-A is the realistic regime; -1.0 → exp(g) ≈ 0.37.
        ssm_a: vec![-1.0f32; n_v_heads],
        ssm_beta: mat(ramp(n_embd * n_v_heads, 0.01)),
        ssm_alpha: mat(ramp(n_embd * n_v_heads, 0.01)),
        ssm_norm: vec![1.0f32; n_state],
        ssm_out: mat(ramp(value_dim * n_embd, 0.01)),
        ffn_gate: mat(ramp(n_embd * n_ff, 0.01)),
        ffn_down: mat(ramp(n_ff * n_embd, 0.01)),
        ffn_up: mat(ramp(n_embd * n_ff, 0.01)),
    }
}

fn full_attn_layer(cfg: &Qwen35Config) -> Qwen35FullAttnLayer {
    let n_embd = cfg.hidden_size;
    let n_head = cfg.num_attention_heads;
    let n_kv_head = cfg.num_key_value_heads;
    let head_dim = cfg.key_length;
    let q_gate_cols = n_head * head_dim * 2;
    let kv_cols = n_kv_head * head_dim;
    let n_ff = cfg.intermediate_size;
    Qwen35FullAttnLayer {
        attn_norm: vec![1.0f32; n_embd],
        attn_post_norm: vec![1.0f32; n_embd],
        attn_q_gate: mat(ramp(n_embd * q_gate_cols, 0.01)),
        attn_k: mat(ramp(n_embd * kv_cols, 0.01)),
        attn_v: mat(ramp(n_embd * kv_cols, 0.01)),
        attn_output: mat(ramp(n_head * head_dim * n_embd, 0.01)),
        attn_q_norm: vec![1.0f32; head_dim],
        attn_k_norm: vec![1.0f32; head_dim],
        ffn_gate: mat(ramp(n_embd * n_ff, 0.01)),
        ffn_down: mat(ramp(n_ff * n_embd, 0.01)),
        ffn_up: mat(ramp(n_embd * n_ff, 0.01)),
    }
}

fn synth_weights(cfg: &Qwen35Config) -> Qwen35Weights {
    let n_embd = cfg.hidden_size;
    let n_vocab = cfg.vocab_size;
    let n_main = cfg.num_hidden_layers - cfg.nextn_predict_layers;
    let interval = cfg.full_attention_interval.max(1);

    let mut trunk = Vec::new();
    for il in 0..n_main {
        let is_full = ((il + 1) % interval) == 0;
        trunk.push(if is_full {
            Qwen35TrunkLayer::FullAttn(full_attn_layer(cfg))
        } else {
            Qwen35TrunkLayer::Linear(linear_layer(cfg))
        });
    }
    let mtp = Qwen35MtpLayer {
        base: full_attn_layer(cfg),
        eh_proj: mat(ramp(2 * n_embd * n_embd, 0.01)),
        enorm: vec![1.0f32; n_embd],
        hnorm: vec![1.0f32; n_embd],
        embed_tokens: None,
        shared_head_head: None,
        shared_head_norm: None,
    };

    Qwen35Weights {
        token_embd: ramp(n_vocab * n_embd, 0.001),
        output_norm: vec![1.0f32; n_embd],
        output: None,
        trunk_layers: trunk,
        mtp_layers: vec![mtp],
    }
}

#[test]
fn qwen35_forward_graph_builds_and_runs_with_mtp() {
    let cfg = tiny_cfg();
    let weights = synth_weights(&cfg);

    let seq = 4;
    let (graph, params, packed) = build_qwen35_graph_sized(
        &cfg,
        weights,
        /*batch*/ 1,
        seq,
        /*with_lm_head*/ true,
        /*last_logits_only*/ true,
        /*enable_mtp_head*/ true,
    )
    .expect("build qwen35 graph");
    assert!(
        packed.is_empty(),
        "synth weights all F32; packed map must be empty (got {} entries)",
        packed.len()
    );

    let session = Session::new(Device::Cpu);
    let mut compiled = session.compile(graph);
    for (name, data) in &params {
        compiled.set_param(name, data);
    }
    // Token ids as F32 input (the embed gather kernel does the
    // implicit cast — host I/O surface is F32-only).
    let ids: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

    // RoPE tables per full-attn layer + MTP layer.
    let _ = seq;
    let half_d = cfg.rope_dim_count / 2;
    let cos = vec![1.0f32; half_d];
    let sin = vec![0.0f32; half_d];

    // Only il=2 is full-attn (interval=3, (2+1)%3==0). Plus MTP at il=3.
    let feeds: Vec<(&str, &[f32])> = vec![
        ("input_ids", &ids),
        ("rope_cos_l2", &cos),
        ("rope_sin_l2", &sin),
        ("rope_cos_l3", &cos),
        ("rope_sin_l3", &sin),
    ];

    let outs = compiled.run(&feeds);
    assert!(
        outs.len() >= 2,
        "expected trunk + MTP outputs, got {}",
        outs.len()
    );
    let n_vocab = cfg.vocab_size;
    assert_eq!(
        outs[0].len(),
        n_vocab,
        "trunk last-token logits len = {} (want {n_vocab})",
        outs[0].len()
    );
    assert_eq!(
        outs[1].len(),
        n_vocab,
        "MTP head logits len = {} (want {n_vocab})",
        outs[1].len()
    );
    // Sanity: all logits must be finite. NaN / Inf would mean the
    // graph has a numerical blowup (e.g. unguarded division, bad
    // L2 norm denom).
    for (i, v) in outs[0].iter().enumerate() {
        assert!(
            v.is_finite(),
            "trunk logits[{i}] = {v} (non-finite — math blew up)"
        );
    }
    for (i, v) in outs[1].iter().enumerate() {
        assert!(
            v.is_finite(),
            "MTP logits[{i}] = {v} (non-finite — math blew up)"
        );
    }
}

#[test]
fn qwen35_runner_builder_works_without_mtp() {
    // Smoke: same as above but with `enable_mtp=false`, and only
    // verifying the trunk path.
    let cfg = tiny_cfg();
    let weights = synth_weights(&cfg);

    let (graph, params, _packed) =
        build_qwen35_graph_sized(&cfg, weights, 1, 4, true, true, false)
            .expect("build qwen35 graph (no mtp)");
    let session = Session::new(Device::Cpu);
    let mut compiled = session.compile(graph);
    for (name, data) in &params {
        compiled.set_param(name, data);
    }
    let ids: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let half_d = cfg.rope_dim_count / 2;
    let cos = vec![1.0f32; half_d];
    let sin = vec![0.0f32; half_d];
    let feeds: Vec<(&str, &[f32])> = vec![
        ("input_ids", &ids),
        ("rope_cos_l2", &cos),
        ("rope_sin_l2", &sin),
    ];
    let outs = compiled.run(&feeds);
    assert_eq!(outs.len(), 1);
    assert_eq!(outs[0].len(), cfg.vocab_size);
    for v in &outs[0] {
        assert!(v.is_finite(), "logit = {v}");
    }
    let _ = Qwen35Runner::builder(); // just verifies the type exists
}

/// Larger-dims smoke: realistic ratios (hidden = n_head * head_dim,
/// GQA group = 2, dt_rank = 4×group, 5 trunk layers with two
/// full-attn slots at intervals matching the production config).
/// Catches shape-mismatch bugs (e.g. reshape rank mismatches) that
/// the tiny test misses because `1 * 1 * h * n == h * n` looks
/// right even when the rank is wrong.
fn medium_cfg() -> Qwen35Config {
    // hidden = 64 = 4 heads × 16 head_dim; kv_head = 2, GQA=2.
    // ssm: state=16, group=4, dt_rank=8 → key_dim=64, value_dim=128.
    // full_attn every 3 layers (so il=2 is full-attn).
    Qwen35Config {
        vocab_size: 64,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 6, // 5 trunk + 1 MTP
        nextn_predict_layers: 1,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        key_length: 16,
        value_length: 16,
        max_position_embeddings: 128,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        rope_dim_count: 16,
        rope_dim_sections: vec![],
        full_attention_interval: 3,
        ssm_conv_kernel: 4,
        ssm_group_count: 4,
        ssm_inner_size: 128,
        ssm_state_size: 16,
        ssm_time_step_rank: 8,
        tie_word_embeddings: true,
    }
}

#[test]
fn qwen35_forward_medium_dims_runs_with_mtp() {
    let cfg = medium_cfg();
    let weights = synth_weights(&cfg);
    let seq = 8;
    let (graph, params, _packed) =
        build_qwen35_graph_sized(&cfg, weights, 1, seq, true, true, true)
            .expect("build qwen35 medium graph");

    let session = Session::new(Device::Cpu);
    let mut compiled = session.compile(graph);
    for (name, data) in &params {
        compiled.set_param(name, data);
    }

    let ids: Vec<f32> = (0..seq as u32).map(|i| (i + 1) as f32).collect();
    let half_d = cfg.rope_dim_count / 2;
    let cos = vec![1.0f32; half_d];
    let sin = vec![0.0f32; half_d];

    // Full-attn layers at il=2 (trunk) and the MTP at il=5
    // (interval=3 → (il+1)%3==0 at il=2; MTP head is il=5).
    let feeds: Vec<(&str, &[f32])> = vec![
        ("input_ids", &ids),
        ("rope_cos_l2", &cos),
        ("rope_sin_l2", &sin),
        ("rope_cos_l5", &cos),
        ("rope_sin_l5", &sin),
    ];
    let outs = compiled.run(&feeds);
    assert_eq!(outs.len(), 2, "trunk + MTP outputs");

    let trunk = &outs[0];
    let mtp = &outs[1];
    assert_eq!(trunk.len(), cfg.vocab_size);
    assert_eq!(mtp.len(), cfg.vocab_size);

    let nan_trunk = trunk.iter().filter(|v| !v.is_finite()).count();
    let nan_mtp = mtp.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(nan_trunk, 0, "trunk has {nan_trunk} non-finite logits");
    assert_eq!(nan_mtp, 0, "MTP has {nan_mtp} non-finite logits");

    // Sanity: logits must not be all-zero (would indicate a
    // collapsed forward — e.g. RMS-norm divide-by-zero, or all
    // matmuls producing zero from a broken transpose).
    let trunk_nonzero = trunk.iter().filter(|v| **v != 0.0).count();
    assert!(
        trunk_nonzero > 0,
        "trunk logits collapsed to all-zero (broken forward)"
    );

    // Sanity: argmax should be a unique-ish token, not a degenerate
    // tie across vocab.
    let max_val = trunk.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let ties = trunk.iter().filter(|&&v| v == max_val).count();
    assert!(
        ties < cfg.vocab_size / 4,
        "trunk argmax tied across {ties}/{} tokens — likely all-equal logits",
        cfg.vocab_size
    );
}
