// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// (license header truncated — see workspace root.)

//! Qwen3.5 — hybrid Mamba-2 + Attention architecture.
//!
//! **Status (0.2.0): architecture detection + config parsing only.**
//! The actual forward graph builder is not yet implemented; loading a
//! Qwen3.5 GGUF and asking for a forward will return a clear error
//! listing exactly which primitives the qwen35 block needs.
//!
//! # What makes Qwen3.5 different from Qwen3
//!
//! Qwen3 (and Qwen3.6) is a pure transformer: every layer is a
//! standard `[RMSNorm → Q/K/V → SDPA → out_proj → residual → RMSNorm
//! → SwiGLU MLP → residual]` block.
//!
//! Qwen3.5 replaces the attention block with a **Mamba-2 SSM gated
//! by an attention-style projection**. Per layer (from `blk.0.*` of
//! `unsloth/Qwen3.5-0.8B-MTP-GGUF`):
//!
//! | tensor               | shape           | role |
//! |----------------------|-----------------|---|
//! | `attn_norm`          | `[1024]`        | RMS norm input |
//! | `attn_qkv`           | `[1024, 6144]`  | fused projection → `[gate, x, B, C, dt]` (6× hidden) |
//! | `attn_gate`          | `[1024, 2048]`  | extra gating projection (inner=2048) |
//! | `ssm_conv1d`         | `[4, 6144]`     | depthwise 1D conv on the fused channels (kernel=4) |
//! | `ssm_dt.bias`        | `[16]`          | per-rank delta-t bias (dt_rank=16) |
//! | `ssm_a`              | `[16]`          | A diagonal log per rank |
//! | `ssm_alpha.weight`   | `[1024, 16]`    | α projection into dt_rank |
//! | `ssm_beta.weight`    | `[1024, 16]`    | β projection into dt_rank |
//! | `ssm_norm.weight`    | `[128]`         | norm over SSM state (state_size=128) |
//! | `ssm_out.weight`     | `[2048, 1024]`  | output projection (inner=2048 → hidden=1024) |
//! | `post_attention_norm`| `[1024]`        | post-SSM RMS norm |
//! | `ffn_{gate,up,down}` | standard SwiGLU | hidden=1024 ↔ intermediate=3584 |
//!
//! Plus the layer at index `block_count - nextn_predict_layers`
//! switches to **standard attention** for the MTP head (it uses
//! `attn_q`, `attn_k`, `attn_v`, `attn_output` like Qwen3 — plus
//! the `nextn.{enorm, hnorm, eh_proj, shared_head_norm}` tensors
//! specific to multi-token prediction).
//!
//! # What's needed to wire this end-to-end
//!
//! Most of the IR primitives already exist:
//!
//! - **Mamba SSM scan**: `Op::SelectiveScan { state_size }` lives in
//!   `rlx-ir/src/ops/special.rs`. Implements `h[t] = exp(Δ[t]·A)·h[t-1]
//!   + Δ[t]·B[t]·x[t]; y[t] = C[t]·h[t]`. Reasonable starting point
//!   though Qwen3.5 uses Mamba-2 specifically (A is scalar-per-head,
//!   not a matrix; state grouping differs from Mamba-1).
//! - **1D depthwise convolution**: `Op::Conv` handles 1D with the
//!   right kernel/groups parameters.
//! - **Norms / gather / RoPE**: all standard.
//! - **The fused 6-way split** of `attn_qkv` into `[gate, x, B, C, dt]`
//!   uses `Op::Narrow` (already standard).
//!
//! What's missing:
//!
//! 1. The qwen35 builder itself — wiring the tensors above into the
//!    above op set, layer by layer, with the MTP-aware index-24
//!    branch that switches to standard attention.
//! 2. A reference oracle for parity checks. The current
//!    `rlx-models/tests/qwen3_parity.rs` uses candle as ground
//!    truth for Qwen3; Qwen3.5 is too new for the same approach.
//!    Recommend: drive llama-cpp-rs from a parity test, dump
//!    top-N logits for a fixed prompt + compare cosine.
//! 3. A Mamba-2-specific scan implementation if `SelectiveScan`'s
//!    Mamba-1-style recurrence isn't numerically equivalent. The
//!    structured-SSM variant Mamba-2 uses is a related but
//!    distinct kernel.

mod builder;
mod config;
mod runner;
mod weights;

pub use builder::{PackedParams, build_qwen35_graph_sized};
pub use config::Qwen35Config;
pub use runner::{Qwen35PrefillOutput, Qwen35Runner, Qwen35RunnerBuilder};
pub use weights::{
    MatWeight, Qwen35FullAttnLayer, Qwen35LinearLayer, Qwen35MtpLayer, Qwen35TrunkLayer,
    Qwen35Weights,
};

/// Stand-in builder. Returns an [`anyhow::Error`] explaining what
/// would need to be wired to actually run Qwen3.5 / Qwen3.5-MTP
/// inference end-to-end. Exists so the `Qwen3Runner` can give a
/// precise error instead of "tensor not found" when handed a
/// Qwen3.5 file.
///
/// See the module docs for the full picture of what's missing.
pub fn build_qwen35_graph_sized_stub(
    _cfg: &Qwen35Config,
) -> anyhow::Result<()> {
    Err(anyhow::anyhow!(
        "Qwen3.5 / Qwen3.5-MTP forward graph not yet wired.\n\
         \n\
         What's in place (this release):\n\
           • `Qwen35Config::from_gguf` parses all `qwen35.*` metadata.\n\
           • `Qwen35Weights::from_loader` reads every tensor named by\n\
             llama.cpp's `qwen35.cpp` (trunk linear/full-attn split +\n\
             MTP layer with NextN eh_proj/enorm/hnorm).\n\
           • `Op::GatedDeltaNet` IR op + CPU autoregressive scan,\n\
             parity-tested against a scalar reference of the recurrence\n\
             in `delta-net-base.cpp::build_delta_net_autoregressive`.\n\
         \n\
         What's still missing:\n\
           • The forward graph builder that wires `attn_qkv` (fused\n\
             6-way projection), `ssm_conv1d` (kernel=4 depthwise) +\n\
             `Op::GatedDeltaNet` (state_size=128) + ssm_norm/ssm_out\n\
             together per layer, plus the layer-N MTP head's switch\n\
             to standard attention.\n\
           • A parity oracle (likely llama-cpp-rs driving the same\n\
             GGUF) so the builder can be verified before being\n\
             exposed via `Qwen3Runner`.\n\
         \n\
         See `rlx_models::qwen35` module docs for the full picture."
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal in-memory GGUF with `general.architecture =
    /// "qwen35"` + the keys [`Qwen35Config::from_gguf`] needs, then
    /// verify parsing succeeds and the stub builder returns a
    /// non-empty error.
    #[test]
    fn parses_qwen35_config_and_stub_errors_cleanly() {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&rlx_gguf::GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // 0 tensors
        let kv_count_off = buf.len();
        buf.extend_from_slice(&0u64.to_le_bytes()); // placeholder KV count

        let write_string = |buf: &mut Vec<u8>, k: &str, v: &str| {
            buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
            buf.extend_from_slice(k.as_bytes());
            buf.extend_from_slice(&8u32.to_le_bytes());
            buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
            buf.extend_from_slice(v.as_bytes());
        };
        let write_u32 = |buf: &mut Vec<u8>, k: &str, v: u32| {
            buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
            buf.extend_from_slice(k.as_bytes());
            buf.extend_from_slice(&4u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        };

        let mut n_kv: u64 = 0;
        write_string(&mut buf, "general.architecture", "qwen35");
        n_kv += 1;
        write_u32(&mut buf, "qwen35.block_count", 25);
        n_kv += 1;
        write_u32(&mut buf, "qwen35.nextn_predict_layers", 1);
        n_kv += 1;
        write_u32(&mut buf, "qwen35.embedding_length", 1024);
        n_kv += 1;
        write_u32(&mut buf, "qwen35.feed_forward_length", 3584);
        n_kv += 1;
        write_u32(&mut buf, "qwen35.attention.head_count", 16);
        n_kv += 1;
        write_u32(&mut buf, "qwen35.attention.head_count_kv", 4);
        n_kv += 1;

        buf[kv_count_off..kv_count_off + 8].copy_from_slice(&n_kv.to_le_bytes());
        while !buf.len().is_multiple_of(rlx_gguf::DEFAULT_ALIGNMENT as usize) {
            buf.push(0);
        }
        let path = std::env::temp_dir().join("rlx_qwen35_config_test.gguf");
        std::fs::write(&path, &buf).unwrap();
        let raw = rlx_gguf::GgufFile::from_path(&path).unwrap();

        let cfg = Qwen35Config::from_gguf(&raw).unwrap();
        assert_eq!(cfg.num_hidden_layers, 25);
        assert_eq!(cfg.nextn_predict_layers, 1);
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.intermediate_size, 3584);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 4);
        assert_eq!(cfg.mtp_layer_start(), Some(24));

        let err = build_qwen35_graph_sized_stub(&cfg).unwrap_err().to_string();
        assert!(err.contains("Qwen3.5"));
        assert!(err.contains("forward graph"));
        assert!(err.contains("GatedDeltaNet"));

        std::fs::remove_file(&path).ok();
    }
}
