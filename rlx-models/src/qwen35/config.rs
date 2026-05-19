// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// (license header truncated — see workspace root.)

//! Qwen3.5 configuration — parsed from the GGUF metadata keys
//! under the `qwen35.*` prefix. Captures every field needed to
//! eventually wire the hybrid Mamba+Attention forward.
//!
//! Keys (observed on `unsloth/Qwen3.5-0.8B-MTP-GGUF`):
//!   `qwen35.block_count`, `qwen35.nextn_predict_layers`,
//!   `qwen35.embedding_length`, `qwen35.feed_forward_length`,
//!   `qwen35.attention.head_count`, `qwen35.attention.head_count_kv`,
//!   `qwen35.attention.key_length`, `qwen35.attention.value_length`,
//!   `qwen35.attention.layer_norm_rms_epsilon`,
//!   `qwen35.context_length`,
//!   `qwen35.full_attention_interval`,
//!   `qwen35.rope.dimension_count`, `qwen35.rope.freq_base`,
//!   `qwen35.rope.dimension_sections`,
//!   `qwen35.ssm.conv_kernel`, `qwen35.ssm.group_count`,
//!   `qwen35.ssm.inner_size`, `qwen35.ssm.state_size`,
//!   `qwen35.ssm.time_step_rank`.

use anyhow::{Result, anyhow};
use rlx_gguf::{GgufFile, MetaValue};

/// Qwen3.5 model config — fields covering both the per-layer Mamba+
/// Attention block and the MTP head.
#[derive(Debug, Clone)]
pub struct Qwen35Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    /// Total layer count (= main layers + `nextn_predict_layers` MTP heads).
    pub num_hidden_layers: usize,
    /// Layers at index `< num_hidden_layers - nextn_predict_layers`
    /// use the hybrid Mamba+Attention block. The remaining
    /// `nextn_predict_layers` layers use standard attention for MTP.
    pub nextn_predict_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Per-head Q dim. The MTP attention head uses this.
    pub key_length: usize,
    /// Per-head V dim.
    pub value_length: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_dim_count: usize,
    pub rope_dim_sections: Vec<usize>,
    /// Some Qwen3.5 layers do full attention every N blocks
    /// (interspersed with the Mamba-style blocks). Read but not yet
    /// acted on.
    pub full_attention_interval: usize,
    pub ssm_conv_kernel: usize,
    pub ssm_group_count: usize,
    pub ssm_inner_size: usize,
    pub ssm_state_size: usize,
    pub ssm_time_step_rank: usize,
    pub tie_word_embeddings: bool,
}

impl Qwen35Config {
    /// Read from a GGUF file with `general.architecture = "qwen35"`.
    /// Returns an error when any required key is missing.
    pub fn from_gguf(raw: &GgufFile) -> Result<Self> {
        let arch = raw
            .metadata
            .get("general.architecture")
            .and_then(MetaValue::as_str)
            .ok_or_else(|| anyhow!("missing general.architecture"))?;
        if arch != "qwen35" {
            return Err(anyhow!("expected arch=qwen35, got {arch}"));
        }
        let u32k = |k: &str| -> Result<u32> {
            raw.metadata
                .get(k)
                .and_then(MetaValue::as_u32)
                .ok_or_else(|| anyhow!("missing qwen35 metadata key: {k}"))
        };
        let f32k = |k: &str| -> Option<f32> {
            raw.metadata.get(k).and_then(|v| match v {
                MetaValue::F32(x) => Some(*x),
                _ => None,
            })
        };
        let boolk = |k: &str| -> Option<bool> {
            raw.metadata.get(k).and_then(|v| match v {
                MetaValue::Bool(b) => Some(*b),
                _ => None,
            })
        };
        let arr_u32k = |k: &str| -> Vec<usize> {
            raw.metadata
                .get(k)
                .and_then(|v| match v {
                    MetaValue::Array(a) => Some(
                        a.iter()
                            .filter_map(|x| match x {
                                MetaValue::U32(u) => Some(*u as usize),
                                MetaValue::U64(u) => Some(*u as usize),
                                MetaValue::I32(i) => Some(*i as usize),
                                _ => None,
                            })
                            .collect(),
                    ),
                    _ => None,
                })
                .unwrap_or_default()
        };
        Ok(Self {
            vocab_size: u32k("qwen35.vocab_size").unwrap_or(151_936) as usize,
            hidden_size: u32k("qwen35.embedding_length")? as usize,
            intermediate_size: u32k("qwen35.feed_forward_length")? as usize,
            num_hidden_layers: u32k("qwen35.block_count")? as usize,
            nextn_predict_layers: u32k("qwen35.nextn_predict_layers").unwrap_or(0) as usize,
            num_attention_heads: u32k("qwen35.attention.head_count")? as usize,
            num_key_value_heads: u32k("qwen35.attention.head_count_kv")? as usize,
            key_length: u32k("qwen35.attention.key_length").unwrap_or(128) as usize,
            value_length: u32k("qwen35.attention.value_length").unwrap_or(128) as usize,
            max_position_embeddings: u32k("qwen35.context_length").unwrap_or(40_960) as usize,
            rms_norm_eps: f32k("qwen35.attention.layer_norm_rms_epsilon").unwrap_or(1e-6) as f64,
            rope_theta: f32k("qwen35.rope.freq_base").unwrap_or(10_000_000.0) as f64,
            rope_dim_count: u32k("qwen35.rope.dimension_count").unwrap_or(64) as usize,
            rope_dim_sections: arr_u32k("qwen35.rope.dimension_sections"),
            full_attention_interval: u32k("qwen35.full_attention_interval").unwrap_or(0) as usize,
            ssm_conv_kernel: u32k("qwen35.ssm.conv_kernel").unwrap_or(4) as usize,
            ssm_group_count: u32k("qwen35.ssm.group_count").unwrap_or(0) as usize,
            ssm_inner_size: u32k("qwen35.ssm.inner_size").unwrap_or(0) as usize,
            ssm_state_size: u32k("qwen35.ssm.state_size").unwrap_or(0) as usize,
            ssm_time_step_rank: u32k("qwen35.ssm.time_step_rank").unwrap_or(0) as usize,
            tie_word_embeddings: boolk("qwen35.tie_word_embeddings").unwrap_or(true),
        })
    }

    /// Index of the first MTP layer (= `num_hidden_layers -
    /// nextn_predict_layers`). Returns `None` when the file has no
    /// MTP heads.
    pub fn mtp_layer_start(&self) -> Option<usize> {
        if self.nextn_predict_layers == 0 {
            None
        } else {
            Some(self.num_hidden_layers - self.nextn_predict_layers)
        }
    }
}
