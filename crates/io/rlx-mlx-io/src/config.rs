// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `config.json` fields relevant to MLX / mlx-lm weight loading.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::fs;
use std::path::Path;

/// Quantization mode written by mlx-lm / `nn.quantize`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlxQuantMode {
    Affine,
    Mxfp4,
    Nvfp4,
    Mxfp8,
}

impl MlxQuantMode {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "affine" => Ok(Self::Affine),
            "mxfp4" => Ok(Self::Mxfp4),
            "nvfp4" => Ok(Self::Nvfp4),
            "mxfp8" => Ok(Self::Mxfp8),
            other => bail!("unsupported MLX quantization mode {other:?}"),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Affine => "affine",
            Self::Mxfp4 => "mxfp4",
            Self::Nvfp4 => "nvfp4",
            Self::Mxfp8 => "mxfp8",
        }
    }
}

/// Subset of mlx-lm `config.json` → `quantization`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlxQuantConfig {
    pub group_size: u32,
    pub bits: u32,
    pub mode: MlxQuantMode,
}

impl MlxQuantConfig {
    pub fn defaults_for(mode: MlxQuantMode) -> Self {
        match mode {
            MlxQuantMode::Affine => Self {
                group_size: 64,
                bits: 4,
                mode,
            },
            MlxQuantMode::Mxfp4 => Self {
                group_size: 32,
                bits: 4,
                mode,
            },
            MlxQuantMode::Nvfp4 => Self {
                group_size: 16,
                bits: 4,
                mode,
            },
            MlxQuantMode::Mxfp8 => Self {
                group_size: 32,
                bits: 8,
                mode,
            },
        }
    }
}

/// Parsed mlx-community / mlx-lm `config.json` (quantization + arch + passthrough).
#[derive(Debug, Clone, Default)]
pub struct MlxConfig {
    /// Global quantization config (the scalar `group_size`/`bits`/`mode`).
    pub quantization: Option<MlxQuantConfig>,
    /// Per-module quant overrides — mlx-lm mixed-precision checkpoints (e.g.
    /// gpt-oss: experts mxfp4 gs=32 globally, but `model.embed_tokens` /
    /// attention projections affine gs=64). Keyed by module base (no `.weight`).
    pub per_module_quant: std::collections::HashMap<String, MlxQuantConfig>,
    pub arch: Option<MlxArchConfig>,
    /// Raw JSON for sidecars / debugging.
    pub raw: Option<serde_json::Value>,
}

impl MlxConfig {
    /// Resolve the quant config for a tensor base name (`{module}` without the
    /// `.weight` suffix): the per-module override if present, else the global.
    pub fn quant_for(&self, base: &str) -> Option<MlxQuantConfig> {
        self.per_module_quant
            .get(base)
            .cloned()
            .or_else(|| self.quantization.clone())
    }
}

/// Llama / Qwen / SmolLM-style architecture fields from `config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct MlxArchConfig {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub head_dim: Option<usize>,
}

impl MlxArchConfig {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads.max(1))
    }
}

#[derive(Debug, Deserialize)]
struct RawQuant {
    #[serde(default)]
    group_size: Option<u32>,
    #[serde(default)]
    bits: Option<u32>,
    #[serde(default)]
    mode: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    #[serde(default)]
    quantization: Option<RawQuant>,
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    vocab_size: Option<usize>,
    #[serde(default)]
    hidden_size: Option<usize>,
    #[serde(default)]
    intermediate_size: Option<usize>,
    #[serde(default)]
    num_hidden_layers: Option<usize>,
    #[serde(default)]
    num_attention_heads: Option<usize>,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    #[serde(default)]
    rms_norm_eps: Option<f32>,
    #[serde(default)]
    rope_theta: Option<f32>,
    #[serde(default)]
    max_position_embeddings: Option<usize>,
    #[serde(default)]
    head_dim: Option<usize>,
}

impl MlxConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let bytes =
            fs::read(path.as_ref()).with_context(|| format!("read {}", path.as_ref().display()))?;
        Self::from_slice(&bytes)
    }

    pub fn from_slice(bytes: &[u8]) -> Result<Self> {
        let raw: serde_json::Value = serde_json::from_slice(bytes).context("parse config.json")?;
        let parsed: RawConfig =
            serde_json::from_value(raw.clone()).context("parse config quantization")?;
        let quantization = match parsed.quantization {
            None => None,
            Some(q) => {
                let mode = match q.mode.as_deref() {
                    None | Some("") => MlxQuantMode::Affine,
                    Some(s) => MlxQuantMode::parse(s)?,
                };
                let defaults = MlxQuantConfig::defaults_for(mode);
                Some(MlxQuantConfig {
                    group_size: q.group_size.unwrap_or(defaults.group_size),
                    bits: q.bits.unwrap_or(defaults.bits),
                    mode,
                })
            }
        };
        let arch = match (
            parsed.hidden_size,
            parsed.num_attention_heads,
            parsed.num_hidden_layers,
            parsed.vocab_size,
        ) {
            (
                Some(hidden_size),
                Some(num_attention_heads),
                Some(num_hidden_layers),
                Some(vocab_size),
            ) => Some(MlxArchConfig {
                model_type: parsed.model_type.unwrap_or_else(|| "llama".into()),
                vocab_size,
                hidden_size,
                intermediate_size: parsed.intermediate_size.unwrap_or(hidden_size * 4),
                num_hidden_layers,
                num_attention_heads,
                num_key_value_heads: parsed.num_key_value_heads.unwrap_or(num_attention_heads),
                rms_norm_eps: parsed.rms_norm_eps.unwrap_or(1e-5),
                rope_theta: parsed.rope_theta.unwrap_or(10_000.0),
                max_position_embeddings: parsed.max_position_embeddings.unwrap_or(2048),
                head_dim: parsed.head_dim,
            }),
            _ => None,
        };
        // Per-module quant overrides: any key under `quantization` whose value
        // is an object carrying group_size/bits/mode (mlx-lm mixed precision).
        let mut per_module_quant = std::collections::HashMap::new();
        if let Some(obj) = raw.get("quantization").and_then(|v| v.as_object()) {
            for (module, val) in obj {
                let Some(mobj) = val.as_object() else {
                    continue;
                };
                if !(mobj.contains_key("group_size")
                    || mobj.contains_key("bits")
                    || mobj.contains_key("mode"))
                {
                    continue;
                }
                if let Ok(rq) = serde_json::from_value::<RawQuant>(val.clone()) {
                    let mode = match rq.mode.as_deref() {
                        None | Some("") => MlxQuantMode::Affine,
                        Some(s) => match MlxQuantMode::parse(s) {
                            Ok(m) => m,
                            Err(_) => continue,
                        },
                    };
                    let d = MlxQuantConfig::defaults_for(mode);
                    per_module_quant.insert(
                        module.clone(),
                        MlxQuantConfig {
                            group_size: rq.group_size.unwrap_or(d.group_size),
                            bits: rq.bits.unwrap_or(d.bits),
                            mode,
                        },
                    );
                }
            }
        }
        Ok(Self {
            quantization,
            per_module_quant,
            arch,
            raw: Some(raw),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::QuantScheme;

    #[test]
    fn nvfp4_maps_to_mlx_mxfp4() {
        let cfg =
            MlxConfig::from_slice(br#"{"quantization":{"group_size":16,"bits":4,"mode":"nvfp4"}}"#)
                .unwrap();
        let q = cfg.quantization.unwrap();
        assert_eq!(q.mode, MlxQuantMode::Nvfp4);
        assert_eq!(q.group_size, 16);
        // Same pack as mxfp4; not NVIDIA Nvfp4Block.
        let scheme = QuantScheme::MlxMxfp4 {
            group_size: q.group_size,
        };
        assert!(scheme.is_mlx());
        assert_eq!(scheme.mlx_group_size(), 16);
    }
}
