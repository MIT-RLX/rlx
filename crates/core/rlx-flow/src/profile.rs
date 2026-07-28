// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compile profile — tier-1 config for fusion, passes, precision, backends.

use rlx_ir::hir::FusionPolicy;
use serde::{Deserialize, Serialize};

/// Tier-1 compile configuration. Load from `*.rlx.toml` or use Rust presets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CompileProfile {
    pub fusion: FusionProfile,
    pub passes: PassProfile,
    pub precision: PrecisionProfile,
    #[serde(default)]
    pub backend: BackendOverrides,
}

impl Default for CompileProfile {
    fn default() -> Self {
        Self::llama32_prefill()
    }
}

impl CompileProfile {
    /// Fusion-first prefill defaults (Direct lowering, fusion passes on).
    pub fn llama32_prefill() -> Self {
        Self {
            fusion: FusionProfile {
                policy: FusionPolicyKind::Direct,
                target: FusionTargetKind::Auto,
                assert_clean: false,
                skip: false,
            },
            passes: PassProfile::default(),
            precision: PrecisionProfile::default(),
            backend: BackendOverrides::default(),
        }
    }

    /// Decode graphs: Fusable lowering so KV-cache concat patterns fuse cleanly.
    pub fn llama32_decode() -> Self {
        Self {
            fusion: FusionProfile {
                policy: FusionPolicyKind::Fusable,
                ..FusionProfile::default()
            },
            ..Self::llama32_prefill()
        }
    }

    /// Qwen3.5 prefill — same fusion-first defaults as LLaMA prefill.
    pub fn qwen35_prefill() -> Self {
        Self::llama32_prefill()
    }

    /// Qwen3.5 decode — fusable policy for GDN / full-attn KV patterns.
    pub fn qwen35_decode() -> Self {
        Self::llama32_decode()
    }

    /// Qwen3 dense LM prefill (GQA + SwiGLU).
    pub fn qwen3_prefill() -> Self {
        Self::llama32_prefill()
    }

    /// Qwen3 decode — fusable policy for bucketed KV-cache graphs.
    pub fn qwen3_decode() -> Self {
        Self::llama32_decode()
    }

    /// Gemma 2 / Gemma 3 causal LM prefill (GQA + RMSNorm + softcap).
    pub fn gemma_prefill() -> Self {
        Self::llama32_prefill()
    }

    /// Gemma decode — fusable policy for bucketed KV-cache graphs.
    pub fn gemma_decode() -> Self {
        Self::llama32_decode()
    }

    /// FLUX.2 diffusion transformer + VAE/text-encoder graphs.
    pub fn flux2() -> Self {
        Self::encoder()
    }

    /// SAM / SAM2 image encoder and mask-decoder subgraphs (ConvNeXt-style stacks).
    pub fn sam_encoder() -> Self {
        Self::encoder()
    }

    /// SAM3 detector encoder/decoder layers (ViT + deformable-style decoder).
    pub fn sam3() -> Self {
        Self::sam_encoder()
    }

    /// SAM2 image + mask-decoder + memory subgraphs (Hiera encoder uses same tier-1 knobs).
    pub fn sam2() -> Self {
        Self::sam_encoder()
    }

    /// SAM2 memory-attention layers — fusion off (host RoPE between subgraphs).
    pub fn sam2_memory_attention() -> Self {
        Self {
            fusion: FusionProfile {
                skip: true,
                ..FusionProfile::default()
            },
            ..Self::encoder()
        }
    }

    /// LLaDA2 / TIDE block-diffusion MoE (bidirectional attention + grouped MoE).
    ///
    /// Fusion is off so graphs legalize on wgpu/CUDA without unfused
    /// `FusedResidualRmsNorm` lowerings.
    pub fn llada2_diffusion() -> Self {
        Self {
            fusion: FusionProfile {
                skip: true,
                ..FusionProfile::default()
            },
            ..Self::encoder()
        }
    }

    /// Bidirectional encoder defaults (BERT, NomicBERT, vision encoders).
    pub fn encoder() -> Self {
        Self {
            fusion: FusionProfile {
                policy: FusionPolicyKind::Direct,
                ..FusionProfile::default()
            },
            passes: PassProfile {
                dce: true,
                constant_folding: true,
                verbose: false,
            },
            precision: PrecisionProfile::default(),
            backend: BackendOverrides::default(),
        }
    }

    pub fn fusion_policy(&self) -> FusionPolicy {
        self.fusion.policy.into()
    }

    pub fn from_toml_str(s: &str) -> anyhow::Result<Self> {
        Ok(toml::from_str(s)?)
    }

    pub fn from_toml_path(path: &std::path::Path) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        Self::from_toml_str(&data)
    }

    /// Load `<family>.rlx.toml` from the directory containing
    /// `weights`. Falls back to the built-in preset for `(family, mode)`
    /// when the sidecar is missing or unreadable.
    ///
    /// Replaces the per-crate `*_profile_near_weights` helpers (one per
    /// family today: `llama32_profile_near_weights`, `qwen3_profile_*`,
    /// `gemma_profile_*`). New families only need to register their
    /// built-in presets via [`Self::default_for`].
    pub fn near_weights(weights: &std::path::Path, family: &str, mode: ProfileMode) -> Self {
        let default = Self::default_for(family, mode);
        let dir = weights
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."));
        let sidecar = dir.join(format!("{family}.rlx.toml"));
        Self::from_toml_path(&sidecar).unwrap_or(default)
    }

    /// Built-in preset for `(family, mode)`. Unknown families fall back
    /// to the Llama 3.2 presets — same behavior the per-crate helpers
    /// used before this method existed.
    pub fn default_for(family: &str, mode: ProfileMode) -> Self {
        match (family, mode) {
            ("llama32", ProfileMode::Prefill) => Self::llama32_prefill(),
            ("llama32", ProfileMode::Decode) => Self::llama32_decode(),
            ("qwen3", ProfileMode::Prefill) => Self::qwen3_prefill(),
            ("qwen3", ProfileMode::Decode) => Self::qwen3_decode(),
            ("qwen35", ProfileMode::Prefill) => Self::qwen35_prefill(),
            ("qwen35", ProfileMode::Decode) => Self::qwen35_decode(),
            ("gemma", ProfileMode::Prefill) => Self::gemma_prefill(),
            ("gemma", ProfileMode::Decode) => Self::gemma_decode(),
            (_, ProfileMode::Prefill) => Self::llama32_prefill(),
            (_, ProfileMode::Decode) => Self::llama32_decode(),
            (_, ProfileMode::Encoder) => Self::encoder(),
        }
    }
}

/// Whether the graph being compiled is a prefill, decode, or
/// encoder-style pass. Selects the right built-in preset in
/// [`CompileProfile::near_weights`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileMode {
    Prefill,
    Decode,
    Encoder,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FusionProfile {
    pub policy: FusionPolicyKind,
    pub target: FusionTargetKind,
    pub assert_clean: bool,
    pub skip: bool,
}

impl Default for FusionProfile {
    fn default() -> Self {
        Self {
            policy: FusionPolicyKind::Direct,
            target: FusionTargetKind::Auto,
            assert_clean: false,
            skip: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum FusionPolicyKind {
    #[default]
    Direct,
    Fusable,
}

impl From<FusionPolicyKind> for FusionPolicy {
    fn from(k: FusionPolicyKind) -> Self {
        match k {
            FusionPolicyKind::Direct => FusionPolicy::Direct,
            FusionPolicyKind::Fusable => FusionPolicy::Fusable,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum FusionTargetKind {
    #[default]
    Auto,
    Cpu,
    Metal,
    Mlx,
    Cuda,
    Rocm,
    Wgpu,
    Tpu,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PassProfile {
    pub dce: bool,
    pub constant_folding: bool,
    pub verbose: bool,
}

impl Default for PassProfile {
    fn default() -> Self {
        Self {
            dce: true,
            constant_folding: true,
            verbose: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PrecisionProfile {
    pub compute: PrecisionKind,
    pub mixed: MixedPrecisionKind,
}

impl Default for PrecisionProfile {
    fn default() -> Self {
        Self {
            compute: PrecisionKind::F32,
            mixed: MixedPrecisionKind::None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PrecisionKind {
    #[default]
    F32,
    F16,
    Bf16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MixedPrecisionKind {
    #[default]
    None,
    Auto,
}

/// Per-backend hint table (env-style toggles without touching IR).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct BackendOverrides {
    #[serde(default)]
    pub metal: MetalBackendProfile,
    #[serde(default)]
    pub cpu: CpuBackendProfile,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct MetalBackendProfile {
    pub skip_fusion: bool,
    pub unfuse_regions: bool,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct CpuBackendProfile {
    pub unfuse_regions: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_profile_toml() {
        let toml = r#"
[fusion]
policy = "direct"
target = "metal"
assert_clean = true

[passes]
dce = true
constant_folding = false

[precision]
compute = "f16"
mixed = "auto"
"#;
        let p = CompileProfile::from_toml_str(toml).unwrap();
        assert_eq!(p.fusion.policy, FusionPolicyKind::Direct);
        assert_eq!(p.fusion.target, FusionTargetKind::Metal);
        assert!(p.fusion.assert_clean);
        assert!(!p.passes.constant_folding);
        assert_eq!(p.precision.compute, PrecisionKind::F16);
        assert_eq!(p.precision.mixed, MixedPrecisionKind::Auto);
    }
}
