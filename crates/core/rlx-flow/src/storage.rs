// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! How dense (non-quantized) weight matrices are stored once uploaded.
//!
//! Loaders hand builders f32, so a bf16 checkpoint used to land in the arena
//! at 4 B/param — twice its file size, for nothing: widening bf16 to f32 is
//! exact, and every backend's matmul reads an f16 / bf16 weight and computes
//! in f32. Storing the matrix at its own width gives the same products at half
//! the memory.
//!
//! A flow opts in with [`crate::ModelFlow::with_weight_storage`]; a process
//! can opt every flow in with `RLX_WEIGHT_STORAGE=auto|f32|f16|bf16`. With
//! neither, flows keep f32 — the behaviour every model was validated with.
//! The policy applies to rank-2 matmul weights loaded through
//! [`crate::FlowCtx::linear`] / [`crate::FlowCtx::load_matrix`] /
//! [`crate::Emit::load_matrix`]; packed quantized weights stay packed, and
//! norms, biases, embeddings and conv kernels stay f32.

use anyhow::{Result, bail};
use rlx_ir::DType;

/// Storage format for dense weight matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WeightStorage {
    /// Keep each tensor's stored float width: a bf16 / f16 tensor stays bf16 /
    /// f16 (lossless), an f32 tensor stays f32. Needs a weight source that
    /// reports [`crate::WeightSource::stored_dtype`]; one that cannot is
    /// treated as f32.
    #[default]
    Auto,
    /// Widen everything to f32 (2x the memory of a half-precision checkpoint,
    /// identical results).
    F32,
    /// Store as IEEE f16. Lossless for f16 checkpoints; for bf16 ones it keeps
    /// 3 more mantissa bits but clips magnitudes above 65504.
    F16,
    /// Store as bf16. Lossless for bf16 checkpoints; rounds f32 / f16 ones to
    /// 8 mantissa bits.
    Bf16,
}

/// The process-wide setting key.
pub const WEIGHT_STORAGE_ENV: &str = "RLX_WEIGHT_STORAGE";

impl WeightStorage {
    /// `auto`, `f32`, `f16`, `bf16` (also `native`, `fp16`, `half`, `bfloat16`).
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.trim().to_ascii_lowercase().as_str() {
            "auto" | "native" => Self::Auto,
            "f32" | "fp32" | "float32" => Self::F32,
            "f16" | "fp16" | "half" | "float16" => Self::F16,
            "bf16" | "bfloat16" => Self::Bf16,
            other => bail!("unknown weight storage `{other}` (auto, f32, f16, bf16)"),
        })
    }

    /// `RLX_WEIGHT_STORAGE`, if set (an unparseable value is reported, then
    /// ignored rather than silently guessed).
    pub fn from_env() -> Option<Self> {
        let v = rlx_ir::env::var(WEIGHT_STORAGE_ENV)?;
        match Self::parse(&v) {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("[rlx-flow] {WEIGHT_STORAGE_ENV}: {e}; using f32");
                None
            }
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }

    /// The dtype to store a tensor in, given the dtype it has on disk
    /// (`None` = unknown, treated as f32).
    pub fn resolve(self, stored: Option<DType>) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
            Self::Bf16 => DType::BF16,
            Self::Auto => match stored {
                Some(d @ (DType::F16 | DType::BF16)) => d,
                _ => DType::F32,
            },
        }
    }

    /// Bytes per element of [`Self::resolve`].
    pub fn bytes_per_elem(self, stored: Option<DType>) -> usize {
        match self.resolve(stored) {
            DType::F16 | DType::BF16 => 2,
            _ => 4,
        }
    }
}

impl std::fmt::Display for WeightStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl std::str::FromStr for WeightStorage {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Self::parse(s)
    }
}

/// `data` in `dtype` as little-endian bytes (f16 / bf16 round to nearest
/// even; any other dtype keeps f32).
pub fn encode_f32(data: &[f32], dtype: DType) -> Vec<u8> {
    match dtype {
        DType::F16 => data
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect(),
        DType::BF16 => data
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect(),
        _ => data.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_keeps_the_stored_half_width() {
        assert_eq!(WeightStorage::Auto.resolve(Some(DType::BF16)), DType::BF16);
        assert_eq!(WeightStorage::Auto.resolve(Some(DType::F16)), DType::F16);
        assert_eq!(WeightStorage::Auto.resolve(Some(DType::F32)), DType::F32);
        assert_eq!(WeightStorage::Auto.resolve(None), DType::F32);
        assert_eq!(WeightStorage::F16.resolve(Some(DType::BF16)), DType::F16);
        assert_eq!(WeightStorage::F32.resolve(Some(DType::BF16)), DType::F32);
        assert_eq!(
            "BF16".parse::<WeightStorage>().unwrap(),
            WeightStorage::Bf16
        );
    }
}
