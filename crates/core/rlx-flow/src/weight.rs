// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Weight loading trait — implemented by model-builder `WeightLoader` adapters.

use anyhow::Result;

use crate::GgufPackedLinear;

/// Abstract weight source for block emission. Keeps `rlx-flow` independent of
/// safetensors / GGUF file formats.
pub trait WeightSource {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)>;

    /// Optional probe for arch-specific key layout detection.
    fn has(&self, key: &str) -> bool {
        let _ = key;
        false
    }

    /// Optional packed (quantized) weight for the linear projection `key`.
    ///
    /// When this returns `Some`, [`crate::FlowCtx::linear`] emits a fused
    /// `DequantMatMul` over the U8 quant blob instead of loading an F32 weight —
    /// the *same* flow topology then runs packed at any `m` (m=1 decode, m=N
    /// prefill) with no F32 weight ever materialized. Default: `None`, so
    /// F32-only sources are unaffected. Any model whose loader can hand out
    /// GGUF/MLX quant blobs opts into packed matmuls just by overriding this —
    /// no per-model graph surgery.
    ///
    /// May be called more than once for the same `key` (KV-tap + decoder stages
    /// both load a projection); implementations should return the same weight
    /// each time. The flow dedups the underlying graph param, so the U8 bytes
    /// are registered only once.
    fn take_packed(&mut self, key: &str) -> Result<Option<GgufPackedLinear>> {
        let _ = key;
        Ok(None)
    }
}

impl<T: WeightSource + ?Sized> WeightSource for &mut T {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        (*self).take(key, transpose)
    }
    // Forward `has` too; otherwise a `&mut dyn WeightSource` silently falls
    // back to the trait default (`false`) and key-layout probing breaks.
    fn has(&self, key: &str) -> bool {
        (**self).has(key)
    }
}

/// In-memory weight map for tests and tooling.
#[derive(Debug, Default, Clone)]
pub struct MapWeights {
    pub tensors: std::collections::HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl MapWeights {
    pub fn insert(&mut self, key: impl Into<String>, data: Vec<f32>, shape: Vec<usize>) {
        self.tensors.insert(key.into(), (data, shape));
    }
}

impl WeightSource for MapWeights {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape) = self
            .tensors
            .remove(key)
            .ok_or_else(|| anyhow::anyhow!("missing weight: {key}"))?;
        if !transpose {
            return Ok((data, shape));
        }
        if shape.len() != 2 {
            return Err(anyhow::anyhow!("transpose requires rank-2 weight: {key}"));
        }
        let rows = shape[0];
        let cols = shape[1];
        let mut out = vec![0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = data[r * cols + c];
            }
        }
        Ok((out, vec![cols, rows]))
    }
}
