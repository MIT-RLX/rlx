// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Weight loading trait — implemented by model-builder `WeightLoader` adapters.

use anyhow::Result;

/// Abstract weight source for block emission. Keeps `rlx-flow` independent of
/// safetensors / GGUF file formats.
pub trait WeightSource {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)>;

    /// Optional probe for arch-specific key layout detection.
    fn has(&self, key: &str) -> bool {
        let _ = key;
        false
    }
}

impl<T: WeightSource + ?Sized> WeightSource for &mut T {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        (*self).take(key, transpose)
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
