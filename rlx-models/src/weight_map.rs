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

//! Safetensors weight loading — standalone, no framework dependency.

use anyhow::{Context, Result};
use std::collections::HashMap;

/// Map of tensor name → (f32 data, shape).
pub struct WeightMap {
    tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl WeightMap {
    /// Load weights from a safetensors file. Auto-converts bf16/f16 to f32.
    pub fn from_file(path: &str) -> Result<Self> {
        let data = std::fs::read(path).with_context(|| format!("reading {path}"))?;
        let st =
            safetensors::SafeTensors::deserialize(&data).with_context(|| "parsing safetensors")?;

        let mut tensors = HashMap::new();
        for (name, view) in st.tensors() {
            let shape: Vec<usize> = view.shape().to_vec();
            let bytes = view.data();
            let f32_data = match view.dtype() {
                safetensors::Dtype::F32 => bytemuck_cast_f32(bytes),
                safetensors::Dtype::F16 => bytes
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect(),
                safetensors::Dtype::BF16 => bytes
                    .chunks_exact(2)
                    .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect(),
                safetensors::Dtype::I64 => bytes
                    .chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
                    .collect(),
                safetensors::Dtype::I32 => bytes
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                    .collect(),
                safetensors::Dtype::C64 => {
                    // Some checkpoints (SAM3) include complex RoPE caches
                    // such as `freqs_cis`. Native code regenerates/handles
                    // those separately; keep loading usable for the real
                    // float weights instead of rejecting the entire file.
                    continue;
                }
                other => anyhow::bail!("unsupported dtype: {other:?}"),
            };
            tensors.insert(name.to_string(), (f32_data, shape));
        }

        Ok(Self { tensors })
    }

    /// Take a tensor by name (removes from map). Returns (data, shape).
    pub fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.tensors
            .remove(key)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {key}"))
    }

    /// Take and transpose a 2D weight: [out, in] → [in, out] for row-major matmul.
    pub fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape) = self.take(key)?;
        if shape.len() != 2 {
            anyhow::bail!("transpose requires 2D, got {shape:?}");
        }
        let (rows, cols) = (shape[0], shape[1]);
        let mut transposed = vec![0f32; data.len()];
        for i in 0..rows {
            for j in 0..cols {
                transposed[j * rows + i] = data[i * cols + j];
            }
        }
        Ok((transposed, vec![cols, rows]))
    }

    /// Check if a key exists.
    pub fn has(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }

    /// List all keys.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }

    /// Number of tensors remaining.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Create from pre-built HashMap (for testing without safetensors files).
    pub fn from_tensors(tensors: HashMap<String, (Vec<f32>, Vec<usize>)>) -> Self {
        Self { tensors }
    }
}

/// Convert a raw byte slice to a `Vec<f32>`. Safetensors stores tensor
/// data at arbitrary byte offsets — when an f32 tensor doesn't land on
/// a 4-byte boundary, `bytemuck::cast_slice` panics with
/// `TargetAlignmentGreaterAndInputNotAligned`. SAM ViT-B is one such
/// file. Fall back to a manual little-endian decode in that case.
fn bytemuck_cast_f32(bytes: &[u8]) -> Vec<f32> {
    debug_assert!(
        bytes.len() % 4 == 0,
        "f32 byte slice length must be multiple of 4 (got {})",
        bytes.len()
    );
    if (bytes.as_ptr() as usize) % std::mem::align_of::<f32>() == 0 {
        let f32s: &[f32] = bytemuck::cast_slice(bytes);
        f32s.to_vec()
    } else {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_2d() {
        let mut wm = WeightMap {
            tensors: HashMap::from([(
                "w".to_string(),
                (vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
            )]),
        };
        let (data, shape) = wm.take_transposed("w").unwrap();
        assert_eq!(shape, vec![3, 2]);
        // Original: [[1,2,3],[4,5,6]] → Transposed: [[1,4],[2,5],[3,6]]
        assert_eq!(data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
