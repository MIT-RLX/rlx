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

//! Per-layer K/V cache for autoregressive decode (Whisper, Qwen, Gemma, …).

use crate::compile_cache::pad_rows;

/// Layer-wise past K/V tensors in row-major `[past_len * kv_dim]` layout per layer.
#[derive(Debug, Clone)]
pub struct LayerKvCache {
    pub past_len: usize,
    pub layers_k: Vec<Vec<f32>>,
    pub layers_v: Vec<Vec<f32>>,
}

impl LayerKvCache {
    pub fn from_layer_outputs(
        num_layers: usize,
        batch: usize,
        past_seq: usize,
        kv_dim: usize,
        outputs: &[Vec<f32>],
    ) -> Result<Self, String> {
        if outputs.len() != 2 * num_layers {
            return Err(format!(
                "from_layer_outputs: expected {} K/V tensors, got {}",
                2 * num_layers,
                outputs.len()
            ));
        }
        let expected = batch * past_seq * kv_dim;
        let mut layers_k = Vec::with_capacity(num_layers);
        let mut layers_v = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            let k = &outputs[2 * layer];
            let v = &outputs[2 * layer + 1];
            if k.len() != expected || v.len() != expected {
                return Err(format!(
                    "layer {layer}: k.len={} v.len={} expected {expected}",
                    k.len(),
                    v.len()
                ));
            }
            layers_k.push(k.clone());
            layers_v.push(v.clone());
        }
        Ok(Self {
            past_len: past_seq,
            layers_k,
            layers_v,
        })
    }

    /// Pad each layer's K/V to `upper` rows along the sequence axis (`kv_dim` inner).
    pub fn pad_layers_to_upper(&self, upper: u64, kv_dim: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let padded_k = self
            .layers_k
            .iter()
            .map(|k| pad_rows(k, kv_dim, upper))
            .collect();
        let padded_v = self
            .layers_v
            .iter()
            .map(|v| pad_rows(v, kv_dim, upper))
            .collect();
        (padded_k, padded_v)
    }

    /// Update cache from decode outputs: `[logits, k0, v0, k1, v1, …]` (bucket-padded).
    pub fn advance_from_decode_outputs(
        &mut self,
        outputs: Vec<Vec<f32>>,
        _batch: usize,
        kv_dim: usize,
    ) -> Result<(), String> {
        let n = self.layers_k.len();
        if outputs.len() != 1 + 2 * n {
            return Err(format!(
                "advance_from_decode_outputs: expected {} outputs, got {}",
                1 + 2 * n,
                outputs.len()
            ));
        }
        let new_len = self.past_len + 1;
        let real_len = new_len * kv_dim;
        let mut iter = outputs.into_iter();
        let _logits = iter.next().ok_or("missing logits")?;
        for i in 0..n {
            let k = iter.next().ok_or("missing k")?;
            let v = iter.next().ok_or("missing v")?;
            self.layers_k[i] = k[..real_len.min(k.len())].to_vec();
            self.layers_v[i] = v[..real_len.min(v.len())].to_vec();
        }
        self.past_len = new_len;
        Ok(())
    }
}
