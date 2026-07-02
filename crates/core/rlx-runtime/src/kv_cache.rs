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
    /// Absolute token index of row `0` in each layer's K/V buffer after
    /// sliding-window trim (Gemma 3/4 ISWA). Zero when the buffer holds
    /// the full prefix from position `0`.
    pub layers_kv_base: Vec<usize>,
}

impl LayerKvCache {
    pub fn from_layer_outputs(
        num_layers: usize,
        batch: usize,
        past_seq: usize,
        kv_dim: usize,
        outputs: &[Vec<f32>],
    ) -> Result<Self, String> {
        let dims: Vec<usize> = vec![kv_dim; num_layers];
        Self::from_layer_outputs_per_layer(num_layers, batch, past_seq, &dims, outputs)
    }

    /// Like [`Self::from_layer_outputs`] but accepts a per-layer
    /// `kv_dim` vector. Gemma 4 12B's full-attention layers have
    /// `kv_dim = 1 * 512 = 512` while sliding layers have `8 * 256 =
    /// 2048`; this constructor handles that heterogeneity.
    pub fn from_layer_outputs_per_layer(
        num_layers: usize,
        batch: usize,
        past_seq: usize,
        kv_dims: &[usize],
        outputs: &[Vec<f32>],
    ) -> Result<Self, String> {
        if outputs.len() != 2 * num_layers {
            return Err(format!(
                "from_layer_outputs_per_layer: expected {} K/V tensors, got {}",
                2 * num_layers,
                outputs.len()
            ));
        }
        if kv_dims.len() != num_layers {
            return Err(format!(
                "from_layer_outputs_per_layer: expected {} kv_dims, got {}",
                num_layers,
                kv_dims.len()
            ));
        }
        let mut layers_k = Vec::with_capacity(num_layers);
        let mut layers_v = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            let kv_dim = kv_dims[layer];
            let expected = batch * past_seq * kv_dim;
            let k = &outputs[2 * layer];
            let v = &outputs[2 * layer + 1];
            if k.len() != expected || v.len() != expected {
                return Err(format!(
                    "layer {layer}: k.len={} v.len={} expected {expected} (kv_dim={kv_dim})",
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
            layers_kv_base: vec![0; num_layers],
        })
    }

    /// Pad each layer's K/V to `upper` rows along the sequence axis (`kv_dim` inner).
    pub fn pad_layers_to_upper(&self, upper: u64, kv_dim: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let dims: Vec<usize> = vec![kv_dim; self.layers_k.len()];
        self.pad_layers_to_upper_per_layer(upper, &dims)
    }

    /// Like [`Self::pad_layers_to_upper`] but pads each layer to its
    /// own `kv_dim`. The number of dims must equal the number of
    /// cached layers.
    pub fn pad_layers_to_upper_per_layer(
        &self,
        upper: u64,
        kv_dims: &[usize],
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        assert_eq!(
            kv_dims.len(),
            self.layers_k.len(),
            "pad_layers_to_upper_per_layer: kv_dims len {} != layers {}",
            kv_dims.len(),
            self.layers_k.len(),
        );
        let padded_k = self
            .layers_k
            .iter()
            .zip(kv_dims.iter())
            .map(|(k, &d)| pad_rows(k, d, upper))
            .collect();
        let padded_v = self
            .layers_v
            .iter()
            .zip(kv_dims.iter())
            .map(|(v, &d)| pad_rows(v, d, upper))
            .collect();
        (padded_k, padded_v)
    }

    /// Update cache from decode outputs: `[logits, k0, v0, k1, v1, …]` (bucket-padded).
    pub fn advance_from_decode_outputs(
        &mut self,
        outputs: Vec<Vec<f32>>,
        batch: usize,
        kv_dim: usize,
    ) -> Result<(), String> {
        let dims: Vec<usize> = vec![kv_dim; self.layers_k.len()];
        self.advance_from_decode_outputs_per_layer(outputs, batch, &dims)
    }

    /// Trim each layer's K/V history to at most `window` rows on
    /// the sequence axis, keeping the most recent rows. Used by
    /// Gemma 3/4 sliding-attention layers — long contexts can keep
    /// only the last `window` (e.g. 1024) tokens per sliding layer
    /// without affecting attention semantics (those layers mask out
    /// older positions anyway).
    ///
    /// `kv_dims_keep` selects which layers to trim and at what dim:
    /// `kv_dims_keep[i] = Some((dim, window))` trims layer `i`,
    /// `None` leaves the layer untouched. Pass-through for layers
    /// whose attention is full-causal.
    ///
    /// Note: `past_len` is unchanged — the per-layer K/V buffers
    /// just hold fewer real rows now; the decode flow's per-layer
    /// `past_k_{i}` input shape will see the trimmed length. Caller
    /// is responsible for ensuring the graph's declared `past_seq`
    /// matches the trimmed length OR the trimmed layer is bound
    /// dynamically.
    pub fn trim_sliding_window_per_layer(
        &mut self,
        kv_dims_keep: &[Option<(usize, usize)>],
    ) -> Result<(), String> {
        if kv_dims_keep.len() != self.layers_k.len() {
            return Err(format!(
                "trim_sliding_window_per_layer: kv_dims_keep len {} != layers {}",
                kv_dims_keep.len(),
                self.layers_k.len(),
            ));
        }
        for (i, spec) in kv_dims_keep.iter().enumerate() {
            let Some((kv_dim, window)) = spec else {
                continue;
            };
            let kv_dim = *kv_dim;
            let window = *window;
            if window == 0 || kv_dim == 0 {
                continue;
            }
            let rows = self.layers_k[i].len() / kv_dim;
            if rows <= window {
                continue;
            }
            let drop_rows = rows - window;
            let drop_bytes = drop_rows * kv_dim;
            self.layers_k[i].drain(..drop_bytes);
            self.layers_v[i].drain(..drop_bytes);
            if self.layers_kv_base.len() <= i {
                self.layers_kv_base.resize(self.layers_k.len(), 0);
            }
            self.layers_kv_base[i] = self.layers_kv_base[i].saturating_add(drop_rows);
        }
        Ok(())
    }

    /// Per-layer variant of [`Self::advance_from_decode_outputs`].
    pub fn advance_from_decode_outputs_per_layer(
        &mut self,
        outputs: Vec<Vec<f32>>,
        _batch: usize,
        kv_dims: &[usize],
    ) -> Result<(), String> {
        let n = self.layers_k.len();
        if outputs.len() != 1 + 2 * n {
            return Err(format!(
                "advance_from_decode_outputs_per_layer: expected {} outputs, got {}",
                1 + 2 * n,
                outputs.len()
            ));
        }
        if kv_dims.len() != n {
            return Err(format!(
                "advance_from_decode_outputs_per_layer: kv_dims len {} != layers {n}",
                kv_dims.len()
            ));
        }
        let new_len = self.past_len + 1;
        let mut iter = outputs.into_iter();
        let _logits = iter.next().ok_or("missing logits")?;
        for i in 0..n {
            let k = iter.next().ok_or("missing k")?;
            let v = iter.next().ok_or("missing v")?;
            let real_len = new_len * kv_dims[i];
            self.layers_k[i] = k[..real_len.min(k.len())].to_vec();
            self.layers_v[i] = v[..real_len.min(v.len())].to_vec();
        }
        self.past_len = new_len;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sliding_window_trim_keeps_last_w_rows() {
        // 3 layers, each storing 6 rows of kv_dim=4 = 24 floats.
        let kv_dim = 4;
        let rows = 6;
        let mut cache = LayerKvCache {
            past_len: rows,
            layers_k: vec![(0..(rows * kv_dim)).map(|x| x as f32).collect(); 3],
            layers_v: vec![(0..(rows * kv_dim)).map(|x| x as f32).collect(); 3],
            layers_kv_base: vec![0; 3],
        };
        // Trim layer 0 to last 2 rows; layer 1 untouched; layer 2 to last 4.
        let spec = [Some((kv_dim, 2)), None, Some((kv_dim, 4))];
        cache.trim_sliding_window_per_layer(&spec).unwrap();
        assert_eq!(cache.layers_k[0].len(), 2 * kv_dim);
        assert_eq!(cache.layers_kv_base[0], 4);
        assert_eq!(cache.layers_kv_base[1], 0);
        assert_eq!(cache.layers_kv_base[2], 2);
        // Layer 0 should now hold the LAST 2 rows: rows 4 and 5.
        assert_eq!(
            cache.layers_k[0],
            vec![16., 17., 18., 19., 20., 21., 22., 23.]
        );
        assert_eq!(
            cache.layers_k[1].len(),
            6 * kv_dim,
            "untouched layer keeps full history"
        );
        assert_eq!(cache.layers_k[2].len(), 4 * kv_dim);
    }

    #[test]
    fn sliding_window_trim_no_op_when_under_window() {
        let kv_dim = 4;
        let rows = 3;
        let mut cache = LayerKvCache {
            past_len: rows,
            layers_k: vec![vec![1.0f32; rows * kv_dim]],
            layers_v: vec![vec![2.0f32; rows * kv_dim]],
            layers_kv_base: vec![0],
        };
        cache
            .trim_sliding_window_per_layer(&[Some((kv_dim, 10))])
            .unwrap();
        assert_eq!(cache.layers_k[0].len(), rows * kv_dim);
    }
}
