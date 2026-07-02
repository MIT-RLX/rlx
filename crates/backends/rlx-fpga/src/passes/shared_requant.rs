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

//! Detect Conv2d / Dense layers where every per-channel `(M0, shift)`
//! pair is identical. When that's true, the per-OC requant ROMs collapse
//! to two `localparam`s — saves two BRAMs per qualifying layer.
//!
//! Common in:
//! * **Per-tensor weight quantization** — the trainer's first cut on
//!   small models. Every output channel has the same scale, so every
//!   `(x_scale · w_scale) / out_scale` is the same float, and quantizing
//!   it gives the same `(M0, shift)`.
//! * **Layers where the trainer assigned a single global scale post-hoc**.

use super::conv_dense_requant;
use crate::model::Layer;

pub fn uniform_requant(layer: &Layer) -> Option<(i32, i32)> {
    let table = conv_dense_requant(layer)?;
    let first = *table.first()?;
    if table.iter().all(|&p| p == first) {
        Some(first)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Layer;
    use crate::quant::quantize_multiplier;

    fn dense_with_requant(table: Vec<(i32, i32)>) -> Layer {
        Layer::Dense {
            name: "d",
            in_features: 4,
            out_features: table.len(),
            x_zp: 0,
            w_zp: 0,
            out_zp: 0,
            weight_bits: 8,
            requant: table,
            weights: vec![1; 4 * 4],
            bias: None,
        }
    }

    #[test]
    fn uniform_table_detected() {
        let (m0, sh) = quantize_multiplier(0.5);
        let l = dense_with_requant(vec![(m0, sh); 4]);
        assert_eq!(uniform_requant(&l), Some((m0, sh)));
    }

    #[test]
    fn nonuniform_table_rejected() {
        let (m0a, sha) = quantize_multiplier(0.5);
        let (m0b, shb) = quantize_multiplier(0.25);
        let l = dense_with_requant(vec![(m0a, sha), (m0b, shb)]);
        assert_eq!(uniform_requant(&l), None);
    }
}
