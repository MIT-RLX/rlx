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

//! Detect Conv2d / Dense layers where both the input and weight zero
//! points are zero. When that's true, the MAC simplifies from
//!
//! ```text
//!     acc += (x - x_zp) * (w - w_zp)        // 2 subs + 1 mul per MAC
//! ```
//!
//! to
//!
//! ```text
//!     acc += x * w                          // 1 mul per MAC
//! ```
//!
//! For symmetric per-tensor quantization (the cortexm trainer's default
//! today) every layer qualifies. The optimizer always sets the hint
//! when `tune.fold_zero_zp` is on; codegen drops the subtractors.

use super::conv_dense_zps;
use crate::model::Layer;

/// True when both zero points are exactly zero.
pub fn layer_has_zero_zps(layer: &Layer) -> bool {
    matches!(conv_dense_zps(layer), Some((0, 0)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::tinyconv_mnist_from_cortexm;

    #[test]
    fn tinyconv_mnist_every_conv_dense_qualifies() {
        let m = tinyconv_mnist_from_cortexm();
        for l in &m.layers {
            if matches!(l, Layer::Conv2d { .. } | Layer::Dense { .. }) {
                assert!(
                    layer_has_zero_zps(l),
                    "expected fold_zero_zp on {}",
                    l.name()
                );
            }
        }
    }

    #[test]
    fn nonzero_zp_disqualifies() {
        use crate::model::Layer;
        use crate::quant::quantize_multiplier;
        let (m0, sh) = quantize_multiplier(0.5);
        let l = Layer::Dense {
            name: "d",
            in_features: 4,
            out_features: 1,
            x_zp: 3,
            w_zp: 0,
            out_zp: 0,
            weight_bits: 8,
            requant: vec![(m0, sh)],
            weights: vec![1, 2, 3, 4],
            bias: None,
        };
        assert!(!layer_has_zero_zps(&l));
    }
}
