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

//! Detect ternary (weight_bits=2) Conv2d / Dense layers — eligible for
//! the add/sub/skip codegen path that drops the multiplier entirely.
//!
//! With `weight ∈ {-1, 0, 1}` the MAC becomes a 4-way mux on the crumb:
//!
//! ```text
//!     case (crumb)
//!         2'b00: ;                      // *0   — skip
//!         2'b01: acc += x_signed;       // *1   — add
//!         2'b10: acc -= x_signed << 1;  // *-2  — unused, kept for safety
//!         2'b11: acc -= x_signed;       // *-1  — sub
//!     endcase
//! ```
//!
//! No DSP slice is needed; the operation is a few muxes plus a 32-bit
//! adder. On real silicon this is the largest single energy / area
//! win in the whole tuning surface for ternary models.

use crate::model::Layer;

/// True when the layer is `weight_bits = 2` *and* `w_zp = 0` — the
/// ternary fast path assumes weights already encode `{-1, 0, 1}` directly,
/// without a zero-point shift to subtract first.
pub fn is_ternary(layer: &Layer) -> bool {
    matches!(
        layer,
        Layer::Conv2d {
            weight_bits: 2,
            w_zp: 0,
            ..
        } | Layer::Dense {
            weight_bits: 2,
            w_zp: 0,
            ..
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Layer;
    use crate::quant::quantize_multiplier;

    #[test]
    fn ternary_dense_detected() {
        let (m0, sh) = quantize_multiplier(0.5);
        let l = Layer::Dense {
            name: "d",
            in_features: 4,
            out_features: 1,
            x_zp: 0,
            w_zp: 0,
            out_zp: 0,
            weight_bits: 2,
            requant: vec![(m0, sh)],
            weights: vec![0x4D],
            bias: None,
        };
        assert!(is_ternary(&l));
    }

    #[test]
    fn eight_bit_dense_rejected() {
        let (m0, sh) = quantize_multiplier(0.5);
        let l = Layer::Dense {
            name: "d",
            in_features: 4,
            out_features: 1,
            x_zp: 0,
            w_zp: 0,
            out_zp: 0,
            weight_bits: 8,
            requant: vec![(m0, sh)],
            weights: vec![1; 4],
            bias: None,
        };
        assert!(!is_ternary(&l));
    }
}
