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

//! Attention-mask helpers for bucketed decode (pad-to-upper, slice-back).

/// Causal decode mask padded to bucket `upper`.
///
/// **K layout** (after `concat(past_k, new_k_rope, dim=1)` in the decode graph):
/// - `K[0..past_seq]`: real prompt K from the cache
/// - `K[past_seq..upper]`: zero padding from `pad_layers_to_upper`
/// - `K[upper]`: the newly rope-rotated K for the current token
///
/// So `mask[upper] = 1.0` (attend to self) and `mask[past_seq..upper] = 0.0`
/// (mask the padding). An earlier off-by-one version attended padding at
/// `past_seq` AND masked the new K at `upper`, which removed self-attention
/// to the current token's own K and collapsed decode to degenerate tokens —
/// `\n\n\n…` for short prompts, `attention attention attention…` for longer
/// ones.
///
/// **Convention**: the IR's `MaskKind::Custom` is a **binary keep mask** —
/// the CPU executor and Metal/MLX lowering all gate scores with a
/// `m[ki] < 0.5` threshold.
pub fn bucket_decode_mask(past_seq: usize, upper: usize) -> Vec<f32> {
    // Graph mask shape is `[batch, upper + 1]` (past keys + new token).
    (0..=upper)
        .map(|i| {
            if i < past_seq || i == upper {
                // real past K, or newly-rope'd K for the current decode position
                1.0
            } else {
                0.0 // padding
            }
        })
        .collect()
}
