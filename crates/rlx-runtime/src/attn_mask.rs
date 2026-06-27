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

/// Like [`bucket_decode_mask`] but for **sliding-window** attention: the
/// current token (absolute position `past_seq`) attends only to cached keys in
/// `[past_seq - window, past_seq]` — older real-past keys are masked out too,
/// matching `MaskKind::SlidingWindow(window)` in prefill. `window == 0` or
/// `window >= past_seq` reduces to the full causal [`bucket_decode_mask`].
pub fn bucket_decode_mask_windowed(past_seq: usize, upper: usize, window: usize) -> Vec<f32> {
    let lo = if window == 0 {
        0
    } else {
        past_seq.saturating_sub(window)
    };
    (0..=upper)
        .map(|i| {
            let real_past_in_window = i >= lo && i < past_seq;
            if real_past_in_window || i == upper {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windowed_mask_reduces_to_causal_when_wide() {
        // window >= past_seq → identical to the full causal mask.
        assert_eq!(
            bucket_decode_mask_windowed(5, 8, 100),
            bucket_decode_mask(5, 8)
        );
    }

    #[test]
    fn windowed_mask_drops_old_keys() {
        // past_seq=5, window=2 → attend to keys [3,4] + the new token at upper.
        let m = bucket_decode_mask_windowed(5, 6, 2);
        assert_eq!(m, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        //                  0    1    2    3    4   pad  new(=upper)
    }
}
