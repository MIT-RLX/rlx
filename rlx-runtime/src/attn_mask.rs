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

/// Causal decode mask padded to bucket `upper`: `0` for positions `0..=past_seq`,
/// large negative elsewhere (matches CPU `attn_mask_neg_inf` default).
pub fn bucket_decode_mask(past_seq: usize, upper: usize) -> Vec<f32> {
    const NEG: f32 = -1e9;
    (0..upper)
        .map(|i| if i <= past_seq { 0.0 } else { NEG })
        .collect()
}
