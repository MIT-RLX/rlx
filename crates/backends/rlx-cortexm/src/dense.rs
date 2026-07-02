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

//! INT8 dense (fully-connected) layer.
//!
//! `out[m] = relu_optional( requantize( bias[m] + sum_k (x[k] - x_zp) * (w[m,k] - w_zp) ) )`
//!
//! Bias is `i32` in the same scale as the accumulator (`x_scale * w_scale`)
//! — same convention TFLite Micro uses.

use crate::quant::{read_weight, requantize, sat_i8, ternary_code};

/// Compute one dense layer.
///
/// * `x`           — `[in_features]` activations, i8.
/// * `w`           — `[out_features, in_features]` weights, row-major,
///                   packed at `weight_bits` bits per element.
/// * `bias`        — `[out_features]` i32 (per-row acc-scale `x_scale * w_scale[m]`), or `None`.
/// * `x_zp`        — input zero point.
/// * `w_zp`        — weight zero point.
/// * `out_zp`      — output zero point.
/// * `mult`        — `[out_features]` per-row `(x_scale * w_scale[m]) / out_scale`.
/// * `weight_bits` — bits per logical weight: 8, 4 or 2.
/// * `out`         — `[out_features]` output buffer, i8.
pub fn dense_i8(
    x: &[i8],
    w: &[i8],
    bias: Option<&[i32]>,
    x_zp: i32,
    w_zp: i32,
    out_zp: i32,
    mult: &[f32],
    weight_bits: u8,
    out: &mut [i8],
) {
    let out_features = out.len();
    let in_features = x.len();
    let n_logical = out_features * in_features;
    debug_assert_eq!(
        w.len(),
        crate::quant::packed_byte_len(n_logical, weight_bits)
    );
    debug_assert_eq!(mult.len(), out_features);

    for m in 0..out_features {
        let mut acc: i32 = bias.map(|b| b[m]).unwrap_or(0);
        let row_base = m * in_features;
        for k in 0..in_features {
            let wv = read_weight(w, row_base + k, weight_bits) - w_zp;
            acc += (x[k] as i32 - x_zp) * wv;
        }
        out[m] = requantize(acc, mult[m], out_zp);
    }

    let _ = sat_i8; // keep warning-free if requantize ever inlines away.
}

/// Ternary-specialized dense layer.
///
/// Same arguments as [`dense_i8`] minus `weight_bits` (always 2).
/// Replaces `xv * wv` with conditional add/sub and skips the `xv`
/// load when `wv == 0`. Caller dispatches from `model.rs`:
///
/// ```ignore
/// match w::WEIGHT_BITS {
///     2 => dense_ternary(...),
///     _ => dense_i8(..., WEIGHT_BITS, ...),
/// }
/// ```
pub fn dense_ternary(
    x: &[i8],
    w: &[i8],
    bias: Option<&[i32]>,
    x_zp: i32,
    w_zp: i32,
    out_zp: i32,
    mult: &[f32],
    out: &mut [i8],
) {
    let out_features = out.len();
    let in_features = x.len();
    let n_logical = out_features * in_features;
    debug_assert_eq!(w_zp, 0, "ternary scheme is symmetric (zero_point=0)");
    debug_assert_eq!(w.len(), crate::quant::packed_byte_len(n_logical, 2));
    debug_assert_eq!(mult.len(), out_features);

    for m in 0..out_features {
        let mut acc: i32 = bias.map(|b| b[m]).unwrap_or(0);
        let row_base = m * in_features;
        for k in 0..in_features {
            match ternary_code(w, row_base + k) {
                0 => {}
                1 => acc += x[k] as i32 - x_zp,
                3 => acc -= x[k] as i32 - x_zp,
                _ => debug_assert!(false, "unreachable ternary code"),
            }
        }
        out[m] = requantize(acc, mult[m], out_zp);
    }
}
