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

//! INT8 2D convolution, NHWC layout, "valid" or zero-padded.
//!
//! Layout choice: NHWC matches CMSIS-NN and TFLite Micro, and keeps
//! the inner loop unit-stride across input channels — the right shape
//! for ARMv7E-M's SMLAD/SMLALD packed multiply-accumulate (we don't
//! emit those yet, but the layout is the prerequisite).

use crate::quant::{read_weight, requantize, ternary_code};

/// Per-channel INT8 conv2d.
///
/// Shapes:
/// * `x`        — `[h_in, w_in, c_in]`
/// * `w`        — `[c_out, kh, kw, c_in]`
/// * `bias`     — `[c_out]` i32 (per-channel acc-scale `x_scale * w_scale[oc]`) or `None`
/// * `out`      — `[h_out, w_out, c_out]`
///
/// `h_out = (h_in + 2*pad_h - kh) / stride_h + 1` (and same for w).
///
/// `mult` is one f32 per output channel: `mult[oc] = (x_scale * w_scale[oc]) / out_scale`.
/// Per-tensor quantization is the special case where every entry is the same value;
/// per-channel typically buys 0.5–2 pp accuracy on small conv stacks because the
/// "loud" output channels stop saturating the rest into noise.
pub struct Conv2dParams<'m> {
    pub h_in: usize,
    pub w_in: usize,
    pub c_in: usize,
    pub c_out: usize,
    pub kh: usize,
    pub kw: usize,
    pub pad_h: usize,
    pub pad_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub x_zp: i32,
    pub w_zp: i32,
    pub out_zp: i32,
    /// Per-output-channel requantization multiplier; length must equal `c_out`.
    pub mult: &'m [f32],
    /// Bits per packed weight: 8 (raw i8), 4 (nibble-packed), 2 (crumb-packed).
    pub weight_bits: u8,
}

impl Conv2dParams<'_> {
    pub fn h_out(&self) -> usize {
        (self.h_in + 2 * self.pad_h - self.kh) / self.stride_h + 1
    }
    pub fn w_out(&self) -> usize {
        (self.w_in + 2 * self.pad_w - self.kw) / self.stride_w + 1
    }
}

pub fn conv2d_i8(p: &Conv2dParams, x: &[i8], w: &[i8], bias: Option<&[i32]>, out: &mut [i8]) {
    let h_out = p.h_out();
    let w_out = p.w_out();
    let n_logical_weights = p.c_out * p.kh * p.kw * p.c_in;
    debug_assert_eq!(x.len(), p.h_in * p.w_in * p.c_in);
    debug_assert_eq!(
        w.len(),
        crate::quant::packed_byte_len(n_logical_weights, p.weight_bits)
    );
    debug_assert_eq!(out.len(), h_out * w_out * p.c_out);
    debug_assert_eq!(p.mult.len(), p.c_out);

    for oh in 0..h_out {
        for ow in 0..w_out {
            for oc in 0..p.c_out {
                let mut acc: i32 = bias.map(|b| b[oc]).unwrap_or(0);
                for kh in 0..p.kh {
                    let ih = oh * p.stride_h + kh;
                    if ih < p.pad_h || ih >= p.pad_h + p.h_in {
                        continue;
                    }
                    let ih = ih - p.pad_h;
                    for kw in 0..p.kw {
                        let iw = ow * p.stride_w + kw;
                        if iw < p.pad_w || iw >= p.pad_w + p.w_in {
                            continue;
                        }
                        let iw = iw - p.pad_w;
                        let x_base = (ih * p.w_in + iw) * p.c_in;
                        let w_base = ((oc * p.kh + kh) * p.kw + kw) * p.c_in;
                        for ic in 0..p.c_in {
                            let xv = x[x_base + ic] as i32 - p.x_zp;
                            let wv = read_weight(w, w_base + ic, p.weight_bits) - p.w_zp;
                            acc += xv * wv;
                        }
                    }
                }
                out[(oh * w_out + ow) * p.c_out + oc] = requantize(acc, p.mult[oc], p.out_zp);
            }
        }
    }
}

/// Ternary-specialized INT8 conv2d.
///
/// Same shapes and parameters as [`conv2d_i8`], but **assumes
/// `weight_bits == 2`** and that the trainer emitted only the three
/// ternary codes `0` (= 0), `1` (= +1), `3` (= -1). Replaces the
/// `xv * wv` MAC with conditional add/sub, and skips the `xv` load
/// entirely when `wv == 0` — typically ~50 % of weights for ternary
/// PTQ on a Gaussian distribution. Net cycle savings on M4F are
/// ~30–35 %.
///
/// `p.weight_bits` is ignored (asserted == 2 in debug). Caller is
/// expected to dispatch from `model.rs`:
///
/// ```ignore
/// match w::WEIGHT_BITS {
///     2 => conv2d_ternary(&p, x, w, b, out),
///     _ => conv2d_i8(&p, x, w, b, out),
/// }
/// ```
pub fn conv2d_ternary(p: &Conv2dParams, x: &[i8], w: &[i8], bias: Option<&[i32]>, out: &mut [i8]) {
    let h_out = p.h_out();
    let w_out = p.w_out();
    let n_logical_weights = p.c_out * p.kh * p.kw * p.c_in;
    debug_assert_eq!(p.weight_bits, 2, "conv2d_ternary requires weight_bits=2");
    debug_assert_eq!(p.w_zp, 0, "ternary scheme is symmetric (zero_point=0)");
    debug_assert_eq!(x.len(), p.h_in * p.w_in * p.c_in);
    debug_assert_eq!(w.len(), crate::quant::packed_byte_len(n_logical_weights, 2));
    debug_assert_eq!(out.len(), h_out * w_out * p.c_out);
    debug_assert_eq!(p.mult.len(), p.c_out);

    for oh in 0..h_out {
        for ow in 0..w_out {
            for oc in 0..p.c_out {
                let mut acc: i32 = bias.map(|b| b[oc]).unwrap_or(0);
                for kh in 0..p.kh {
                    let ih = oh * p.stride_h + kh;
                    if ih < p.pad_h || ih >= p.pad_h + p.h_in {
                        continue;
                    }
                    let ih = ih - p.pad_h;
                    for kw in 0..p.kw {
                        let iw = ow * p.stride_w + kw;
                        if iw < p.pad_w || iw >= p.pad_w + p.w_in {
                            continue;
                        }
                        let iw = iw - p.pad_w;
                        let x_base = (ih * p.w_in + iw) * p.c_in;
                        let w_base = ((oc * p.kh + kh) * p.kw + kw) * p.c_in;
                        for ic in 0..p.c_in {
                            // Branch on the raw 2-bit code; skip the
                            // x load entirely on zero, replace mul
                            // with add/sub on ±1. Code 2 (= -2) is
                            // never emitted by the trainer's ternary
                            // path so we don't bother handling it.
                            match ternary_code(w, w_base + ic) {
                                0 => {}
                                1 => acc += x[x_base + ic] as i32 - p.x_zp,
                                3 => acc -= x[x_base + ic] as i32 - p.x_zp,
                                _ => debug_assert!(false, "unreachable ternary code"),
                            }
                        }
                    }
                }
                out[(oh * w_out + ow) * p.c_out + oc] = requantize(acc, p.mult[oc], p.out_zp);
            }
        }
    }
}
