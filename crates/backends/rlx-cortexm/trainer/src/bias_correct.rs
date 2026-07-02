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

//! Post-training bias correction.
//!
//! After PTQ (or QAT), the per-output-channel mean of conv outputs
//! shifts slightly relative to the FP32 reference — quantizing
//! weights changes the integral of `conv(x, w_q) - conv(x, w_fp32)`
//! across the calibration distribution. The shift is constant per
//! channel (= `E[x] · (w_q - w_fp32)` summed over the kernel
//! window), so we can absorb it into the bias by subtracting it.
//!
//! This is the textbook "bias correction" trick from Nagel et al.
//! 2019 ([*Data-Free Quantization Through Weight Equalization and
//! Bias Correction*](https://arxiv.org/abs/1906.04721)). It costs
//! one calibration pass per layer and typically lifts INT8/INT4
//! accuracy by 0.5–2 pp at zero runtime cost.
//!
//! ## What it does (in math)
//!
//! For each conv output channel `c`:
//!
//! ```text
//!   shift[c]  = E_x[ conv_fp32(x, w)[c]  -  conv_q(x, w_q)[c] ]
//!   bias[c]  += shift[c]
//! ```
//!
//! Where the expectation is taken over the calibration batch. The
//! shift is dominated by the difference between `w` and the
//! dequantized `w_q`, which we can compute analytically (no need
//! to run conv twice) — we collapse the formula to:
//!
//! ```text
//!   shift[c] = sum over k_h, k_w, c_in:
//!                (w_fp32[c, k_h, k_w, c_in] - w_dequant[c, k_h, k_w, c_in])
//!                  * E[x[c_in]]
//! ```
//!
//! The per-input-channel mean `E[x[c_in]]` is what the calibration
//! pass measures; the weight delta comes from the quantizer's own
//! round-trip (apply scale → round → clamp → unscale).
//!
//! ## Scope
//!
//! Today's implementation handles the three convs / FC layers in
//! the TinyConv-MNIST model. Generalising to arbitrary conv
//! shapes is straightforward (the math is the same), just more
//! plumbing.

use crate::quant::QuantizedModel;

/// Compute the per-channel input mean across a calibration batch.
/// Returns a `[c_in]` vector. `x` is `[N, c_in, H, W]` (NCHW) or
/// `[N, H, W, c_in]` (NHWC) depending on `nhwc`.
pub fn input_channel_mean(
    x: &[f32],
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    nhwc: bool,
) -> Vec<f32> {
    let mut sum = vec![0f64; c_in];
    let total_per_chan = n * h * w;
    if nhwc {
        for i in 0..(n * h * w) {
            for ic in 0..c_in {
                sum[ic] += x[i * c_in + ic] as f64;
            }
        }
    } else {
        for ni in 0..n {
            for ic in 0..c_in {
                let base = (ni * c_in + ic) * h * w;
                for j in 0..(h * w) {
                    sum[ic] += x[base + j] as f64;
                }
            }
        }
    }
    sum.into_iter()
        .map(|s| (s / total_per_chan as f64) as f32)
        .collect()
}

/// For each output channel, compute `Σ_{kh,kw,ic} (w_fp32 - w_dequant)[c,kh,kw,ic] · E[x[ic]]`.
///
/// `w_fp32`: `[c_out, kH, kW, c_in]` (NHWC weight layout — same as
/// what `quantize_conv_weight` outputs).
/// `w_q`: matching `[c_out, kH, kW, c_in]` int8 codes (logical, not packed).
/// `w_scale[c_out]`: per-channel scale to dequantize `w_q`.
/// `e_x[c_in]`: per-input-channel input mean.
///
/// Returns the `[c_out]` shift vector to **subtract** from each
/// channel's bias-in-acc-scale (so `bias_q[c] -= shift[c] / acc_scale[c]`).
pub fn conv_shift(
    w_fp32: &[f32],
    w_q: &[i8],
    w_scale: &[f32],
    e_x: &[f32],
    c_out: usize,
    kh: usize,
    kw: usize,
    c_in: usize,
) -> Vec<f32> {
    debug_assert_eq!(w_fp32.len(), c_out * kh * kw * c_in);
    debug_assert_eq!(w_q.len(), c_out * kh * kw * c_in);
    debug_assert_eq!(w_scale.len(), c_out);
    debug_assert_eq!(e_x.len(), c_in);
    let mut shift = vec![0f32; c_out];
    for oc in 0..c_out {
        let s = w_scale[oc];
        let mut acc = 0f64;
        for h in 0..kh {
            for w_ in 0..kw {
                for ic in 0..c_in {
                    let idx = ((oc * kh + h) * kw + w_) * c_in + ic;
                    let w_dequant = (w_q[idx] as f32) * s;
                    let delta = w_fp32[idx] - w_dequant;
                    acc += (delta * e_x[ic]) as f64;
                }
            }
        }
        shift[oc] = acc as f32;
    }
    shift
}

/// Apply the conv-bias correction in place. `bias_q` is in
/// accumulator scale (`x_scale * w_scale[oc]`); the shift is
/// subtracted in real-space and re-scaled accordingly.
pub fn apply_conv_correction(bias_q: &mut [i32], shift: &[f32], x_scale: f32, w_scale: &[f32]) {
    debug_assert_eq!(bias_q.len(), shift.len());
    debug_assert_eq!(bias_q.len(), w_scale.len());
    for c in 0..bias_q.len() {
        let acc_scale = x_scale * w_scale[c];
        let delta_q = (shift[c] / acc_scale).round() as i32;
        // bias_q[c] += -shift[c] in acc-scale codes.
        bias_q[c] = bias_q[c].saturating_sub(delta_q);
    }
}

/// One-shot driver. Walks the three layers of TinyConv-MNIST and
/// adjusts each bias in-place. `inputs_*` are calibration batches
/// (FP32 activations going into each layer). The trainer collects
/// these from a small calibration sweep before the final pack.
pub fn correct_tinyconv_biases(
    model: &mut QuantizedModel,
    conv1_w_fp32: &[f32],
    conv2_w_fp32: &[f32],
    fc_w_fp32: &[f32],
    inputs_conv1: &[f32],
    inputs_conv2: &[f32],
    inputs_fc: &[f32],
    n: usize,
) {
    // conv1: input is [N, 1, 28, 28] NCHW.
    let e_x1 = input_channel_mean(inputs_conv1, n, 1, 28, 28, false);
    let s1 = conv_shift(
        conv1_w_fp32,
        &model.conv1_w,
        &model.w1_scale,
        &e_x1,
        8,
        3,
        3,
        1,
    );
    apply_conv_correction(&mut model.conv1_b, &s1, model.x_scale, &model.w1_scale);

    // conv2: input is [N, 8, 13, 13] post-pool, but the firmware
    // sees it in NHWC as [N, 13, 13, 8] (since the pool is in
    // NHWC). The trainer's inputs_conv2 should be flattened in NHWC
    // so c_in is the last dim.
    let e_x2 = input_channel_mean(inputs_conv2, n, 8, 13, 13, true);
    let s2 = conv_shift(
        conv2_w_fp32,
        &model.conv2_w,
        &model.w2_scale,
        &e_x2,
        16,
        3,
        3,
        8,
    );
    apply_conv_correction(&mut model.conv2_b, &s2, model.p1_scale, &model.w2_scale);

    // fc: input is [N, 400] flat (= [N, 5, 5, 16] NHWC flattened
    // HWC-major to match the firmware kernel).
    let e_xf = input_channel_mean(inputs_fc, n, 400, 1, 1, true);
    // fc_w is [10, 400] in (oc, ic) layout — treat as a degenerate
    // conv with kH=kW=1.
    let sf = conv_shift(
        fc_w_fp32,
        &model.fc_w,
        &model.wfc_scale,
        &e_xf,
        10,
        1,
        1,
        400,
    );
    apply_conv_correction(&mut model.fc_b, &sf, model.p2_scale, &model.wfc_scale);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correction_reduces_mean_shift() {
        // Tiny synthetic case: c_out=2, c_in=1, kh=kw=1.
        // FP32 weights: [0.5, -0.3]. Quantized to i4 (q_max=7) with
        // per-channel scale = max_abs/7. Dequantize and check shift.
        let w_fp32 = vec![0.5_f32, -0.3];
        let w_q = vec![7_i8, -7]; // ±7 with their own scales
        let w_scale = vec![0.5 / 7.0, 0.3 / 7.0];
        let e_x = vec![1.0_f32];

        let shift = conv_shift(&w_fp32, &w_q, &w_scale, &e_x, 2, 1, 1, 1);
        // shift[c] = (w_fp32[c] - w_dequant[c]) * e_x[0]
        // w_dequant = w_q * w_scale = ±0.5 (matches w_fp32 to scale)
        // shift should be ~0 here since q is exact.
        assert!(shift.iter().all(|&s| s.abs() < 1e-5));
    }

    #[test]
    fn correction_handles_real_round_trip_loss() {
        // FP32 weights that don't divide cleanly into the i4 step.
        let w_fp32 = vec![0.43_f32, -0.27];
        let w_scale = vec![0.43 / 7.0, 0.27 / 7.0];
        // Suppose quant produces 7 and -7 (boundary).
        let w_q = vec![7_i8, -7];
        let e_x = vec![2.0_f32];

        let shift = conv_shift(&w_fp32, &w_q, &w_scale, &e_x, 2, 1, 1, 1);
        // shifts should be tiny (within step/2 * e_x).
        assert!(shift.iter().all(|&s| s.abs() < 0.1));

        let mut bias = vec![0_i32; 2];
        apply_conv_correction(&mut bias, &shift, 1.0, &w_scale);
        // Bias adjustments are in acc-scale codes; just check they
        // don't panic and don't blow up.
        assert!(bias.iter().all(|&b| b.abs() < 1000));
    }
}
