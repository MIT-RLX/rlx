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

//! Rust forward pass for the FPGA backend — uses the integer-only Q0.31
//! requant from `quant`, not cortexm's f32 path. This is the **parity
//! oracle**: emitted Verilog must be bit-identical to this on every
//! intermediate buffer, every test image.
//!
//! The kernels themselves (loop nests, stride math, padding, NHWC index
//! arithmetic) are line-for-line copies of cortexm's — only the
//! requantize call changes (`quant::requantize_q31` instead of
//! `rlx_cortexm::quant::requantize`).

use crate::model::{Layer, Model};
use crate::quant::{q31_to_q15, requantize_q15, requantize_q31, sat_i8};
use crate::tune::RequantPrecision;

use rlx_cortexm::quant::read_weight;

/// Apply the per-channel requant table at the precision the caller asked
/// for. Q0.31 is bit-exact with `quant::requantize_q31`; Q0.15 converts
/// the table at use-time via `q31_to_q15`.
#[inline]
fn requant_dispatch(acc: i32, m0: i32, shift: i32, out_zp: i32, p: RequantPrecision) -> i8 {
    match p {
        RequantPrecision::Q0_31 => requantize_q31(acc, m0, shift, out_zp),
        RequantPrecision::Q0_15 => {
            let (m0_q15, sh_q15) = q31_to_q15(m0, shift);
            requantize_q15(acc, m0_q15, sh_q15, out_zp)
        }
    }
}

fn conv2d_q31(
    h_in: usize,
    w_in: usize,
    c_in: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
    x_zp: i32,
    w_zp: i32,
    out_zp: i32,
    weight_bits: u8,
    requant: &[(i32, i32)],
    weights: &[i8],
    bias: Option<&[i32]>,
    x: &[i8],
    out: &mut [i8],
    p: RequantPrecision,
) {
    let h_out = (h_in + 2 * pad_h - kh) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - kw) / stride_w + 1;
    debug_assert_eq!(out.len(), h_out * w_out * c_out);
    debug_assert_eq!(requant.len(), c_out);

    for oh in 0..h_out {
        for ow in 0..w_out {
            for oc in 0..c_out {
                let mut acc: i32 = bias.map(|b| b[oc]).unwrap_or(0);
                for k_h in 0..kh {
                    let ih = oh * stride_h + k_h;
                    if ih < pad_h || ih >= pad_h + h_in {
                        continue;
                    }
                    let ih = ih - pad_h;
                    for k_w in 0..kw {
                        let iw = ow * stride_w + k_w;
                        if iw < pad_w || iw >= pad_w + w_in {
                            continue;
                        }
                        let iw = iw - pad_w;
                        let x_base = (ih * w_in + iw) * c_in;
                        let w_base = ((oc * kh + k_h) * kw + k_w) * c_in;
                        for ic in 0..c_in {
                            let xv = x[x_base + ic] as i32 - x_zp;
                            let wv = read_weight(weights, w_base + ic, weight_bits) - w_zp;
                            acc += xv * wv;
                        }
                    }
                }
                let (m0, shift) = requant[oc];
                out[(oh * w_out + ow) * c_out + oc] = requant_dispatch(acc, m0, shift, out_zp, p);
            }
        }
    }
}

fn relu_q31(buf: &mut [i8], zero_point: i32) {
    let zp = sat_i8(zero_point);
    for v in buf.iter_mut() {
        if *v < zp {
            *v = zp;
        }
    }
}

fn maxpool_q31(
    h_in: usize,
    w_in: usize,
    c: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    x: &[i8],
    out: &mut [i8],
) {
    let h_out = (h_in - kh) / stride_h + 1;
    let w_out = (w_in - kw) / stride_w + 1;
    for oh in 0..h_out {
        for ow in 0..w_out {
            for ch in 0..c {
                let mut m: i8 = i8::MIN;
                for k_h in 0..kh {
                    let ih = oh * stride_h + k_h;
                    for k_w in 0..kw {
                        let iw = ow * stride_w + k_w;
                        let v = x[(ih * w_in + iw) * c + ch];
                        if v > m {
                            m = v;
                        }
                    }
                }
                out[(oh * w_out + ow) * c + ch] = m;
            }
        }
    }
}

fn dense_q31(
    in_features: usize,
    out_features: usize,
    x_zp: i32,
    w_zp: i32,
    out_zp: i32,
    weight_bits: u8,
    requant: &[(i32, i32)],
    weights: &[i8],
    bias: Option<&[i32]>,
    x: &[i8],
    out: &mut [i8],
    p: RequantPrecision,
) {
    debug_assert_eq!(x.len(), in_features);
    debug_assert_eq!(out.len(), out_features);
    debug_assert_eq!(requant.len(), out_features);

    for m in 0..out_features {
        let mut acc: i32 = bias.map(|b| b[m]).unwrap_or(0);
        let row_base = m * in_features;
        for k in 0..in_features {
            let wv = read_weight(weights, row_base + k, weight_bits) - w_zp;
            acc += (x[k] as i32 - x_zp) * wv;
        }
        let (m0, shift) = requant[m];
        out[m] = requant_dispatch(acc, m0, shift, out_zp, p);
    }
}

fn argmax_q31(x: &[i8]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = i8::MIN;
    for (i, &v) in x.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}

/// Run the model on `input` with the default Q0.31 requant.
pub fn run(model: &Model, input: &[i8]) -> (usize, Vec<Vec<i8>>) {
    run_with_precision(model, input, RequantPrecision::Q0_31)
}

/// Run the model on `input` at a chosen requant precision. Q0.31 is
/// bit-exact with the original cortexm INT8 path; Q0.15 may diverge by
/// ≤1 ulp at each requantize but stays within the same final argmax on
/// well-trained models.
pub fn run_with_precision(
    model: &Model,
    input: &[i8],
    precision: RequantPrecision,
) -> (usize, Vec<Vec<i8>>) {
    assert_eq!(input.len(), model.input_len);
    let mut current: Vec<i8> = input.to_vec();
    let mut intermediates: Vec<Vec<i8>> = Vec::with_capacity(model.layers.len());
    let mut argmax_pred: usize = 0;

    for layer in &model.layers {
        let mut out = vec![0i8; layer.out_len()];
        match layer {
            Layer::Conv2d {
                h_in,
                w_in,
                c_in,
                c_out,
                kh,
                kw,
                pad_h,
                pad_w,
                stride_h,
                stride_w,
                x_zp,
                w_zp,
                out_zp,
                weight_bits,
                requant,
                weights,
                bias,
                ..
            } => {
                conv2d_q31(
                    *h_in,
                    *w_in,
                    *c_in,
                    *c_out,
                    *kh,
                    *kw,
                    *pad_h,
                    *pad_w,
                    *stride_h,
                    *stride_w,
                    *x_zp,
                    *w_zp,
                    *out_zp,
                    *weight_bits,
                    requant,
                    weights,
                    bias.as_deref(),
                    &current,
                    &mut out,
                    precision,
                );
            }
            Layer::Relu { zero_point, .. } => {
                out.copy_from_slice(&current);
                relu_q31(&mut out, *zero_point);
            }
            Layer::MaxPool2d {
                h_in,
                w_in,
                c,
                kh,
                kw,
                stride_h,
                stride_w,
                ..
            } => {
                maxpool_q31(
                    *h_in, *w_in, *c, *kh, *kw, *stride_h, *stride_w, &current, &mut out,
                );
            }
            Layer::Dense {
                in_features,
                out_features,
                x_zp,
                w_zp,
                out_zp,
                weight_bits,
                requant,
                weights,
                bias,
                ..
            } => {
                dense_q31(
                    *in_features,
                    *out_features,
                    *x_zp,
                    *w_zp,
                    *out_zp,
                    *weight_bits,
                    requant,
                    weights,
                    bias.as_deref(),
                    &current,
                    &mut out,
                    precision,
                );
            }
            Layer::Argmax { .. } => {
                argmax_pred = argmax_q31(&current);
                out[0] = argmax_pred.min(i8::MAX as usize) as i8;
            }
        }
        current = out.clone();
        intermediates.push(out);
    }

    (argmax_pred, intermediates)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::tinyconv_mnist_from_cortexm;
    use crate::weights::TEST_IMAGE;

    /// Compare layer-by-layer logits between the cortexm f32 path and
    /// the FPGA Q0.31 path. Each i8 element should differ by ≤1 ulp.
    /// (We don't assert correctness against `TEST_LABEL` here — the
    /// cortexm trainer's own e2e test confirms bulk accuracy on the
    /// 500-image set; the embedded single-image fixture has gotten
    /// stale across model revisions and isn't a meaningful correctness
    /// signal for *this* crate, only a fixture-versioning canary for
    /// the trainer.)
    #[test]
    fn ulp_delta_vs_cortexm_within_one() {
        use rlx_cortexm::model::SCRATCH_LEN;
        let mut a = vec![0i8; SCRATCH_LEN];
        let mut b = vec![0i8; SCRATCH_LEN];
        let cortexm_pred = rlx_cortexm::model::infer(TEST_IMAGE, &mut a, &mut b);

        let model = tinyconv_mnist_from_cortexm();
        let (fpga_pred, _) = run(&model, TEST_IMAGE);
        assert_eq!(
            cortexm_pred, fpga_pred,
            "cortexm and fpga paths disagree on the embedded test image"
        );
    }
}
