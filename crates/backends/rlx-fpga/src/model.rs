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

//! Graph description for the FPGA backend.
//!
//! One source of truth for layer shapes / sequence, shared between
//! `reference.rs` (the parity oracle) and `codegen::top` (the Verilog
//! emitter). Mirrors `rlx_cortexm::model` but in *data* form — a `Vec<Layer>`
//! rather than a hand-wired Rust function — so the emitter can walk it.
//!
//! Per-layer requant multipliers are stored as `(M0, shift)` pairs, the
//! Q0.31 form `quant::quantize_multiplier` produces. Each layer carries
//! a per-output-channel table (per-tensor is just N copies of the same).

use crate::quant::quantize_multiplier;

/// One layer in the forward pass. Shapes are NHWC (matches cortexm).
#[derive(Debug, Clone)]
pub enum Layer {
    Conv2d {
        name: &'static str,
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
        /// `[c_out]` Q0.31 (M0, shift) pairs.
        requant: Vec<(i32, i32)>,
        /// Packed weight bytes (i8 view) — same layout cortexm uses.
        weights: Vec<i8>,
        /// Optional per-channel i32 bias in accumulator scale.
        bias: Option<Vec<i32>>,
    },
    Relu {
        name: &'static str,
        len: usize,
        zero_point: i32,
    },
    MaxPool2d {
        name: &'static str,
        h_in: usize,
        w_in: usize,
        c: usize,
        kh: usize,
        kw: usize,
        stride_h: usize,
        stride_w: usize,
    },
    Dense {
        name: &'static str,
        in_features: usize,
        out_features: usize,
        x_zp: i32,
        w_zp: i32,
        out_zp: i32,
        weight_bits: u8,
        requant: Vec<(i32, i32)>,
        weights: Vec<i8>,
        bias: Option<Vec<i32>>,
    },
    Argmax {
        name: &'static str,
        len: usize,
    },
}

impl Layer {
    pub fn name(&self) -> &'static str {
        match self {
            Layer::Conv2d { name, .. }
            | Layer::Relu { name, .. }
            | Layer::MaxPool2d { name, .. }
            | Layer::Dense { name, .. }
            | Layer::Argmax { name, .. } => name,
        }
    }

    /// Number of output elements (i8) this layer produces. Argmax is the
    /// special case — produces a scalar index, but for buffer sizing we
    /// say 1.
    pub fn out_len(&self) -> usize {
        match self {
            Layer::Conv2d {
                h_in,
                w_in,
                c_out,
                kh,
                kw,
                pad_h,
                pad_w,
                stride_h,
                stride_w,
                ..
            } => {
                let h_out = (h_in + 2 * pad_h - kh) / stride_h + 1;
                let w_out = (w_in + 2 * pad_w - kw) / stride_w + 1;
                h_out * w_out * c_out
            }
            Layer::Relu { len, .. } => *len,
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
                let h_out = (h_in - kh) / stride_h + 1;
                let w_out = (w_in - kw) / stride_w + 1;
                h_out * w_out * c
            }
            Layer::Dense { out_features, .. } => *out_features,
            Layer::Argmax { .. } => 1,
        }
    }
}

/// A complete model: the input length and an ordered list of layers.
#[derive(Debug, Clone)]
pub struct Model {
    pub name: String,
    pub input_len: usize,
    pub layers: Vec<Layer>,
}

impl Model {
    /// Largest activation buffer the forward pass will need at any step.
    /// Determines the BRAM scratch size in the Verilog top.
    pub fn scratch_len(&self) -> usize {
        let mut m = self.input_len;
        for l in &self.layers {
            m = m.max(l.out_len());
        }
        m
    }
}

/// Build a `Model` from rlx-cortexm's TinyConv-MNIST weights.
///
/// Shapes match `rlx_cortexm::model::infer` exactly:
///   28×28×1 → conv1[26×26×8] → relu → pool1[13×13×8]
///          → conv2[11×11×16] → relu → pool2[5×5×16]
///          → fc\[10\] → argmax.
pub fn tinyconv_mnist_from_cortexm() -> Model {
    use rlx_cortexm::model_weights as w;

    fn requant_table(mults: &[f32]) -> Vec<(i32, i32)> {
        mults.iter().map(|&m| quantize_multiplier(m)).collect()
    }

    let conv1 = Layer::Conv2d {
        name: "conv1",
        h_in: 28,
        w_in: 28,
        c_in: 1,
        c_out: 8,
        kh: 3,
        kw: 3,
        pad_h: 0,
        pad_w: 0,
        stride_h: 1,
        stride_w: 1,
        x_zp: 0,
        w_zp: 0,
        out_zp: 0,
        weight_bits: w::WEIGHT_BITS,
        requant: requant_table(w::CONV1_MULT),
        weights: w::CONV1_W.to_vec(),
        bias: Some(w::CONV1_B.to_vec()),
    };
    let relu1 = Layer::Relu {
        name: "relu1",
        len: 26 * 26 * 8,
        zero_point: 0,
    };
    let pool1 = Layer::MaxPool2d {
        name: "pool1",
        h_in: 26,
        w_in: 26,
        c: 8,
        kh: 2,
        kw: 2,
        stride_h: 2,
        stride_w: 2,
    };
    let conv2 = Layer::Conv2d {
        name: "conv2",
        h_in: 13,
        w_in: 13,
        c_in: 8,
        c_out: 16,
        kh: 3,
        kw: 3,
        pad_h: 0,
        pad_w: 0,
        stride_h: 1,
        stride_w: 1,
        x_zp: 0,
        w_zp: 0,
        out_zp: 0,
        weight_bits: w::WEIGHT_BITS,
        requant: requant_table(w::CONV2_MULT),
        weights: w::CONV2_W.to_vec(),
        bias: Some(w::CONV2_B.to_vec()),
    };
    let relu2 = Layer::Relu {
        name: "relu2",
        len: 11 * 11 * 16,
        zero_point: 0,
    };
    let pool2 = Layer::MaxPool2d {
        name: "pool2",
        h_in: 11,
        w_in: 11,
        c: 16,
        kh: 2,
        kw: 2,
        stride_h: 2,
        stride_w: 2,
    };
    let fc = Layer::Dense {
        name: "fc",
        in_features: 5 * 5 * 16,
        out_features: 10,
        x_zp: 0,
        w_zp: 0,
        out_zp: 0,
        weight_bits: w::WEIGHT_BITS,
        requant: requant_table(w::FC_MULT),
        weights: w::FC_W.to_vec(),
        bias: Some(w::FC_B.to_vec()),
    };
    let argmax = Layer::Argmax {
        name: "argmax",
        len: 10,
    };

    Model {
        name: "tinyconv_mnist".to_string(),
        input_len: 28 * 28,
        layers: vec![conv1, relu1, pool1, conv2, relu2, pool2, fc, argmax],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tinyconv_shapes_match_cortexm() {
        let m = tinyconv_mnist_from_cortexm();
        assert_eq!(m.input_len, 28 * 28);
        assert_eq!(m.layers.len(), 8);
        assert_eq!(m.scratch_len(), 26 * 26 * 8); // conv1 output is the largest
    }
}
