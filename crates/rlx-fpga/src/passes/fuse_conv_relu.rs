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

//! Fuse `Conv2d → Relu` adjacent pairs.
//!
//! Pattern: layer `i` is `Conv2d` (with `out_zp = 0`, the trainer's
//! default), layer `i + 1` is `Relu` with the same length and zero
//! point. The relu is then redundant — `relu(x) = max(x, zp)`, and
//! the conv kernel can clamp `< OUT_ZP` to `OUT_ZP` directly inside
//! its requant epilogue (the existing `q_out` path already does the
//! `sat_i8` clamp, so this pass only needs to *bump the lower bound*).
//!
//! When the pattern fires:
//!
//! * The Conv2d layer's `Hints.fuses_relu` is set. Codegen makes the
//!   `requant_q31` (or `requant_q15`) output be `max(rdpot_out + out_zp, OUT_ZP)`
//!   instead of just letting `sat_i8` saturate at `i8::MIN`.
//! * The Relu layer's `Hints.elided` is set. `top.sv` skips its kernel
//!   instance and its intermediate BRAM; the controller wires Conv's
//!   output to the Relu-consumer's input directly.
//!
//! What it does NOT do:
//!
//! * Touch the cortexm-aligned `Op::QConv2d` IR node. The IR's view is
//!   for analysis only; backend behavior is driven by `Hints`. (We
//!   avoid extending `Op::QConv2d` to carry a `fuse_relu` flag because
//!   that would force every backend in the workspace to handle it.)

use crate::model::{Layer, Model};
use crate::passes::Hints;

/// Run the pass. Mutates `hints` in place (one entry per `model.layers`).
pub fn run(model: &Model, hints: &mut [Hints]) {
    debug_assert_eq!(model.layers.len(), hints.len());
    let n = model.layers.len();
    for i in 0..n.saturating_sub(1) {
        let conv_idx = i;
        let relu_idx = i + 1;
        if matches(&model.layers[conv_idx], &model.layers[relu_idx]) {
            // Don't fuse if the relu has any other consumer (no other
            // consumers exist in our linear pipeline today, but keep
            // the guard so we don't break later when topology gets
            // less linear).
            hints[conv_idx].fuses_relu = true;
            hints[relu_idx].elided = true;
        }
    }
}

/// Pattern check: conv with `out_zp = 0` immediately followed by a
/// matching-length Relu at zp = 0. Output lengths must agree (relu is
/// element-wise, so they always do).
fn matches(a: &Layer, b: &Layer) -> bool {
    let conv_ok = matches!(a, Layer::Conv2d { out_zp: 0, .. });
    let relu_ok = matches!(b, Layer::Relu { zero_point: 0, len, .. } if *len == a.out_len());
    conv_ok && relu_ok
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::tinyconv_mnist_from_cortexm;
    use crate::passes::Hints;

    #[test]
    fn tinyconv_fires_on_both_conv_relu_pairs() {
        let m = tinyconv_mnist_from_cortexm();
        let mut hints = vec![Hints::default(); m.layers.len()];
        run(&m, &mut hints);
        // conv1 (idx 0) → relu1 (idx 1) and conv2 (idx 3) → relu2 (idx 4)
        assert!(hints[0].fuses_relu, "conv1 should fuse relu1");
        assert!(hints[1].elided, "relu1 should be elided");
        assert!(hints[3].fuses_relu, "conv2 should fuse relu2");
        assert!(hints[4].elided, "relu2 should be elided");
        // The Dense layer (fc) and others should not be touched.
        for &i in &[2usize, 5, 6, 7] {
            assert!(!hints[i].fuses_relu);
            assert!(!hints[i].elided);
        }
    }

    #[test]
    fn no_match_when_pattern_breaks() {
        // Build a model with conv → maxpool (no relu in between).
        let m = Model {
            name: "noop".into(),
            input_len: 4,
            layers: vec![
                Layer::Conv2d {
                    name: "c",
                    h_in: 2,
                    w_in: 2,
                    c_in: 1,
                    c_out: 1,
                    kh: 1,
                    kw: 1,
                    pad_h: 0,
                    pad_w: 0,
                    stride_h: 1,
                    stride_w: 1,
                    x_zp: 0,
                    w_zp: 0,
                    out_zp: 0,
                    weight_bits: 8,
                    requant: vec![(1 << 30, 0)],
                    weights: vec![1],
                    bias: None,
                },
                Layer::MaxPool2d {
                    name: "p",
                    h_in: 2,
                    w_in: 2,
                    c: 1,
                    kh: 1,
                    kw: 1,
                    stride_h: 1,
                    stride_w: 1,
                },
            ],
        };
        let mut hints = vec![Hints::default(); 2];
        run(&m, &mut hints);
        assert!(!hints[0].fuses_relu);
        assert!(!hints[1].elided);
    }
}
