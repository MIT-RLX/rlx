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

//! TinyConv-MNIST forward pass, wired up from this crate's INT8 kernels.
//!
//! Shapes (NHWC):
//!     input  [28, 28, 1]  →  conv1 [26, 26, 8]  →  pool1 [13, 13, 8]
//!     →  conv2 [11, 11, 16]  →  pool2 [5, 5, 16]  →  fc \[10\]
//!
//! No heap allocation. Caller supplies two scratch buffers of at least
//! `SCRATCH_LEN` bytes each (the host test passes Vecs; the firmware
//! will pass `static mut [i8; SCRATCH_LEN]`).

use crate::argmax::argmax_i8;
use crate::conv2d::{Conv2dParams, conv2d_i8, conv2d_ternary};
use crate::dense::{dense_i8, dense_ternary};
use crate::maxpool::{MaxPool2dParams, maxpool2d_i8};
use crate::model_weights as w;
use crate::relu::relu_i8;

pub const INPUT_LEN: usize = 28 * 28;
pub const CONV1_OUT_LEN: usize = 26 * 26 * 8; // 5408 — largest activation
pub const POOL1_OUT_LEN: usize = 13 * 13 * 8; // 1352
pub const CONV2_OUT_LEN: usize = 11 * 11 * 16; // 1936
pub const POOL2_OUT_LEN: usize = 5 * 5 * 16; // 400
pub const FC_OUT_LEN: usize = 10;

/// Both scratch buffers must be at least this many bytes.
pub const SCRATCH_LEN: usize = CONV1_OUT_LEN;

/// Run inference. Returns the predicted digit (0..=9).
///
/// `input` is `[28, 28, 1]` i8 in the same scale the training script
/// calibrated (`X_SCALE`, symmetric, zero_point = 0).
pub fn infer(input: &[i8], buf_a: &mut [i8], buf_b: &mut [i8]) -> usize {
    assert_eq!(input.len(), INPUT_LEN);
    assert!(buf_a.len() >= SCRATCH_LEN);
    assert!(buf_b.len() >= SCRATCH_LEN);

    // ── conv1: input → buf_a[..5408] ────────────────────────────
    let p = Conv2dParams {
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
        mult: w::CONV1_MULT,
        weight_bits: w::WEIGHT_BITS,
    };
    if w::WEIGHT_BITS == 2 {
        conv2d_ternary(
            &p,
            input,
            w::CONV1_W,
            Some(w::CONV1_B),
            &mut buf_a[..CONV1_OUT_LEN],
        );
    } else {
        conv2d_i8(
            &p,
            input,
            w::CONV1_W,
            Some(w::CONV1_B),
            &mut buf_a[..CONV1_OUT_LEN],
        );
    }
    relu_i8(&mut buf_a[..CONV1_OUT_LEN], 0);

    // ── pool1: buf_a → buf_b[..1352] ────────────────────────────
    let pp = MaxPool2dParams {
        h_in: 26,
        w_in: 26,
        c: 8,
        kh: 2,
        kw: 2,
        stride_h: 2,
        stride_w: 2,
    };
    maxpool2d_i8(&pp, &buf_a[..CONV1_OUT_LEN], &mut buf_b[..POOL1_OUT_LEN]);

    // ── conv2: buf_b → buf_a[..1936] ────────────────────────────
    let p = Conv2dParams {
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
        mult: w::CONV2_MULT,
        weight_bits: w::WEIGHT_BITS,
    };
    if w::WEIGHT_BITS == 2 {
        conv2d_ternary(
            &p,
            &buf_b[..POOL1_OUT_LEN],
            w::CONV2_W,
            Some(w::CONV2_B),
            &mut buf_a[..CONV2_OUT_LEN],
        );
    } else {
        conv2d_i8(
            &p,
            &buf_b[..POOL1_OUT_LEN],
            w::CONV2_W,
            Some(w::CONV2_B),
            &mut buf_a[..CONV2_OUT_LEN],
        );
    }
    relu_i8(&mut buf_a[..CONV2_OUT_LEN], 0);

    // ── pool2: buf_a → buf_b[..400] ─────────────────────────────
    let pp = MaxPool2dParams {
        h_in: 11,
        w_in: 11,
        c: 16,
        kh: 2,
        kw: 2,
        stride_h: 2,
        stride_w: 2,
    };
    maxpool2d_i8(&pp, &buf_a[..CONV2_OUT_LEN], &mut buf_b[..POOL2_OUT_LEN]);

    // ── fc: buf_b → buf_a[..10] ─────────────────────────────────
    if w::WEIGHT_BITS == 2 {
        dense_ternary(
            &buf_b[..POOL2_OUT_LEN],
            w::FC_W,
            Some(w::FC_B),
            0,
            0,
            0,
            w::FC_MULT,
            &mut buf_a[..FC_OUT_LEN],
        );
    } else {
        dense_i8(
            &buf_b[..POOL2_OUT_LEN],
            w::FC_W,
            Some(w::FC_B),
            0,
            0,
            0,
            w::FC_MULT,
            w::WEIGHT_BITS,
            &mut buf_a[..FC_OUT_LEN],
        );
    }

    argmax_i8(&buf_a[..FC_OUT_LEN])
}
