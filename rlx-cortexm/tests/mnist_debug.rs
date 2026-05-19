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

//! Diagnostic: dump fc logits and a few activations to compare against
//! the PyTorch reference. Ignored by default — run with --ignored.

use rlx_cortexm::conv2d::{Conv2dParams, conv2d_i8};
use rlx_cortexm::dense::dense_i8;
use rlx_cortexm::maxpool::{MaxPool2dParams, maxpool2d_i8};
use rlx_cortexm::model::SCRATCH_LEN;
use rlx_cortexm::model_weights as w;
use rlx_cortexm::relu::relu_i8;

#[test]
#[ignore]
fn dump_fc_logits() {
    let mut a = vec![0i8; SCRATCH_LEN];
    let mut b = vec![0i8; SCRATCH_LEN];

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
    conv2d_i8(
        &p,
        w::TEST_IMAGE,
        w::CONV1_W,
        Some(w::CONV1_B),
        &mut a[..26 * 26 * 8],
    );
    println!("conv1 out [0..16] = {:?}", &a[..16]);
    relu_i8(&mut a[..26 * 26 * 8], 0);

    let pp = MaxPool2dParams {
        h_in: 26,
        w_in: 26,
        c: 8,
        kh: 2,
        kw: 2,
        stride_h: 2,
        stride_w: 2,
    };
    maxpool2d_i8(&pp, &a[..26 * 26 * 8], &mut b[..13 * 13 * 8]);
    println!("pool1 out [0..16] = {:?}", &b[..16]);

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
    conv2d_i8(
        &p,
        &b[..13 * 13 * 8],
        w::CONV2_W,
        Some(w::CONV2_B),
        &mut a[..11 * 11 * 16],
    );
    relu_i8(&mut a[..11 * 11 * 16], 0);

    let pp = MaxPool2dParams {
        h_in: 11,
        w_in: 11,
        c: 16,
        kh: 2,
        kw: 2,
        stride_h: 2,
        stride_w: 2,
    };
    maxpool2d_i8(&pp, &a[..11 * 11 * 16], &mut b[..5 * 5 * 16]);
    println!("pool2 out [0..16] = {:?}", &b[..16]);

    dense_i8(
        &b[..5 * 5 * 16],
        w::FC_W,
        Some(w::FC_B),
        0,
        0,
        0,
        w::FC_MULT,
        w::WEIGHT_BITS,
        &mut a[..10],
    );
    println!("fc logits = {:?}", &a[..10]);
    println!("expected label = {}", w::TEST_LABEL);
}
