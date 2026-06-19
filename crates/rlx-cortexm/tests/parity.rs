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

//! Cross-checks: i8 kernels vs the f32 reference. Tolerance is loose
//! (we expect quantization noise) but the *direction* must match —
//! e.g. argmax of the dense output should agree with the float oracle.
//!
//! Random inputs use a fixed-seed LCG so failures are reproducible.

use rlx_cortexm::{
    argmax,
    conv2d::{Conv2dParams, conv2d_i8},
    dense::dense_i8,
    maxpool::{MaxPool2dParams, maxpool2d_i8},
    quant::{QParams, dequant, quant},
    reference::{Conv2dParamsF32, conv2d_f32, dense_f32},
    relu::relu_i8,
};

struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_f32_unit(&mut self) -> f32 {
        // [-1, 1)
        ((self.next_u32() as f32) / (u32::MAX as f32)) * 2.0 - 1.0
    }
}

fn quant_vec(xs: &[f32], p: QParams) -> Vec<i8> {
    xs.iter().map(|&x| quant(x, p)).collect()
}

fn dequant_vec(qs: &[i8], p: QParams) -> Vec<f32> {
    qs.iter().map(|&q| dequant(q, p)).collect()
}

#[test]
fn dense_i8_tracks_f32() {
    let in_f = 32;
    let out_f = 16;
    let mut rng = Lcg(0xDEADBEEF);

    let x_f: Vec<f32> = (0..in_f).map(|_| rng.next_f32_unit()).collect();
    let w_f: Vec<f32> = (0..out_f * in_f)
        .map(|_| rng.next_f32_unit() * 0.5)
        .collect();

    let mut y_f = vec![0.0f32; out_f];
    dense_f32(&x_f, &w_f, None, &mut y_f);

    let x_p = QParams::symmetric(1.0 / 127.0);
    let w_p = QParams::symmetric(0.5 / 127.0);
    // Choose output scale from observed range to keep most of the dynamic range.
    let max_abs = y_f.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-6);
    let y_p = QParams::symmetric(max_abs / 127.0);

    let x_q = quant_vec(&x_f, x_p);
    let w_q = quant_vec(&w_f, w_p);

    let mut y_q = vec![0i8; out_f];
    let mult = vec![(x_p.scale * w_p.scale) / y_p.scale; out_f];
    dense_i8(
        &x_q,
        &w_q,
        None,
        x_p.zero_point,
        w_p.zero_point,
        y_p.zero_point,
        &mult,
        8,
        &mut y_q,
    );
    let y_dq = dequant_vec(&y_q, y_p);

    // Per-element tolerance: a few output-scale steps.
    let tol = 4.0 * y_p.scale;
    for (a, b) in y_dq.iter().zip(y_f.iter()) {
        assert!(
            (a - b).abs() <= tol,
            "dense parity off: {a} vs {b} (tol {tol})"
        );
    }
}

#[test]
fn conv2d_i8_tracks_f32() {
    // 8x8x3 → 6x6x4, 3x3 valid conv.
    let p_f = Conv2dParamsF32 {
        h_in: 8,
        w_in: 8,
        c_in: 3,
        c_out: 4,
        kh: 3,
        kw: 3,
        pad_h: 0,
        pad_w: 0,
        stride_h: 1,
        stride_w: 1,
    };
    let h_out = p_f.h_out();
    let w_out = p_f.w_out();

    let mut rng = Lcg(0xC0FFEE);
    let x_f: Vec<f32> = (0..p_f.h_in * p_f.w_in * p_f.c_in)
        .map(|_| rng.next_f32_unit())
        .collect();
    let w_f: Vec<f32> = (0..p_f.c_out * p_f.kh * p_f.kw * p_f.c_in)
        .map(|_| rng.next_f32_unit() * 0.3)
        .collect();

    let mut y_f = vec![0.0f32; h_out * w_out * p_f.c_out];
    conv2d_f32(&p_f, &x_f, &w_f, None, &mut y_f);

    let x_p = QParams::symmetric(1.0 / 127.0);
    let w_p = QParams::symmetric(0.3 / 127.0);
    let max_abs = y_f.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-6);
    let y_p = QParams::symmetric(max_abs / 127.0);

    let x_q = quant_vec(&x_f, x_p);
    let w_q = quant_vec(&w_f, w_p);
    let mut y_q = vec![0i8; h_out * w_out * p_f.c_out];

    let mult = vec![(x_p.scale * w_p.scale) / y_p.scale; p_f.c_out];
    let p_q = Conv2dParams {
        h_in: p_f.h_in,
        w_in: p_f.w_in,
        c_in: p_f.c_in,
        c_out: p_f.c_out,
        kh: p_f.kh,
        kw: p_f.kw,
        pad_h: 0,
        pad_w: 0,
        stride_h: 1,
        stride_w: 1,
        x_zp: x_p.zero_point,
        w_zp: w_p.zero_point,
        out_zp: y_p.zero_point,
        mult: &mult,
        weight_bits: 8,
    };
    conv2d_i8(&p_q, &x_q, &w_q, None, &mut y_q);
    let y_dq = dequant_vec(&y_q, y_p);

    let tol = 4.0 * y_p.scale;
    for (a, b) in y_dq.iter().zip(y_f.iter()) {
        assert!(
            (a - b).abs() <= tol,
            "conv2d parity off: {a} vs {b} (tol {tol})"
        );
    }
}

#[test]
fn relu_clamps_at_zero_point() {
    let mut buf: Vec<i8> = vec![-10, -1, 0, 5, 100];
    relu_i8(&mut buf, 0);
    assert_eq!(buf, vec![0, 0, 0, 5, 100]);

    // With zp = -5, anything < -5 becomes -5.
    let mut buf: Vec<i8> = vec![-10, -1, 0, 5, 100];
    relu_i8(&mut buf, -5);
    assert_eq!(buf, vec![-5, -1, 0, 5, 100]);
}

#[test]
fn maxpool_matches_handworked() {
    // 4x4x1, 2x2 stride 2 → 2x2x1.
    #[rustfmt::skip]
    let x: Vec<i8> = vec![
        1, 2, 3, 4,
        5, 6, 7, 8,
        9,10,11,12,
       13,14,15,16,
    ];
    let p = MaxPool2dParams {
        h_in: 4,
        w_in: 4,
        c: 1,
        kh: 2,
        kw: 2,
        stride_h: 2,
        stride_w: 2,
    };
    let mut y = vec![0i8; 2 * 2];
    maxpool2d_i8(&p, &x, &mut y);
    assert_eq!(y, vec![6, 8, 14, 16]);
}

#[test]
fn argmax_picks_largest() {
    assert_eq!(argmax::argmax_i8(&[-3, 7, 2, 7, 1]), 1); // first max wins
    assert_eq!(argmax::argmax_i8(&[i8::MIN, 0]), 1);
}
