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

//! FP32 reference kernels used by the test suite to validate the i8
//! versions. Not compiled into firmware (`std` feature only).

pub fn dense_f32(x: &[f32], w: &[f32], bias: Option<&[f32]>, out: &mut [f32]) {
    let out_features = out.len();
    let in_features = x.len();
    for m in 0..out_features {
        let mut acc = bias.map(|b| b[m]).unwrap_or(0.0);
        for k in 0..in_features {
            acc += x[k] * w[m * in_features + k];
        }
        out[m] = acc;
    }
}

pub struct Conv2dParamsF32 {
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
}

impl Conv2dParamsF32 {
    pub fn h_out(&self) -> usize {
        (self.h_in + 2 * self.pad_h - self.kh) / self.stride_h + 1
    }
    pub fn w_out(&self) -> usize {
        (self.w_in + 2 * self.pad_w - self.kw) / self.stride_w + 1
    }
}

pub fn conv2d_f32(
    p: &Conv2dParamsF32,
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    let h_out = p.h_out();
    let w_out = p.w_out();
    for oh in 0..h_out {
        for ow in 0..w_out {
            for oc in 0..p.c_out {
                let mut acc = bias.map(|b| b[oc]).unwrap_or(0.0);
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
                        for ic in 0..p.c_in {
                            let xv = x[(ih * p.w_in + iw) * p.c_in + ic];
                            let wv = w[((oc * p.kh + kh) * p.kw + kw) * p.c_in + ic];
                            acc += xv * wv;
                        }
                    }
                }
                out[(oh * w_out + ow) * p.c_out + oc] = acc;
            }
        }
    }
}
