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

//! INT8 2D max-pool, NHWC, no padding (the only case we need for
//! TinyConv-MNIST).

pub struct MaxPool2dParams {
    pub h_in: usize,
    pub w_in: usize,
    pub c: usize,
    pub kh: usize,
    pub kw: usize,
    pub stride_h: usize,
    pub stride_w: usize,
}

impl MaxPool2dParams {
    pub fn h_out(&self) -> usize {
        (self.h_in - self.kh) / self.stride_h + 1
    }
    pub fn w_out(&self) -> usize {
        (self.w_in - self.kw) / self.stride_w + 1
    }
}

pub fn maxpool2d_i8(p: &MaxPool2dParams, x: &[i8], out: &mut [i8]) {
    let h_out = p.h_out();
    let w_out = p.w_out();
    debug_assert_eq!(x.len(), p.h_in * p.w_in * p.c);
    debug_assert_eq!(out.len(), h_out * w_out * p.c);

    for oh in 0..h_out {
        for ow in 0..w_out {
            for c in 0..p.c {
                let mut m: i8 = i8::MIN;
                for kh in 0..p.kh {
                    let ih = oh * p.stride_h + kh;
                    for kw in 0..p.kw {
                        let iw = ow * p.stride_w + kw;
                        let v = x[(ih * p.w_in + iw) * p.c + c];
                        if v > m {
                            m = v;
                        }
                    }
                }
                out[(oh * w_out + ow) * p.c + c] = m;
            }
        }
    }
}
