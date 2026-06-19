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

//! NCHW conv2d forward (matches `Thunk::Conv2D` / PyTorch cross-correlation).

#[allow(clippy::too_many_arguments)]
pub fn conv2d_forward_nchw_f32(
    inp: &[f32],
    wt: &[f32],
    out: &mut [f32],
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
    groups: usize,
) {
    let c_in_per_g = c_in / groups;
    let c_out_per_g = c_out / groups;
    debug_assert_eq!(inp.len(), n * c_in * h * w);
    debug_assert_eq!(wt.len(), c_out * c_in_per_g * kh * kw);
    debug_assert_eq!(out.len(), n * c_out * h_out * w_out);
    for ni in 0..n {
        for co in 0..c_out {
            let g = co / c_out_per_g;
            let ci_start = g * c_in_per_g;
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let mut acc = 0f32;
                    for ci_off in 0..c_in_per_g {
                        let ci = ci_start + ci_off;
                        let in_chan = (ni * c_in + ci) * h * w;
                        let wt_chan = (co * c_in_per_g + ci_off) * kh * kw;
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let hi = ho * sh + ki * dh;
                                let wi = wo * sw + kj * dw;
                                if hi < ph || wi < pw {
                                    continue;
                                }
                                let hi = hi - ph;
                                let wi = wi - pw;
                                if hi >= h || wi >= w {
                                    continue;
                                }
                                acc += inp[in_chan + hi * w + wi] * wt[wt_chan + ki * kw + kj];
                            }
                        }
                    }
                    out[(ni * c_out + co) * h_out * w_out + ho * w_out + wo] = acc;
                }
            }
        }
    }
}
