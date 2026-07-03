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

    // im2col + BLAS GEMM. Per (batch, group) we gather the receptive fields into
    // a `[K, N]` column matrix (K = c_in_per_g·kh·kw, N = h_out·w_out) and compute
    // `out_g[M,N] = W_g[M,K] @ col[K,N]` via sgemm (M = c_out_per_g). The weight's
    // inner layout `[c_in_per_g, kh, kw]` matches the col row order, and the
    // output block for (batch, group) is contiguous `[c_out_per_g, N]` row-major,
    // so sgemm writes straight into `out`. Replaces a naive 6-deep loop that made
    // conv-heavy models (the deform host kernel had the same class of bug).
    let k = c_in_per_g * kh * kw;
    let hw_out = h_out * w_out;
    let mut col = vec![0f32; k * hw_out];
    for ni in 0..n {
        for g in 0..groups {
            let ci_start = g * c_in_per_g;
            // Build col[row, p] where row = (ci_off·kh + ki)·kw + kj, p = ho·w_out + wo.
            for ci_off in 0..c_in_per_g {
                let in_chan = (ni * c_in + ci_start + ci_off) * h * w;
                for ki in 0..kh {
                    for kj in 0..kw {
                        let row = (ci_off * kh + ki) * kw + kj;
                        let col_row = &mut col[row * hw_out..(row + 1) * hw_out];
                        for ho in 0..h_out {
                            let hi = ho * sh + ki * dh;
                            let in_y = hi.wrapping_sub(ph);
                            let y_ok = hi >= ph && in_y < h;
                            for wo in 0..w_out {
                                let wi = wo * sw + kj * dw;
                                let in_x = wi.wrapping_sub(pw);
                                col_row[ho * w_out + wo] = if y_ok && wi >= pw && in_x < w {
                                    inp[in_chan + in_y * w + in_x]
                                } else {
                                    0.0
                                };
                            }
                        }
                    }
                }
            }
            let wt_g = &wt[g * c_out_per_g * k..(g * c_out_per_g + c_out_per_g) * k];
            let out_off = (ni * c_out + g * c_out_per_g) * hw_out;
            let out_g = &mut out[out_off..out_off + c_out_per_g * hw_out];
            crate::blas::sgemm(wt_g, &col, out_g, c_out_per_g, k, hw_out);
        }
    }
}

/// NCDHW conv3d forward (PyTorch cross-correlation; the depth-axis analogue of
/// [`conv2d_forward_nchw_f32`]). Naive direct nested loop — correctness over
/// speed, sufficient for inference parity. Weight `[C_out, C_in/g, kD, kH, kW]`.
#[allow(clippy::too_many_arguments)]
pub fn conv3d_forward_ncdhw_f32(
    inp: &[f32],
    wt: &[f32],
    out: &mut [f32],
    n: usize,
    c_in: usize,
    d: usize,
    h: usize,
    w: usize,
    c_out: usize,
    d_out: usize,
    h_out: usize,
    w_out: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    sd: usize,
    sh: usize,
    sw: usize,
    pd: usize,
    ph: usize,
    pw: usize,
    dd: usize,
    dh: usize,
    dw: usize,
    groups: usize,
) {
    let c_in_per_g = c_in / groups;
    let c_out_per_g = c_out / groups;
    debug_assert_eq!(inp.len(), n * c_in * d * h * w);
    debug_assert_eq!(wt.len(), c_out * c_in_per_g * kd * kh * kw);
    debug_assert_eq!(out.len(), n * c_out * d_out * h_out * w_out);
    for ni in 0..n {
        for co in 0..c_out {
            let g = co / c_out_per_g;
            let ci_start = g * c_in_per_g;
            for od in 0..d_out {
                for ho in 0..h_out {
                    for wo in 0..w_out {
                        let mut acc = 0f32;
                        for ci_off in 0..c_in_per_g {
                            let ci = ci_start + ci_off;
                            let in_chan = ((ni * c_in) + ci) * d * h * w;
                            let wt_chan = ((co * c_in_per_g) + ci_off) * kd * kh * kw;
                            for kdi in 0..kd {
                                let di = od * sd + kdi * dd;
                                if di < pd {
                                    continue;
                                }
                                let di = di - pd;
                                if di >= d {
                                    continue;
                                }
                                for ki in 0..kh {
                                    let hi = ho * sh + ki * dh;
                                    if hi < ph {
                                        continue;
                                    }
                                    let hi = hi - ph;
                                    if hi >= h {
                                        continue;
                                    }
                                    for kj in 0..kw {
                                        let wi = wo * sw + kj * dw;
                                        if wi < pw {
                                            continue;
                                        }
                                        let wi = wi - pw;
                                        if wi >= w {
                                            continue;
                                        }
                                        acc += inp[in_chan + (di * h + hi) * w + wi]
                                            * wt[wt_chan + (kdi * kh + ki) * kw + kj];
                                    }
                                }
                            }
                        }
                        out[((ni * c_out) + co) * d_out * h_out * w_out
                            + (od * h_out + ho) * w_out
                            + wo] = acc;
                    }
                }
            }
        }
    }
}
