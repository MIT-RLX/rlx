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

//! NCHW im2col in `[M, C·kH·kW]` row layout (`M = N · H_out · W_out`).

/// im2col in `[M, C·kH·kW]` row layout (`M = N · H_out · W_out`).
#[allow(clippy::too_many_arguments)]
pub fn im2col_rows_layout(
    x: &[f32],
    col: &mut [f32],
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw_dil: usize,
) {
    let k = c_in * kh * kw;
    let h_isz = h as isize;
    let w_isz = w as isize;
    let ph_isz = ph as isize;
    let pw_isz = pw as isize;
    for ni in 0..n {
        let x_base = ni * c_in * h * w;
        for ho in 0..h_out {
            for wo in 0..w_out {
                let row = (ni * h_out * w_out + ho * w_out + wo) * k;
                let mut elem = 0usize;
                for ci in 0..c_in {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let hi = (ho * sh + ki * dh) as isize - ph_isz;
                            let wi = (wo * sw + kj * dw_dil) as isize - pw_isz;
                            col[row + elem] = if hi < 0 || hi >= h_isz || wi < 0 || wi >= w_isz {
                                0.0
                            } else {
                                let hi = hi as usize;
                                let wi = wi as usize;
                                x[x_base + (ci * h + hi) * w + wi]
                            };
                            elem += 1;
                        }
                    }
                }
            }
        }
    }
}

/// Run im2col against a contiguous f32 arena (`base` is byte offset 0).
#[allow(clippy::too_many_arguments)]
pub unsafe fn execute_im2col_rows_layout(
    x_off: usize,
    col_off: usize,
    n: u32,
    c_in: u32,
    h: u32,
    w: u32,
    h_out: u32,
    w_out: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    dh: u32,
    dw_dil: u32,
    base: *mut u8,
) {
    unsafe {
        let c_in = c_in as usize;
        let h = h as usize;
        let w = w as usize;
        let per_batch = c_in * h * w;
        let n_eff = if n == 0 { 0usize } else { n as usize };
        let x_floats = if n_eff == 0 {
            per_batch.max(1)
        } else {
            n_eff * per_batch
        };
        let xs = std::slice::from_raw_parts(base.add(x_off) as *const f32, x_floats);
        let n = if n == 0 {
            xs.len() / per_batch.max(1)
        } else {
            n_eff
        };
        let m = n * h_out as usize * w_out as usize;
        let k = c_in * kh as usize * kw as usize;
        let cols = std::slice::from_raw_parts_mut(base.add(col_off) as *mut f32, m * k);
        im2col_rows_layout(
            xs,
            cols,
            n,
            c_in,
            h,
            w,
            h_out as usize,
            w_out as usize,
            kh as usize,
            kw as usize,
            sh as usize,
            sw as usize,
            ph as usize,
            pw as usize,
            dh as usize,
            dw_dil as usize,
        );
    }
}
