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

//! 2-D conv backward kernels shared by CPU thunks and Metal host fallback.

#[allow(clippy::too_many_arguments)]
fn im2col(
    x: &[f32],
    col: &mut [f32],
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
    let n_dim = h_out * w_out;
    debug_assert_eq!(col.len(), c_in * kh * kw * n_dim);
    debug_assert_eq!(x.len(), c_in * h * w);
    let h_isz = h as isize;
    let w_isz = w as isize;
    let ph_isz = ph as isize;
    let pw_isz = pw as isize;
    for ci in 0..c_in {
        for ki in 0..kh {
            for kj in 0..kw {
                let row = ((ci * kh) + ki) * kw + kj;
                let row_off = row * n_dim;
                for ho in 0..h_out {
                    let hi = (ho * sh + ki * dh) as isize - ph_isz;
                    if hi < 0 || hi >= h_isz {
                        for wo in 0..w_out {
                            col[row_off + ho * w_out + wo] = 0.0;
                        }
                        continue;
                    }
                    let hi = hi as usize;
                    let in_row_off = (ci * h + hi) * w;
                    for wo in 0..w_out {
                        let wi = (wo * sw + kj * dw_dil) as isize - pw_isz;
                        col[row_off + ho * w_out + wo] = if wi < 0 || wi >= w_isz {
                            0.0
                        } else {
                            x[in_row_off + wi as usize]
                        };
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn col2im(
    col: &[f32],
    x: &mut [f32],
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
    let n_dim = h_out * w_out;
    debug_assert_eq!(col.len(), c_in * kh * kw * n_dim);
    debug_assert_eq!(x.len(), c_in * h * w);
    let h_isz = h as isize;
    let w_isz = w as isize;
    let ph_isz = ph as isize;
    let pw_isz = pw as isize;
    for ci in 0..c_in {
        for ki in 0..kh {
            for kj in 0..kw {
                let row = ((ci * kh) + ki) * kw + kj;
                let row_off = row * n_dim;
                for ho in 0..h_out {
                    let hi = (ho * sh + ki * dh) as isize - ph_isz;
                    if hi < 0 || hi >= h_isz {
                        continue;
                    }
                    let hi = hi as usize;
                    let in_row_off = (ci * h + hi) * w;
                    for wo in 0..w_out {
                        let wi = (wo * sw + kj * dw_dil) as isize - pw_isz;
                        if wi < 0 || wi >= w_isz {
                            continue;
                        }
                        x[in_row_off + wi as usize] += col[row_off + ho * w_out + wo];
                    }
                }
            }
        }
    }
}

/// `Op::Conv2dBackwardInput` on a unified-memory arena slice (F32).
#[allow(clippy::too_many_arguments)]
pub unsafe fn execute_conv2d_backward_input_f32(
    base: *mut u8,
    dy: usize,
    w: usize,
    dx: usize,
    n: u32,
    c_in: u32,
    h: u32,
    w_in: u32,
    c_out: u32,
    h_out: u32,
    w_out: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    dh: u32,
    dw: u32,
    groups: u32,
) {
    let n = n as usize;
    let c_in = c_in as usize;
    let h = h as usize;
    let w_in = w_in as usize;
    let c_out = c_out as usize;
    let h_out = h_out as usize;
    let w_out = w_out as usize;
    let kh = kh as usize;
    let kw = kw as usize;
    let sh = sh as usize;
    let sw = sw as usize;
    let ph = ph as usize;
    let pw = pw as usize;
    let dh = dh as usize;
    let dw = dw as usize;
    let groups = groups as usize;
    let c_in_per_g = c_in / groups;
    let c_out_per_g = c_out / groups;

    let m_dim = c_in_per_g * kh * kw;
    let n_dim = h_out * w_out;
    let k_dim = c_out_per_g;

    let dy_stride_n = c_out * h_out * w_out;
    let w_stride_g = c_out_per_g * c_in_per_g * kh * kw;
    let dx_stride_n = c_in * h * w_in;
    let dx_stride_g = c_in_per_g * h * w_in;

    let base_f = base as *mut f32;
    unsafe {
        let dys = std::slice::from_raw_parts(base_f.add(dy / 4), n * c_out * h_out * w_out);
        let ws = std::slice::from_raw_parts(base_f.add(w / 4), c_out * c_in_per_g * kh * kw);
        let dxs = std::slice::from_raw_parts_mut(base_f.add(dx / 4), n * c_in * h * w_in);
        dxs.fill(0.0);

        let mut dcol = vec![0f32; m_dim * n_dim];
        for ni in 0..n {
            for g in 0..groups {
                let w_g_off = g * w_stride_g;
                let dy_n_g_off = ni * dy_stride_n + g * c_out_per_g * h_out * w_out;
                let dx_n_g_off = ni * dx_stride_n + g * dx_stride_g;
                crate::blas::sgemm_general(
                    ws.as_ptr().add(w_g_off),
                    dys.as_ptr().add(dy_n_g_off),
                    dcol.as_mut_ptr(),
                    m_dim,
                    n_dim,
                    k_dim,
                    1.0,
                    0.0,
                    m_dim,
                    n_dim,
                    n_dim,
                    true,
                    false,
                );
                col2im(
                    &dcol,
                    &mut dxs[dx_n_g_off..dx_n_g_off + dx_stride_g],
                    c_in_per_g,
                    h,
                    w_in,
                    h_out,
                    w_out,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    dh,
                    dw,
                );
            }
        }
    }
}

/// `Op::Conv2dBackwardWeight` on a unified-memory arena slice (F32).
#[allow(clippy::too_many_arguments)]
pub unsafe fn execute_conv2d_backward_weight_f32(
    base: *mut u8,
    x: usize,
    dy: usize,
    dw: usize,
    n: u32,
    c_in: u32,
    h: u32,
    w: u32,
    c_out: u32,
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
    groups: u32,
) {
    let n = n as usize;
    let c_in = c_in as usize;
    let h = h as usize;
    let w = w as usize;
    let c_out = c_out as usize;
    let h_out = h_out as usize;
    let w_out = w_out as usize;
    let kh = kh as usize;
    let kw = kw as usize;
    let sh = sh as usize;
    let sw = sw as usize;
    let ph = ph as usize;
    let pw = pw as usize;
    let dh = dh as usize;
    let dw_dil = dw_dil as usize;
    let groups = groups as usize;
    let c_in_per_g = c_in / groups;
    let c_out_per_g = c_out / groups;

    let m_dim = c_out_per_g;
    let n_dim = c_in_per_g * kh * kw;
    let k_dim = h_out * w_out;

    let x_stride_n = c_in * h * w;
    let x_stride_g = c_in_per_g * h * w;
    let dy_stride_n = c_out * h_out * w_out;
    let dy_stride_g = c_out_per_g * h_out * w_out;
    let dw_stride_g = c_out_per_g * c_in_per_g * kh * kw;

    let base_f = base as *mut f32;
    unsafe {
        let xs = std::slice::from_raw_parts(base_f.add(x / 4), n * c_in * h * w);
        let dys = std::slice::from_raw_parts(base_f.add(dy / 4), n * c_out * h_out * w_out);
        let dws = std::slice::from_raw_parts_mut(base_f.add(dw / 4), c_out * c_in_per_g * kh * kw);
        dws.fill(0.0);

        let mut col = vec![0f32; n_dim * k_dim];
        for ni in 0..n {
            for g in 0..groups {
                let x_n_g_off = ni * x_stride_n + g * x_stride_g;
                im2col(
                    &xs[x_n_g_off..x_n_g_off + x_stride_g],
                    &mut col,
                    c_in_per_g,
                    h,
                    w,
                    h_out,
                    w_out,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    dh,
                    dw_dil,
                );
                let dy_n_g_off = ni * dy_stride_n + g * dy_stride_g;
                let dw_g_off = g * dw_stride_g;
                crate::blas::sgemm_general(
                    dys.as_ptr().add(dy_n_g_off),
                    col.as_ptr(),
                    dws.as_mut_ptr().add(dw_g_off),
                    m_dim,
                    n_dim,
                    k_dim,
                    1.0,
                    1.0,
                    k_dim,
                    k_dim,
                    n_dim,
                    false,
                    true,
                );
            }
        }
    }
}
