// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Naive NCDHW 3-D conv / max-pool backward (CPU reference + host fallback).

#[allow(clippy::too_many_arguments)]
pub fn conv3d_backward_input_ncdhw(
    dy: &[f32],
    weight: &[f32],
    dx: &mut [f32],
    n: usize,
    c_in: usize,
    c_out: usize,
    d: usize,
    h: usize,
    w: usize,
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
    let c_in_per_g = c_in / groups.max(1);
    let c_out_per_g = c_out / groups.max(1);
    dx.fill(0.0);
    for ni in 0..n {
        for g in 0..groups {
            for co_off in 0..c_out_per_g {
                let co = g * c_out_per_g + co_off;
                for do_ in 0..d_out {
                    for ho in 0..h_out {
                        for wo in 0..w_out {
                            let dyv =
                                dy[(((ni * c_out + co) * d_out + do_) * h_out + ho) * w_out + wo];
                            for ci_off in 0..c_in_per_g {
                                let ci = g * c_in_per_g + ci_off;
                                for kz in 0..kd {
                                    let id = do_ as isize * sd as isize + kz as isize * dd as isize
                                        - pd as isize;
                                    if id < 0 || id >= d as isize {
                                        continue;
                                    }
                                    for ki in 0..kh {
                                        let ih = ho as isize * sh as isize
                                            + ki as isize * dh as isize
                                            - ph as isize;
                                        if ih < 0 || ih >= h as isize {
                                            continue;
                                        }
                                        for kj in 0..kw {
                                            let iw = wo as isize * sw as isize
                                                + kj as isize * dw as isize
                                                - pw as isize;
                                            if iw < 0 || iw >= w as isize {
                                                continue;
                                            }
                                            let wv = weight[(((co * c_in_per_g + ci_off) * kd
                                                + kz)
                                                * kh
                                                + ki)
                                                * kw
                                                + kj];
                                            let dx_i = (((ni * c_in + ci) * d + id as usize) * h
                                                + ih as usize)
                                                * w
                                                + iw as usize;
                                            dx[dx_i] += dyv * wv;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn conv3d_backward_weight_ncdhw(
    x: &[f32],
    dy: &[f32],
    dw: &mut [f32],
    n: usize,
    c_in: usize,
    c_out: usize,
    d: usize,
    h: usize,
    w: usize,
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
    dw_dil: usize,
    groups: usize,
) {
    let c_in_per_g = c_in / groups.max(1);
    let c_out_per_g = c_out / groups.max(1);
    dw.fill(0.0);
    for ni in 0..n {
        for g in 0..groups {
            for co_off in 0..c_out_per_g {
                let co = g * c_out_per_g + co_off;
                for do_ in 0..d_out {
                    for ho in 0..h_out {
                        for wo in 0..w_out {
                            let dyv =
                                dy[(((ni * c_out + co) * d_out + do_) * h_out + ho) * w_out + wo];
                            for ci_off in 0..c_in_per_g {
                                let ci = g * c_in_per_g + ci_off;
                                for kz in 0..kd {
                                    let id = do_ as isize * sd as isize + kz as isize * dd as isize
                                        - pd as isize;
                                    if id < 0 || id >= d as isize {
                                        continue;
                                    }
                                    for ki in 0..kh {
                                        let ih = ho as isize * sh as isize
                                            + ki as isize * dh as isize
                                            - ph as isize;
                                        if ih < 0 || ih >= h as isize {
                                            continue;
                                        }
                                        for kj in 0..kw {
                                            let iw = wo as isize * sw as isize
                                                + kj as isize * dw_dil as isize
                                                - pw as isize;
                                            if iw < 0 || iw >= w as isize {
                                                continue;
                                            }
                                            let xv = x[(((ni * c_in + ci) * d + id as usize) * h
                                                + ih as usize)
                                                * w
                                                + iw as usize];
                                            let dw_i =
                                                (((co * c_in_per_g + ci_off) * kd + kz) * kh + ki)
                                                    * kw
                                                    + kj;
                                            dw[dw_i] += dyv * xv;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn maxpool3d_backward_ncdhw(
    x: &[f32],
    dy: &[f32],
    dx: &mut [f32],
    n: usize,
    c: usize,
    d: usize,
    h: usize,
    w: usize,
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
) {
    dx.fill(0.0);
    for ni in 0..n {
        for cc in 0..c {
            let base = (ni * c + cc) * d * h * w;
            for do_ in 0..d_out {
                for ho in 0..h_out {
                    for wo in 0..w_out {
                        let dstart = do_ as isize * sd as isize - pd as isize;
                        let hstart = ho as isize * sh as isize - ph as isize;
                        let wstart = wo as isize * sw as isize - pw as isize;
                        let mut best = f32::NEG_INFINITY;
                        let mut best_idx = 0usize;
                        let mut found = false;
                        for kz in 0..kd {
                            let id = dstart + kz as isize;
                            if id < 0 || id >= d as isize {
                                continue;
                            }
                            for ki in 0..kh {
                                let ih = hstart + ki as isize;
                                if ih < 0 || ih >= h as isize {
                                    continue;
                                }
                                for kj in 0..kw {
                                    let iw = wstart + kj as isize;
                                    if iw < 0 || iw >= w as isize {
                                        continue;
                                    }
                                    let idx =
                                        base + ((id as usize * h + ih as usize) * w + iw as usize);
                                    let v = x[idx];
                                    if !found || v > best {
                                        best = v;
                                        best_idx = idx;
                                        found = true;
                                    }
                                }
                            }
                        }
                        if found {
                            let dyv = dy[(((ni * c + cc) * d_out + do_) * h_out + ho) * w_out + wo];
                            dx[best_idx] += dyv;
                        }
                    }
                }
            }
        }
    }
}
