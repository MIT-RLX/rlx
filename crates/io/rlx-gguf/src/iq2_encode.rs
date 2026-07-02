// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Fast IQ2 encoders — llama.cpp kmap + sign-extraction path (uniform
//! weights; no imatrix required).

use std::sync::OnceLock;

use crate::QK_K;
use crate::iq_encode_common::{
    GROUP_MAX_EPS, build_kmap_2bit, extract_signs_parity, extract_signs_raw, f16_bytes,
    find_best_grid_8, grid_u64_to_i8x8, lookup_grid_8, nearest_int, weighted_scale_refine,
};
use crate::iq_grids::{IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID};

const K_MAX_Q: i32 = 3;

struct Iq2Tables {
    grids: Vec<[i8; 8]>,
    kmap: Vec<i32>,
}

fn tables_from_raw(raw: &[u64]) -> Iq2Tables {
    let grids: Vec<[i8; 8]> = raw.iter().map(|&e| grid_u64_to_i8x8(e)).collect();
    let kmap = build_kmap_2bit(&grids);
    Iq2Tables { grids, kmap }
}

fn iq2_xxs_tables() -> &'static Iq2Tables {
    static T: OnceLock<Iq2Tables> = OnceLock::new();
    T.get_or_init(|| tables_from_raw(&IQ2XXS_GRID))
}

fn iq2_xs_tables() -> &'static Iq2Tables {
    static T: OnceLock<Iq2Tables> = OnceLock::new();
    T.get_or_init(|| tables_from_raw(&IQ2XS_GRID))
}

fn iq2_s_tables() -> &'static Iq2Tables {
    static T: OnceLock<Iq2Tables> = OnceLock::new();
    T.get_or_init(|| tables_from_raw(&IQ2S_GRID))
}

pub fn quantize_iq2_xxs_block(src: &[f32], out: &mut [u8]) {
    const QS_LEN: usize = (QK_K / 8) * 2;
    assert!(src.len() >= QK_K && out.len() >= 2 + QS_LEN);
    let tab = iq2_xxs_tables();

    let mut q2 = [0u32; 2 * (QK_K / 32)];
    let mut scales = [0f32; QK_K / 32];
    let mut max_scale = 0f32;

    let sumx2: f32 = src[..QK_K].iter().map(|v| v * v).sum();
    let sigma2 = sumx2 / QK_K as f32;

    for ib in 0..QK_K / 32 {
        let xb = &src[ib * 32..(ib + 1) * 32];
        let mut weight = [0f32; 32];
        let mut xval = [0f32; 32];
        let mut block_signs = [0u8; 4];
        for i in 0..32 {
            weight[i] = sigma2 + xb[i] * xb[i];
        }
        extract_signs_parity(xb, &weight, &mut xval, &mut block_signs, 4);

        let mut max = xval[0];
        for &v in &xval[1..] {
            max = max.max(v);
        }
        if max < GROUP_MAX_EPS {
            scales[ib] = 0.0;
            continue;
        }

        let mut l = [0i8; 32];
        let mut laux = [0i8; 32];
        let mut scale = {
            let mut ltmp = [0i8; 32];
            for k in 0..4 {
                let mut lv = [0i8; 8];
                lookup_grid_8(
                    &tab.kmap,
                    &tab.grids,
                    (&xval[8 * k..8 * k + 8]).try_into().unwrap(),
                    (&weight[8 * k..8 * k + 8]).try_into().unwrap(),
                    max / (2 * K_MAX_Q - 1) as f32,
                    &mut lv,
                );
                ltmp[8 * k..8 * k + 8].copy_from_slice(&lv);
            }
            weighted_scale_refine(&xval, &weight, &ltmp)
        };
        let mut best = scale * weighted_scale_refine(&xval, &weight, &laux);

        for is in -6..=6 {
            let id = (2 * K_MAX_Q - 1) as f32 + is as f32 * 0.1;
            let id = id / max;
            let this_scale = 1.0 / id;
            for k in 0..4 {
                let mut lv = [0i8; 8];
                lookup_grid_8(
                    &tab.kmap,
                    &tab.grids,
                    (&xval[8 * k..8 * k + 8]).try_into().unwrap(),
                    (&weight[8 * k..8 * k + 8]).try_into().unwrap(),
                    this_scale,
                    &mut lv,
                );
                laux[8 * k..8 * k + 8].copy_from_slice(&lv);
            }
            let sumqx = weighted_scale_refine(&xval, &weight, &laux);
            let sumq2 = {
                let mut s = 0f32;
                for i in 0..32 {
                    let q = 2.0 * laux[i] as f32 + 1.0;
                    s += weight[i] * q * q;
                }
                s
            };
            if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                l.copy_from_slice(&laux);
            }
        }

        if scale > 0.0 {
            for k in 0..4 {
                let mut lv = [0i8; 8];
                lookup_grid_8(
                    &tab.kmap,
                    &tab.grids,
                    (&xval[8 * k..8 * k + 8]).try_into().unwrap(),
                    (&weight[8 * k..8 * k + 8]).try_into().unwrap(),
                    scale,
                    &mut lv,
                );
                l[8 * k..8 * k + 8].copy_from_slice(&lv);
            }
            scale = weighted_scale_refine(&xval, &weight, &l);
        }

        let mut signs = block_signs;
        if scale < 0.0 {
            scale = -scale;
            for s in &mut signs {
                *s = (!*s) & 127;
            }
        }

        for k in 0..4 {
            let mut lv = [0i8; 8];
            lv.copy_from_slice(&l[8 * k..8 * k + 8]);
            let mut u = 0u16;
            for i in 0..8 {
                u |= (lv[i] as u16) << (2 * i);
            }
            let gi = tab.kmap[u as usize];
            let grid_index = if gi >= 0 {
                gi as u32
            } else {
                find_best_grid_8(
                    &tab.grids,
                    (&xval[8 * k..8 * k + 8]).try_into().unwrap(),
                    (&weight[8 * k..8 * k + 8]).try_into().unwrap(),
                    scale,
                    &mut lv,
                ) as u32
            };
            q2[2 * ib] |= grid_index << (8 * k);
            q2[2 * ib + 1] |= (signs[k] as u32) << (7 * k);
        }
        scales[ib] = scale;
        max_scale = max_scale.max(scale);
    }

    if max_scale <= 0.0 {
        out[2..].fill(0);
        out[0..2].copy_from_slice(&f16_bytes(0.0));
        return;
    }

    let d = max_scale / 31.0;
    out[0..2].copy_from_slice(&f16_bytes(d));
    let id = 1.0 / d;
    for ib in 0..QK_K / 32 {
        let l = nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 15) as u32;
        q2[2 * ib + 1] |= l << 28;
    }
    for (i, chunk) in q2.iter().flat_map(|v| v.to_le_bytes()).enumerate() {
        out[2 + i] = chunk;
    }
}

pub fn quantize_iq2_xs_block(src: &[f32], out: &mut [u8]) {
    const QS0: usize = 2;
    const SC0: usize = 2 + (QK_K / 8) * 2;
    assert!(src.len() >= QK_K && out.len() >= SC0 + QK_K / 32);
    let tab = iq2_xs_tables();

    let mut q2 = [0u16; 2 * (QK_K / 16)];
    let mut scales = [0f32; QK_K / 16];
    let mut max_scale = 0f32;

    let sumx2: f32 = src[..QK_K].iter().map(|v| v * v).sum();
    let sigma2 = sumx2 / QK_K as f32;

    for ib in 0..QK_K / 16 {
        let xb = &src[ib * 16..(ib + 1) * 16];
        let mut weight = [0f32; 16];
        let mut xval = [0f32; 16];
        let mut block_signs = [0u8; 2];
        for i in 0..16 {
            weight[i] = sigma2 + xb[i] * xb[i];
        }
        extract_signs_parity(xb, &weight, &mut xval, &mut block_signs, 2);

        let mut max = xval[0];
        for &v in &xval[1..] {
            max = max.max(v);
        }
        if max < GROUP_MAX_EPS {
            scales[ib] = 0.0;
            continue;
        }

        let mut l = [0i8; 16];
        let mut laux = [0i8; 16];
        let mut scale = max / (2 * K_MAX_Q - 1) as f32;
        let mut best = 0f32;
        let mut on_grid = [true, true];
        let mut on_grid_aux = [true, true];

        for is in -9..=9 {
            let id = (2 * K_MAX_Q - 1) as f32 + is as f32 * 0.1;
            let id = id / max;
            let this_scale = 1.0 / id;
            for k in 0..2 {
                let mut lv = [0i8; 8];
                let gi = lookup_grid_8(
                    &tab.kmap,
                    &tab.grids,
                    (&xval[8 * k..8 * k + 8]).try_into().unwrap(),
                    (&weight[8 * k..8 * k + 8]).try_into().unwrap(),
                    this_scale,
                    &mut lv,
                );
                on_grid_aux[k] = tab.kmap[{
                    let mut u = 0u16;
                    for i in 0..8 {
                        u |= (lv[i] as u16) << (2 * i);
                    }
                    u as usize
                }] >= 0;
                let _ = gi;
                laux[8 * k..8 * k + 8].copy_from_slice(&lv);
            }
            let sumqx = weighted_scale_refine(&xval, &weight, &laux);
            let mut sumq2 = 0f32;
            for i in 0..16 {
                let q = 2.0 * laux[i] as f32 + 1.0;
                sumq2 += weight[i] * q * q;
            }
            if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                l.copy_from_slice(&laux);
                on_grid.copy_from_slice(&on_grid_aux);
            }
        }

        if (!on_grid[0] || !on_grid[1]) && scale > 0.0 {
            for k in 0..2 {
                if on_grid[k] {
                    continue;
                }
                let mut lv = [0i8; 8];
                lookup_grid_8(
                    &tab.kmap,
                    &tab.grids,
                    (&xval[8 * k..8 * k + 8]).try_into().unwrap(),
                    (&weight[8 * k..8 * k + 8]).try_into().unwrap(),
                    scale,
                    &mut lv,
                );
                l[8 * k..8 * k + 8].copy_from_slice(&lv);
            }
            scale = weighted_scale_refine(&xval, &weight, &l);
        }

        let mut signs = block_signs;
        if scale < 0.0 {
            scale = -scale;
            for s in &mut signs {
                *s = (!*s) & 127;
            }
        }

        for k in 0..2 {
            let mut lv = [0i8; 8];
            lv.copy_from_slice(&l[8 * k..8 * k + 8]);
            let mut u = 0u16;
            for i in 0..8 {
                u |= (lv[i] as u16) << (2 * i);
            }
            let gi = tab.kmap[u as usize];
            let grid_index = if gi >= 0 {
                gi as u16
            } else {
                find_best_grid_8(
                    &tab.grids,
                    (&xval[8 * k..8 * k + 8]).try_into().unwrap(),
                    (&weight[8 * k..8 * k + 8]).try_into().unwrap(),
                    scale,
                    &mut lv,
                ) as u16
            };
            q2[2 * ib + k] = grid_index | ((signs[k] as u16) << 9);
        }
        scales[ib] = scale;
        max_scale = max_scale.max(scale);
    }

    if max_scale <= 0.0 {
        out.fill(0);
        return;
    }

    let d = max_scale / 31.0;
    out[0..2].copy_from_slice(&f16_bytes(d));
    let id = 1.0 / d;
    for ib in 0..QK_K / 16 {
        let l = nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 15) as u8;
        if ib % 2 == 0 {
            out[SC0 + ib / 2] = l;
        } else {
            out[SC0 + ib / 2] |= l << 4;
        }
    }
    for (i, chunk) in q2.iter().flat_map(|v| v.to_le_bytes()).enumerate() {
        out[QS0 + i] = chunk;
    }
}

pub fn quantize_iq2_s_block(src: &[f32], out: &mut [u8]) {
    const QS0: usize = 2;
    const QH0: usize = 2 + QK_K / 4;
    const SC0: usize = QH0 + QK_K / 32;
    assert!(src.len() >= QK_K && out.len() >= SC0 + QK_K / 32);
    let tab = iq2_s_tables();

    let sumx2: f32 = src[..QK_K].iter().map(|v| v * v).sum();
    let sigma2 = 2.0 * sumx2 / QK_K as f32;

    let mut max_scale = 0f32;
    let mut scales = [0f32; QK_K / 16];

    for ib in 0..QK_K / 16 {
        let xb = &src[ib * 16..(ib + 1) * 16];
        let mut weight = [0f32; 16];
        let mut waux = [0f32; 16];
        let mut xval = [0f32; 16];
        let mut block_signs = [0u8; 2];
        for i in 0..16 {
            weight[i] = 0.25 * sigma2 + xb[i] * xb[i];
            waux[i] = weight[i].sqrt();
        }
        extract_signs_raw(xb, &mut xval, &mut block_signs, 2);

        let mut max = xval[0];
        for &v in &xval[1..] {
            max = max.max(v);
        }
        if max < GROUP_MAX_EPS {
            scales[ib] = 0.0;
            continue;
        }

        let mut l = [0i8; 16];
        let mut laux = [0i8; 16];
        let mut scale = max / (2 * K_MAX_Q - 1) as f32;
        let mut best = 0f32;
        let mut on_grid = [true, true];
        let mut on_grid_aux = [true, true];

        for is in -9..=9 {
            let id = (2 * K_MAX_Q - 1) as f32 + is as f32 * 0.1;
            let id = id / max;
            let this_scale = 1.0 / id;
            for k in 0..2 {
                let mut lv = [0i8; 8];
                lookup_grid_8(
                    &tab.kmap,
                    &tab.grids,
                    (&xval[8 * k..8 * k + 8]).try_into().unwrap(),
                    (&weight[8 * k..8 * k + 8]).try_into().unwrap(),
                    this_scale,
                    &mut lv,
                );
                on_grid_aux[k] = true;
                laux[8 * k..8 * k + 8].copy_from_slice(&lv);
            }
            let sumqx = weighted_scale_refine(&xval, &weight, &laux);
            let mut sumq2 = 0f32;
            for i in 0..16 {
                let q = 2.0 * laux[i] as f32 + 1.0;
                sumq2 += weight[i] * q * q;
            }
            if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                l.copy_from_slice(&laux);
                on_grid.copy_from_slice(&on_grid_aux);
            }
        }

        if (!on_grid[0] || !on_grid[1]) && scale > 0.0 {
            for k in 0..2 {
                if on_grid[k] {
                    continue;
                }
                let mut lv = [0i8; 8];
                lookup_grid_8(
                    &tab.kmap,
                    &tab.grids,
                    (&xval[8 * k..8 * k + 8]).try_into().unwrap(),
                    (&weight[8 * k..8 * k + 8]).try_into().unwrap(),
                    scale,
                    &mut lv,
                );
                l[8 * k..8 * k + 8].copy_from_slice(&lv);
            }
            scale = weighted_scale_refine(&xval, &weight, &l);
        }

        let mut signs = block_signs;
        if scale < 0.0 {
            scale = -scale;
            for s in &mut signs {
                *s = !*s;
            }
        }

        for k in 0..2 {
            let mut lv = [0i8; 8];
            lv.copy_from_slice(&l[8 * k..8 * k + 8]);
            let mut u = 0u16;
            for i in 0..8 {
                u |= (lv[i] as u16) << (2 * i);
            }
            let gi = tab.kmap[u as usize];
            let grid_index = if gi >= 0 {
                gi as usize
            } else {
                find_best_grid_8(
                    &tab.grids,
                    (&xval[8 * k..8 * k + 8]).try_into().unwrap(),
                    (&weight[8 * k..8 * k + 8]).try_into().unwrap(),
                    scale,
                    &mut lv,
                )
            };
            let i8 = 2 * ib + k;
            out[QS0 + i8] = (grid_index & 0xFF) as u8;
            out[QH0 + i8 / 4] |= (((grid_index >> 8) & 3) as u8) << (2 * (i8 % 4));
            out[QS0 + QK_K / 8 + i8] = signs[k];
        }
        scales[ib] = scale;
        max_scale = max_scale.max(scale);
    }

    if max_scale <= 0.0 {
        out.fill(0);
        return;
    }

    let d = max_scale / 31.0 * 0.9875;
    out[0..2].copy_from_slice(&f16_bytes(d));
    let id = 1.0 / (max_scale / 31.0);
    for ib in 0..QK_K / 16 {
        let l = nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 15) as u8;
        if ib % 2 == 0 {
            out[SC0 + ib / 2] = l;
        } else {
            out[SC0 + ib / 2] |= l << 4;
        }
    }
}
