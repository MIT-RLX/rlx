// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Fast IQ3 encoders — sign extraction + precomputed 4-wide grids.

use std::sync::OnceLock;

use crate::QK_K;
use crate::iq_encode_common::{
    GROUP_MAX_EPS, build_kmap_3bit, extract_signs_parity, extract_signs_raw, f16_bytes,
    grid_u32_to_i8x4, lookup_grid_4, nearest_int, weighted_scale_refine,
};
use crate::iq_grids::{IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};

const K_MAX_Q_IQ3: i32 = 8;

struct Iq3Tables {
    grids: Vec<[i8; 4]>,
    kmap: Vec<i32>,
}

fn tables_from_raw(raw: &[u32]) -> Iq3Tables {
    let grids: Vec<[i8; 4]> = raw.iter().map(|&e| grid_u32_to_i8x4(e)).collect();
    let kmap = build_kmap_3bit(&grids);
    Iq3Tables { grids, kmap }
}

fn iq3_xxs_tables() -> &'static Iq3Tables {
    static T: OnceLock<Iq3Tables> = OnceLock::new();
    T.get_or_init(|| tables_from_raw(&IQ3XXS_GRID))
}

fn iq3_s_tables() -> &'static Iq3Tables {
    static T: OnceLock<Iq3Tables> = OnceLock::new();
    T.get_or_init(|| tables_from_raw(&IQ3S_GRID))
}

fn mse_group8_xxs(
    xg: &[f32; 8],
    weight: &[f32; 8],
    db: f32,
    g1: &[i8; 4],
    g2: &[i8; 4],
    sign_idx: u8,
) -> f32 {
    let mask = KSIGNS_IQ2XS[(sign_idx & 127) as usize];
    mse_group8_mask(xg, weight, db, g1, g2, mask)
}

fn mse_group8_s(
    xg: &[f32; 8],
    weight: &[f32; 8],
    db: f32,
    g1: &[i8; 4],
    g2: &[i8; 4],
    sign: u8,
) -> f32 {
    mse_group8_mask(xg, weight, db, g1, g2, sign)
}

fn mse_group8_mask(
    xg: &[f32; 8],
    weight: &[f32; 8],
    db: f32,
    g1: &[i8; 4],
    g2: &[i8; 4],
    mask: u8,
) -> f32 {
    let mut err = 0f32;
    for j in 0..4 {
        let s0 = if mask & KMASK_IQ2XS[j] != 0 {
            -1.0
        } else {
            1.0
        };
        let s1 = if mask & KMASK_IQ2XS[j + 4] != 0 {
            -1.0
        } else {
            1.0
        };
        let d0 = db * g1[j] as f32 * s0 - xg[j];
        let d1 = db * g2[j] as f32 * s1 - xg[j + 4];
        err += weight[j] * d0 * d0 + weight[j + 4] * d1 * d1;
    }
    err
}

fn pick_pair_xxs(
    tab: &Iq3Tables,
    xg: &[f32; 8],
    weight: &[f32; 8],
    sign_idx: u8,
    db: f32,
) -> (u8, u8) {
    let mut best = (0u8, 0u8);
    let mut best_err = f32::MAX;
    for g1 in 0..256u16 {
        let grid1 = tab.grids[g1 as usize];
        for g2 in 0..256u16 {
            let grid2 = tab.grids[g2 as usize];
            let err = mse_group8_xxs(xg, weight, db, &grid1, &grid2, sign_idx);
            if err < best_err {
                best_err = err;
                best = (g1 as u8, g2 as u8);
            }
        }
    }
    best
}

pub fn quantize_iq3_xxs_block(src: &[f32], out: &mut [u8]) {
    const QS0: usize = 2;
    const SAS0: usize = 2 + QK_K / 4;
    assert!(src.len() >= QK_K && out.len() >= SAS0 + QK_K / 8);
    let tab = iq3_xxs_tables();

    let mut max_scale = 0f32;
    let mut scales = [0f32; QK_K / 32];
    let mut qbytes = [0u8; QK_K / 4];
    let mut aux = [0u32; QK_K / 32];

    for ib in 0..QK_K / 32 {
        let xb = &src[ib * 32..(ib + 1) * 32];
        let mut weight = [0f32; 32];
        let mut xval = [0f32; 32];
        let mut block_signs = [0u8; 4];
        for i in 0..32 {
            weight[i] = xb[i] * xb[i];
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
        let mut scale = max / (2 * K_MAX_Q_IQ3 - 1) as f32;
        let mut best = 0f32;

        for is in -15..=15 {
            let id = (2 * K_MAX_Q_IQ3 - 1) as f32 + is as f32 * 0.2;
            let id = id / max;
            let this_scale = 1.0 / id;
            for k in 0..4 {
                let mut lv = [0i8; 4];
                lookup_grid_4(
                    &tab.kmap,
                    &tab.grids,
                    (&xval[4 * k..4 * k + 4]).try_into().unwrap(),
                    (&weight[4 * k..4 * k + 4]).try_into().unwrap(),
                    this_scale,
                    &mut lv,
                );
                laux[4 * k..4 * k + 4].copy_from_slice(&lv);
            }
            let sumqx = weighted_scale_refine(&xval, &weight, &laux);
            let mut sumq2 = 0f32;
            for i in 0..32 {
                let q = 2.0 * laux[i] as f32 + 1.0;
                sumq2 += weight[i] * q * q;
            }
            if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                l.copy_from_slice(&laux);
            }
        }

        if scale > 0.0 {
            scale = weighted_scale_refine(&xval, &weight, &l);
        }

        let db = scale;
        for l in 0..4 {
            let xg: [f32; 8] = xval[l * 8..l * 8 + 8].try_into().unwrap();
            let wg: [f32; 8] = weight[l * 8..l * 8 + 8].try_into().unwrap();
            let (g1, g2) = pick_pair_xxs(tab, &xg, &wg, block_signs[l], db);
            qbytes[ib * 8 + 2 * l] = g1;
            qbytes[ib * 8 + 2 * l + 1] = g2;
            aux[ib] |= (block_signs[l] as u32) << (7 * l);
        }
        scales[ib] = scale;
        max_scale = max_scale.max(scale);
    }

    if max_scale <= 0.0 {
        out.fill(0);
        return;
    }

    let d = max_scale / 63.0;
    out[0..2].copy_from_slice(&f16_bytes(d));
    let id = 1.0 / d;
    for ib in 0..QK_K / 32 {
        let l = nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 15) as u32;
        aux[ib] |= l << 28;
    }
    out[QS0..QS0 + QK_K / 4].copy_from_slice(&qbytes);
    for ib in 0..QK_K / 32 {
        out[SAS0 + 4 * ib..SAS0 + 4 * ib + 4].copy_from_slice(&aux[ib].to_le_bytes());
    }
}

fn pick_pair_s(
    tab: &Iq3Tables,
    xg: &[f32; 8],
    weight: &[f32; 8],
    db: f32,
    sign: u8,
    l: usize,
) -> (u8, u8, u8) {
    let mut best = (0u8, 0u8, 0u8);
    let mut best_err = f32::MAX;
    for g1 in 0..256u16 {
        for g2 in 0..256u16 {
            for qh_b in 0..4u8 {
                let idx1 = g1 as usize | (((qh_b as usize) << (8 - 2 * l)) & 256);
                let idx2 = g2 as usize | (((qh_b as usize) << (7 - 2 * l)) & 256);
                let grid1 = tab.grids[idx1.min(tab.grids.len() - 1)];
                let grid2 = tab.grids[idx2.min(tab.grids.len() - 1)];
                let err = mse_group8_s(xg, weight, db, &grid1, &grid2, sign);
                if err < best_err {
                    best_err = err;
                    best = (g1 as u8, g2 as u8, qh_b);
                }
            }
        }
    }
    best
}

fn approx_err_half(xval: &[f32; 32], db: f32) -> f32 {
    xval.iter()
        .map(|&x| {
            let id = if db != 0.0 { x / db } else { 0.0 };
            let l = nearest_int(0.5 * (id - 1.0)).clamp(0, 7) as f32;
            let q = 2.0 * l + 1.0;
            (x - db * q).powi(2)
        })
        .sum()
}

fn estimate_scale_nibble(xval: &[f32], d: f32) -> u8 {
    let max = xval.iter().copied().fold(0f32, f32::max);
    if max < GROUP_MAX_EPS {
        return 0;
    }
    ((max / (d * 15.0) - 1.0) / 2.0).round().clamp(0.0, 15.0) as u8
}

pub fn quantize_iq3_s_block(src: &[f32], out: &mut [u8]) {
    const QS0: usize = 2;
    const QH0: usize = 2 + QK_K / 4;
    const SG0: usize = QH0 + QK_K / 32;
    const SC0: usize = SG0 + QK_K / 8;
    assert!(src.len() >= QK_K && out.len() >= SC0 + QK_K / 64);
    let tab = iq3_s_tables();
    out.fill(0);

    let sumx2: f32 = src[..QK_K].iter().map(|v| v * v).sum();
    let sigma2 = 2.0 * sumx2 / QK_K as f32;

    let mut max_scale = 0f32;
    let mut amax = 0f32;
    for &v in &src[..QK_K] {
        amax = amax.max(v.abs());
    }
    let d = amax.max(GROUP_MAX_EPS);
    out[0..2].copy_from_slice(&f16_bytes(d));

    let mut qs_walk = 0usize;
    let mut signs_walk = 0usize;
    let mut qh_walk = 0usize;

    for ib32 in (0..QK_K / 32).step_by(2) {
        let mut xval0 = [0f32; 32];
        let mut xval1 = [0f32; 32];
        let mut signs0 = [0u8; 4];
        let mut signs1 = [0u8; 4];
        extract_signs_raw(&src[ib32 * 32..ib32 * 32 + 32], &mut xval0, &mut signs0, 4);
        extract_signs_raw(
            &src[(ib32 + 1) * 32..(ib32 + 2) * 32],
            &mut xval1,
            &mut signs1,
            4,
        );
        let lo_est = estimate_scale_nibble(&xval0, d);
        let hi_est = estimate_scale_nibble(&xval1, d);
        let mut sb = lo_est | (hi_est << 4);
        let mut best_err = f32::MAX;
        for dlo in -1i32..=1 {
            for dhi in -1i32..=1 {
                let lo = (lo_est as i32 + dlo).clamp(0, 15) as u8;
                let hi = (hi_est as i32 + dhi).clamp(0, 15) as u8;
                let err = approx_err_half(&xval0, d * (1.0 + 2.0 * lo as f32))
                    + approx_err_half(&xval1, d * (1.0 + 2.0 * hi as f32));
                if err < best_err {
                    best_err = err;
                    sb = lo | (hi << 4);
                }
            }
        }
        out[SC0 + ib32 / 2] = sb;

        let db1 = d * (1.0 + 2.0 * (sb & 0xF) as f32);
        let db2 = d * (1.0 + 2.0 * (sb >> 4) as f32);

        for half in 0..2usize {
            let xoff = (ib32 + half) * 32;
            let xval = if half == 0 { &xval0 } else { &xval1 };
            let block_signs = if half == 0 { &signs0 } else { &signs1 };
            let mut weight = [0f32; 32];
            for i in 0..32 {
                weight[i] = sigma2 + src[xoff + i] * src[xoff + i];
            }
            let db = if half == 0 { db1 } else { db2 };
            for l in 0..4 {
                let xg: [f32; 8] = xval[l * 8..l * 8 + 8].try_into().unwrap();
                let wg: [f32; 8] = weight[l * 8..l * 8 + 8].try_into().unwrap();
                let (g1, g2, qh_b) = pick_pair_s(tab, &xg, &wg, db, block_signs[l], l);
                out[QS0 + qs_walk + 2 * l] = g1;
                out[QS0 + qs_walk + 2 * l + 1] = g2;
                out[SG0 + signs_walk + l] = block_signs[l];
                out[QH0 + qh_walk + half] |= qh_b << (2 * l);
            }
            qs_walk += 8;
            signs_walk += 4;
            max_scale = max_scale.max(db1.max(db2));
        }
        qh_walk += 2;
    }
}
