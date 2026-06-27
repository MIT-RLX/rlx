// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Fast IQ1 encoders — kmap grid lookup + reduced scale/delta search.

use std::sync::OnceLock;

use crate::QK_K;
use crate::iq_encode_common::{GROUP_MAX_EPS, f16_bytes, grid_u64_to_i8x8};
use crate::iq_grids::IQ1S_GRID;

const IQ1S_DELTA: f32 = 0.125;

struct Iq1Tables {
    grids: Vec<[i8; 8]>,
}

fn iq1_tables() -> &'static Iq1Tables {
    static T: OnceLock<Iq1Tables> = OnceLock::new();
    T.get_or_init(|| {
        let grids: Vec<[i8; 8]> = IQ1S_GRID.iter().map(|&e| grid_u64_to_i8x8(e)).collect();
        Iq1Tables { grids }
    })
}

fn mse_iq1(xg: &[f32; 8], grid: &[i8; 8], dl: f32, delta: f32) -> f32 {
    xg.iter()
        .zip(grid.iter())
        .map(|(&x, &g)| {
            let r = dl * (g as f32 + delta) - x;
            r * r
        })
        .sum()
}

/// Search idx 0..2047 — low 8 bits land in `qs`, bits 8..10 land in `qh`.
fn best_iq1_group(tab: &Iq1Tables, xg: &[f32; 8], dl: f32, delta: f32) -> u16 {
    let mut best_idx = 0u16;
    let mut best_err = f32::MAX;
    for idx in 0..2048u16 {
        let grid_idx = (idx as usize & 0xFF) | (((idx as usize >> 8) & 7) << 8);
        let grid = tab.grids[grid_idx.min(tab.grids.len() - 1)];
        let err = mse_iq1(xg, &grid, dl, delta);
        if err < best_err {
            best_err = err;
            best_idx = idx;
        }
    }
    best_idx
}

pub fn quantize_iq1_s_block(src: &[f32], out: &mut [u8]) {
    const QS0: usize = 2;
    const QH0: usize = 2 + QK_K / 8;
    assert!(src.len() >= QK_K && out.len() >= QH0 + QK_K / 16);
    let tab = iq1_tables();

    let mut amax = 0f32;
    for &v in &src[..QK_K] {
        amax = amax.max(v.abs());
    }
    let d = amax.max(GROUP_MAX_EPS);
    out[0..2].copy_from_slice(&f16_bytes(d));

    let mut qs_idx = 0usize;
    for ib in 0..QK_K / 32 {
        let xb = &src[ib * 32..(ib + 1) * 32];
        let mut best_qh = 0u16;
        let mut best_qs = [0u8; 4];
        let mut best_err = f32::MAX;

        for scale3 in 0..8u16 {
            for delta_neg in [false, true] {
                let dl = d * (2.0 * scale3 as f32 + 1.0);
                let delta = if delta_neg { -IQ1S_DELTA } else { IQ1S_DELTA };
                let mut qh = scale3 << 12;
                if delta_neg {
                    qh |= 0x8000;
                }
                let mut recon_err = 0f32;
                let mut qtmp = [0u8; 4];
                for l in 0..4 {
                    let xg: [f32; 8] = xb[l * 8..l * 8 + 8].try_into().unwrap();
                    let best_idx = best_iq1_group(tab, &xg, dl, delta);
                    qtmp[l] = (best_idx & 0xFF) as u8;
                    qh |= ((best_idx >> 8) & 7) << (3 * l);
                    let grid_idx =
                        (best_idx as usize & 0xFF) | (((best_idx as usize >> 8) & 7) << 8);
                    let grid = tab.grids[grid_idx.min(tab.grids.len() - 1)];
                    recon_err += mse_iq1(&xg, &grid, dl, delta);
                }
                if recon_err < best_err {
                    best_err = recon_err;
                    best_qh = qh;
                    best_qs = qtmp;
                }
            }
        }
        out[QH0 + 2 * ib..QH0 + 2 * ib + 2].copy_from_slice(&best_qh.to_le_bytes());
        out[QS0 + qs_idx..QS0 + qs_idx + 4].copy_from_slice(&best_qs);
        qs_idx += 4;
    }
}

pub fn quantize_iq1_m_block(src: &[f32], out: &mut [u8]) {
    const QS0: usize = 0;
    const QH0: usize = QK_K / 8;
    const SC0: usize = QK_K / 8 + QK_K / 16;
    assert!(src.len() >= QK_K && out.len() >= SC0 + 8);
    let tab = iq1_tables();

    let mut amax = 0f32;
    for &v in &src[..QK_K] {
        amax = amax.max(v.abs());
    }
    let d = amax.max(GROUP_MAX_EPS);
    let scale_u16 = half::f16::from_f32(d).to_bits();
    let sc = [
        scale_u16 | 0xF000,
        (scale_u16 & 0x0FF0) | 0x000F,
        (scale_u16 & 0x00FF) | 0x0F00,
        scale_u16 & 0xFFF0,
    ];
    out[SC0..SC0 + 2].copy_from_slice(&sc[0].to_le_bytes());
    out[SC0 + 2..SC0 + 4].copy_from_slice(&sc[1].to_le_bytes());
    out[SC0 + 4..SC0 + 6].copy_from_slice(&sc[2].to_le_bytes());
    out[SC0 + 6..SC0 + 8].copy_from_slice(&sc[3].to_le_bytes());

    let mut qs_walk = 0usize;
    let mut qh_walk = 0usize;
    for ib in 0..QK_K / 32 {
        let sc_word = sc[ib / 2];
        let dl1 = d * (2.0 * ((sc_word >> (6 * (ib % 2))) & 0x7) as f32 + 1.0);
        let dl2 = d * (2.0 * ((sc_word >> (6 * (ib % 2) + 3)) & 0x7) as f32 + 1.0);
        for (half, dl) in [(0usize, dl1), (1, dl2)] {
            let xoff = ib * 32 + half * 16;
            for g in 0..2usize {
                let xg: [f32; 8] = src[xoff + g * 8..xoff + g * 8 + 8].try_into().unwrap();
                let mut gbest = (0u16, false, f32::MAX);
                for idx in 0..2048u16 {
                    for delta_neg in [false, true] {
                        let grid_idx = (idx as usize & 0xFF) | (((idx as usize >> 8) & 7) << 8);
                        let grid = tab.grids[grid_idx.min(tab.grids.len() - 1)];
                        let delta = if delta_neg { -IQ1S_DELTA } else { IQ1S_DELTA };
                        let err = mse_iq1(&xg, &grid, dl, delta);
                        if err < gbest.2 {
                            gbest = (idx, delta_neg, err);
                        }
                    }
                }
                out[QS0 + qs_walk + g + half * 2] = (gbest.0 & 0xFF) as u8;
                let hi = ((gbest.0 >> 8) & 7) as u8;
                let qh_slot = QH0 + qh_walk + half;
                if g == 0 {
                    out[qh_slot] = (out[qh_slot] & 0xF0) | hi;
                    if gbest.1 {
                        out[qh_slot] |= 0x08;
                    } else {
                        out[qh_slot] &= !0x08;
                    }
                } else {
                    out[qh_slot] = (out[qh_slot] & 0x0F) | (hi << 4);
                    if gbest.1 {
                        out[qh_slot] |= 0x80;
                    } else {
                        out[qh_slot] &= !0x80;
                    }
                }
            }
        }
        qs_walk += 4;
        qh_walk += 2;
    }
}
