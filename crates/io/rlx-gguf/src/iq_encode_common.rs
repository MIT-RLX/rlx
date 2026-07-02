// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Shared helpers for fast IQ-family encoders (llama.cpp-style sign
//! extraction + kmap grid lookup, uniform weights when no imatrix).

pub(crate) const GROUP_MAX_EPS: f32 = 1e-15;

#[inline]
pub(crate) fn f16_bytes(x: f32) -> [u8; 2] {
    half::f16::from_f32(x).to_le_bytes()
}

#[inline]
pub(crate) fn nearest_int(x: f32) -> i32 {
    if x >= 0.0 {
        (x + 0.5) as i32
    } else {
        (x - 0.5) as i32
    }
}

#[inline]
pub(crate) fn grid_u64_to_i8x8(entry: u64) -> [i8; 8] {
    let b = entry.to_le_bytes();
    [
        b[0] as i8, b[1] as i8, b[2] as i8, b[3] as i8, b[4] as i8, b[5] as i8, b[6] as i8,
        b[7] as i8,
    ]
}

#[inline]
pub(crate) fn grid_u32_to_i8x4(entry: u32) -> [i8; 4] {
    let b = entry.to_le_bytes();
    [b[0] as i8, b[1] as i8, b[2] as i8, b[3] as i8]
}

#[inline]
pub(crate) fn quant_index_2bit(g: &[i8; 8]) -> u16 {
    let mut index = 0u16;
    for (k, &gv) in g.iter().enumerate() {
        let l = ((gv as i32 - 1) / 2).clamp(0, 3) as u16;
        index |= l << (2 * k);
    }
    index
}

#[inline]
pub(crate) fn quant_index_3bit(g: &[i8; 4]) -> u16 {
    let mut index = 0u16;
    for (k, &gv) in g.iter().enumerate() {
        let l = ((gv as i32 - 1) / 2).clamp(0, 7) as u16;
        index |= l << (3 * k);
    }
    index
}

/// Extract absolute magnitudes + 7-bit sign indices (llama.cpp parity fix).
pub(crate) fn extract_signs_parity(
    xb: &[f32],
    weight: &[f32],
    xval: &mut [f32],
    block_signs: &mut [u8],
    groups: usize,
) {
    for k in 0..groups {
        let base = 8 * k;
        let mut nflip = 0u8;
        let mut s = 0u8;
        for i in 0..8 {
            if xb[base + i] >= 0.0 {
                xval[base + i] = xb[base + i];
            } else {
                xval[base + i] = -xb[base + i];
                nflip += 1;
                s |= 1 << i;
            }
        }
        if nflip & 1 == 1 {
            let mut imin = 0usize;
            let mut min = weight[base] * xb[base] * xb[base];
            for i in 1..8 {
                let ax = weight[base + i] * xb[base + i] * xb[base + i];
                if ax < min {
                    min = ax;
                    imin = i;
                }
            }
            xval[base + imin] = -xval[base + imin];
            s ^= 1 << imin;
        }
        block_signs[k] = s & 127;
    }
}

/// IQ2_S signs: no parity constraint, full byte per 8-group.
pub(crate) fn extract_signs_raw(
    xb: &[f32],
    xval: &mut [f32],
    block_signs: &mut [u8],
    groups: usize,
) {
    for k in 0..groups {
        let base = 8 * k;
        let mut s = 0u8;
        for i in 0..8 {
            if xb[base + i] >= 0.0 {
                xval[base + i] = xb[base + i];
            } else {
                xval[base + i] = -xb[base + i];
                s |= 1 << i;
            }
        }
        block_signs[k] = s;
    }
}

pub(crate) fn build_kmap_2bit(grids: &[[i8; 8]]) -> Vec<i32> {
    let mut kmap = vec![-1i32; 1 << 16];
    for (i, g) in grids.iter().enumerate() {
        kmap[quant_index_2bit(g) as usize] = i as i32;
    }
    kmap
}

pub(crate) fn build_kmap_3bit(grids: &[[i8; 4]]) -> Vec<i32> {
    let mut kmap = vec![-1i32; 1 << 12];
    for (i, g) in grids.iter().enumerate() {
        kmap[quant_index_3bit(g) as usize] = i as i32;
    }
    kmap
}

pub(crate) fn find_best_grid_8(
    grids: &[[i8; 8]],
    xval: &[f32; 8],
    weight: &[f32; 8],
    scale: f32,
    l_out: &mut [i8; 8],
) -> usize {
    let mut best_d2 = f32::MAX;
    let mut best_idx = 0usize;
    for (idx, g) in grids.iter().enumerate() {
        let mut d2 = 0f32;
        for i in 0..8 {
            let diff = scale * g[i] as f32 - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if d2 < best_d2 {
            best_d2 = d2;
            best_idx = idx;
        }
    }
    let g = &grids[best_idx];
    for i in 0..8 {
        l_out[i] = ((g[i] as i32 - 1) / 2).clamp(0, 3) as i8;
    }
    best_idx
}

pub(crate) fn find_best_grid_4(
    grids: &[[i8; 4]],
    xval: &[f32; 4],
    weight: &[f32; 4],
    scale: f32,
    l_out: &mut [i8; 4],
) -> usize {
    let mut best_d2 = f32::MAX;
    let mut best_idx = 0usize;
    for (idx, g) in grids.iter().enumerate() {
        let mut d2 = 0f32;
        for i in 0..4 {
            let diff = scale * g[i] as f32 - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if d2 < best_d2 {
            best_d2 = d2;
            best_idx = idx;
        }
    }
    let g = &grids[best_idx];
    for i in 0..4 {
        l_out[i] = ((g[i] as i32 - 1) / 2).clamp(0, 7) as i8;
    }
    best_idx
}

pub(crate) fn lookup_grid_8(
    kmap: &[i32],
    grids: &[[i8; 8]],
    xval: &[f32; 8],
    weight: &[f32; 8],
    scale: f32,
    l_out: &mut [i8; 8],
) -> usize {
    let mut u = 0u16;
    for i in 0..8 {
        let l = nearest_int(0.5 * (scale.recip() * xval[i] - 1.0)).clamp(0, 2) as u16;
        u |= l << (2 * i);
        l_out[i] = l as i8;
    }
    let gi = kmap[u as usize];
    if gi >= 0 {
        return gi as usize;
    }
    find_best_grid_8(grids, xval, weight, scale, l_out)
}

pub(crate) fn lookup_grid_4(
    kmap: &[i32],
    grids: &[[i8; 4]],
    xval: &[f32; 4],
    weight: &[f32; 4],
    scale: f32,
    l_out: &mut [i8; 4],
) -> usize {
    let mut u = 0u16;
    for i in 0..4 {
        let l = nearest_int(0.5 * (scale.recip() * xval[i] - 1.0)).clamp(0, 7) as u16;
        u |= l << (3 * i);
        l_out[i] = l as i8;
    }
    let gi = kmap[u as usize];
    if gi >= 0 {
        return gi as usize;
    }
    find_best_grid_4(grids, xval, weight, scale, l_out)
}

pub(crate) fn weighted_scale_refine(xval: &[f32], weight: &[f32], l: &[i8]) -> f32 {
    let mut sumqx = 0f32;
    let mut sumq2 = 0f32;
    for i in 0..xval.len() {
        let q = 2.0 * l[i] as f32 + 1.0;
        sumqx += weight[i] * xval[i] * q;
        sumq2 += weight[i] * q * q;
    }
    if sumq2 > 0.0 { sumqx / sumq2 } else { 0.0 }
}
