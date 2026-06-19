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

//! IQ-family dequant kernels — `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`,
//! `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`.
//!
//! These formats compress sub-byte weights via grid lookup. The grid
//! tables ([`crate::iq_grids`]) are taken verbatim from llama.cpp's
//! `ggml-common.h`. The dequant routines below transliterate the
//! reference `dequantize_row_iq*` functions in `ggml-quants.c`; each
//! `u64`/`u32` grid entry is a packed 8-byte (or 4-byte) i8 vector
//! interpreted by viewing the integer in little-endian byte order.

use anyhow::{Result, bail};

use crate::iq_grids::{
    IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS,
    KSIGNS_IQ2XS, KVALUES_IQ4NL,
};
use crate::{QK_K, read_f16_le};

/// IQ1S / IQ1M nudge constant — matches `IQ1S_DELTA` / `IQ1M_DELTA` in
/// llama.cpp.
const IQ1S_DELTA: f32 = 0.125;

/// 32-element block size for `IQ4_NL`.
pub const QK4_NL: usize = 32;

const IQ4_NL_BLOCK_BYTES: usize = 2 + QK4_NL / 2; // 18
const IQ4XS_BLOCK_BYTES: usize = 2 + 2 + QK_K / 64 + QK_K / 2; // 136
const IQ2XXS_BLOCK_BYTES: usize = 2 + (QK_K / 8) * 2; // 66
const IQ2XS_BLOCK_BYTES: usize = 2 + (QK_K / 8) * 2 + QK_K / 32; // 74
const IQ2S_BLOCK_BYTES: usize = 2 + QK_K / 4 + QK_K / 32 + QK_K / 32; // 82
const IQ3XXS_BLOCK_BYTES: usize = 2 + 3 * (QK_K / 8); // 98
const IQ3S_N_SCALE: usize = QK_K / 64; // 4
const IQ3S_BLOCK_BYTES: usize = 2 + QK_K / 4 + QK_K / 32 + QK_K / 8 + IQ3S_N_SCALE; // 110
const IQ1S_BLOCK_BYTES: usize = 2 + QK_K / 8 + (QK_K / 32) * 2; // 50
const IQ1M_BLOCK_BYTES: usize = QK_K / 8 + QK_K / 16 + QK_K / 32; // 56

// ─── byte-count helpers ─────────────────────────────────────────────

pub fn iq4_nl_bytes(n: usize) -> Option<usize> {
    if !n.is_multiple_of(QK4_NL) {
        return None;
    }
    Some((n / QK4_NL) * IQ4_NL_BLOCK_BYTES)
}

pub fn iq4_xs_bytes(n: usize) -> Option<usize> {
    if !n.is_multiple_of(QK_K) {
        return None;
    }
    Some((n / QK_K) * IQ4XS_BLOCK_BYTES)
}

pub fn iq2_xxs_bytes(n: usize) -> Option<usize> {
    if !n.is_multiple_of(QK_K) {
        return None;
    }
    Some((n / QK_K) * IQ2XXS_BLOCK_BYTES)
}

pub fn iq2_xs_bytes(n: usize) -> Option<usize> {
    if !n.is_multiple_of(QK_K) {
        return None;
    }
    Some((n / QK_K) * IQ2XS_BLOCK_BYTES)
}

pub fn iq2_s_bytes(n: usize) -> Option<usize> {
    if !n.is_multiple_of(QK_K) {
        return None;
    }
    Some((n / QK_K) * IQ2S_BLOCK_BYTES)
}

pub fn iq3_xxs_bytes(n: usize) -> Option<usize> {
    if !n.is_multiple_of(QK_K) {
        return None;
    }
    Some((n / QK_K) * IQ3XXS_BLOCK_BYTES)
}

pub fn iq3_s_bytes(n: usize) -> Option<usize> {
    if !n.is_multiple_of(QK_K) {
        return None;
    }
    Some((n / QK_K) * IQ3S_BLOCK_BYTES)
}

pub fn iq1_s_bytes(n: usize) -> Option<usize> {
    if !n.is_multiple_of(QK_K) {
        return None;
    }
    Some((n / QK_K) * IQ1S_BLOCK_BYTES)
}

pub fn iq1_m_bytes(n: usize) -> Option<usize> {
    if !n.is_multiple_of(QK_K) {
        return None;
    }
    Some((n / QK_K) * IQ1M_BLOCK_BYTES)
}

// ─── shared helpers ─────────────────────────────────────────────────

#[inline]
fn grid_u64_to_i8x8(entry: u64) -> [i8; 8] {
    let b = entry.to_le_bytes();
    [
        b[0] as i8, b[1] as i8, b[2] as i8, b[3] as i8, b[4] as i8, b[5] as i8, b[6] as i8,
        b[7] as i8,
    ]
}

#[inline]
fn grid_u32_to_i8x4(entry: u32) -> [i8; 4] {
    let b = entry.to_le_bytes();
    [b[0] as i8, b[1] as i8, b[2] as i8, b[3] as i8]
}

#[inline]
fn read_u16_le(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}

#[inline]
fn read_u32_le(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

// ─── IQ4_NL ──────────────────────────────────────────────────────────

pub fn dequant_iq4_nl(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if !n.is_multiple_of(QK4_NL) {
        bail!("IQ4_NL: n={n} not divisible by {QK4_NL}");
    }
    let nb = n / QK4_NL;
    if bytes.len() != nb * IQ4_NL_BLOCK_BYTES {
        bail!(
            "IQ4_NL: expected {} bytes, got {}",
            nb * IQ4_NL_BLOCK_BYTES,
            bytes.len()
        );
    }
    let mut out = vec![0f32; n];
    for i in 0..nb {
        let off = i * IQ4_NL_BLOCK_BYTES;
        let d = read_f16_le(&bytes[off..off + 2]);
        let qs = &bytes[off + 2..off + 2 + QK4_NL / 2];
        let dst = &mut out[i * QK4_NL..(i + 1) * QK4_NL];
        for j in 0..QK4_NL / 2 {
            dst[j] = d * KVALUES_IQ4NL[(qs[j] & 0xF) as usize] as f32;
            dst[j + QK4_NL / 2] = d * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
        }
    }
    Ok(out)
}

// ─── IQ4_XS ──────────────────────────────────────────────────────────

pub fn dequant_iq4_xs(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if !n.is_multiple_of(QK_K) {
        bail!("IQ4_XS: n={n} not divisible by {QK_K}");
    }
    let nb = n / QK_K;
    if bytes.len() != nb * IQ4XS_BLOCK_BYTES {
        bail!(
            "IQ4_XS: expected {} bytes, got {}",
            nb * IQ4XS_BLOCK_BYTES,
            bytes.len()
        );
    }
    let mut out = vec![0f32; n];
    let scales_l_len = QK_K / 64; // 4
    for i in 0..nb {
        let off = i * IQ4XS_BLOCK_BYTES;
        let d = read_f16_le(&bytes[off..off + 2]);
        let scales_h = read_u16_le(&bytes[off + 2..off + 4]);
        let scales_l = &bytes[off + 4..off + 4 + scales_l_len];
        let qs = &bytes[off + 4 + scales_l_len..off + IQ4XS_BLOCK_BYTES];
        let dst = &mut out[i * QK_K..(i + 1) * QK_K];
        let mut y = 0usize;
        let mut qs_off = 0usize;
        for ib in 0..QK_K / 32 {
            let lo = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF;
            let hi = ((scales_h >> (2 * ib)) & 0x3) as u8;
            let ls = (lo as i32) | ((hi as i32) << 4);
            let dl = d * (ls - 32) as f32;
            for j in 0..16 {
                let b = qs[qs_off + j];
                dst[y + j] = dl * KVALUES_IQ4NL[(b & 0xF) as usize] as f32;
                dst[y + j + 16] = dl * KVALUES_IQ4NL[(b >> 4) as usize] as f32;
            }
            y += 32;
            qs_off += 16;
        }
    }
    Ok(out)
}

// ─── IQ2_XXS ─────────────────────────────────────────────────────────

pub fn dequant_iq2_xxs(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if !n.is_multiple_of(QK_K) {
        bail!("IQ2_XXS: n={n} not divisible by {QK_K}");
    }
    let nb = n / QK_K;
    if bytes.len() != nb * IQ2XXS_BLOCK_BYTES {
        bail!(
            "IQ2_XXS: expected {} bytes, got {}",
            nb * IQ2XXS_BLOCK_BYTES,
            bytes.len()
        );
    }
    let mut out = vec![0f32; n];
    for i in 0..nb {
        let off = i * IQ2XXS_BLOCK_BYTES;
        let d = read_f16_le(&bytes[off..off + 2]);
        let qs = &bytes[off + 2..off + IQ2XXS_BLOCK_BYTES]; // 64 bytes
        let dst = &mut out[i * QK_K..(i + 1) * QK_K];
        let mut y = 0usize;
        for ib32 in 0..QK_K / 32 {
            let base = 8 * ib32;
            let aux32_0 = read_u32_le(&qs[base..base + 4]);
            let aux32_1 = read_u32_le(&qs[base + 4..base + 8]);
            let aux8 = aux32_0.to_le_bytes();
            let db = d * (0.5 + (aux32_1 >> 28) as f32) * 0.25;
            for l in 0..4 {
                let grid_idx = aux8[l] as usize;
                let grid = grid_u64_to_i8x8(IQ2XXS_GRID[grid_idx]);
                let signs = KSIGNS_IQ2XS[((aux32_1 >> (7 * l)) & 127) as usize];
                for j in 0..8 {
                    let sign = if signs & KMASK_IQ2XS[j] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    dst[y + j] = db * grid[j] as f32 * sign;
                }
                y += 8;
            }
        }
    }
    Ok(out)
}

// ─── IQ2_XS ──────────────────────────────────────────────────────────

pub fn dequant_iq2_xs(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if !n.is_multiple_of(QK_K) {
        bail!("IQ2_XS: n={n} not divisible by {QK_K}");
    }
    let nb = n / QK_K;
    if bytes.len() != nb * IQ2XS_BLOCK_BYTES {
        bail!(
            "IQ2_XS: expected {} bytes, got {}",
            nb * IQ2XS_BLOCK_BYTES,
            bytes.len()
        );
    }
    let mut out = vec![0f32; n];
    for i in 0..nb {
        let off = i * IQ2XS_BLOCK_BYTES;
        let d = read_f16_le(&bytes[off..off + 2]);
        let qs_off = off + 2;
        let qs = &bytes[qs_off..qs_off + (QK_K / 8) * 2]; // 64 bytes
        let scales = &bytes[qs_off + (QK_K / 8) * 2..off + IQ2XS_BLOCK_BYTES];
        let dst = &mut out[i * QK_K..(i + 1) * QK_K];
        let mut y = 0usize;
        for ib32 in 0..QK_K / 32 {
            let db0 = d * (0.5 + (scales[ib32] & 0xF) as f32) * 0.25;
            let db1 = d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25;
            for l in 0..4 {
                let pos = (4 * ib32 + l) * 2;
                let q = read_u16_le(&qs[pos..pos + 2]);
                let grid = grid_u64_to_i8x8(IQ2XS_GRID[(q & 511) as usize]);
                let signs = KSIGNS_IQ2XS[(q >> 9) as usize];
                let dl = if l / 2 == 0 { db0 } else { db1 };
                for j in 0..8 {
                    let sign = if signs & KMASK_IQ2XS[j] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    dst[y + j] = dl * grid[j] as f32 * sign;
                }
                y += 8;
            }
        }
    }
    Ok(out)
}

// ─── IQ2_S ───────────────────────────────────────────────────────────
//
// Block layout: f16 d (2) | qs[QK_K/4 = 64] | qh[QK_K/32 = 8] |
// scales[QK_K/32 = 8]. qs is split internally: the first QK_K/8 = 32
// bytes hold grid indices; the next 32 bytes hold sign masks (one byte
// per 8-element group).

pub fn dequant_iq2_s(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if !n.is_multiple_of(QK_K) {
        bail!("IQ2_S: n={n} not divisible by {QK_K}");
    }
    let nb = n / QK_K;
    if bytes.len() != nb * IQ2S_BLOCK_BYTES {
        bail!(
            "IQ2_S: expected {} bytes, got {}",
            nb * IQ2S_BLOCK_BYTES,
            bytes.len()
        );
    }
    let mut out = vec![0f32; n];
    for i in 0..nb {
        let off = i * IQ2S_BLOCK_BYTES;
        let d = read_f16_le(&bytes[off..off + 2]);
        let qs_off = off + 2;
        let qs = &bytes[qs_off..qs_off + QK_K / 4]; // 64
        let qh = &bytes[qs_off + QK_K / 4..qs_off + QK_K / 4 + QK_K / 32]; // 8
        let scales =
            &bytes[qs_off + QK_K / 4 + QK_K / 32..qs_off + QK_K / 4 + QK_K / 32 + QK_K / 32];
        let dst = &mut out[i * QK_K..(i + 1) * QK_K];
        let mut y = 0usize;
        let mut qs_idx = 0usize;
        let mut signs_idx = QK_K / 8; // 32, into qs
        for ib32 in 0..QK_K / 32 {
            let db0 = d * (0.5 + (scales[ib32] & 0xF) as f32) * 0.25;
            let db1 = d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25;
            for l in 0..4 {
                let dl = if l / 2 == 0 { db0 } else { db1 };
                let q = qs[qs_idx + l] as u16;
                let qh_b = qh[ib32] as u16;
                let idx = (q | ((qh_b << (8 - 2 * l)) & 0x300)) as usize;
                let grid = grid_u64_to_i8x8(IQ2S_GRID[idx]);
                let sign = qs[signs_idx + l];
                for j in 0..8 {
                    let s = if sign & KMASK_IQ2XS[j] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    dst[y + j] = dl * grid[j] as f32 * s;
                }
                y += 8;
            }
            qs_idx += 4;
            signs_idx += 4;
        }
    }
    Ok(out)
}

// ─── IQ3_XXS ─────────────────────────────────────────────────────────

pub fn dequant_iq3_xxs(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if !n.is_multiple_of(QK_K) {
        bail!("IQ3_XXS: n={n} not divisible by {QK_K}");
    }
    let nb = n / QK_K;
    if bytes.len() != nb * IQ3XXS_BLOCK_BYTES {
        bail!(
            "IQ3_XXS: expected {} bytes, got {}",
            nb * IQ3XXS_BLOCK_BYTES,
            bytes.len()
        );
    }
    let mut out = vec![0f32; n];
    for i in 0..nb {
        let off = i * IQ3XXS_BLOCK_BYTES;
        let d = read_f16_le(&bytes[off..off + 2]);
        let qs = &bytes[off + 2..off + 2 + QK_K / 4]; // 64
        let sas = &bytes[off + 2 + QK_K / 4..off + IQ3XXS_BLOCK_BYTES]; // 32
        let dst = &mut out[i * QK_K..(i + 1) * QK_K];
        let mut y = 0usize;
        let mut qs_idx = 0usize;
        for ib32 in 0..QK_K / 32 {
            let aux32 = read_u32_le(&sas[4 * ib32..4 * ib32 + 4]);
            let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
            for l in 0..4 {
                let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
                let g1 = grid_u32_to_i8x4(IQ3XXS_GRID[qs[qs_idx + 2 * l] as usize]);
                let g2 = grid_u32_to_i8x4(IQ3XXS_GRID[qs[qs_idx + 2 * l + 1] as usize]);
                for j in 0..4 {
                    let s0 = if signs & KMASK_IQ2XS[j] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    let s1 = if signs & KMASK_IQ2XS[j + 4] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    dst[y + j] = db * g1[j] as f32 * s0;
                    dst[y + j + 4] = db * g2[j] as f32 * s1;
                }
                y += 8;
            }
            qs_idx += 8;
        }
    }
    Ok(out)
}

// ─── IQ3_S ───────────────────────────────────────────────────────────

pub fn dequant_iq3_s(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if !n.is_multiple_of(QK_K) {
        bail!("IQ3_S: n={n} not divisible by {QK_K}");
    }
    let nb = n / QK_K;
    if bytes.len() != nb * IQ3S_BLOCK_BYTES {
        bail!(
            "IQ3_S: expected {} bytes, got {}",
            nb * IQ3S_BLOCK_BYTES,
            bytes.len()
        );
    }
    let mut out = vec![0f32; n];
    for i in 0..nb {
        let off = i * IQ3S_BLOCK_BYTES;
        let d = read_f16_le(&bytes[off..off + 2]);
        let qs_off = off + 2;
        let qs = &bytes[qs_off..qs_off + QK_K / 4]; // 64
        let qh = &bytes[qs_off + QK_K / 4..qs_off + QK_K / 4 + QK_K / 32]; // 8
        let signs = &bytes[qs_off + QK_K / 4 + QK_K / 32..qs_off + QK_K / 4 + QK_K / 32 + QK_K / 8]; // 32
        let scales = &bytes[qs_off + QK_K / 4 + QK_K / 32 + QK_K / 8..off + IQ3S_BLOCK_BYTES];
        let dst = &mut out[i * QK_K..(i + 1) * QK_K];
        let mut y = 0usize;
        let mut qs_walk = 0usize;
        let mut signs_walk = 0usize;
        let mut qh_walk = 0usize;
        for ib32 in (0..QK_K / 32).step_by(2) {
            let db1 = d * (1.0 + 2.0 * (scales[ib32 / 2] & 0xF) as f32);
            let db2 = d * (1.0 + 2.0 * (scales[ib32 / 2] >> 4) as f32);
            for l in 0..4 {
                let g1 = grid_u32_to_i8x4(
                    IQ3S_GRID[(qs[qs_walk + 2 * l] as usize)
                        | (((qh[qh_walk] as usize) << (8 - 2 * l)) & 256)],
                );
                let g2 = grid_u32_to_i8x4(
                    IQ3S_GRID[(qs[qs_walk + 2 * l + 1] as usize)
                        | (((qh[qh_walk] as usize) << (7 - 2 * l)) & 256)],
                );
                let sign = signs[signs_walk + l];
                for j in 0..4 {
                    let s0 = if sign & KMASK_IQ2XS[j] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    let s1 = if sign & KMASK_IQ2XS[j + 4] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    dst[y + j] = db1 * g1[j] as f32 * s0;
                    dst[y + j + 4] = db1 * g2[j] as f32 * s1;
                }
                y += 8;
            }
            qs_walk += 8;
            signs_walk += 4;
            for l in 0..4 {
                let g1 = grid_u32_to_i8x4(
                    IQ3S_GRID[(qs[qs_walk + 2 * l] as usize)
                        | (((qh[qh_walk + 1] as usize) << (8 - 2 * l)) & 256)],
                );
                let g2 = grid_u32_to_i8x4(
                    IQ3S_GRID[(qs[qs_walk + 2 * l + 1] as usize)
                        | (((qh[qh_walk + 1] as usize) << (7 - 2 * l)) & 256)],
                );
                let sign = signs[signs_walk + l];
                for j in 0..4 {
                    let s0 = if sign & KMASK_IQ2XS[j] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    let s1 = if sign & KMASK_IQ2XS[j + 4] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    dst[y + j] = db2 * g1[j] as f32 * s0;
                    dst[y + j + 4] = db2 * g2[j] as f32 * s1;
                }
                y += 8;
            }
            qs_walk += 8;
            signs_walk += 4;
            qh_walk += 2;
            let _ = ib32;
        }
    }
    Ok(out)
}

// ─── IQ1_S ───────────────────────────────────────────────────────────

pub fn dequant_iq1_s(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if !n.is_multiple_of(QK_K) {
        bail!("IQ1_S: n={n} not divisible by {QK_K}");
    }
    let nb = n / QK_K;
    if bytes.len() != nb * IQ1S_BLOCK_BYTES {
        bail!(
            "IQ1_S: expected {} bytes, got {}",
            nb * IQ1S_BLOCK_BYTES,
            bytes.len()
        );
    }
    let mut out = vec![0f32; n];
    for i in 0..nb {
        let off = i * IQ1S_BLOCK_BYTES;
        let d = read_f16_le(&bytes[off..off + 2]);
        let qs = &bytes[off + 2..off + 2 + QK_K / 8]; // 32
        let qh_bytes = &bytes[off + 2 + QK_K / 8..off + IQ1S_BLOCK_BYTES]; // 16
        let dst = &mut out[i * QK_K..(i + 1) * QK_K];
        let mut y = 0usize;
        let mut qs_idx = 0usize;
        for ib in 0..QK_K / 32 {
            let qh = read_u16_le(&qh_bytes[2 * ib..2 * ib + 2]);
            let dl = d * (2.0 * ((qh >> 12) & 7) as f32 + 1.0);
            let delta = if qh & 0x8000 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            };
            for l in 0..4 {
                let idx = (qs[qs_idx + l] as usize) | ((((qh >> (3 * l)) & 7) as usize) << 8);
                let grid = grid_u64_to_i8x8(IQ1S_GRID[idx]);
                for j in 0..8 {
                    dst[y + j] = dl * (grid[j] as f32 + delta);
                }
                y += 8;
            }
            qs_idx += 4;
        }
    }
    Ok(out)
}

// ─── IQ1_M ───────────────────────────────────────────────────────────

pub fn dequant_iq1_m(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if !n.is_multiple_of(QK_K) {
        bail!("IQ1_M: n={n} not divisible by {QK_K}");
    }
    let nb = n / QK_K;
    if bytes.len() != nb * IQ1M_BLOCK_BYTES {
        bail!(
            "IQ1_M: expected {} bytes, got {}",
            nb * IQ1M_BLOCK_BYTES,
            bytes.len()
        );
    }
    let mut out = vec![0f32; n];
    for i in 0..nb {
        let off = i * IQ1M_BLOCK_BYTES;
        let qs = &bytes[off..off + QK_K / 8]; // 32
        let qh = &bytes[off + QK_K / 8..off + QK_K / 8 + QK_K / 16]; // 16
        let scales_bytes = &bytes[off + QK_K / 8 + QK_K / 16..off + IQ1M_BLOCK_BYTES]; // 8
        let sc: [u16; 4] = [
            read_u16_le(&scales_bytes[0..2]),
            read_u16_le(&scales_bytes[2..4]),
            read_u16_le(&scales_bytes[4..6]),
            read_u16_le(&scales_bytes[6..8]),
        ];
        let scale_u16 =
            (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
        let d = half::f16::from_le_bytes(scale_u16.to_le_bytes()).to_f32();
        let dst = &mut out[i * QK_K..(i + 1) * QK_K];
        let mut y = 0usize;
        let mut qs_walk = 0usize;
        let mut qh_walk = 0usize;
        for ib in 0..QK_K / 32 {
            let dl1 = d * (2.0 * ((sc[ib / 2] >> (6 * (ib % 2))) & 0x7) as f32 + 1.0);
            let dl2 = d * (2.0 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) as f32 + 1.0);
            let idx0 = qs[qs_walk] as u16 | ((qh[qh_walk] as u16) << 8 & 0x700);
            let idx1 = qs[qs_walk + 1] as u16 | ((qh[qh_walk] as u16) << 4 & 0x700);
            let idx2 = qs[qs_walk + 2] as u16 | ((qh[qh_walk + 1] as u16) << 8 & 0x700);
            let idx3 = qs[qs_walk + 3] as u16 | ((qh[qh_walk + 1] as u16) << 4 & 0x700);
            let deltas = [
                if qh[qh_walk] & 0x08 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                },
                if qh[qh_walk] & 0x80 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                },
                if qh[qh_walk + 1] & 0x08 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                },
                if qh[qh_walk + 1] & 0x80 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                },
            ];
            let groups = [
                (idx0, deltas[0], dl1),
                (idx1, deltas[1], dl1),
                (idx2, deltas[2], dl2),
                (idx3, deltas[3], dl2),
            ];
            for (idx, delta, dl) in groups.iter() {
                let grid = grid_u64_to_i8x8(IQ1S_GRID[*idx as usize]);
                for j in 0..8 {
                    dst[y + j] = *dl * (grid[j] as f32 + *delta);
                }
                y += 8;
            }
            qs_walk += 4;
            qh_walk += 2;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iq4_nl_unity_recovers_kvalues() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        for i in 0..16u8 {
            bytes.push(i);
        }
        let out = dequant_iq4_nl(&bytes, QK4_NL).unwrap();
        // Low nibbles → first half: kvalues_iq4nl[i]. High nibble = 0 → second half: kvalues_iq4nl[0].
        for i in 0..16 {
            assert!((out[i] - KVALUES_IQ4NL[i] as f32).abs() < 1e-5);
            assert!((out[i + 16] - KVALUES_IQ4NL[0] as f32).abs() < 1e-5);
        }
    }

    #[test]
    fn iq4_xs_zero_block_is_zero() {
        let bytes = vec![0u8; IQ4XS_BLOCK_BYTES];
        let out = dequant_iq4_xs(&bytes, QK_K).unwrap();
        assert_eq!(out.len(), QK_K);
        for v in out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn iq2_xxs_zero_block_is_zero() {
        let bytes = vec![0u8; IQ2XXS_BLOCK_BYTES];
        let out = dequant_iq2_xxs(&bytes, QK_K).unwrap();
        for v in out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn iq2_xs_zero_block_is_zero() {
        let bytes = vec![0u8; IQ2XS_BLOCK_BYTES];
        let out = dequant_iq2_xs(&bytes, QK_K).unwrap();
        for v in out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn iq2_s_zero_block_is_zero() {
        let bytes = vec![0u8; IQ2S_BLOCK_BYTES];
        let out = dequant_iq2_s(&bytes, QK_K).unwrap();
        for v in out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn iq3_xxs_zero_block_is_zero() {
        let bytes = vec![0u8; IQ3XXS_BLOCK_BYTES];
        let out = dequant_iq3_xxs(&bytes, QK_K).unwrap();
        for v in out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn iq3_s_zero_block_is_zero() {
        let bytes = vec![0u8; IQ3S_BLOCK_BYTES];
        let out = dequant_iq3_s(&bytes, QK_K).unwrap();
        for v in out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn iq1_s_zero_block_is_zero() {
        let bytes = vec![0u8; IQ1S_BLOCK_BYTES];
        let out = dequant_iq1_s(&bytes, QK_K).unwrap();
        for v in out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn iq1_m_zero_block_is_finite() {
        let bytes = vec![0u8; IQ1M_BLOCK_BYTES];
        let out = dequant_iq1_m(&bytes, QK_K).unwrap();
        for v in out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn rejects_bad_byte_count() {
        assert!(dequant_iq4_nl(&[0u8; 10], QK4_NL).is_err());
        assert!(dequant_iq4_xs(&[0u8; 10], QK_K).is_err());
        assert!(dequant_iq2_xxs(&[0u8; 10], QK_K).is_err());
        assert!(dequant_iq1_s(&[0u8; 10], QK_K).is_err());
    }
}
