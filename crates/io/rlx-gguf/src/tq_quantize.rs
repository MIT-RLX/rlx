// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! TQ1_0 / TQ2_0 encoders — mirrors `quantize_row_tq1_0_ref` /
//! `quantize_row_tq2_0_ref` in llama.cpp's `ggml-quants.c`.

use anyhow::{Result, bail};

use crate::QK_K;

const TQ1_0_QS_LEN: usize = (QK_K - 4 * QK_K / 64) / 5;
const TQ1_0_QH_LEN: usize = QK_K / 64;
const TQ1_0_BLOCK_BYTES: usize = TQ1_0_QS_LEN + TQ1_0_QH_LEN + 2;
const TQ2_0_QS_LEN: usize = QK_K / 4;
const TQ2_0_BLOCK_BYTES: usize = TQ2_0_QS_LEN + 2;

#[inline]
fn f16_bytes(x: f32) -> [u8; 2] {
    half::f16::from_f32(x).to_le_bytes()
}

#[inline]
fn lround_f32(x: f32) -> i32 {
    if x >= 0.0 {
        (x + 0.5) as i32
    } else {
        (x - 0.5) as i32
    }
}

/// Quantize one TQ2_0 block (256 fp32 → 66 bytes).
pub fn quantize_tq2_0_block(src: &[f32], out: &mut [u8]) {
    assert!(src.len() >= QK_K && out.len() >= TQ2_0_BLOCK_BYTES);
    let mut amax = 0f32;
    for &v in &src[..QK_K] {
        amax = amax.max(v.abs());
    }
    let d = amax;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    out[TQ2_0_QS_LEN..].copy_from_slice(&f16_bytes(d));
    let mut x = 0usize;
    let mut j = 0usize;
    while j < TQ2_0_QS_LEN {
        for m in 0..32 {
            let mut q = 0u8;
            for n in 0..4 {
                let xi = (lround_f32(src[x + m + n * 32] * id) + 1).clamp(0, 2) as u8;
                q |= (xi & 3) << (2 * n);
            }
            out[j + m] = q;
        }
        x += 4 * 32;
        j += 32;
    }
}

/// Quantize one TQ1_0 block (256 fp32 → 54 bytes).
pub fn quantize_tq1_0_block(src: &[f32], out: &mut [u8]) {
    assert!(src.len() >= QK_K && out.len() >= TQ1_0_BLOCK_BYTES);
    let mut amax = 0f32;
    for &v in &src[..QK_K] {
        amax = amax.max(v.abs());
    }
    let d = amax;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    let mut x = 0usize;
    for m in 0..32 {
        let mut q: u32 = 0;
        for n in 0..5 {
            let xi = (lround_f32(src[x + m + n * 32] * id) + 1) as u32;
            q *= 3;
            q += xi;
        }
        q = (q * 256).div_ceil(243);
        out[m] = q as u8;
    }
    x += 5 * 32;
    for m in 0..16 {
        let mut q: u32 = 0;
        for n in 0..5 {
            let xi = (lround_f32(src[x + m + n * 16] * id) + 1) as u32;
            q *= 3;
            q += xi;
        }
        q = (q * 256).div_ceil(243);
        out[32 + m] = q as u8;
    }
    x += 5 * 16;
    for j in 0..TQ1_0_QH_LEN {
        let mut q: u32 = 0;
        for m in 0..4 {
            let xi = (lround_f32(src[x + j + m * TQ1_0_QH_LEN] * id) + 1) as u32;
            q *= 3;
            q += xi;
        }
        q *= 3;
        q = (q * 256).div_ceil(243);
        out[TQ1_0_QS_LEN + j] = q as u8;
    }
    out[TQ1_0_QS_LEN + TQ1_0_QH_LEN..].copy_from_slice(&f16_bytes(d));
}

pub fn quantize_tq1_0(src: &[f32]) -> Result<Vec<u8>> {
    if !src.len().is_multiple_of(QK_K) {
        bail!("TQ1_0: n={} not divisible by {QK_K}", src.len());
    }
    let nb = src.len() / QK_K;
    let mut out = vec![0u8; nb * TQ1_0_BLOCK_BYTES];
    for i in 0..nb {
        quantize_tq1_0_block(
            &src[i * QK_K..(i + 1) * QK_K],
            &mut out[i * TQ1_0_BLOCK_BYTES..(i + 1) * TQ1_0_BLOCK_BYTES],
        );
    }
    Ok(out)
}

pub fn quantize_tq2_0(src: &[f32]) -> Result<Vec<u8>> {
    if !src.len().is_multiple_of(QK_K) {
        bail!("TQ2_0: n={} not divisible by {QK_K}", src.len());
    }
    let nb = src.len() / QK_K;
    let mut out = vec![0u8; nb * TQ2_0_BLOCK_BYTES];
    for i in 0..nb {
        quantize_tq2_0_block(
            &src[i * QK_K..(i + 1) * QK_K],
            &mut out[i * TQ2_0_BLOCK_BYTES..(i + 1) * TQ2_0_BLOCK_BYTES],
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tq_dequant::{dequant_tq1_0, dequant_tq2_0};

    #[test]
    fn tq2_roundtrip() {
        let x: Vec<f32> = (0..256)
            .map(|i| match i % 3 {
                0 => -0.5,
                1 => 0.0,
                _ => 0.5,
            })
            .collect();
        let q = quantize_tq2_0(&x).unwrap();
        let out = dequant_tq2_0(&q, 256).unwrap();
        for i in 0..256 {
            assert!((out[i] - x[i]).abs() < 0.05, "i={i}");
        }
    }

    #[test]
    fn tq1_roundtrip_alternating() {
        let x: Vec<f32> = (0..256)
            .map(|i| match i % 3 {
                0 => -0.5,
                1 => 0.0,
                _ => 0.5,
            })
            .collect();
        let q = quantize_tq1_0(&x).unwrap();
        let out = dequant_tq1_0(&q, 256).unwrap();
        for i in 0..256 {
            assert!((out[i] - x[i]).abs() < 0.05, "i={i}");
        }
    }
}
