// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! MXFP4 / NVFP4 encoders — mirrors `quantize_row_mxfp4_ref` /
//! `quantize_row_nvfp4_ref` in llama.cpp's `ggml-quants.c`.

use anyhow::{Result, bail};

use crate::mx_dequant::{
    MXFP4_BLOCK_BYTES, NVFP4_BLOCK_BYTES, QK_MXFP4, QK_NVFP4, QK_NVFP4_SUB, e8m0_scale_to_f32,
    ue4m3_scale_to_f32,
};

const FP4_E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

#[inline]
fn fp4(nibble: u8) -> f32 {
    FP4_E2M1[(nibble & 0x0F) as usize]
}

fn best_index_fp4(x: f32, scale: f32) -> u8 {
    let mut best = 0u8;
    let mut best_err = (fp4(0) * scale - x).abs();
    for i in 1..16u8 {
        let err = (fp4(i) * scale - x).abs();
        if err < best_err {
            best = i;
            best_err = err;
        }
    }
    best
}

/// E8M0 scale byte: `2^(byte - 127)` (matches [`e8m0_scale_to_f32`]).
fn e8m0_from_amax(amax: f32) -> u8 {
    if amax <= 0.0 {
        return 0;
    }
    (amax.log2().floor() as i32 - 2 + 127).clamp(0, 254) as u8
}

/// Unsigned E4M3 scale byte for `x` — ggml's `ggml_fp32_to_ue4m3`, rounding
/// and saturation included.
fn fp32_to_ue4m3(x: f32) -> u8 {
    // Also catches NaN, like ggml's `!(x > 0.0f)`.
    if x.is_nan() || x <= 0.0 {
        return 0;
    }
    let x = x.min(448.0);
    let bits = x.to_bits();
    let fp32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let fp32_man = ((bits >> 20) & 0x7) as i32;
    let mut exp = fp32_exp + 7;
    if exp <= 0 {
        // Subnormal: value = man · 2^-9.
        let man = ((x * 512.0 + 0.5) as i32).min(7);
        return if man < 1 { 0 } else { man as u8 };
    }
    if exp >= 15 {
        return 0x7E;
    }
    let mut man = fp32_man + ((bits >> 19) & 1) as i32;
    if man > 7 {
        man = 0;
        exp += 1;
        if exp >= 15 {
            return 0x7E;
        }
    }
    ((exp << 3) | man) as u8
}

/// Quantize one MXFP4 block (32 fp32 → 17 bytes).
pub fn quantize_mxfp4_block(src: &[f32], out: &mut [u8]) {
    assert!(src.len() >= QK_MXFP4 && out.len() >= MXFP4_BLOCK_BYTES);
    let mut amax = 0f32;
    for &v in &src[..QK_MXFP4] {
        amax = amax.max(v.abs());
    }
    let e = e8m0_from_amax(amax);
    let d = e8m0_scale_to_f32(e);
    out[0] = e;
    for j in 0..QK_MXFP4 / 2 {
        let x0 = best_index_fp4(src[j], d);
        let x1 = best_index_fp4(src[j + QK_MXFP4 / 2], d);
        out[1 + j] = x0 | (x1 << 4);
    }
}

/// Quantize one NVFP4 block (64 fp32 → 36 bytes), ggml layout: four UE4M3
/// scales (one per 16 values), then per scale group `s` byte `8s + j` holds
/// values `16s + j` (low nibble) and `16s + j + 8` (high).
pub fn quantize_nvfp4_block(src: &[f32], out: &mut [u8]) {
    const HALF: usize = QK_NVFP4_SUB / 2;
    const N_SUB: usize = QK_NVFP4 / QK_NVFP4_SUB;
    assert!(src.len() >= QK_NVFP4 && out.len() >= NVFP4_BLOCK_BYTES);
    for s in 0..N_SUB {
        let xb = &src[s * QK_NVFP4_SUB..(s + 1) * QK_NVFP4_SUB];
        let amax = xb.iter().fold(0f32, |m, v| m.max(v.abs()));
        let ue = fp32_to_ue4m3(amax / 6.0);
        out[s] = ue;
        let d = ue4m3_scale_to_f32(ue);
        for j in 0..HALF {
            let x0 = best_index_fp4(xb[j], d);
            let x1 = best_index_fp4(xb[j + HALF], d);
            out[N_SUB + s * HALF + j] = x0 | (x1 << 4);
        }
    }
}

pub fn quantize_mxfp4(src: &[f32]) -> Result<Vec<u8>> {
    if !src.len().is_multiple_of(QK_MXFP4) {
        bail!("MXFP4: n={} not divisible by {QK_MXFP4}", src.len());
    }
    let nb = src.len() / QK_MXFP4;
    let mut out = vec![0u8; nb * MXFP4_BLOCK_BYTES];
    for i in 0..nb {
        quantize_mxfp4_block(
            &src[i * QK_MXFP4..(i + 1) * QK_MXFP4],
            &mut out[i * MXFP4_BLOCK_BYTES..(i + 1) * MXFP4_BLOCK_BYTES],
        );
    }
    Ok(out)
}

pub fn quantize_nvfp4(src: &[f32]) -> Result<Vec<u8>> {
    if !src.len().is_multiple_of(QK_NVFP4) {
        bail!("NVFP4: n={} not divisible by {QK_NVFP4}", src.len());
    }
    let nb = src.len() / QK_NVFP4;
    let mut out = vec![0u8; nb * NVFP4_BLOCK_BYTES];
    for i in 0..nb {
        quantize_nvfp4_block(
            &src[i * QK_NVFP4..(i + 1) * QK_NVFP4],
            &mut out[i * NVFP4_BLOCK_BYTES..(i + 1) * NVFP4_BLOCK_BYTES],
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mx_dequant::{dequant_mxfp4, dequant_nvfp4};

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 {
            1.0
        } else {
            dot / (na * nb)
        }
    }

    #[test]
    fn mxfp4_roundtrip() {
        let x: Vec<f32> = (0..256).map(|i| (i as f32 * 0.013).sin()).collect();
        let q = quantize_mxfp4(&x).unwrap();
        let out = dequant_mxfp4(&q, x.len()).unwrap();
        assert!(cosine(&x, &out) > 0.95, "cos={}", cosine(&x, &out));
    }

    #[test]
    fn nvfp4_roundtrip() {
        let x: Vec<f32> = (0..128).map(|i| (i as f32 * 0.017).cos()).collect();
        let q = quantize_nvfp4(&x).unwrap();
        let out = dequant_nvfp4(&q, x.len()).unwrap();
        assert!(cosine(&x, &out) > 0.95, "cos={}", cosine(&x, &out));
    }
}
