// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! OCP microscaling FP4 formats `MXFP4` and `NVFP4` (sometimes named
//! `MXFP4_NV` in llama.cpp's `ggml-quants.c`).
//!
//! Both formats encode weights as 4-bit E2M1 codes (LUT below) with block
//! scales, laid out exactly as ggml's `block_mxfp4` / `block_nvfp4`:
//!
//! | format | elems/blk | scales                    | block bytes | nibbles |
//! |--------|-----------|---------------------------|-------------|---------|
//! | MXFP4  | 32        | 1 × E8M0, first           | 17          | byte `j` = elements `j` (low), `j + 16` (high) |
//! | NVFP4  | 64        | 4 × UE4M3 (one per 16), first | 36      | sub-block `s`, byte `8s + j` = elements `16s + j` (low), `16s + j + 8` (high) |
//!
//! Values: MXFP4 `e2m1 · 2^(e − 127)`; NVFP4 `e2m1 · ue4m3(d[s])`. (ggml
//! writes the same products as a doubled E2M1 table times a halved scale.)
//!
//! The OCP E2M1 LUT, the E4M3 decoder, and the constant `NVFP4_GROUP_SIZE`
//! already live in `rlx-ir::nvfp4`, but `rlx-gguf` must stay standalone
//! (no `rlx-*` deps) so the small constants are duplicated here. They
//! match `rlx-ir`'s implementation byte-for-byte.

use anyhow::{Result, bail};

/// Elements per MXFP4 block.
pub const QK_MXFP4: usize = 32;
/// Elements per NVFP4 block (ggml `QK_NVFP4`).
pub const QK_NVFP4: usize = 64;
/// Elements per NVFP4 scale group (ggml `QK_NVFP4_SUB`).
pub const QK_NVFP4_SUB: usize = 16;

pub(crate) const MXFP4_BLOCK_BYTES: usize = 1 + QK_MXFP4 / 2; // 1 + 16 = 17
pub(crate) const NVFP4_BLOCK_BYTES: usize = QK_NVFP4 / QK_NVFP4_SUB + QK_NVFP4 / 2; // 4 + 32 = 36

/// OCP E2M1 FP4 decode LUT (sign in the high bit; magnitudes
/// 0, 0.5, 1, 1.5, 2, 3, 4, 6).
const FP4_E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

#[inline]
fn fp4(nibble: u8) -> f32 {
    FP4_E2M1[(nibble & 0x0F) as usize]
}

/// Decode an E8M0 scale byte: purely 8-bit unsigned exponent, value =
/// `2^(byte − 127)`. Byte `0xFF` is reserved as NaN; we return 0 for
/// matmul stability.
#[inline]
pub fn e8m0_scale_to_f32(byte: u8) -> f32 {
    if byte == 0xFF {
        return 0.0;
    }
    // 2^(byte - 127). `byte` ∈ [0, 254].
    let exp = byte as i32 - 127;
    if exp >= 128 {
        f32::INFINITY
    } else if exp < -126 {
        2f32.powi(exp.max(-149))
    } else {
        2f32.powi(exp)
    }
}

/// Decode an OCP E4M3 (FP8) scale byte. Matches `rlx_ir::nvfp4`.
#[inline]
pub fn e4m3_scale_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = (byte >> 3) & 0x0F;
    let mant = byte & 0x07;
    let v = if exp == 0 {
        if mant == 0 {
            0.0
        } else {
            (mant as f32 / 8.0) * 2f32.powi(-6)
        }
    } else if exp == 0x0F && mant == 0x07 {
        0.0 // NaN → 0
    } else {
        (1.0 + mant as f32 / 8.0) * 2f32.powi(exp as i32 - 7)
    };
    sign * v
}

/// Decode an unsigned E4M3 NVFP4 scale byte as ggml does (`ggml_ue4m3_to_fp32`,
/// without its ×½ — that belongs to ggml's doubled E2M1 table): `0` and
/// `0x7F` decode to 0, the sign bit is ignored.
#[inline]
pub fn ue4m3_scale_to_f32(byte: u8) -> f32 {
    if byte == 0 || byte == 0x7F {
        return 0.0;
    }
    let exp = (byte >> 3) & 0x0F;
    let man = (byte & 0x07) as f32;
    if exp == 0 {
        man * 2f32.powi(-9)
    } else {
        (1.0 + man / 8.0) * 2f32.powi(exp as i32 - 7)
    }
}

pub fn mxfp4_bytes(n: usize) -> Option<usize> {
    if !n.is_multiple_of(QK_MXFP4) {
        return None;
    }
    Some((n / QK_MXFP4) * MXFP4_BLOCK_BYTES)
}

pub fn nvfp4_bytes(n: usize) -> Option<usize> {
    if !n.is_multiple_of(QK_NVFP4) {
        return None;
    }
    Some((n / QK_NVFP4) * NVFP4_BLOCK_BYTES)
}

/// Dequantize one MXFP4 block into `out` (ggml `dequantize_row_mxfp4`).
pub fn dequant_mxfp4_block(block: &[u8], out: &mut [f32; QK_MXFP4]) {
    let s = e8m0_scale_to_f32(block[0]);
    let qs = &block[1..1 + QK_MXFP4 / 2];
    for (j, &b) in qs.iter().enumerate() {
        out[j] = s * fp4(b & 0x0F);
        out[j + QK_MXFP4 / 2] = s * fp4(b >> 4);
    }
}

/// Dequantize one NVFP4 block into `out` (ggml `dequantize_row_nvfp4`).
pub fn dequant_nvfp4_block(block: &[u8], out: &mut [f32; QK_NVFP4]) {
    const HALF: usize = QK_NVFP4_SUB / 2;
    let (d, qs) = block.split_at(QK_NVFP4 / QK_NVFP4_SUB);
    for (s, sub) in out.as_chunks_mut::<QK_NVFP4_SUB>().0.iter_mut().enumerate() {
        let scale = ue4m3_scale_to_f32(d[s]);
        for (j, &b) in qs[s * HALF..(s + 1) * HALF].iter().enumerate() {
            sub[j] = scale * fp4(b & 0x0F);
            sub[j + HALF] = scale * fp4(b >> 4);
        }
    }
}

pub fn dequant_mxfp4(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if !n.is_multiple_of(QK_MXFP4) {
        bail!("MXFP4: n={n} not divisible by {QK_MXFP4}");
    }
    let nb = n / QK_MXFP4;
    if bytes.len() != nb * MXFP4_BLOCK_BYTES {
        bail!(
            "MXFP4: expected {} bytes, got {}",
            nb * MXFP4_BLOCK_BYTES,
            bytes.len()
        );
    }
    let mut out = vec![0f32; n];
    for i in 0..nb {
        let off = i * MXFP4_BLOCK_BYTES;
        dequant_mxfp4_block(
            &bytes[off..off + MXFP4_BLOCK_BYTES],
            (&mut out[i * QK_MXFP4..(i + 1) * QK_MXFP4])
                .try_into()
                .unwrap(),
        );
    }
    Ok(out)
}

pub fn dequant_nvfp4(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if !n.is_multiple_of(QK_NVFP4) {
        bail!("NVFP4: n={n} not divisible by {QK_NVFP4}");
    }
    let nb = n / QK_NVFP4;
    if bytes.len() != nb * NVFP4_BLOCK_BYTES {
        bail!(
            "NVFP4: expected {} bytes, got {}",
            nb * NVFP4_BLOCK_BYTES,
            bytes.len()
        );
    }
    let mut out = vec![0f32; n];
    for i in 0..nb {
        let off = i * NVFP4_BLOCK_BYTES;
        dequant_nvfp4_block(
            &bytes[off..off + NVFP4_BLOCK_BYTES],
            (&mut out[i * QK_NVFP4..(i + 1) * QK_NVFP4])
                .try_into()
                .unwrap(),
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e8m0_one_is_unity() {
        // Byte 127 → exponent 0 → 2^0 = 1.
        assert_eq!(e8m0_scale_to_f32(127), 1.0);
        assert_eq!(e8m0_scale_to_f32(128), 2.0);
        assert_eq!(e8m0_scale_to_f32(126), 0.5);
    }

    /// ggml's `dequantize_row_mxfp4` / `dequantize_row_nvfp4`, transcribed with
    /// its doubled `kvalues_mxfp4` table and halved scales.
    const KVALUES: [f32; 16] = [
        0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0,
    ];

    fn ggml_ue4m3_half(x: u8) -> f32 {
        if x == 0 || x == 0x7F {
            return 0.0;
        }
        let (exp, man) = (((x >> 3) & 0xF) as i32, (x & 7) as f32);
        let raw = if exp == 0 {
            man * 2f32.powi(-9)
        } else {
            (1.0 + man / 8.0) * 2f32.powi(exp - 7)
        };
        raw * 0.5
    }

    fn ggml_mxfp4(src: &[u8]) -> Vec<f32> {
        let mut y = Vec::new();
        for b in src.chunks(17) {
            // GGML_E8M0_TO_FP32_HALF
            let d = 2f32.powi(b[0] as i32 - 128);
            let mut blk = [0f32; 32];
            for j in 0..16 {
                blk[j] = KVALUES[(b[1 + j] & 0x0F) as usize] * d;
                blk[j + 16] = KVALUES[(b[1 + j] >> 4) as usize] * d;
            }
            y.extend(blk);
        }
        y
    }

    fn ggml_nvfp4(src: &[u8]) -> Vec<f32> {
        let mut y = Vec::new();
        for b in src.chunks(36) {
            for s in 0..4 {
                let d = ggml_ue4m3_half(b[s]);
                let qs = &b[4 + 8 * s..4 + 8 * s + 8];
                let mut sub = [0f32; 16];
                for j in 0..8 {
                    sub[j] = KVALUES[(qs[j] & 0x0F) as usize] * d;
                    sub[j + 8] = KVALUES[(qs[j] >> 4) as usize] * d;
                }
                y.extend(sub);
            }
        }
        y
    }

    /// Bytes covering every nibble at every position and scales across the
    /// normal and subnormal range (sign bit clear: ggml ignores it).
    fn pattern(blocks: usize, block: usize) -> Vec<u8> {
        (0..blocks * block)
            .map(|i| ((i * 37 + i / block * 11) % 256) as u8)
            .collect()
    }

    #[test]
    fn mxfp4_matches_ggml() {
        let mut src = pattern(40, 17);
        for b in src.chunks_mut(17) {
            b[0] = 100 + (b[0] % 50); // finite E8M0 scales
        }
        assert_eq!(dequant_mxfp4(&src, 40 * 32).unwrap(), ggml_mxfp4(&src));
    }

    #[test]
    fn nvfp4_matches_ggml() {
        let mut src = pattern(40, 36);
        for b in src.chunks_mut(36) {
            for s in &mut b[..4] {
                *s &= 0x7F;
            }
        }
        assert_eq!(dequant_nvfp4(&src, 40 * 64).unwrap(), ggml_nvfp4(&src));
    }

    #[test]
    fn nvfp4_zero_scale_zeroes_out() {
        let block = vec![0u8; NVFP4_BLOCK_BYTES];
        let out = dequant_nvfp4(&block, QK_NVFP4).unwrap();
        assert!(out.iter().all(|&v| v == 0.0));
    }
}
