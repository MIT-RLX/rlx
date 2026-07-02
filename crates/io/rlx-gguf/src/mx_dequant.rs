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

//! OCP microscaling FP4 formats `MXFP4` and `NVFP4` (sometimes named
//! `MXFP4_NV` in llama.cpp's `ggml-quants.c`).
//!
//! Both formats encode weights as 4-bit E2M1 codes (LUT below) plus one
//! shared block scale. The two differ in block size and scale dtype:
//!
//! | format | elems/blk | scale       | block bytes |
//! |--------|-----------|-------------|-------------|
//! | MXFP4  | 32        | E8M0 (1 B)  | 17          |
//! | NVFP4  | 16        | E4M3 (1 B)  | 9           |
//!
//! The OCP E2M1 LUT, the E4M3 decoder, and the constant `NVFP4_GROUP_SIZE`
//! already live in `rlx-ir::nvfp4`, but `rlx-gguf` must stay standalone
//! (no `rlx-*` deps) so the small constants are duplicated here. They
//! match `rlx-ir`'s implementation byte-for-byte.

use anyhow::{Result, bail};

/// Elements per MXFP4 block.
pub const QK_MXFP4: usize = 32;
/// Elements per NVFP4 block (the on-disk GGUF layout, **not** the
/// MLX/Blackwell 16-along-K group used for matmul).
pub const QK_NVFP4: usize = 16;

const MXFP4_BLOCK_BYTES: usize = 1 + QK_MXFP4 / 2; // 1 + 16 = 17
const NVFP4_BLOCK_BYTES: usize = 1 + QK_NVFP4 / 2; // 1 + 8 = 9

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

/// Dequantize one MXFP4 block into `out`.
pub fn dequant_mxfp4_block(block: &[u8], out: &mut [f32; QK_MXFP4]) {
    let s = e8m0_scale_to_f32(block[0]);
    let qs = &block[1..1 + QK_MXFP4 / 2];
    for (i, &b) in qs.iter().enumerate() {
        out[2 * i] = s * fp4(b & 0x0F);
        out[2 * i + 1] = s * fp4(b >> 4);
    }
}

/// Dequantize one NVFP4 block into `out`.
pub fn dequant_nvfp4_block(block: &[u8], out: &mut [f32; QK_NVFP4]) {
    let s = e4m3_scale_to_f32(block[0]);
    let qs = &block[1..1 + QK_NVFP4 / 2];
    for (i, &b) in qs.iter().enumerate() {
        out[2 * i] = s * fp4(b & 0x0F);
        out[2 * i + 1] = s * fp4(b >> 4);
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

    #[test]
    fn mxfp4_unity_scale_recovers_lut() {
        let mut block = vec![127u8]; // E8M0 = 1.0
        // First two nibbles 0x02 (=1.0) and 0x05 (=3.0) → byte = 0x52
        let mut bytes = vec![0u8; QK_MXFP4 / 2];
        for i in 0..QK_MXFP4 / 2 {
            // alternate: low=2 (1.0), high=14 (-4.0)
            bytes[i] = 0xE2;
        }
        block.extend_from_slice(&bytes);
        let out = dequant_mxfp4(&block, QK_MXFP4).unwrap();
        for i in 0..QK_MXFP4 / 2 {
            assert_eq!(out[2 * i], 1.0);
            assert_eq!(out[2 * i + 1], -4.0);
        }
    }

    #[test]
    fn nvfp4_zero_scale_zeroes_out() {
        let block = vec![0u8; NVFP4_BLOCK_BYTES];
        let out = dequant_nvfp4(&block, QK_NVFP4).unwrap();
        for v in out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn nvfp4_unity_scale() {
        // E4M3 unity = 0x38.
        let mut block = vec![0x38u8];
        block.extend(std::iter::repeat_n(0x02u8, QK_NVFP4 / 2)); // nibbles 2, 0 → 1.0, 0.0
        let out = dequant_nvfp4(&block, QK_NVFP4).unwrap();
        for i in 0..QK_NVFP4 / 2 {
            assert_eq!(out[2 * i], 1.0);
            assert_eq!(out[2 * i + 1], 0.0);
        }
    }
}
