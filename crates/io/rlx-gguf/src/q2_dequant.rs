// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Custom 2-bit GGUF format `Q2_0` (a.k.a. `Q2_0_g128`), ggml type 42.
//!
//! Introduced by the PrismML llama.cpp fork for Ternary Bonsai
//! (`prism-ml/Ternary-Bonsai-*-gguf`). Every weight is a 2-bit code
//! `q ∈ {0,1,2,3}` mapped to `(q − 1) · d`, with one f16 scale `d`
//! shared across a group of 128 weights — 2.125 bits/weight deployed.
//! Ternary models only use `{−1,0,+1}` (codes 0/1/2); code 3 (`+2d`) is
//! reserved for future 2-bit content.
//!
//! Block layout (byte-for-byte with `block_q2_0` in the fork's
//! `ggml-common.h`):
//!
//! ```text
//!   d   (f16, 2 bytes)           # group scale
//!   qs  (32 bytes = 128×2 bits)  # LSB-first 2-bit codes within each byte
//! ```
//!
//! = 34 bytes / 128 elements. Dequant mirrors `dequantize_row_q2_0`.

use crate::read_f16_le;
use anyhow::{Result, bail};

/// Group size for the `Q2_0` format (weights sharing one f16 scale).
pub const QK2_0: usize = 128;
/// Bytes per `Q2_0` block: f16 scale + 128×2-bit codes.
pub const Q2_0_BLOCK_BYTES: usize = 2 + QK2_0 / 4; // 34

/// Storage bytes for `n` `Q2_0` elements. `None` if `n` isn't a
/// multiple of the 128-element group.
pub fn q2_0_bytes(n: usize) -> Option<usize> {
    if !n.is_multiple_of(QK2_0) {
        return None;
    }
    Some((n / QK2_0) * Q2_0_BLOCK_BYTES)
}

/// Dequantize one `Q2_0` block (34 bytes) into `out` (128 f32 values).
///
/// Layout: `d` (f16) then `qs[32]` 2-bit codes, LSB-first within each
/// byte (`(qs[j / 4] >> ((j % 4) * 2)) & 3`). Value = `(q − 1) * d`.
pub fn dequant_q2_0_block(block: &[u8], out: &mut [f32; QK2_0]) {
    let d = read_f16_le(&block[0..2]);
    let qs = &block[2..2 + QK2_0 / 4];
    for (j, slot) in out.iter_mut().enumerate() {
        let q = (qs[j / 4] >> ((j % 4) * 2)) & 0x03;
        *slot = (q as i32 - 1) as f32 * d;
    }
}

/// Dequantize a full `Q2_0` tensor of `n` elements to f32.
pub fn dequant_q2_0(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if !n.is_multiple_of(QK2_0) {
        bail!("Q2_0: n={n} not divisible by {QK2_0}");
    }
    let nb = n / QK2_0;
    if bytes.len() != nb * Q2_0_BLOCK_BYTES {
        bail!(
            "Q2_0: expected {} bytes, got {}",
            nb * Q2_0_BLOCK_BYTES,
            bytes.len()
        );
    }
    let mut out = vec![0f32; n];
    for i in 0..nb {
        let off = i * Q2_0_BLOCK_BYTES;
        dequant_q2_0_block(
            &bytes[off..off + Q2_0_BLOCK_BYTES],
            (&mut out[i * QK2_0..(i + 1) * QK2_0]).try_into().unwrap(),
        );
    }
    Ok(out)
}

/// Quantize `n` f32 values to `Q2_0` (128-element groups). Scale `d` is
/// the max absolute value; each weight becomes
/// `clamp(round(w/d) + 1, 0..=3)`. Lossy — used for tests / transcode.
pub fn quantize_q2_0(src: &[f32]) -> Result<Vec<u8>> {
    let n = src.len();
    if !n.is_multiple_of(QK2_0) {
        bail!("Q2_0: n={n} not divisible by {QK2_0}");
    }
    let nb = n / QK2_0;
    let mut out = vec![0u8; nb * Q2_0_BLOCK_BYTES];
    for i in 0..nb {
        let group = &src[i * QK2_0..(i + 1) * QK2_0];
        let amax = group.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let id = if amax > 0.0 { 1.0 / amax } else { 0.0 };
        let off = i * Q2_0_BLOCK_BYTES;
        out[off..off + 2].copy_from_slice(&half::f16::from_f32(amax).to_le_bytes());
        for (j, &v) in group.iter().enumerate() {
            let mut q = (v * id).round() as i32 + 1;
            q = q.clamp(0, 3);
            out[off + 2 + j / 4] |= (q as u8) << ((j % 4) * 2);
        }
    }
    Ok(out)
}

/// Dequantize only the given rows of a packed `Q2_0` matrix stored
/// row-major as `[n_rows, row_len]` (`row_len` a multiple of 128).
/// Returns `[indices.len(), row_len]` f32.
pub fn gather_rows_q2_0(bytes: &[u8], row_len: usize, indices: &[usize]) -> Result<Vec<f32>> {
    if !row_len.is_multiple_of(QK2_0) {
        bail!("Q2_0 gather: row_len={row_len} not a multiple of {QK2_0}");
    }
    let blocks_per_row = row_len / QK2_0;
    let row_bytes = blocks_per_row * Q2_0_BLOCK_BYTES;
    let mut out = vec![0f32; indices.len() * row_len];
    for (i, &r) in indices.iter().enumerate() {
        let off = r * row_bytes;
        let end = off + row_bytes;
        if end > bytes.len() {
            bail!("Q2_0 gather: row {r} past packed length {}", bytes.len());
        }
        let row_out = &mut out[i * row_len..(i + 1) * row_len];
        for b in 0..blocks_per_row {
            let boff = off + b * Q2_0_BLOCK_BYTES;
            dequant_q2_0_block(
                &bytes[boff..boff + Q2_0_BLOCK_BYTES],
                (&mut row_out[b * QK2_0..(b + 1) * QK2_0])
                    .try_into()
                    .unwrap(),
            );
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack_q2_0_block(d: f32, codes: &[u8; QK2_0]) -> Vec<u8> {
        let mut bytes = vec![0u8; Q2_0_BLOCK_BYTES];
        bytes[0..2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
        for (j, &q) in codes.iter().enumerate() {
            bytes[2 + j / 4] |= (q & 0x03) << ((j % 4) * 2);
        }
        bytes
    }

    #[test]
    fn block_bytes_and_roundtrip() {
        assert_eq!(Q2_0_BLOCK_BYTES, 34);
        let mut codes = [1u8; QK2_0]; // code 1 → 0
        codes[0] = 0; // -d
        codes[1] = 1; // 0
        codes[2] = 2; // +d
        codes[3] = 3; // +2d
        let block = pack_q2_0_block(0.5, &codes);
        let mut out = [0f32; QK2_0];
        dequant_q2_0_block(&block, &mut out);
        assert!((out[0] + 0.5).abs() < 1e-4);
        assert!(out[1].abs() < 1e-4);
        assert!((out[2] - 0.5).abs() < 1e-4);
        assert!((out[3] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn quantize_dequant_ternary() {
        let mut src = vec![0f32; 256];
        for (i, v) in src.iter_mut().enumerate() {
            *v = match i % 3 {
                0 => -0.25,
                1 => 0.0,
                _ => 0.25,
            };
        }
        let packed = quantize_q2_0(&src).unwrap();
        let out = dequant_q2_0(&packed, src.len()).unwrap();
        for (a, b) in src.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn gather_rows() {
        let src: Vec<f32> = (0..6 * 256).map(|i| ((i % 3) - 1) as f32 * 0.1).collect();
        let packed = quantize_q2_0(&src).unwrap();
        let got = gather_rows_q2_0(&packed, 256, &[1, 4]).unwrap();
        assert_eq!(got.len(), 2 * 256);
        let full = dequant_q2_0(&packed, src.len()).unwrap();
        assert_eq!(&got[..256], &full[256..512]);
        assert_eq!(&got[256..], &full[4 * 256..5 * 256]);
    }
}
