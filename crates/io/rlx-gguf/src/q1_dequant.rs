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

//! Custom 1-bit GGUF format `Q1_0` (a.k.a. `Q1_0_g128`), ggml type 41.
//!
//! Introduced by the PrismML llama.cpp fork for `prism-ml/Bonsai-27B`
//! (a Qwen3.6-27B derivative). Every weight is a single sign bit that
//! selects `±d`, with one f16 scale `d` shared across a group of 128
//! weights — 1 sign bit + an amortized 16-bit scale = 1.125 bits/weight.
//!
//! Block layout (byte-for-byte with `block_q1_0` in the fork's
//! `ggml-common.h`):
//!
//! ```text
//!   d   (f16, 2 bytes)          # group scale
//!   qs  (16 bytes = 128 bits)   # one sign bit per weight, LSB-first
//! ```
//!
//! = 18 bytes / 128 elements. Dequant mirrors `dequantize_row_q1_0`:
//! bit `1` → `+d`, bit `0` → `−d`.

use crate::read_f16_le;
use anyhow::{Result, bail};

/// Group size for the `Q1_0` format (weights sharing one f16 scale).
pub const QK1_0: usize = 128;
/// Bytes per `Q1_0` block: f16 scale + 128 sign bits.
pub const Q1_0_BLOCK_BYTES: usize = 2 + QK1_0 / 8; // 18

/// Storage bytes for `n` `Q1_0` elements. `None` if `n` isn't a
/// multiple of the 128-element group.
pub fn q1_0_bytes(n: usize) -> Option<usize> {
    if !n.is_multiple_of(QK1_0) {
        return None;
    }
    Some((n / QK1_0) * Q1_0_BLOCK_BYTES)
}

/// Dequantize one `Q1_0` block (18 bytes) into `out` (128 f32 values).
///
/// Layout: `d` (f16) then `qs[16]` sign bits, LSB-first within each byte
/// (`qs[j / 8] >> (j % 8) & 1`). Bit `1` maps to `+d`, `0` to `−d`.
pub fn dequant_q1_0_block(block: &[u8], out: &mut [f32; QK1_0]) {
    let d = read_f16_le(&block[0..2]);
    let neg_d = -d;
    let qs = &block[2..2 + QK1_0 / 8];
    for (j, slot) in out.iter_mut().enumerate() {
        let bit = (qs[j / 8] >> (j % 8)) & 1;
        *slot = if bit != 0 { d } else { neg_d };
    }
}

/// Dequantize a full `Q1_0` tensor of `n` elements to f32.
pub fn dequant_q1_0(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if !n.is_multiple_of(QK1_0) {
        bail!("Q1_0: n={n} not divisible by {QK1_0}");
    }
    let nb = n / QK1_0;
    if bytes.len() != nb * Q1_0_BLOCK_BYTES {
        bail!(
            "Q1_0: expected {} bytes, got {}",
            nb * Q1_0_BLOCK_BYTES,
            bytes.len()
        );
    }
    let mut out = vec![0f32; n];
    for i in 0..nb {
        let off = i * Q1_0_BLOCK_BYTES;
        dequant_q1_0_block(
            &bytes[off..off + Q1_0_BLOCK_BYTES],
            (&mut out[i * QK1_0..(i + 1) * QK1_0]).try_into().unwrap(),
        );
    }
    Ok(out)
}

/// Quantize `n` f32 values to `Q1_0` (128-element groups). Each group's
/// scale `d` is the mean absolute value (the L2-optimal scalar for a
/// sign-only code); each weight becomes its sign bit (`>= 0 -> 1`).
/// Lossy — the round-trip reproduces `±d`. Used for tests / transcode.
pub fn quantize_q1_0(src: &[f32]) -> Result<Vec<u8>> {
    let n = src.len();
    if !n.is_multiple_of(QK1_0) {
        bail!("Q1_0: n={n} not divisible by {QK1_0}");
    }
    let nb = n / QK1_0;
    let mut out = vec![0u8; nb * Q1_0_BLOCK_BYTES];
    for i in 0..nb {
        let group = &src[i * QK1_0..(i + 1) * QK1_0];
        let mean_abs = group.iter().map(|v| v.abs()).sum::<f32>() / QK1_0 as f32;
        let off = i * Q1_0_BLOCK_BYTES;
        out[off..off + 2].copy_from_slice(&half::f16::from_f32(mean_abs).to_le_bytes());
        for (j, &v) in group.iter().enumerate() {
            if v >= 0.0 {
                out[off + 2 + j / 8] |= 1 << (j % 8);
            }
        }
    }
    Ok(out)
}

/// Dequantize only the given rows of a packed `Q1_0` matrix stored
/// row-major as `[n_rows, row_len]` (`row_len` a multiple of 128).
/// Returns `[indices.len(), row_len]` f32.
///
/// This is the low-footprint embedding-gather primitive: an embedding
/// table is `[vocab, hidden]` Q1_0 (e.g. Bonsai-27B: 248320×5120 = 178 MB
/// packed vs ~5 GB dequantized to F32). Gathering the handful of rows for
/// the prompt tokens dequantizes ~KB instead of materializing the whole
/// table — the core of option (f) for constrained machines.
pub fn gather_rows_q1_0(bytes: &[u8], row_len: usize, indices: &[usize]) -> Result<Vec<f32>> {
    if !row_len.is_multiple_of(QK1_0) {
        bail!("Q1_0 gather: row_len={row_len} not a multiple of {QK1_0}");
    }
    let blocks_per_row = row_len / QK1_0;
    let row_bytes = blocks_per_row * Q1_0_BLOCK_BYTES;
    let mut out = vec![0f32; indices.len() * row_len];
    for (i, &r) in indices.iter().enumerate() {
        let off = r * row_bytes;
        let end = off + row_bytes;
        if end > bytes.len() {
            bail!("Q1_0 gather: row {r} past packed length {}", bytes.len());
        }
        let row_out = &mut out[i * row_len..(i + 1) * row_len];
        for b in 0..blocks_per_row {
            let boff = off + b * Q1_0_BLOCK_BYTES;
            dequant_q1_0_block(
                &bytes[boff..boff + Q1_0_BLOCK_BYTES],
                (&mut row_out[b * QK1_0..(b + 1) * QK1_0])
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

    /// Pack 128 signs in `{-1, +1}` into a `Q1_0` block (mirrors the
    /// reference encoder: `x >= 0 → bit 1`, else `bit 0`).
    fn pack_q1_0_block(d: f32, signs: &[i8; QK1_0]) -> Vec<u8> {
        let mut bytes = vec![0u8; Q1_0_BLOCK_BYTES];
        bytes[0..2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
        for (j, &s) in signs.iter().enumerate() {
            if s >= 0 {
                bytes[2 + j / 8] |= 1 << (j % 8);
            }
        }
        bytes
    }

    #[test]
    fn block_size_matches_llama_cpp() {
        assert_eq!(Q1_0_BLOCK_BYTES, 18);
        assert_eq!(q1_0_bytes(QK1_0), Some(18));
        assert_eq!(q1_0_bytes(4 * QK1_0), Some(72));
    }

    #[test]
    fn q1_0_roundtrip_alternating() {
        let d = 0.375_f32;
        let mut signs = [0i8; QK1_0];
        for (i, s) in signs.iter_mut().enumerate() {
            *s = if i % 2 == 0 { 1 } else { -1 };
        }
        let bytes = pack_q1_0_block(d, &signs);
        let out = dequant_q1_0(&bytes, QK1_0).unwrap();
        for i in 0..QK1_0 {
            let expected = signs[i] as f32 * d;
            assert!(
                (out[i] - expected).abs() < 1e-4,
                "i={i}: out={} expected={expected}",
                out[i],
            );
        }
    }

    #[test]
    fn q1_0_roundtrip_multiblock_pattern() {
        // Two blocks, distinct scales, LSB-first bit ordering exercised
        // across byte boundaries (bit 7 vs bit 8 land in different bytes).
        let mut bytes = Vec::new();
        let mut expected = Vec::new();
        for (b, &d) in [0.5_f32, 1.25_f32].iter().enumerate() {
            let mut signs = [0i8; QK1_0];
            for (i, s) in signs.iter_mut().enumerate() {
                *s = if (i + b) % 3 == 0 { 1 } else { -1 };
                expected.push(*s as f32 * d);
            }
            bytes.extend_from_slice(&pack_q1_0_block(d, &signs));
        }
        let out = dequant_q1_0(&bytes, 2 * QK1_0).unwrap();
        assert_eq!(out.len(), expected.len());
        for (i, (o, e)) in out.iter().zip(&expected).enumerate() {
            assert!((o - e).abs() < 1e-4, "i={i}: out={o} expected={e}");
        }
    }

    #[test]
    fn rejects_bad_byte_count() {
        assert!(dequant_q1_0(&[0u8; 10], QK1_0).is_err());
        assert!(dequant_q1_0(&[0u8; Q1_0_BLOCK_BYTES], 17).is_err());
        assert_eq!(q1_0_bytes(QK1_0 - 1), None);
    }

    #[test]
    fn gather_rows_matches_full_dequant() {
        // [n_rows=6, row_len=256] packed Q1_0; gather a subset of rows and
        // confirm they equal the corresponding slice of a full dequant.
        let n_rows = 6usize;
        let row_len = 2 * QK1_0; // 256
        let n = n_rows * row_len;
        let src: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).sin() - 0.2).collect();
        let packed = quantize_q1_0(&src).unwrap();
        let full = dequant_q1_0(&packed, n).unwrap();

        let indices = [4usize, 0, 4, 2];
        let gathered = gather_rows_q1_0(&packed, row_len, &indices).unwrap();
        assert_eq!(gathered.len(), indices.len() * row_len);
        for (i, &r) in indices.iter().enumerate() {
            let g = &gathered[i * row_len..(i + 1) * row_len];
            let f = &full[r * row_len..(r + 1) * row_len];
            assert_eq!(g, f, "row {r} (gather idx {i}) mismatch");
        }
        // Out-of-range row is rejected.
        assert!(gather_rows_q1_0(&packed, row_len, &[n_rows]).is_err());
        assert!(gather_rows_q1_0(&packed, row_len - 1, &[0]).is_err());
    }
}
