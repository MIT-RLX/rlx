// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pestle exact-ternary GGUF format `G8_0`, ggml type 143.
//!
//! Introduced by the Doses AI `mortar.cpp` llama.cpp fork for
//! `Doses-AI/Pestle-27B-Ternary-GGUF` (a Qwen3.6-27B derivative). Used
//! only for the untied `token_embd` / `output` (lm_head) tables — the
//! transformer linears are Pestle-factorized `Q2_0` pairs instead
//! (see [`crate::q2_dequant`]).
//!
//! Every weight is a 2-bit code `q ∈ {0,1,2}` mapped to `(q − 1) · d`,
//! i.e. exact ternary `{−d, 0, +d}`. Unlike `Q2_0` the scale is *not*
//! shared across the whole block: each **group of 8** weights carries
//! its own bf16 scale, so a 32-element block holds four of them —
//! 2 bits + 16/8 bits amortized = **4 bits/weight**. The much finer
//! scale granularity is what makes the format "exact" enough to hold
//! an embedding table at ternary code width.
//!
//! Block layout (byte-for-byte with `block_g8_0` in the fork's
//! `ggml-common.h`):
//!
//! ```text
//!   d   (4 × u16, 8 bytes)   # raw bf16 scale bits, one per group of 8
//!   qs  (8 bytes = 32×2 bits)  # LSB-first 2-bit codes within each byte
//! ```
//!
//! = 16 bytes / 32 elements. Dequant mirrors `dequantize_row_g8_0`:
//! `y[j] = (q − 1) · bf16(d[j / 8])`.
//!
//! Note code `3` is unreachable from the fork's encoder (it clamps to
//! `0..=2`); we map it to `+2d` anyway so a hand-written block never
//! silently reads as something else.

use anyhow::{Result, bail};

/// Elements per `G8_0` block.
pub const QKG8_0: usize = 32;
/// Weights sharing one bf16 scale.
pub const G8_0_GROUP: usize = 8;
/// Scales per block (`QKG8_0 / G8_0_GROUP`).
pub const G8_0_SCALES: usize = QKG8_0 / G8_0_GROUP; // 4
/// Bytes per `G8_0` block: 4 bf16 scales + 32×2-bit codes.
pub const G8_0_BLOCK_BYTES: usize = 2 * G8_0_SCALES + QKG8_0 / 4; // 16

/// Storage bytes for `n` `G8_0` elements. `None` if `n` isn't a
/// multiple of the 32-element block.
pub fn g8_0_bytes(n: usize) -> Option<usize> {
    if !n.is_multiple_of(QKG8_0) {
        return None;
    }
    Some((n / QKG8_0) * G8_0_BLOCK_BYTES)
}

#[inline]
fn read_bf16_le(b: &[u8]) -> f32 {
    half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32()
}

/// Dequantize one `G8_0` block (16 bytes) into `out` (32 f32 values).
///
/// Layout: `d[4]` (bf16 bits) then `qs[8]` 2-bit codes, LSB-first
/// within each byte (`(qs[j / 4] >> ((j % 4) * 2)) & 3`).
/// Value = `(q − 1) * d[j / 8]`.
pub fn dequant_g8_0_block(block: &[u8], out: &mut [f32; QKG8_0]) {
    let mut d = [0f32; G8_0_SCALES];
    for (g, slot) in d.iter_mut().enumerate() {
        *slot = read_bf16_le(&block[g * 2..g * 2 + 2]);
    }
    let qs = &block[2 * G8_0_SCALES..G8_0_BLOCK_BYTES];
    for (j, slot) in out.iter_mut().enumerate() {
        let q = (qs[j / 4] >> ((j % 4) * 2)) & 0x03;
        *slot = (q as i32 - 1) as f32 * d[j / G8_0_GROUP];
    }
}

/// Dequantize a full `G8_0` tensor of `n` elements to f32.
pub fn dequant_g8_0(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if !n.is_multiple_of(QKG8_0) {
        bail!("G8_0: n={n} not divisible by {QKG8_0}");
    }
    let nb = n / QKG8_0;
    if bytes.len() != nb * G8_0_BLOCK_BYTES {
        bail!(
            "G8_0: expected {} bytes, got {}",
            nb * G8_0_BLOCK_BYTES,
            bytes.len()
        );
    }
    let mut out = vec![0f32; n];
    for i in 0..nb {
        let off = i * G8_0_BLOCK_BYTES;
        dequant_g8_0_block(
            &bytes[off..off + G8_0_BLOCK_BYTES],
            (&mut out[i * QKG8_0..(i + 1) * QKG8_0]).try_into().unwrap(),
        );
    }
    Ok(out)
}

/// Quantize `n` f32 values to `G8_0` (32-element blocks, one bf16
/// scale per 8). Scale is the group's max absolute value; each weight
/// becomes `clamp(round(w / d) + 1, 0..=2)`. Lossy — used for tests /
/// transcode. Mirrors `quantize_row_g8_0_ref`.
pub fn quantize_g8_0(src: &[f32]) -> Result<Vec<u8>> {
    let n = src.len();
    if !n.is_multiple_of(QKG8_0) {
        bail!("G8_0: n={n} not divisible by {QKG8_0}");
    }
    let nb = n / QKG8_0;
    let mut out = vec![0u8; nb * G8_0_BLOCK_BYTES];
    for i in 0..nb {
        let block = &src[i * QKG8_0..(i + 1) * QKG8_0];
        let off = i * G8_0_BLOCK_BYTES;
        for g in 0..G8_0_SCALES {
            let group = &block[g * G8_0_GROUP..(g + 1) * G8_0_GROUP];
            // Round-trip through bf16: the decoder reads the stored
            // bits, so quantize against the value it will actually see.
            let amax = half::bf16::from_f32(group.iter().map(|v| v.abs()).fold(0.0f32, f32::max));
            out[off + g * 2..off + g * 2 + 2].copy_from_slice(&amax.to_bits().to_le_bytes());
            let amax = amax.to_f32();
            let id = if amax > 0.0 { 1.0 / amax } else { 0.0 };
            for (j, &v) in group.iter().enumerate() {
                let p = g * G8_0_GROUP + j;
                let q = ((v * id).round() as i32 + 1).clamp(0, 2);
                out[off + 2 * G8_0_SCALES + p / 4] |= (q as u8) << ((p % 4) * 2);
            }
        }
    }
    Ok(out)
}

/// Dequantize only the given rows of a packed `G8_0` matrix stored
/// row-major as `[n_rows, row_len]` (`row_len` a multiple of 32).
/// Returns `[indices.len(), row_len]` f32.
///
/// This is the embedding-gather path: Pestle's `token_embd` is
/// 248320 × 5120, which is 5.1 GiB dequantized but 636 MiB packed —
/// gathering rows keeps the table off the heap.
pub fn gather_rows_g8_0(bytes: &[u8], row_len: usize, indices: &[usize]) -> Result<Vec<f32>> {
    if !row_len.is_multiple_of(QKG8_0) {
        bail!("G8_0 gather: row_len={row_len} not a multiple of {QKG8_0}");
    }
    let blocks_per_row = row_len / QKG8_0;
    let row_bytes = blocks_per_row * G8_0_BLOCK_BYTES;
    let mut out = vec![0f32; indices.len() * row_len];
    for (i, &r) in indices.iter().enumerate() {
        let off = r * row_bytes;
        let end = off + row_bytes;
        if end > bytes.len() {
            bail!("G8_0 gather: row {r} past packed length {}", bytes.len());
        }
        let row_out = &mut out[i * row_len..(i + 1) * row_len];
        for b in 0..blocks_per_row {
            let boff = off + b * G8_0_BLOCK_BYTES;
            dequant_g8_0_block(
                &bytes[boff..boff + G8_0_BLOCK_BYTES],
                (&mut row_out[b * QKG8_0..(b + 1) * QKG8_0])
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

    #[test]
    fn block_size_matches_fork_layout() {
        // static_assert(sizeof(block_g8_0) == 4*sizeof(uint16_t) + QKG8_0/4)
        assert_eq!(G8_0_BLOCK_BYTES, 16);
        assert_eq!(g8_0_bytes(QKG8_0 * 3), Some(48));
        assert_eq!(g8_0_bytes(QKG8_0 + 1), None);
    }

    #[test]
    fn roundtrip_is_exact_for_ternary_input() {
        // Ternary values scaled per group of 8 survive the round trip
        // exactly: the scale is a bf16-representable power of two and
        // every weight is one of {−d, 0, +d}.
        let mut src = vec![0f32; QKG8_0 * 2];
        for (j, v) in src.iter_mut().enumerate() {
            let d = [0.25f32, 2.0, 0.5, 4.0, 1.0, 0.125, 8.0, 16.0][(j / G8_0_GROUP) % 8];
            *v = [-1.0f32, 0.0, 1.0][j % 3] * d;
        }
        let packed = quantize_g8_0(&src).unwrap();
        assert_eq!(packed.len(), 2 * G8_0_BLOCK_BYTES);
        let back = dequant_g8_0(&packed, src.len()).unwrap();
        assert_eq!(back, src);
    }

    #[test]
    fn scales_are_per_group_of_eight() {
        // Group 0 is ±1, group 1 is ±100: a single block-wide scale
        // would crush the first group to zero.
        let mut src = vec![0f32; QKG8_0];
        for (j, v) in src.iter_mut().enumerate() {
            *v = if j < 8 { 1.0 } else { 100.0 };
        }
        let back = dequant_g8_0(&quantize_g8_0(&src).unwrap(), QKG8_0).unwrap();
        assert_eq!(back[0], 1.0);
        assert_eq!(back[8], 100.0);
    }

    #[test]
    fn gather_rows_matches_full_dequant() {
        let n_rows = 6;
        let row_len = QKG8_0 * 2;
        let src: Vec<f32> = (0..n_rows * row_len)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.5)
            .collect();
        let packed = quantize_g8_0(&src).unwrap();
        let full = dequant_g8_0(&packed, src.len()).unwrap();
        let idx = [4usize, 0, 3];
        let got = gather_rows_g8_0(&packed, row_len, &idx).unwrap();
        for (i, &r) in idx.iter().enumerate() {
            assert_eq!(
                &got[i * row_len..(i + 1) * row_len],
                &full[r * row_len..(r + 1) * row_len]
            );
        }
    }

    #[test]
    fn code_three_reads_as_two_d() {
        // Unreachable from the encoder, but a hand-written block must
        // still decode deterministically.
        let mut block = [0u8; G8_0_BLOCK_BYTES];
        for g in 0..G8_0_SCALES {
            block[g * 2..g * 2 + 2]
                .copy_from_slice(&half::bf16::from_f32(3.0).to_bits().to_le_bytes());
        }
        block[2 * G8_0_SCALES] = 0b11; // element 0 → code 3
        let mut out = [0f32; QKG8_0];
        dequant_g8_0_block(&block, &mut out);
        assert_eq!(out[0], 6.0);
        assert_eq!(out[1], -3.0); // code 0
    }
}
