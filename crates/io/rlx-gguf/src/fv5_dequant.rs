// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fermion five-value ternary GGUF formats `FV5` (ggml type 43) and its
//! int8 companion `FV5B` (ggml type 44).
//!
//! Introduced by the Fermion Research `fermion-fv5` llama.cpp fork to carry
//! their TRTC v4 ternary containers (Neutrino-0.6B / Neutrino-8B) as GGUF.
//! Both models are stock Qwen3 geometry; only the weight storage is novel.
//!
//! # `FV5` — 3.25 bits/weight, 256-element blocks
//!
//! Every weight is one of five values `{0, ±s_lo, ±s_hi}` selected by three
//! bit-planes over a 256-element block, with two f32 per-row scales copied
//! verbatim into each block (no per-block scale search — the scales live in
//! the source container):
//!
//! ```text
//!   s_lo (f32, 4 bytes)          # low-magnitude scale
//!   s_hi (f32, 4 bytes)          # high-magnitude scale
//!   bp   (32 bytes = 256 bits)   # +1 plane      (LSB-first)
//!   bn   (32 bytes = 256 bits)   # -1 plane      (disjoint from bp)
//!   br   (32 bytes = 256 bits)   # hi-mag select (subset of bp|bn)
//! ```
//!
//! = 104 bytes / 256 elements. Reconstruction mirrors the fork's
//! `dequantize_row_fv5`:
//!
//! ```text
//!   sign = (bp bit) - (bn bit)          in {-1, 0, +1}
//!   mag  = (br bit) ? s_hi : s_lo
//!   w    = sign * mag
//! ```
//!
//! Bit order is little (bit `i` of byte `j` selects element `8*j + i`),
//! identical to `Q1_0`. Invariants held by every valid block:
//! `bp & bn == 0` and `br ⊆ (bp | bn)`.
//!
//! # `FV5B` — 8.125 bits/weight, 256-element blocks
//!
//! Plain int8 rows with one f32 per-row scale, used for the untied token
//! embedding and lm_head (`w = s * q`):
//!
//! ```text
//!   s  (f32, 4 bytes)            # row scale
//!   qs (256 bytes, int8)         # one int8 code per weight
//! ```
//!
//! = 260 bytes / 256 elements.
//!
//! There is intentionally no float→FV5 quantizer: the blocks are produced
//! offline by the TRTC v4 → GGUF converter and rlx only *reads* them, so the
//! dequant here is byte-exact against the container's f32 expansion (the
//! same reference the fork's correctness gate certifies against).

use anyhow::{Result, bail};

/// Elements per block for both `FV5` and `FV5B`.
pub const QK_FV5: usize = 256;
/// Bytes per `FV5` block: two f32 scales + three 32-byte bit-planes.
pub const FV5_BLOCK_BYTES: usize = 2 * 4 + 3 * (QK_FV5 / 8); // 104
/// Bytes per `FV5B` block: one f32 scale + 256 int8 codes.
pub const FV5B_BLOCK_BYTES: usize = 4 + QK_FV5; // 260

#[inline]
fn read_f32_le(b: &[u8]) -> f32 {
    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// Storage bytes for `n` `FV5` elements. `None` if `n` isn't a multiple of
/// the 256-element block.
pub fn fv5_bytes(n: usize) -> Option<usize> {
    if !n.is_multiple_of(QK_FV5) {
        return None;
    }
    Some((n / QK_FV5) * FV5_BLOCK_BYTES)
}

/// Storage bytes for `n` `FV5B` elements. `None` if `n` isn't a multiple of
/// the 256-element block.
pub fn fv5b_bytes(n: usize) -> Option<usize> {
    if !n.is_multiple_of(QK_FV5) {
        return None;
    }
    Some((n / QK_FV5) * FV5B_BLOCK_BYTES)
}

/// Dequantize one `FV5` block (104 bytes) into `out` (256 f32 values).
///
/// `w = (bp - bn) * (br ? s_hi : s_lo)`, LSB-first bit order within each
/// plane byte.
pub fn dequant_fv5_block(block: &[u8], out: &mut [f32; QK_FV5]) {
    let s_lo = read_f32_le(&block[0..4]);
    let s_hi = read_f32_le(&block[4..8]);
    let bp = &block[8..8 + QK_FV5 / 8];
    let bn = &block[8 + QK_FV5 / 8..8 + 2 * (QK_FV5 / 8)];
    let br = &block[8 + 2 * (QK_FV5 / 8)..8 + 3 * (QK_FV5 / 8)];
    for (j, slot) in out.iter_mut().enumerate() {
        let byte = j / 8;
        let bit = 1u8 << (j % 8);
        let p = (bp[byte] & bit) != 0;
        let ng = (bn[byte] & bit) != 0;
        let hi = (br[byte] & bit) != 0;
        let sign = (p as i32) - (ng as i32);
        let mag = if hi { s_hi } else { s_lo };
        *slot = sign as f32 * mag;
    }
}

/// Dequantize one `FV5B` block (260 bytes) into `out` (256 f32 values).
/// `w = s * q` with int8 codes.
pub fn dequant_fv5b_block(block: &[u8], out: &mut [f32; QK_FV5]) {
    let s = read_f32_le(&block[0..4]);
    let qs = &block[4..4 + QK_FV5];
    for (j, slot) in out.iter_mut().enumerate() {
        *slot = s * (qs[j] as i8 as f32);
    }
}

/// Dequantize a full `FV5` tensor of `n` elements to f32.
pub fn dequant_fv5(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if !n.is_multiple_of(QK_FV5) {
        bail!("FV5: n={n} not divisible by {QK_FV5}");
    }
    let nb = n / QK_FV5;
    if bytes.len() != nb * FV5_BLOCK_BYTES {
        bail!(
            "FV5: expected {} bytes, got {}",
            nb * FV5_BLOCK_BYTES,
            bytes.len()
        );
    }
    let mut out = vec![0f32; n];
    for i in 0..nb {
        let off = i * FV5_BLOCK_BYTES;
        dequant_fv5_block(
            &bytes[off..off + FV5_BLOCK_BYTES],
            (&mut out[i * QK_FV5..(i + 1) * QK_FV5]).try_into().unwrap(),
        );
    }
    Ok(out)
}

/// Dequantize a full `FV5B` tensor of `n` elements to f32.
pub fn dequant_fv5b(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if !n.is_multiple_of(QK_FV5) {
        bail!("FV5B: n={n} not divisible by {QK_FV5}");
    }
    let nb = n / QK_FV5;
    if bytes.len() != nb * FV5B_BLOCK_BYTES {
        bail!(
            "FV5B: expected {} bytes, got {}",
            nb * FV5B_BLOCK_BYTES,
            bytes.len()
        );
    }
    let mut out = vec![0f32; n];
    for i in 0..nb {
        let off = i * FV5B_BLOCK_BYTES;
        dequant_fv5b_block(
            &bytes[off..off + FV5B_BLOCK_BYTES],
            (&mut out[i * QK_FV5..(i + 1) * QK_FV5]).try_into().unwrap(),
        );
    }
    Ok(out)
}

/// Dequantize only the given rows of a packed `FV5` matrix stored row-major
/// as `[n_rows, row_len]` (`row_len` a multiple of 256). Returns
/// `[indices.len(), row_len]` f32. Low-footprint gather primitive (mirrors
/// [`gather_rows_fv5b`] for the embedding table).
pub fn gather_rows_fv5(bytes: &[u8], row_len: usize, indices: &[usize]) -> Result<Vec<f32>> {
    if !row_len.is_multiple_of(QK_FV5) {
        bail!("FV5 gather: row_len={row_len} not a multiple of {QK_FV5}");
    }
    let blocks_per_row = row_len / QK_FV5;
    let row_bytes = blocks_per_row * FV5_BLOCK_BYTES;
    let mut out = vec![0f32; indices.len() * row_len];
    for (i, &r) in indices.iter().enumerate() {
        let off = r * row_bytes;
        let end = off + row_bytes;
        if end > bytes.len() {
            bail!("FV5 gather: row {r} past packed length {}", bytes.len());
        }
        let row_out = &mut out[i * row_len..(i + 1) * row_len];
        for b in 0..blocks_per_row {
            let boff = off + b * FV5_BLOCK_BYTES;
            dequant_fv5_block(
                &bytes[boff..boff + FV5_BLOCK_BYTES],
                (&mut row_out[b * QK_FV5..(b + 1) * QK_FV5])
                    .try_into()
                    .unwrap(),
            );
        }
    }
    Ok(out)
}

/// Dequantize only the given rows of a packed `FV5B` matrix stored row-major
/// as `[n_rows, row_len]` (`row_len` a multiple of 256). Returns
/// `[indices.len(), row_len]` f32. This is the embedding-gather primitive
/// for Neutrino's int8 `token_embd`: gather the prompt-token rows instead of
/// materializing the whole `[vocab, hidden]` table in f32.
pub fn gather_rows_fv5b(bytes: &[u8], row_len: usize, indices: &[usize]) -> Result<Vec<f32>> {
    if !row_len.is_multiple_of(QK_FV5) {
        bail!("FV5B gather: row_len={row_len} not a multiple of {QK_FV5}");
    }
    let blocks_per_row = row_len / QK_FV5;
    let row_bytes = blocks_per_row * FV5B_BLOCK_BYTES;
    let mut out = vec![0f32; indices.len() * row_len];
    for (i, &r) in indices.iter().enumerate() {
        let off = r * row_bytes;
        let end = off + row_bytes;
        if end > bytes.len() {
            bail!("FV5B gather: row {r} past packed length {}", bytes.len());
        }
        let row_out = &mut out[i * row_len..(i + 1) * row_len];
        for b in 0..blocks_per_row {
            let boff = off + b * FV5B_BLOCK_BYTES;
            dequant_fv5b_block(
                &bytes[boff..boff + FV5B_BLOCK_BYTES],
                (&mut row_out[b * QK_FV5..(b + 1) * QK_FV5])
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

    /// Pack 256 five-value codes into an `FV5` block. `codes[j]` ∈
    /// `{-2,-1,0,1,2}` maps to `{-s_hi,-s_lo,0,+s_lo,+s_hi}` — the same
    /// bp/bn/br assignment the container uses, respecting the invariants.
    fn pack_fv5_block(s_lo: f32, s_hi: f32, codes: &[i8; QK_FV5]) -> Vec<u8> {
        let mut bytes = vec![0u8; FV5_BLOCK_BYTES];
        bytes[0..4].copy_from_slice(&s_lo.to_le_bytes());
        bytes[4..8].copy_from_slice(&s_hi.to_le_bytes());
        let (bp_off, bn_off, br_off) = (8, 8 + QK_FV5 / 8, 8 + 2 * (QK_FV5 / 8));
        for (j, &c) in codes.iter().enumerate() {
            let byte = j / 8;
            let bit = 1u8 << (j % 8);
            match c {
                1 => bytes[bp_off + byte] |= bit,
                2 => {
                    bytes[bp_off + byte] |= bit;
                    bytes[br_off + byte] |= bit;
                }
                -1 => bytes[bn_off + byte] |= bit,
                -2 => {
                    bytes[bn_off + byte] |= bit;
                    bytes[br_off + byte] |= bit;
                }
                0 => {}
                _ => panic!("code out of range: {c}"),
            }
        }
        bytes
    }

    fn expand_code(c: i8, s_lo: f32, s_hi: f32) -> f32 {
        match c {
            0 => 0.0,
            1 => s_lo,
            -1 => -s_lo,
            2 => s_hi,
            -2 => -s_hi,
            _ => unreachable!(),
        }
    }

    fn pack_fv5b_block(s: f32, qs: &[i8; QK_FV5]) -> Vec<u8> {
        let mut bytes = vec![0u8; FV5B_BLOCK_BYTES];
        bytes[0..4].copy_from_slice(&s.to_le_bytes());
        for (j, &q) in qs.iter().enumerate() {
            bytes[4 + j] = q as u8;
        }
        bytes
    }

    #[test]
    fn block_sizes_match_fork() {
        assert_eq!(FV5_BLOCK_BYTES, 104);
        assert_eq!(FV5B_BLOCK_BYTES, 260);
        assert_eq!(fv5_bytes(QK_FV5), Some(104));
        assert_eq!(fv5_bytes(4 * QK_FV5), Some(416));
        assert_eq!(fv5b_bytes(QK_FV5), Some(260));
        assert_eq!(fv5b_bytes(2 * QK_FV5), Some(520));
    }

    #[test]
    fn fv5_roundtrip_five_values() {
        let (s_lo, s_hi) = (0.0125_f32, 0.0875_f32);
        // Cycle through all five codes across the block, crossing byte and
        // plane boundaries so bit-order bugs surface.
        let mut codes = [0i8; QK_FV5];
        for (i, c) in codes.iter_mut().enumerate() {
            *c = [0i8, 1, -1, 2, -2][i % 5];
        }
        let bytes = pack_fv5_block(s_lo, s_hi, &codes);
        let out = dequant_fv5(&bytes, QK_FV5).unwrap();
        for i in 0..QK_FV5 {
            let expected = expand_code(codes[i], s_lo, s_hi);
            assert_eq!(
                out[i], expected,
                "i={i}: out={} expected={expected}",
                out[i]
            );
        }
    }

    #[test]
    fn fv5_multiblock_distinct_scales() {
        let mut bytes = Vec::new();
        let mut expected = Vec::new();
        for (b, &(s_lo, s_hi)) in [(0.01_f32, 0.2_f32), (0.05_f32, 0.5_f32)]
            .iter()
            .enumerate()
        {
            let mut codes = [0i8; QK_FV5];
            for (i, c) in codes.iter_mut().enumerate() {
                *c = [0i8, 1, -1, 2, -2][(i + b) % 5];
                expected.push(expand_code(*c, s_lo, s_hi));
            }
            bytes.extend_from_slice(&pack_fv5_block(s_lo, s_hi, &codes));
        }
        let out = dequant_fv5(&bytes, 2 * QK_FV5).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn fv5b_roundtrip_int8() {
        let s = 0.0003_f32;
        let mut qs = [0i8; QK_FV5];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = (i as i32 - 128) as i8;
        }
        let bytes = pack_fv5b_block(s, &qs);
        let out = dequant_fv5b(&bytes, QK_FV5).unwrap();
        for i in 0..QK_FV5 {
            assert_eq!(out[i], s * qs[i] as f32, "i={i}");
        }
    }

    #[test]
    fn rejects_bad_byte_count() {
        assert!(dequant_fv5(&[0u8; 10], QK_FV5).is_err());
        assert!(dequant_fv5(&[0u8; FV5_BLOCK_BYTES], QK_FV5 - 1).is_err());
        assert!(dequant_fv5b(&[0u8; 10], QK_FV5).is_err());
        assert_eq!(fv5_bytes(QK_FV5 - 1), None);
        assert_eq!(fv5b_bytes(QK_FV5 + 1), None);
    }

    #[test]
    fn gather_rows_match_full_dequant() {
        // FV5: [n_rows=5, row_len=512] gather subset == full-dequant slice.
        let (n_rows, row_len) = (5usize, 2 * QK_FV5);
        let (s_lo, s_hi) = (0.02_f32, 0.3_f32);
        let mut bytes = Vec::new();
        for r in 0..n_rows {
            for _bk in 0..(row_len / QK_FV5) {
                let mut codes = [0i8; QK_FV5];
                for (i, c) in codes.iter_mut().enumerate() {
                    *c = [0i8, 1, -1, 2, -2][(i + r) % 5];
                }
                bytes.extend_from_slice(&pack_fv5_block(s_lo, s_hi, &codes));
            }
        }
        let full = dequant_fv5(&bytes, n_rows * row_len).unwrap();
        let indices = [3usize, 0, 3, 1];
        let gathered = gather_rows_fv5(&bytes, row_len, &indices).unwrap();
        for (i, &r) in indices.iter().enumerate() {
            let g = &gathered[i * row_len..(i + 1) * row_len];
            let f = &full[r * row_len..(r + 1) * row_len];
            assert_eq!(g, f, "FV5 row {r} (idx {i})");
        }
        assert!(gather_rows_fv5(&bytes, row_len, &[n_rows]).is_err());

        // FV5B: same shape.
        let mut b2 = Vec::new();
        for r in 0..n_rows {
            for _bk in 0..(row_len / QK_FV5) {
                let mut qs = [0i8; QK_FV5];
                for (i, q) in qs.iter_mut().enumerate() {
                    *q = ((i as i32 + r as i32) % 251 - 125) as i8;
                }
                b2.extend_from_slice(&pack_fv5b_block(0.001_f32, &qs));
            }
        }
        let full_b = dequant_fv5b(&b2, n_rows * row_len).unwrap();
        let gathered_b = gather_rows_fv5b(&b2, row_len, &indices).unwrap();
        for (i, &r) in indices.iter().enumerate() {
            let g = &gathered_b[i * row_len..(i + 1) * row_len];
            let f = &full_b[r * row_len..(r + 1) * row_len];
            assert_eq!(g, f, "FV5B row {r} (idx {i})");
        }
    }
}
