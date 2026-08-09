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

// ---------------------------------------------------------------------------
// int8 dot path (llama.cpp-style): quantize activations to int8 per 128-group,
// then dot directly against the packed 2-bit codes. Mirrors the Q4_K×Q8_K seam
// in rlx-cpu, but the single-scale Q2_0 layout maps 1:1 onto one VNNI
// `VPDPBUSD` loop (u8 codes × i8 activations). See rlx-cpu `intrinsics::vnni`.
// ---------------------------------------------------------------------------

/// Byte offset of the group-sum field inside a packed activation block.
const Q8_0_G128_XSUM_OFF: usize = 4 + QK2_0;

/// Packed int8 activation block for the `Q2_0` int-dot path (one 128-group):
/// `d_x` f32 scale · `QK2_0`×i8 codes · `xsum` i32 (Σ of the i8 codes).
///
/// `xsum` lets the VNNI kernel fold the `−1` weight offset with a single
/// subtract, so the unsigned codes `{0,1,2,3}` can feed `VPDPBUSD` directly.
pub const Q8_0_G128_BYTES: usize = 4 + QK2_0 + 4; // 136

/// Quantize a row of activations (`x.len()` a multiple of [`QK2_0`]) into the
/// packed int8 group format consumed by the `Q2_0` int-dot kernels.
///
/// Per group: `d_x = amax/127`, `a_j = round(x_j·127/amax)` clamped to
/// `[-127,127]`, and `xsum = Σ a_j`. Quantized once per GEMV and reused across
/// every output row.
pub fn quantize_q8_0_g128_row(x: &[f32], out: &mut [u8]) {
    assert!(
        x.len().is_multiple_of(QK2_0),
        "q8_0_g128: len not multiple of {QK2_0}"
    );
    let nb = x.len() / QK2_0;
    assert_eq!(out.len(), nb * Q8_0_G128_BYTES);
    for b in 0..nb {
        let g = &x[b * QK2_0..(b + 1) * QK2_0];
        let dst = &mut out[b * Q8_0_G128_BYTES..(b + 1) * Q8_0_G128_BYTES];
        let amax = g.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let dx = amax / 127.0;
        let id = if amax > 0.0 { 127.0 / amax } else { 0.0 };
        dst[0..4].copy_from_slice(&dx.to_le_bytes());
        let mut xsum = 0i32;
        for (j, &v) in g.iter().enumerate() {
            let q = (v * id).round().clamp(-127.0, 127.0) as i32;
            dst[4 + j] = q as i8 as u8;
            xsum += q;
        }
        dst[Q8_0_G128_XSUM_OFF..Q8_0_G128_XSUM_OFF + 4].copy_from_slice(&xsum.to_le_bytes());
    }
}

/// Scalar reference dot of one packed `Q2_0` weight block (34 B) with one
/// packed int8 activation block ([`Q8_0_G128_BYTES`]).
///
/// Bit-matches the SIMD kernels by construction: integer accumulation of
/// `Σ_j (q_j − 1)·a_j`, then a single f32 scale `d·d_x`. (SIMD paths instead
/// compute `Σ q_j·a_j − xsum`, which is the identical integer.)
pub fn q2_0_dot_q8_g128(w: &[u8], a: &[u8]) -> f32 {
    let d = read_f16_le(&w[0..2]);
    let dx = f32::from_le_bytes([a[0], a[1], a[2], a[3]]);
    let qs = &w[2..2 + QK2_0 / 4];
    let acts = &a[4..4 + QK2_0];
    let mut raw = 0i32;
    for j in 0..QK2_0 {
        let q = ((qs[j / 4] >> ((j % 4) * 2)) & 0x03) as i32;
        raw += (q - 1) * (acts[j] as i8 as i32);
    }
    d * dx * raw as f32
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

    #[test]
    fn int_dot_matches_f32_reference() {
        // One 128-group: ternary-ish weights, arbitrary activations.
        let mut wf = [0f32; QK2_0];
        let mut xf = [0f32; QK2_0];
        for j in 0..QK2_0 {
            wf[j] = (((j * 7) % 3) as i32 - 1) as f32 * 0.5; // -0.5 / 0 / 0.5
            xf[j] = ((j as f32) * 0.013).sin() * 3.0;
        }
        let w = quantize_q2_0(&wf).unwrap();
        let mut a = vec![0u8; Q8_0_G128_BYTES];
        quantize_q8_0_g128_row(&xf, &mut a);

        // Reference: dequant both sides to f32 and dot.
        let mut wd = [0f32; QK2_0];
        dequant_q2_0_block(&w, &mut wd);
        let dx = f32::from_le_bytes([a[0], a[1], a[2], a[3]]);
        let mut ref_dot = 0f32;
        for j in 0..QK2_0 {
            ref_dot += wd[j] * (a[4 + j] as i8 as f32 * dx);
        }
        let got = q2_0_dot_q8_g128(&w, &a);
        // int8 activation quant → small tolerance vs full-f32 dot.
        assert!(
            (got - ref_dot).abs() < 1e-2 * (1.0 + ref_dot.abs()),
            "{got} vs {ref_dot}"
        );
    }
}
