// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Microsoft BitNet `I2_S` format, ggml type 36.
//!
//! Introduced by Microsoft's `bitnet.cpp` / `VibeASR.cpp` fork for
//! BitNet-b1.58 ternary weights (e.g. `microsoft/VibeVoice-ASR-BitNet`).
//! Every weight is a 2-bit code `q ∈ {0,1,2}` mapped to `(q − 1) · d`
//! (0 → −d, 1 → 0, 2 → +d; code 3 is unused). Unlike the K-quant family
//! the scale is **per-tensor**: a single f32 `d` stored *after* all the
//! packed codes, not per block.
//!
//! On-disk layout for a tensor of `n` elements (`n` a multiple of 128):
//!
//! ```text
//!   codes  (n / 4 bytes)   # 128-element blocks, 32 bytes each
//!   d      (f32, 4 bytes)  # per-tensor scale, at byte offset n/4
//!   pad    (28 bytes)      # → total n/4 + 32 bytes
//! ```
//!
//! Within one 128-element block (bytes `[b*32 .. b*32+32)`), byte `p`
//! (`p ∈ 0..32`) packs four codes at bit shifts `6, 4, 2, 0`:
//! `code(elem b*128 + g*32 + p) = (byte[b*32 + p] >> (6 − 2g)) & 3` for
//! `g ∈ {0,1,2,3}`. This mirrors `quantize_i2_s` in `ggml-lm-mad.cpp`
//! and was verified byte-for-byte against the shipped GGUF.

use anyhow::{Result, bail};

/// Elements per `I2_S` block (packs into 32 bytes).
pub const QK_I2_S: usize = 128;
/// Trailing bytes reserved for the per-tensor f32 scale (4 used, 28 pad).
pub const I2_S_SCALE_BYTES: usize = 32;

/// Storage bytes for `n` `I2_S` elements: `n/4` packed + 32-byte scale
/// trailer. `None` if `n` isn't a multiple of the 128-element block.
pub fn i2_s_bytes(n: usize) -> Option<usize> {
    if !n.is_multiple_of(QK_I2_S) {
        return None;
    }
    Some(n / 4 + I2_S_SCALE_BYTES)
}

/// Dequantize a full `I2_S` tensor of `n` elements to f32.
///
/// Reads the single per-tensor scale from `bytes[n/4 .. n/4+4]` and maps
/// each 2-bit code `q` to `(q − 1) * scale`.
pub fn dequant_i2_s(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if !n.is_multiple_of(QK_I2_S) {
        bail!("I2_S: n={n} not divisible by {QK_I2_S}");
    }
    let packed = n / 4;
    let need = packed + I2_S_SCALE_BYTES;
    if bytes.len() < packed + 4 {
        bail!("I2_S: expected >= {} bytes, got {}", packed + 4, bytes.len());
    }
    if bytes.len() != need {
        // Tolerate slight over-allocation but warn via error only when short.
        // (GGUF stores exactly n/4 + 32; keep strict-ish but permissive up.)
    }
    let scale = f32::from_le_bytes([
        bytes[packed],
        bytes[packed + 1],
        bytes[packed + 2],
        bytes[packed + 3],
    ]);
    let mut out = vec![0f32; n];
    let nb = n / QK_I2_S;
    for b in 0..nb {
        let boff = b * 32;
        let base = b * QK_I2_S;
        for p in 0..32 {
            let byte = bytes[boff + p];
            for g in 0..4 {
                let code = (byte >> (6 - 2 * g)) & 0x03;
                out[base + g * 32 + p] = (code as i32 - 1) as f32 * scale;
            }
        }
    }
    Ok(out)
}

/// Read only the per-tensor scale of an `I2_S` tensor of `n` elements.
pub fn i2_s_scale(bytes: &[u8], n: usize) -> Result<f32> {
    let packed = n / 4;
    if bytes.len() < packed + 4 {
        bail!("I2_S: too few bytes for scale");
    }
    Ok(f32::from_le_bytes([
        bytes[packed],
        bytes[packed + 1],
        bytes[packed + 2],
        bytes[packed + 3],
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack `n` ternary codes (`0/1/2`) + a scale exactly like the shipped
    /// GGUF, then check dequant recovers `(code − 1) * scale`.
    fn pack(codes: &[u8], scale: f32) -> Vec<u8> {
        let n = codes.len();
        let mut bytes = vec![0u8; n / 4 + I2_S_SCALE_BYTES];
        let nb = n / QK_I2_S;
        for b in 0..nb {
            for (p, byte) in bytes[b * 32..b * 32 + 32].iter_mut().enumerate() {
                let mut v = 0u8;
                for g in 0..4 {
                    let code = codes[b * QK_I2_S + g * 32 + p];
                    v |= (code & 0x03) << (6 - 2 * g);
                }
                *byte = v;
            }
        }
        bytes[n / 4..n / 4 + 4].copy_from_slice(&scale.to_le_bytes());
        bytes
    }

    #[test]
    fn bytes_formula() {
        assert_eq!(i2_s_bytes(128), Some(32 + 32));
        assert_eq!(i2_s_bytes(2359296), Some(2359296 / 4 + 32)); // attn_q
        assert_eq!(i2_s_bytes(100), None);
    }

    #[test]
    fn roundtrip_ternary() {
        let mut codes = vec![1u8; 256]; // all zero
        codes[0] = 0; // -d  (elem 0: b0 g0 p0)
        codes[32] = 2; // +d (elem 32: b0 g1 p0)
        codes[64] = 1; // 0
        codes[128] = 2; // +d in block 1
        let scale = 0.0553f32;
        let bytes = pack(&codes, scale);
        assert_eq!(bytes.len(), i2_s_bytes(256).unwrap());
        let out = dequant_i2_s(&bytes, 256).unwrap();
        assert!((out[0] + scale).abs() < 1e-6, "elem0 = {}", out[0]);
        assert!((out[32] - scale).abs() < 1e-6, "elem32 = {}", out[32]);
        assert!(out[64].abs() < 1e-6);
        assert!(out[1].abs() < 1e-6);
        assert!((out[128] - scale).abs() < 1e-6);
        assert_eq!(i2_s_scale(&bytes, 256).unwrap(), scale);
    }
}
