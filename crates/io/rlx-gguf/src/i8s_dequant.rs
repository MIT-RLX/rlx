// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Microsoft BitNet `I8_S` format, ggml type 37.
//!
//! Symmetric 8-bit weights used by Microsoft's `VibeASR.cpp` VAE encoder
//! (`microsoft/VibeVoice-ASR-BitNet`). Plain row-major `int8` values with
//! a single **per-tensor** f32 scale stored after the data:
//!
//! ```text
//!   qs  (n bytes, int8)    # row-major weights
//!   d   (f32, 4 bytes)     # per-tensor scale = amax/127, at offset n
//!   pad (28 bytes)         # → total n + 32 bytes
//! ```
//!
//! Dequant: `w = (int8) * d`. Verified against the shipped GGUF
//! (int8 range hits ±127, `d ≈ amax/127`).

use anyhow::{Result, bail};

/// Trailing bytes reserved for the per-tensor f32 scale (4 used, 28 pad).
pub const I8_S_SCALE_BYTES: usize = 32;

/// Storage bytes for `n` `I8_S` elements: `n` int8 + 32-byte scale trailer.
pub fn i8_s_bytes(n: usize) -> Option<usize> {
    Some(n + I8_S_SCALE_BYTES)
}

/// Dequantize a full `I8_S` tensor of `n` elements to f32 (`int8 * scale`).
pub fn dequant_i8_s(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if bytes.len() < n + 4 {
        bail!("I8_S: expected >= {} bytes, got {}", n + 4, bytes.len());
    }
    let scale =
        f32::from_le_bytes([bytes[n], bytes[n + 1], bytes[n + 2], bytes[n + 3]]);
    let mut out = vec![0f32; n];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = (bytes[i] as i8) as f32 * scale;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_formula() {
        assert_eq!(i8_s_bytes(256), Some(288));
        assert_eq!(i8_s_bytes(8192), Some(8224));
    }

    #[test]
    fn roundtrip() {
        let n = 64;
        let mut bytes = vec![0u8; n + I8_S_SCALE_BYTES];
        // int8 values -3..: store some signed values
        let vals: Vec<i8> = (0..n as i32).map(|i| (i - 32) as i8).collect();
        for (i, &v) in vals.iter().enumerate() {
            bytes[i] = v as u8;
        }
        let scale = 0.0025f32;
        bytes[n..n + 4].copy_from_slice(&scale.to_le_bytes());
        let out = dequant_i8_s(&bytes, n).unwrap();
        for (i, &v) in vals.iter().enumerate() {
            assert!((out[i] - v as f32 * scale).abs() < 1e-9);
        }
    }
}
