// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Ternary GGUF formats `TQ1_0` and `TQ2_0`. Used by BitNet /
//! TernaryLLM-style models. Layouts (byte-for-byte llama.cpp):
//!
//! - **TQ1_0**: `qs[(QK_K − 4·QK_K/64) / 5] = 48` bytes packing 240
//!   elements (5 trits per byte) + `qh[QK_K/64] = 4` bytes packing 16
//!   elements (4 trits per byte) + `d` (f16) = 54 bytes / 256 elements.
//!   Encoding stores `q = ⌈(t₀·81 + t₁·27 + t₂·9 + t₃·3 + t₄)·256/243⌉`
//!   per byte (i.e. left-aligned in the u8) so trit extraction is the
//!   "multiply by `pow3[n]` then `(q·3) >> 8`" trick.
//! - **TQ2_0**: `qs[QK_K/4] = 64` bytes (4 trits per byte at 2 bits each)
//!   + `d` (f16) = 66 bytes / 256 elements. Trits are stored as
//!   `(t + 1) & 3` directly; element layout is sub-block × trit-slot.
//!
//! Trit mapping is `t ∈ {0,1,2}` → `t − 1 ∈ {−1,0,+1}`; the f32 output
//! is `(t − 1) · d`.

use anyhow::{Result, bail};

use crate::{QK_K, read_f16_le};

const TQ1_0_QS_LEN: usize = (QK_K - 4 * QK_K / 64) / 5; // 48
const TQ1_0_QH_LEN: usize = QK_K / 64; // 4
const TQ1_0_BLOCK_BYTES: usize = TQ1_0_QS_LEN + TQ1_0_QH_LEN + 2; // 54

const TQ2_0_QS_LEN: usize = QK_K / 4; // 64
const TQ2_0_BLOCK_BYTES: usize = TQ2_0_QS_LEN + 2; // 66

const POW3: [u8; 5] = [1, 3, 9, 27, 81];

pub fn tq1_0_bytes(n: usize) -> Option<usize> {
    if !n.is_multiple_of(QK_K) {
        return None;
    }
    Some((n / QK_K) * TQ1_0_BLOCK_BYTES)
}

pub fn tq2_0_bytes(n: usize) -> Option<usize> {
    if !n.is_multiple_of(QK_K) {
        return None;
    }
    Some((n / QK_K) * TQ2_0_BLOCK_BYTES)
}

/// Dequantize one TQ1_0 block (54 bytes) into `out`.
///
/// Layout (in order in the file):
///   qs[48]                              # offsets 0..48
///   qh[4]                               # offsets 48..52
///   d  (f16, 2 bytes)                   # offsets 52..54
pub fn dequant_tq1_0_block(block: &[u8], out: &mut [f32; QK_K]) {
    let qs = &block[0..TQ1_0_QS_LEN];
    let qh = &block[TQ1_0_QS_LEN..TQ1_0_QS_LEN + TQ1_0_QH_LEN];
    let d = read_f16_le(&block[TQ1_0_QS_LEN + TQ1_0_QH_LEN..]);
    let mut y = 0usize;
    // First chunk: 32 bytes packing 5×32 = 160 elements.
    let j_split = TQ1_0_QS_LEN - TQ1_0_QS_LEN % 32; // 32
    let mut j = 0usize;
    while j < j_split {
        for n in 0..5 {
            for m in 0..32 {
                let q = qs[j + m].wrapping_mul(POW3[n]); // u8 wrap (mimics C `uint8_t`)
                let xi = ((q as u16 * 3) >> 8) as i32; // 0, 1, or 2
                out[y] = (xi - 1) as f32 * d;
                y += 1;
            }
        }
        j += 32;
    }
    // Second chunk: remaining bytes packing 5×16 = 80 elements per 16 bytes.
    while j < TQ1_0_QS_LEN {
        for n in 0..5 {
            for m in 0..16 {
                let q = qs[j + m].wrapping_mul(POW3[n]);
                let xi = ((q as u16 * 3) >> 8) as i32;
                out[y] = (xi - 1) as f32 * d;
                y += 1;
            }
        }
        j += 16;
    }
    // qh: 4 bytes packing 4×4 = 16 elements. Outer loop walks n in 0..4
    // (the qh bytes encode a 4-trit base-3 word; the encoder shifts the
    // first value to the most-significant trit so we extract via
    // n = 0 → high, n = 3 → low, which `(q*3) >> 8` after multiplication
    // by pow3[n] achieves).
    for n in 0..4 {
        for j in 0..TQ1_0_QH_LEN {
            let q = qh[j].wrapping_mul(POW3[n]);
            let xi = ((q as u16 * 3) >> 8) as i32;
            out[y] = (xi - 1) as f32 * d;
            y += 1;
        }
    }
    debug_assert_eq!(y, QK_K);
}

/// Dequantize one TQ2_0 block (66 bytes) into `out`.
///
/// Layout: `qs[64]` then `d` (f16). Trits emitted as 32-element strides:
/// byte `j+m` slot `l` produces output element `y_base + l*32 + m` where
/// `y_base` walks 128 elements per outer `j += 32` step.
pub fn dequant_tq2_0_block(block: &[u8], out: &mut [f32; QK_K]) {
    let qs = &block[0..TQ2_0_QS_LEN];
    let d = read_f16_le(&block[TQ2_0_QS_LEN..]);
    let mut y = 0usize;
    let mut j = 0usize;
    while j < TQ2_0_QS_LEN {
        for l in 0..4 {
            for m in 0..32 {
                let q = ((qs[j + m] >> (l * 2)) & 0x3) as i32;
                out[y] = (q - 1) as f32 * d;
                y += 1;
            }
        }
        j += 32;
    }
    debug_assert_eq!(y, QK_K);
}

pub fn dequant_tq1_0(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if !n.is_multiple_of(QK_K) {
        bail!("TQ1_0: n={n} not divisible by {QK_K}");
    }
    let nb = n / QK_K;
    if bytes.len() != nb * TQ1_0_BLOCK_BYTES {
        bail!(
            "TQ1_0: expected {} bytes, got {}",
            nb * TQ1_0_BLOCK_BYTES,
            bytes.len()
        );
    }
    let mut out = vec![0f32; n];
    for i in 0..nb {
        let off = i * TQ1_0_BLOCK_BYTES;
        dequant_tq1_0_block(
            &bytes[off..off + TQ1_0_BLOCK_BYTES],
            (&mut out[i * QK_K..(i + 1) * QK_K]).try_into().unwrap(),
        );
    }
    Ok(out)
}

pub fn dequant_tq2_0(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if !n.is_multiple_of(QK_K) {
        bail!("TQ2_0: n={n} not divisible by {QK_K}");
    }
    let nb = n / QK_K;
    if bytes.len() != nb * TQ2_0_BLOCK_BYTES {
        bail!(
            "TQ2_0: expected {} bytes, got {}",
            nb * TQ2_0_BLOCK_BYTES,
            bytes.len()
        );
    }
    let mut out = vec![0f32; n];
    for i in 0..nb {
        let off = i * TQ2_0_BLOCK_BYTES;
        dequant_tq2_0_block(
            &bytes[off..off + TQ2_0_BLOCK_BYTES],
            (&mut out[i * QK_K..(i + 1) * QK_K]).try_into().unwrap(),
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encode 256 trits in `{-1,0,1}` to a TQ2_0 block (mirrors the
    /// reference encoder `quantize_row_tq2_0_ref` byte-for-byte).
    fn pack_tq2_0_block(d: f32, trits: &[i8; QK_K]) -> Vec<u8> {
        let mut bytes = vec![0u8; TQ2_0_BLOCK_BYTES];
        let mut x = 0usize;
        let mut j = 0usize;
        while j < TQ2_0_QS_LEN {
            for m in 0..32 {
                let mut q = 0u8;
                for n in 0..4 {
                    let xi = ((trits[x + m + n * 32]) + 1) as u8 & 0x3;
                    q |= xi << (2 * n);
                }
                bytes[j + m] = q;
            }
            x += 4 * 32;
            j += 32;
        }
        bytes[TQ2_0_QS_LEN..].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
        bytes
    }

    /// Encode 256 trits to a TQ1_0 block (reference encoder).
    fn pack_tq1_0_block(d: f32, trits: &[i8; QK_K]) -> Vec<u8> {
        let mut bytes = vec![0u8; TQ1_0_BLOCK_BYTES];
        let mut x = 0usize;
        // First chunk: 32 bytes packing 5×32 = 160 trits.
        for m in 0..32 {
            let mut q: u32 = 0;
            for n in 0..5 {
                let xi = (trits[x + m + n * 32] + 1) as u32;
                q *= 3;
                q += xi;
            }
            q = (q * 256).div_ceil(243);
            bytes[m] = q as u8;
        }
        x += 5 * 32;
        // Second chunk: 16 bytes packing 5×16 = 80 trits.
        for m in 0..16 {
            let mut q: u32 = 0;
            for n in 0..5 {
                let xi = (trits[x + m + n * 16] + 1) as u32;
                q *= 3;
                q += xi;
            }
            q = (q * 256).div_ceil(243);
            bytes[32 + m] = q as u8;
        }
        x += 5 * 16;
        // qh: 4 bytes packing 4×4 = 16 trits, shifted to high.
        for j in 0..TQ1_0_QH_LEN {
            let mut q: u32 = 0;
            for m in 0..4 {
                let xi = (trits[x + j + m * TQ1_0_QH_LEN] + 1) as u32;
                q *= 3;
                q += xi;
            }
            q *= 3; // shift first value to most-significant trit
            q = (q * 256).div_ceil(243);
            bytes[TQ1_0_QS_LEN + j] = q as u8;
        }
        bytes[TQ1_0_QS_LEN + TQ1_0_QH_LEN..].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
        bytes
    }

    #[test]
    fn tq2_0_roundtrip_random() {
        let mut trits = [0i8; QK_K];
        for i in 0..QK_K {
            trits[i] = ((i * 7) % 3) as i8 - 1;
        }
        let bytes = pack_tq2_0_block(0.25, &trits);
        let out = dequant_tq2_0(&bytes, QK_K).unwrap();
        for i in 0..QK_K {
            assert!(
                (out[i] - 0.25 * trits[i] as f32).abs() < 1e-5,
                "i={i}: out={} expected={}",
                out[i],
                0.25 * trits[i] as f32
            );
        }
    }

    #[test]
    fn tq1_0_roundtrip_alternating() {
        let mut trits = [0i8; QK_K];
        for i in 0..QK_K {
            trits[i] = match i % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            };
        }
        let bytes = pack_tq1_0_block(0.5, &trits);
        let out = dequant_tq1_0(&bytes, QK_K).unwrap();
        for i in 0..QK_K {
            assert!(
                (out[i] - 0.5 * trits[i] as f32).abs() < 1e-5,
                "i={i}: out={} expected={}",
                out[i],
                0.5 * trits[i] as f32
            );
        }
    }

    #[test]
    fn rejects_bad_byte_count() {
        assert!(dequant_tq1_0(&[0u8; 10], QK_K).is_err());
        assert!(dequant_tq2_0(&[0u8; 10], QK_K).is_err());
        assert!(dequant_tq1_0(&[0u8; TQ1_0_BLOCK_BYTES], 17).is_err());
    }

    #[test]
    fn tq1_0_block_size_matches_llama_cpp() {
        assert_eq!(TQ1_0_BLOCK_BYTES, 54);
    }

    #[test]
    fn tq2_0_block_size_matches_llama_cpp() {
        assert_eq!(TQ2_0_BLOCK_BYTES, 66);
    }
}
