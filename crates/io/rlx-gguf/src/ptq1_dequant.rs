// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Custom ternary GGUF format `PTQ1_0`, ggml type 143 in the PrismML
//! llama.cpp dialect.
//!
//! Introduced by the PrismML llama.cpp fork for
//! `prism-ml/Ternary-Bonsai-2-27B-gguf` (a Qwen3.8-27B derivative).
//! Every weight is a trit `{−1, 0, +1}` scaled by one f16 `d` shared
//! across a group of 128 — 1.75 bits/weight deployed.
//!
//! The trits are packed in **base 3**, five per byte (`3^5 = 243 ≤ 256`),
//! exactly like upstream `TQ1_0` (type 34); the difference is the group
//! size — one scale per 128 weights instead of per 256 — which is what
//! lets an already-ternary-at-128 checkpoint round-trip losslessly.
//!
//! Block layout (byte-for-byte with `block_ptq1_0` in the fork's
//! `ggml-common.h` — note `d` comes **last** here, unlike `Q1_0`/`Q2_0`):
//!
//! ```text
//!   qs  (24 bytes, 5 trits each → 120 values)
//!   qh  ( 2 bytes, 4 trits each →   8 values)
//!   d   (f16, 2 bytes)              # group scale
//! ```
//!
//! = 28 bytes / 128 elements. Dequant mirrors `dequantize_row_ptq1_0`:
//! digit `n` of byte `b` is recovered as `((u8)(b · 3ⁿ) · 3) >> 8`,
//! yielding `0/1/2` → `−1/0/+1` after the `− 1`.
//!
//! The traversal order is **not** row-sequential: `qs` is walked in a
//! 16-wide stage (bytes 0..16, 80 values) then an 8-wide stage
//! (bytes 16..24, 40 values), digit-major within each stage, and `qh`
//! supplies the final 8. Getting this wrong still decodes to a valid
//! ternary tensor — it just permutes the weights — so it is checked
//! against the C reference in the tests below.
//!
//! [`QK_PTQ1_0`] is 128, matching [`crate::q1_dequant::QK1_0`].

use crate::read_f16_le;
use anyhow::{Result, bail};

/// Group size for the `PTQ1_0` format (weights sharing one f16 scale).
pub const QK_PTQ1_0: usize = 128;
/// Bytes of base-3 payload holding the first 120 trits (5 per byte).
pub const PTQ1_0_QS_BYTES: usize = (QK_PTQ1_0 - 4 * QK_PTQ1_0 / 64) / 5; // 24
/// Bytes of base-3 payload holding the trailing 8 trits (4 per byte).
pub const PTQ1_0_QH_BYTES: usize = QK_PTQ1_0 / 64; // 2
/// Bytes per `PTQ1_0` block: `qs` + `qh` + f16 scale.
pub const PTQ1_0_BLOCK_BYTES: usize = PTQ1_0_QS_BYTES + PTQ1_0_QH_BYTES + 2; // 28

/// Byte-stage widths `qs` is traversed in. Only the strides that fit in
/// [`PTQ1_0_QS_BYTES`] contribute, so this is effectively `16` then `8`;
/// the leading `32` is carried verbatim from the C reference so the two
/// stay diffable.
const PTQ1_0_STAGES: [usize; 3] = [32, 16, 8];

/// `3ⁿ` for the five base-3 digits a byte can hold.
const POW3: [u16; 5] = [1, 3, 9, 27, 81];

/// Recover base-3 digit `n` of `byte` as a trit in `{−1, 0, +1}`.
///
/// `(u8)(byte · 3ⁿ)` keeps the digit in the top two bits, and the
/// `(· 3) >> 8` maps that window onto `0/1/2` without a division.
#[inline]
fn trit(byte: u8, n: usize) -> i32 {
    let q = (byte as u16).wrapping_mul(POW3[n]) as u8;
    ((q as u16 * 3) >> 8) as i32 - 1
}

/// Storage bytes for `n` `PTQ1_0` elements. `None` if `n` isn't a
/// multiple of the 128-element group.
pub fn ptq1_0_bytes(n: usize) -> Option<usize> {
    if !n.is_multiple_of(QK_PTQ1_0) {
        return None;
    }
    Some((n / QK_PTQ1_0) * PTQ1_0_BLOCK_BYTES)
}

/// Dequantize one `PTQ1_0` block (28 bytes) into `out` (128 f32 values).
pub fn dequant_ptq1_0_block(block: &[u8], out: &mut [f32; QK_PTQ1_0]) {
    let qs = &block[0..PTQ1_0_QS_BYTES];
    let qh = &block[PTQ1_0_QS_BYTES..PTQ1_0_QS_BYTES + PTQ1_0_QH_BYTES];
    let d = read_f16_le(&block[PTQ1_0_QS_BYTES + PTQ1_0_QH_BYTES..PTQ1_0_BLOCK_BYTES]);

    let mut o = 0usize;
    let mut j = 0usize;
    for c in PTQ1_0_STAGES {
        while j + c <= PTQ1_0_QS_BYTES {
            for n in 0..5 {
                for m in 0..c {
                    out[o] = trit(qs[j + m], n) as f32 * d;
                    o += 1;
                }
            }
            j += c;
        }
    }
    for n in 0..4 {
        for &h in qh {
            out[o] = trit(h, n) as f32 * d;
            o += 1;
        }
    }
    debug_assert_eq!(o, QK_PTQ1_0);
}

/// Dequantize a full `PTQ1_0` tensor of `n` elements to f32.
pub fn dequant_ptq1_0(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if !n.is_multiple_of(QK_PTQ1_0) {
        bail!("PTQ1_0: n={n} not divisible by {QK_PTQ1_0}");
    }
    let nb = n / QK_PTQ1_0;
    if bytes.len() != nb * PTQ1_0_BLOCK_BYTES {
        bail!(
            "PTQ1_0: expected {} bytes, got {}",
            nb * PTQ1_0_BLOCK_BYTES,
            bytes.len()
        );
    }
    let mut out = vec![0f32; n];
    for i in 0..nb {
        let off = i * PTQ1_0_BLOCK_BYTES;
        dequant_ptq1_0_block(
            &bytes[off..off + PTQ1_0_BLOCK_BYTES],
            (&mut out[i * QK_PTQ1_0..(i + 1) * QK_PTQ1_0])
                .try_into()
                .unwrap(),
        );
    }
    Ok(out)
}

/// Quantize `n` f32 values to `PTQ1_0` (128-element groups).
///
/// Mirrors `quantize_row_ptq1_0_ref`: `d` is the group's max absolute
/// value and each weight becomes `clamp(round(w/d), -1..=1)`. Exact for
/// input that is already ternary at group 128 — which is the only thing
/// the published checkpoints contain — and lossy otherwise.
///
/// The packing is base 3 with the **first** trit most significant,
/// followed by `ceil(q · 256 / 243)`. That rescale is what makes the
/// decoder's `((u8)(q · 3ⁿ) · 3) >> 8` recover digit `n` with no
/// division; writing the digits in the obvious `Σ xi · 3ⁿ` order instead
/// still produces a well-formed ternary tensor, just a scrambled one.
pub fn quantize_ptq1_0(src: &[f32]) -> Result<Vec<u8>> {
    if !src.len().is_multiple_of(QK_PTQ1_0) {
        bail!("PTQ1_0: len={} not divisible by {QK_PTQ1_0}", src.len());
    }
    // `-1, 0, 1` -> `0, 1, 2`, matching the reference's `lroundf(..) + 1`.
    #[inline]
    fn code(w: f32, id: f32) -> u8 {
        ((w * id).round().clamp(-1.0, 1.0) as i32 + 1) as u8
    }
    // Pack a base-3 number so the decoder's shift trick recovers it.
    #[inline]
    fn rescale(q: u8) -> u8 {
        (q as u16 * 256).div_ceil(243) as u8
    }

    let mut out = Vec::with_capacity(ptq1_0_bytes(src.len()).unwrap());
    for blk in src.chunks_exact(QK_PTQ1_0) {
        let d = blk.iter().fold(0f32, |a, v| a.max(v.abs()));
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };

        let mut qs = [0u8; PTQ1_0_QS_BYTES];
        let mut qh = [0u8; PTQ1_0_QH_BYTES];
        // Offset into the block; each stage consumes `5 * c` elements
        // laid out digit-major, exactly as the decoder emits them.
        let mut base = 0usize;
        let mut j = 0usize;
        for c in PTQ1_0_STAGES {
            while j + c <= PTQ1_0_QS_BYTES {
                for m in 0..c {
                    let mut q = 0u8;
                    for n in 0..5 {
                        q = q * 3 + code(blk[base + m + n * c], id);
                    }
                    qs[j + m] = rescale(q);
                }
                base += 5 * c;
                j += c;
            }
        }
        for h in 0..PTQ1_0_QH_BYTES {
            let mut q = 0u8;
            for m in 0..4 {
                q = q * 3 + code(blk[base + h + m * PTQ1_0_QH_BYTES], id);
            }
            // Only 4 trits, so shift them into the top digit positions.
            q *= 3;
            qh[h] = rescale(q);
        }
        debug_assert_eq!(base + 4 * PTQ1_0_QH_BYTES, QK_PTQ1_0);

        out.extend_from_slice(&qs);
        out.extend_from_slice(&qh);
        out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
    }
    Ok(out)
}

/// Dequantize only rows `indices` of a `[n_rows, row_len]` `PTQ1_0`
/// tensor. `row_len` must be a multiple of the 128-element group so
/// every row starts on a block boundary.
///
/// Used for embedding lookups: `Ternary-Bonsai-2-27B`'s `token_embd` is
/// 248320 × 5120, which is 278 MB packed but 5.1 GB as f32.
pub fn gather_rows_ptq1_0(bytes: &[u8], row_len: usize, indices: &[usize]) -> Result<Vec<f32>> {
    let row_bytes = ptq1_0_bytes(row_len).ok_or_else(|| {
        anyhow::anyhow!("PTQ1_0: row_len={row_len} not a multiple of {QK_PTQ1_0}")
    })?;
    let n_rows = bytes.len() / row_bytes;
    let mut out = vec![0f32; indices.len() * row_len];
    for (i, &r) in indices.iter().enumerate() {
        if r >= n_rows {
            bail!("PTQ1_0: row {r} out of range (n_rows={n_rows})");
        }
        let src = &bytes[r * row_bytes..(r + 1) * row_bytes];
        let dst = &mut out[i * row_len..(i + 1) * row_len];
        for (b, chunk) in src.chunks_exact(PTQ1_0_BLOCK_BYTES).enumerate() {
            dequant_ptq1_0_block(
                chunk,
                (&mut dst[b * QK_PTQ1_0..(b + 1) * QK_PTQ1_0])
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

    /// The C reference's element map, transcribed from the fork's
    /// `tests/test-ptq1_0-element-map.cpp` (itself the CUDA accessor).
    /// Independent of our stage walk, so agreement pins the traversal.
    fn ref_trit(qs: &[u8; 24], qh: &[u8; 2], e: usize) -> i32 {
        let (b, n) = if e < 80 {
            (qs[e & 15], e >> 4)
        } else if e < 120 {
            let t = e - 80;
            (qs[16 + (t & 7)], t >> 3)
        } else {
            let t = e - 120;
            (qh[t & 1], t >> 1)
        };
        let mut v = b as u32;
        for i in 0..4 {
            if i < n {
                v = (v * 3) & 0xFF;
            }
        }
        ((v * 3) >> 8) as i32 - 1
    }

    #[test]
    fn traversal_matches_cuda_element_map() {
        // Same LCG the fork's test uses, so a divergence is comparable
        // block-for-block against it.
        let mut seed: u32 = 12345;
        let mut next = || {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            ((seed >> 16) & 0xFF) as u8
        };
        for _ in 0..2000 {
            let mut qs = [0u8; 24];
            let mut qh = [0u8; 2];
            for b in qs.iter_mut() {
                *b = next();
            }
            for b in qh.iter_mut() {
                *b = next();
            }
            let mut block = Vec::new();
            block.extend_from_slice(&qs);
            block.extend_from_slice(&qh);
            block.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
            let mut out = [0f32; QK_PTQ1_0];
            dequant_ptq1_0_block(&block, &mut out);
            for e in 0..QK_PTQ1_0 {
                assert_eq!(
                    out[e] as i32,
                    ref_trit(&qs, &qh, e),
                    "element {e} diverges from the C element map"
                );
            }
        }
    }

    #[test]
    fn decodes_only_ternary_values() {
        let mut seed: u32 = 99;
        let mut bytes = Vec::new();
        for _ in 0..8 {
            for _ in 0..26 {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                bytes.push(((seed >> 16) & 0xFF) as u8);
            }
            bytes.extend_from_slice(&half::f16::from_f32(0.25).to_le_bytes());
        }
        let out = dequant_ptq1_0(&bytes, 8 * QK_PTQ1_0).unwrap();
        for v in out {
            assert!(
                v == -0.25 || v == 0.0 || v == 0.25,
                "PTQ1_0 decoded a non-ternary value: {v}"
            );
        }
    }

    #[test]
    fn roundtrips_ternary_input_exactly() {
        // Already-ternary at group 128 is the lossless case the format
        // exists for, so the round trip must be exact, not approximate.
        let n = 3 * QK_PTQ1_0;
        let scales = [0.125f32, 0.5, 2.0];
        let src: Vec<f32> = (0..n)
            .map(|i| {
                let t = (i % 3) as f32 - 1.0;
                t * scales[i / QK_PTQ1_0]
            })
            .collect();
        let packed = quantize_ptq1_0(&src).unwrap();
        assert_eq!(packed.len(), ptq1_0_bytes(n).unwrap());
        let back = dequant_ptq1_0(&packed, n).unwrap();
        assert_eq!(back, src);
    }

    #[test]
    fn rejects_bad_byte_count() {
        assert!(dequant_ptq1_0(&[0u8; 10], QK_PTQ1_0).is_err());
        assert!(dequant_ptq1_0(&[0u8; PTQ1_0_BLOCK_BYTES], 17).is_err());
        assert_eq!(ptq1_0_bytes(QK_PTQ1_0 - 1), None);
        assert_eq!(ptq1_0_bytes(QK_PTQ1_0), Some(PTQ1_0_BLOCK_BYTES));
    }

    #[test]
    fn gather_rows_matches_full_dequant() {
        let n_rows = 5usize;
        let row_len = 2 * QK_PTQ1_0;
        let n = n_rows * row_len;
        let src: Vec<f32> = (0..n)
            .map(|i| (((i * 7) % 3) as f32 - 1.0) * 0.75)
            .collect();
        let packed = quantize_ptq1_0(&src).unwrap();
        let full = dequant_ptq1_0(&packed, n).unwrap();
        let indices = [3usize, 0, 3, 1];
        let gathered = gather_rows_ptq1_0(&packed, row_len, &indices).unwrap();
        assert_eq!(gathered.len(), indices.len() * row_len);
        for (i, &r) in indices.iter().enumerate() {
            assert_eq!(
                &gathered[i * row_len..(i + 1) * row_len],
                &full[r * row_len..(r + 1) * row_len],
                "row {r} (gather idx {i}) mismatch"
            );
        }
        assert!(gather_rows_ptq1_0(&packed, row_len, &[n_rows]).is_err());
        assert!(gather_rows_ptq1_0(&packed, row_len - 1, &[0]).is_err());
    }
}
