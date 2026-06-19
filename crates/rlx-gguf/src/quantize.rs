// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Float → GGML quant encoders. Mirrors `quantize_row_*` from
//! llama.cpp's `ggml-quants.c`. Output is byte-identical for the
//! legacy schemes (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0); K-quants use a simpler
//! per-sub-block min/max search than upstream's iterative
//! `make_qx_quants`, so quality is a notch below `llama-quantize`
//! but is fully valid GGUF output that round-trips through the
//! dequant kernels in [`super`].
//!
//! # Entry points
//!
//! * [`quantize`] — pick a scheme at runtime; returns the encoded
//!   byte payload sized for the writer.
//! * `quantize_q*_block` — per-block primitives, for callers that
//!   want to stream encode without an intermediate `Vec`.
//! * `quantize_q*` — full-tensor wrappers for the legacy formats
//!   (Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0).
//!
//! # Example
//!
//! ```ignore
//! use rlx_gguf::{GgmlType, quantize};
//!
//! let x: Vec<f32> = /* … */;
//! let bytes = quantize(&x, GgmlType::Q4K)?;
//! // `bytes` is now ready for `GgufWriter::add_tensor_bytes`.
//! ```
//!
//! # Quality
//!
//! Round-trip cosine on production weights (see
//! `rlx-gguf-convert`'s benchmark table) hits 0.997 at Q4_K, 0.9998
//! at Q6_K, and 0.99998 at Q8_0 — close enough for inference, slightly
//! worse than `llama-quantize`'s output on the same inputs. If
//! peak-quality matters more than pure-Rust simplicity, prefer the
//! upstream C++ tool.

use anyhow::{Result, bail};

use crate::{GgmlType, K_SCALE_SIZE, QK_K, QK4_0, QK8_0};

const QK4_1: usize = 32;
const QK5_0: usize = 32;
const QK5_1: usize = 32;

#[inline]
fn f16_bytes(x: f32) -> [u8; 2] {
    half::f16::from_f32(x).to_le_bytes()
}

#[inline]
fn nearest_i32(x: f32) -> i32 {
    // Match upstream: `roundf` (round half-away-from-zero).
    if x >= 0.0 {
        (x + 0.5) as i32
    } else {
        (x - 0.5) as i32
    }
}

// ─── byte-count for encoder output ────────────────────────────────

/// Exact byte count produced by [`quantize`] for `n` elements at
/// `dtype`. Returns `None` if `n` doesn't divide the block size.
pub fn encoded_bytes(dtype: GgmlType, n: usize) -> Option<usize> {
    crate::bytes_for_public(dtype, n)
}

// ─── legacy formats ───────────────────────────────────────────────

/// Quantize one Q8_0 block.
pub fn quantize_q8_0_block(src: &[f32], out: &mut [u8]) {
    assert!(src.len() >= QK8_0 && out.len() >= 2 + QK8_0);
    let mut amax = 0f32;
    for &v in &src[..QK8_0] {
        amax = amax.max(v.abs());
    }
    let d = amax / 127.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    out[0..2].copy_from_slice(&f16_bytes(d));
    for j in 0..QK8_0 {
        let q = nearest_i32(src[j] * id).clamp(-128, 127) as i8;
        out[2 + j] = q as u8;
    }
}

/// Quantize one Q4_0 block (32 fp32 → 18 bytes).
pub fn quantize_q4_0_block(src: &[f32], out: &mut [u8]) {
    assert!(src.len() >= QK4_0 && out.len() >= 2 + QK4_0 / 2);
    let mut amax = 0f32;
    let mut max = 0f32;
    for &v in &src[..QK4_0] {
        if amax < v.abs() {
            amax = v.abs();
            max = v;
        }
    }
    let d = max / -8.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    out[0..2].copy_from_slice(&f16_bytes(d));
    for j in 0..QK4_0 / 2 {
        let x0 = src[j] * id;
        let x1 = src[QK4_0 / 2 + j] * id;
        let xi0 = (nearest_i32(x0 + 8.0)).clamp(0, 15) as u8;
        let xi1 = (nearest_i32(x1 + 8.0)).clamp(0, 15) as u8;
        out[2 + j] = xi0 | (xi1 << 4);
    }
}

/// Quantize one Q4_1 block (32 fp32 → 20 bytes).
pub fn quantize_q4_1_block(src: &[f32], out: &mut [u8]) {
    assert!(src.len() >= QK4_1 && out.len() >= 2 + 2 + QK4_1 / 2);
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for &v in &src[..QK4_1] {
        if v < mn {
            mn = v;
        }
        if v > mx {
            mx = v;
        }
    }
    let d = (mx - mn) / 15.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    out[0..2].copy_from_slice(&f16_bytes(d));
    out[2..4].copy_from_slice(&f16_bytes(mn));
    for j in 0..QK4_1 / 2 {
        let x0 = (src[j] - mn) * id;
        let x1 = (src[QK4_1 / 2 + j] - mn) * id;
        let xi0 = (nearest_i32(x0)).clamp(0, 15) as u8;
        let xi1 = (nearest_i32(x1)).clamp(0, 15) as u8;
        out[4 + j] = xi0 | (xi1 << 4);
    }
}

/// Quantize one Q5_0 block (32 fp32 → 22 bytes).
pub fn quantize_q5_0_block(src: &[f32], out: &mut [u8]) {
    assert!(src.len() >= QK5_0 && out.len() >= 2 + 4 + QK5_0 / 2);
    let mut amax = 0f32;
    let mut max = 0f32;
    for &v in &src[..QK5_0] {
        if amax < v.abs() {
            amax = v.abs();
            max = v;
        }
    }
    let d = max / -16.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    out[0..2].copy_from_slice(&f16_bytes(d));
    let mut qh: u32 = 0;
    for j in 0..QK5_0 / 2 {
        let x0 = src[j] * id;
        let x1 = src[QK5_0 / 2 + j] * id;
        let xi0 = (nearest_i32(x0 + 16.0)).clamp(0, 31) as u32;
        let xi1 = (nearest_i32(x1 + 16.0)).clamp(0, 31) as u32;
        out[6 + j] = ((xi0 & 0x0F) | ((xi1 & 0x0F) << 4)) as u8;
        qh |= ((xi0 & 0x10) >> 4) << j;
        qh |= ((xi1 & 0x10) >> 4) << (j + QK5_0 / 2);
    }
    out[2..6].copy_from_slice(&qh.to_le_bytes());
}

/// Quantize one Q5_1 block (32 fp32 → 24 bytes).
pub fn quantize_q5_1_block(src: &[f32], out: &mut [u8]) {
    assert!(src.len() >= QK5_1 && out.len() >= 2 + 2 + 4 + QK5_1 / 2);
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for &v in &src[..QK5_1] {
        if v < mn {
            mn = v;
        }
        if v > mx {
            mx = v;
        }
    }
    let d = (mx - mn) / 31.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    out[0..2].copy_from_slice(&f16_bytes(d));
    out[2..4].copy_from_slice(&f16_bytes(mn));
    let mut qh: u32 = 0;
    for j in 0..QK5_1 / 2 {
        let x0 = (src[j] - mn) * id;
        let x1 = (src[QK5_1 / 2 + j] - mn) * id;
        let xi0 = (nearest_i32(x0)).clamp(0, 31) as u32;
        let xi1 = (nearest_i32(x1)).clamp(0, 31) as u32;
        out[8 + j] = ((xi0 & 0x0F) | ((xi1 & 0x0F) << 4)) as u8;
        qh |= ((xi0 & 0x10) >> 4) << j;
        qh |= ((xi1 & 0x10) >> 4) << (j + QK5_1 / 2);
    }
    out[4..8].copy_from_slice(&qh.to_le_bytes());
}

// ─── per-tensor wrappers ──────────────────────────────────────────

fn check_div(name: &str, n: usize, blk: usize) -> Result<()> {
    if !n.is_multiple_of(blk) {
        bail!("{name}: n={n} not divisible by {blk}");
    }
    Ok(())
}

pub fn quantize_q8_0(src: &[f32]) -> Result<Vec<u8>> {
    let n = src.len();
    check_div("Q8_0", n, QK8_0)?;
    let nb = n / QK8_0;
    let blk = 2 + QK8_0;
    let mut out = vec![0u8; nb * blk];
    for i in 0..nb {
        let off = i * blk;
        quantize_q8_0_block(&src[i * QK8_0..(i + 1) * QK8_0], &mut out[off..off + blk]);
    }
    Ok(out)
}

pub fn quantize_q4_0(src: &[f32]) -> Result<Vec<u8>> {
    let n = src.len();
    check_div("Q4_0", n, QK4_0)?;
    let nb = n / QK4_0;
    let blk = 2 + QK4_0 / 2;
    let mut out = vec![0u8; nb * blk];
    for i in 0..nb {
        let off = i * blk;
        quantize_q4_0_block(&src[i * QK4_0..(i + 1) * QK4_0], &mut out[off..off + blk]);
    }
    Ok(out)
}

pub fn quantize_q4_1(src: &[f32]) -> Result<Vec<u8>> {
    let n = src.len();
    check_div("Q4_1", n, QK4_1)?;
    let nb = n / QK4_1;
    let blk = 2 + 2 + QK4_1 / 2;
    let mut out = vec![0u8; nb * blk];
    for i in 0..nb {
        let off = i * blk;
        quantize_q4_1_block(&src[i * QK4_1..(i + 1) * QK4_1], &mut out[off..off + blk]);
    }
    Ok(out)
}

pub fn quantize_q5_0(src: &[f32]) -> Result<Vec<u8>> {
    let n = src.len();
    check_div("Q5_0", n, QK5_0)?;
    let nb = n / QK5_0;
    let blk = 2 + 4 + QK5_0 / 2;
    let mut out = vec![0u8; nb * blk];
    for i in 0..nb {
        let off = i * blk;
        quantize_q5_0_block(&src[i * QK5_0..(i + 1) * QK5_0], &mut out[off..off + blk]);
    }
    Ok(out)
}

pub fn quantize_q5_1(src: &[f32]) -> Result<Vec<u8>> {
    let n = src.len();
    check_div("Q5_1", n, QK5_1)?;
    let nb = n / QK5_1;
    let blk = 2 + 2 + 4 + QK5_1 / 2;
    let mut out = vec![0u8; nb * blk];
    for i in 0..nb {
        let off = i * blk;
        quantize_q5_1_block(&src[i * QK5_1..(i + 1) * QK5_1], &mut out[off..off + blk]);
    }
    Ok(out)
}

// ─── K-quants (simplified per-sub-block min/max) ──────────────────
//
// Upstream `quantize_row_q*_K` runs an inner search across candidate
// scales (`make_qx_quants`) and an importance matrix; we use plain
// min/max. Output is valid GGUF and round-trips losslessly through
// the matching dequant in [`super`], but for very lopsided
// distributions some accuracy is left on the table.

#[inline]
fn write_packed_scales_mins(scales: &[u8; 8], mins: &[u8; 8], dst: &mut [u8; K_SCALE_SIZE]) {
    // Reverse of `get_scale_min_k4`:
    //   j < 4:  dst[j]   low 6 bits = scales[j];   dst[j+4] low 6 bits = mins[j]
    //   j >= 4: dst[j+4] = (scales[j] & 0xF) | ((mins[j] & 0xF) << 4)
    //           top 2 bits of scales[j] → top 2 bits of dst[j-4]
    //           top 2 bits of mins[j]   → top 2 bits of dst[j]
    for j in 0..4 {
        dst[j] = scales[j] & 0x3F;
        dst[j + 4] = mins[j] & 0x3F;
    }
    for j in 4..8 {
        dst[j + 4] = (scales[j] & 0x0F) | ((mins[j] & 0x0F) << 4);
        dst[j - 4] |= (scales[j] & 0x30) << 2;
        dst[j] |= (mins[j] & 0x30) << 2;
    }
}

/// Quantize one Q4_K super-block (256 fp32 → 144 bytes).
pub fn quantize_q4_k_block(src: &[f32], out: &mut [u8]) {
    assert!(src.len() >= QK_K && out.len() >= 2 + 2 + K_SCALE_SIZE + QK_K / 2);
    // Per-sub-block (8 × 32 elements). Encoded range respects the
    // unsigned-`m` constraint of the decoder (`out = d*sc*q -
    // dmin*m`, m ≥ 0): when mn ≥ 0 we can't express a positive
    // offset, so collapse mn_eff to 0 (slightly coarser quant for
    // all-positive sub-blocks, but correct).
    let mut sub_d = [0f32; 8];
    let mut sub_min = [0f32; 8];
    for j in 0..8 {
        let sub = &src[j * 32..(j + 1) * 32];
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        for &v in sub {
            if v < mn {
                mn = v;
            }
            if v > mx {
                mx = v;
            }
        }
        if mn >= 0.0 {
            sub_d[j] = mx / 15.0;
            sub_min[j] = 0.0;
        } else {
            sub_d[j] = (mx - mn) / 15.0;
            sub_min[j] = -mn;
        }
    }
    let d_outer = sub_d.iter().cloned().fold(0f32, f32::max) / 63.0;
    let dmin_outer = sub_min.iter().cloned().fold(0f32, f32::max) / 63.0;
    let id = if d_outer != 0.0 { 1.0 / d_outer } else { 0.0 };
    let idm = if dmin_outer != 0.0 {
        1.0 / dmin_outer
    } else {
        0.0
    };
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];
    for j in 0..8 {
        sc[j] = (nearest_i32(sub_d[j] * id)).clamp(0, 63) as u8;
        mn[j] = (nearest_i32(sub_min[j] * idm)).clamp(0, 63) as u8;
    }
    out[0..2].copy_from_slice(&f16_bytes(d_outer));
    out[2..4].copy_from_slice(&f16_bytes(dmin_outer));
    let mut packed = [0u8; K_SCALE_SIZE];
    write_packed_scales_mins(&sc, &mn, &mut packed);
    out[4..4 + K_SCALE_SIZE].copy_from_slice(&packed);
    let qs = &mut out[4 + K_SCALE_SIZE..4 + K_SCALE_SIZE + QK_K / 2];
    // Pair sub-blocks (j, j+1): low nibble = sub-block j, high nibble = j+1.
    let mut is = 0usize;
    for j in (0..8).step_by(2) {
        let d0 = d_outer * sc[j] as f32;
        let m0 = dmin_outer * mn[j] as f32;
        let d1 = d_outer * sc[j + 1] as f32;
        let m1 = dmin_outer * mn[j + 1] as f32;
        let id0 = if d0 != 0.0 { 1.0 / d0 } else { 0.0 };
        let id1 = if d1 != 0.0 { 1.0 / d1 } else { 0.0 };
        for l in 0..32 {
            let q0 = (nearest_i32((src[j * 32 + l] + m0) * id0)).clamp(0, 15) as u8;
            let q1 = (nearest_i32((src[(j + 1) * 32 + l] + m1) * id1)).clamp(0, 15) as u8;
            qs[is + l] = q0 | (q1 << 4);
        }
        is += 32;
    }
}

/// Quantize one Q5_K super-block (256 fp32 → 176 bytes).
pub fn quantize_q5_k_block(src: &[f32], out: &mut [u8]) {
    let blk = 2 + 2 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2;
    assert!(src.len() >= QK_K && out.len() >= blk);
    let mut sub_d = [0f32; 8];
    let mut sub_min = [0f32; 8];
    for j in 0..8 {
        let sub = &src[j * 32..(j + 1) * 32];
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        for &v in sub {
            if v < mn {
                mn = v;
            }
            if v > mx {
                mx = v;
            }
        }
        if mn >= 0.0 {
            sub_d[j] = mx / 31.0;
            sub_min[j] = 0.0;
        } else {
            sub_d[j] = (mx - mn) / 31.0;
            sub_min[j] = -mn;
        }
    }
    let d_outer = sub_d.iter().cloned().fold(0f32, f32::max) / 63.0;
    let dmin_outer = sub_min.iter().cloned().fold(0f32, f32::max) / 63.0;
    let id = if d_outer != 0.0 { 1.0 / d_outer } else { 0.0 };
    let idm = if dmin_outer != 0.0 {
        1.0 / dmin_outer
    } else {
        0.0
    };
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];
    for j in 0..8 {
        sc[j] = (nearest_i32(sub_d[j] * id)).clamp(0, 63) as u8;
        mn[j] = (nearest_i32(sub_min[j] * idm)).clamp(0, 63) as u8;
    }
    out[0..2].copy_from_slice(&f16_bytes(d_outer));
    out[2..4].copy_from_slice(&f16_bytes(dmin_outer));
    let mut packed = [0u8; K_SCALE_SIZE];
    write_packed_scales_mins(&sc, &mn, &mut packed);
    out[4..4 + K_SCALE_SIZE].copy_from_slice(&packed);
    let qh_off = 4 + K_SCALE_SIZE;
    let qs_off = qh_off + QK_K / 8;
    // Zero qh first; we OR bits in below.
    for b in &mut out[qh_off..qh_off + QK_K / 8] {
        *b = 0;
    }
    // Pair sub-blocks. qh interleaves bits: for the pair (j, j+1),
    // qh bit `u1` (initially 1) marks the high bit of low-nibble
    // elements, qh bit `u2` (initially 2) marks high bit of
    // high-nibble elements; both shift left by 2 each iteration.
    let mut is = 0usize;
    let mut u1: u8 = 1;
    let mut u2: u8 = 2;
    for j in (0..8).step_by(2) {
        let d0 = d_outer * sc[j] as f32;
        let m0 = dmin_outer * mn[j] as f32;
        let d1 = d_outer * sc[j + 1] as f32;
        let m1 = dmin_outer * mn[j + 1] as f32;
        let id0 = if d0 != 0.0 { 1.0 / d0 } else { 0.0 };
        let id1 = if d1 != 0.0 { 1.0 / d1 } else { 0.0 };
        for l in 0..32 {
            let q0 = (nearest_i32((src[j * 32 + l] + m0) * id0)).clamp(0, 31) as u8;
            let q1 = (nearest_i32((src[(j + 1) * 32 + l] + m1) * id1)).clamp(0, 31) as u8;
            out[qs_off + is + l] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
            if q0 & 0x10 != 0 {
                out[qh_off + l] |= u1;
            }
            if q1 & 0x10 != 0 {
                out[qh_off + l] |= u2;
            }
        }
        is += 32;
        u1 <<= 2;
        u2 <<= 2;
    }
}

/// Quantize one Q6_K super-block (256 fp32 → 210 bytes).
pub fn quantize_q6_k_block(src: &[f32], out: &mut [u8]) {
    let ql_len = QK_K / 2;
    let qh_len = QK_K / 4;
    let sc_len = QK_K / 16;
    let blk = ql_len + qh_len + sc_len + 2;
    assert!(src.len() >= QK_K && out.len() >= blk);
    // 16 sub-blocks of 16 elements. Per sub-block: amax, signed
    // scale. d = max(|sub_scale|) / 127 (i8). sc_i = sub_scale / d.
    let mut sub_scale = [0f32; 16];
    for j in 0..16 {
        let sub = &src[j * 16..(j + 1) * 16];
        let mut amax = 0f32;
        let mut maxv = 0f32;
        for &v in sub {
            if amax < v.abs() {
                amax = v.abs();
                maxv = v;
            }
        }
        // Symmetric 6-bit range is [-32, 31]; pick the divisor that
        // maps |max| onto 32.
        sub_scale[j] = maxv / -32.0;
    }
    let amax = sub_scale.iter().fold(0f32, |a, &v| a.max(v.abs()));
    let d_outer = amax / 127.0;
    let id_outer = if d_outer != 0.0 { 1.0 / d_outer } else { 0.0 };
    let mut sc = [0i8; 16];
    for j in 0..16 {
        sc[j] = (nearest_i32(sub_scale[j] * id_outer)).clamp(-128, 127) as i8;
    }
    // Quantize values per sub-block.
    let mut q = [0i8; QK_K];
    for j in 0..16 {
        let s = d_outer * sc[j] as f32;
        let isf = if s != 0.0 { 1.0 / s } else { 0.0 };
        for l in 0..16 {
            q[j * 16 + l] = (nearest_i32(src[j * 16 + l] * isf)).clamp(-32, 31) as i8;
        }
    }
    // Pack into the layout the dequant kernel expects. From the
    // dequant decoder:
    //   for h in 0..2 {
    //     for l in 0..32 {
    //       q1 = (ql[ql_off+l]&0xF) | ((qh[qh_off_h+l] & 3) << 4) - 32  → dst[base+l]
    //       q2 = (ql[ql_off+l+32]&0xF) | (((qh[..]>>2)&3)<<4) - 32       → dst[base+l+32]
    //       q3 = (ql[ql_off+l]>>4) | (((qh[..]>>4)&3)<<4) - 32           → dst[base+l+64]
    //       q4 = (ql[ql_off+l+32]>>4) | (((qh[..]>>6)&3)<<4) - 32        → dst[base+l+96]
    // i.e. each 64-byte ql half stores low nibbles of 64 elements, and
    // each 32-byte qh half packs 2 high bits for those same 4 groups.
    for b in out[..blk].iter_mut() {
        *b = 0;
    }
    for h in 0..2 {
        let dst_base = h * 128;
        let ql_off = h * 64;
        let qh_off_h = h * 32;
        for l in 0..32 {
            let v1 = (q[dst_base + l] as i32 + 32) as u8; // [0, 63]
            let v2 = (q[dst_base + l + 32] as i32 + 32) as u8;
            let v3 = (q[dst_base + l + 64] as i32 + 32) as u8;
            let v4 = (q[dst_base + l + 96] as i32 + 32) as u8;
            out[ql_off + l] = (v1 & 0x0F) | ((v3 & 0x0F) << 4);
            out[ql_off + l + 32] = (v2 & 0x0F) | ((v4 & 0x0F) << 4);
            out[ql_len + qh_off_h + l] =
                (v1 >> 4) | ((v2 >> 4) << 2) | ((v3 >> 4) << 4) | ((v4 >> 4) << 6);
        }
    }
    let sc_off = ql_len + qh_len;
    for j in 0..16 {
        out[sc_off + j] = sc[j] as u8;
    }
    out[sc_off + sc_len..sc_off + sc_len + 2].copy_from_slice(&f16_bytes(d_outer));
}

/// Quantize one Q8_K super-block (256 fp32 → 292 bytes).
pub fn quantize_q8_k_block(src: &[f32], out: &mut [u8]) {
    let blk = 4 + QK_K + (QK_K / 16) * 2;
    assert!(src.len() >= QK_K && out.len() >= blk);
    let amax = src[..QK_K].iter().fold(0f32, |a, &v| a.max(v.abs()));
    let d = amax / 127.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    out[0..4].copy_from_slice(&d.to_le_bytes());
    for j in 0..QK_K {
        let q = (nearest_i32(src[j] * id)).clamp(-128, 127) as i8;
        out[4 + j] = q as u8;
    }
    // bsums[16] = sum of qs in each 16-element group (i16, LE).
    for k in 0..QK_K / 16 {
        let mut s: i32 = 0;
        for l in 0..16 {
            s += out[4 + k * 16 + l] as i8 as i32;
        }
        let s = s.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        out[4 + QK_K + k * 2..4 + QK_K + k * 2 + 2].copy_from_slice(&s.to_le_bytes());
    }
}

/// Quantize one Q2_K super-block (256 fp32 → 84 bytes). Simplified
/// per-sub-block quantizer; valid output but lower quality than
/// upstream's iterative search.
pub fn quantize_q2_k_block(src: &[f32], out: &mut [u8]) {
    let blk = 2 + 2 + QK_K / 16 + QK_K / 4;
    assert!(src.len() >= QK_K && out.len() >= blk);
    // 16 sub-blocks of 16 elements; 2-bit quants with 4-bit per-sub-block
    // scale + 4-bit per-sub-block min, all multiplied by f16 d/min.
    let mut sub_d = [0f32; 16];
    let mut sub_min = [0f32; 16];
    for j in 0..16 {
        let sub = &src[j * 16..(j + 1) * 16];
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        for &v in sub {
            if v < mn {
                mn = v;
            }
            if v > mx {
                mx = v;
            }
        }
        if mn >= 0.0 {
            sub_d[j] = mx / 3.0;
            sub_min[j] = 0.0;
        } else {
            sub_d[j] = (mx - mn) / 3.0;
            sub_min[j] = -mn;
        }
    }
    let d_outer = sub_d.iter().cloned().fold(0f32, f32::max) / 15.0;
    let min_outer = sub_min.iter().cloned().fold(0f32, f32::max) / 15.0;
    let id = if d_outer != 0.0 { 1.0 / d_outer } else { 0.0 };
    let idm = if min_outer != 0.0 {
        1.0 / min_outer
    } else {
        0.0
    };
    let mut sc = [0u8; 16];
    for j in 0..16 {
        let s = (nearest_i32(sub_d[j] * id)).clamp(0, 15) as u8;
        let m = (nearest_i32(sub_min[j] * idm)).clamp(0, 15) as u8;
        sc[j] = s | (m << 4);
    }
    // Layout: scales[16] | qs[64] | d (f16) | dmin (f16) — total 84 bytes.
    out[0..16].copy_from_slice(&sc);
    let qs_off = QK_K / 16; // 16
    let d_off = qs_off + QK_K / 4; // 80
    out[d_off..d_off + 2].copy_from_slice(&f16_bytes(d_outer));
    out[d_off + 2..d_off + 4].copy_from_slice(&f16_bytes(min_outer));
    for b in &mut out[qs_off..qs_off + QK_K / 4] {
        *b = 0;
    }
    let mut sub_idx = 0usize;
    for h in 0..2 {
        let base_byte = qs_off + h * 32;
        for s_iter in 0..4 {
            let shift = (s_iter * 2) as u32;
            // sub-block A (low half): elements src[(sub_idx)*16 .. *16+16]
            let sub_a = &src[sub_idx * 16..(sub_idx + 1) * 16];
            let dl = d_outer * (sc[sub_idx] & 0x0F) as f32;
            let ml = min_outer * (sc[sub_idx] >> 4) as f32;
            let idla = if dl != 0.0 { 1.0 / dl } else { 0.0 };
            for l in 0..16 {
                let q = (nearest_i32((sub_a[l] + ml) * idla)).clamp(0, 3) as u8;
                out[base_byte + l] |= q << shift;
            }
            sub_idx += 1;
            let sub_b = &src[sub_idx * 16..(sub_idx + 1) * 16];
            let dl = d_outer * (sc[sub_idx] & 0x0F) as f32;
            let ml = min_outer * (sc[sub_idx] >> 4) as f32;
            let idlb = if dl != 0.0 { 1.0 / dl } else { 0.0 };
            for l in 0..16 {
                let q = (nearest_i32((sub_b[l] + ml) * idlb)).clamp(0, 3) as u8;
                out[base_byte + 16 + l] |= q << shift;
            }
            sub_idx += 1;
        }
    }
}

/// Quantize one Q3_K super-block. Simplified — symmetric 3-bit
/// quants with a single super-block scale per the 16 sub-blocks.
pub fn quantize_q3_k_block(src: &[f32], out: &mut [u8]) {
    let blk = 2 + K_SCALE_SIZE + QK_K / 8 + QK_K / 4;
    assert!(src.len() >= QK_K && out.len() >= blk);
    // Signed 3-bit; per-sub-block 6-bit signed scale; one super-scale d.
    let mut sub_scale = [0f32; 16];
    for j in 0..16 {
        let sub = &src[j * 16..(j + 1) * 16];
        let mut amax = 0f32;
        let mut maxv = 0f32;
        for &v in sub {
            if amax < v.abs() {
                amax = v.abs();
                maxv = v;
            }
        }
        sub_scale[j] = maxv / -4.0; // 3-bit signed range [-4, 3]
    }
    let amax = sub_scale.iter().fold(0f32, |a, &v| a.max(v.abs()));
    let d_outer = amax / 31.0;
    let id_outer = if d_outer != 0.0 { 1.0 / d_outer } else { 0.0 };
    let mut sc6 = [0i8; 16];
    for j in 0..16 {
        sc6[j] = (nearest_i32(sub_scale[j] * id_outer)).clamp(-32, 31) as i8;
    }
    // Q3_K layout: hmask[32] | qs[64] | scales[12] | d (f16). Total 110.
    let hm_off = 0usize;
    let qs_off = QK_K / 8; // 32
    let scales_off = qs_off + QK_K / 4; // 96
    let d_off = scales_off + K_SCALE_SIZE; // 108
    out[d_off..d_off + 2].copy_from_slice(&f16_bytes(d_outer));
    // Pack 16 × 6-bit unsigned scales (= signed scale + 32) into 12
    // bytes matching the decoder's `aux[]` reverse transform.
    let mut aux = [0u32; 3];
    for j in 0..16 {
        let s = (sc6[j] + 32) as u8 & 0x3F;
        let low4 = (s & 0x0F) as u32;
        let hi2 = ((s >> 4) & 0x03) as u32;
        let (aux_idx, low_shift, hi2_shift) = match j {
            0..=3 => (0, j * 8, j * 8),
            4..=7 => (1, (j - 4) * 8, (j - 4) * 8 + 2),
            8..=11 => (0, (j - 8) * 8 + 4, (j - 8) * 8 + 4),
            _ => (1, (j - 12) * 8 + 4, (j - 12) * 8 + 6),
        };
        aux[aux_idx] |= low4 << low_shift;
        aux[2] |= hi2 << hi2_shift;
    }
    out[scales_off..scales_off + 4].copy_from_slice(&aux[0].to_le_bytes());
    out[scales_off + 4..scales_off + 8].copy_from_slice(&aux[1].to_le_bytes());
    out[scales_off + 8..scales_off + 12].copy_from_slice(&aux[2].to_le_bytes());
    // Quantize values and pack: hm holds bit 2 of each 3-bit quant
    // (inverted: hm bit set = +0 shift, clear = +4 added back, per
    // decoder); qs holds the low 2 bits packed in groups.
    for b in &mut out[hm_off..hm_off + QK_K / 8] {
        *b = 0;
    }
    for b in &mut out[qs_off..qs_off + QK_K / 4] {
        *b = 0;
    }
    // Build per-element 3-bit signed q[i] ∈ [-4, 3].
    let mut q3 = [0i8; QK_K];
    for j in 0..16 {
        let s = d_outer * sc6[j] as f32;
        let isf = if s != 0.0 { 1.0 / s } else { 0.0 };
        for l in 0..16 {
            q3[j * 16 + l] = (nearest_i32(src[j * 16 + l] * isf)).clamp(-4, 3) as i8;
        }
    }
    // Pack matching the decoder. Reading dequant_q3_k_block:
    //   let h = if hm[l] & m != 0 { 0 } else { 4 };
    //   out = dl * (((q[l]>>shift)&3) as i8 - h);
    // Therefore for signed q3 ∈ [-4, 3]:
    //   q3 ≥ 0  ⇒  low2 = q3,       hm bit SET
    //   q3 < 0  ⇒  low2 = q3 + 4,   hm bit CLEAR
    //
    // qs layout (64 bytes total): per super-half h, bytes
    // [h*32 .. h*32+32]. Within a super-half, byte index uses
    //   `l` directly for sub_idx-even (decoder reads `q[l]`)
    //   `l + 16` for sub_idx-odd (decoder reads `q[l + 16]`)
    // and the shift selects which of 4 sub-block pairs that byte
    // holds — sub-blocks (h*8 + s*2 + which) use shift = s*2.
    let mut m: u8 = 1;
    let mut sub_idx = 0usize;
    for h in 0..2 {
        let base_byte = qs_off + h * 32;
        let mut shift = 0u32;
        for _s in 0..4 {
            for which in 0..2 {
                let l_base = which * 16;
                for l in 0..16 {
                    let v = q3[sub_idx * 16 + l] as i32;
                    let (low2, hm_bit) = if v >= 0 {
                        ((v as u8) & 3, true)
                    } else {
                        (((v + 4) as u8) & 3, false)
                    };
                    out[base_byte + l_base + l] |= low2 << shift;
                    if hm_bit {
                        out[hm_off + l_base + l] |= m;
                    }
                }
                sub_idx += 1;
            }
            shift += 2;
            m <<= 1;
        }
    }
}

/// Per-tensor encoder for any supported scheme. Element count must
/// divide the scheme's block size.
pub fn quantize(src: &[f32], dtype: GgmlType) -> Result<Vec<u8>> {
    match dtype {
        GgmlType::F32 => Ok(bytemuck::cast_slice(src).to_vec()),
        GgmlType::F16 => {
            let mut out = Vec::with_capacity(src.len() * 2);
            for &v in src {
                out.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
            Ok(out)
        }
        GgmlType::BF16 => {
            let mut out = Vec::with_capacity(src.len() * 2);
            for &v in src {
                out.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
            }
            Ok(out)
        }
        GgmlType::Q8_0 => quantize_q8_0(src),
        GgmlType::Q4_0 => quantize_q4_0(src),
        GgmlType::Q4_1 => quantize_q4_1(src),
        GgmlType::Q5_0 => quantize_q5_0(src),
        GgmlType::Q5_1 => quantize_q5_1(src),
        GgmlType::Q4K => block_quantize(
            src,
            "Q4_K",
            2 + 2 + K_SCALE_SIZE + QK_K / 2,
            quantize_q4_k_block,
        ),
        GgmlType::Q5K => block_quantize(
            src,
            "Q5_K",
            2 + 2 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2,
            quantize_q5_k_block,
        ),
        GgmlType::Q6K => block_quantize(
            src,
            "Q6_K",
            QK_K / 2 + QK_K / 4 + QK_K / 16 + 2,
            quantize_q6_k_block,
        ),
        GgmlType::Q8K => {
            block_quantize(src, "Q8_K", 4 + QK_K + (QK_K / 16) * 2, quantize_q8_k_block)
        }
        GgmlType::Q2K => block_quantize(
            src,
            "Q2_K",
            2 + 2 + QK_K / 16 + QK_K / 4,
            quantize_q2_k_block,
        ),
        GgmlType::Q3K => block_quantize(
            src,
            "Q3_K",
            2 + K_SCALE_SIZE + QK_K / 8 + QK_K / 4,
            quantize_q3_k_block,
        ),
        other => bail!("quantize: encoder for {other:?} not implemented"),
    }
}

fn block_quantize<F: FnMut(&[f32], &mut [u8])>(
    src: &[f32],
    name: &str,
    blk: usize,
    mut f: F,
) -> Result<Vec<u8>> {
    check_div(name, src.len(), QK_K)?;
    let nb = src.len() / QK_K;
    let mut out = vec![0u8; nb * blk];
    for i in 0..nb {
        f(
            &src[i * QK_K..(i + 1) * QK_K],
            &mut out[i * blk..(i + 1) * blk],
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dequant_q2_k, dequant_q3_k, dequant_q4_0, dequant_q4_1, dequant_q4_k, dequant_q5_0,
        dequant_q5_1, dequant_q5_k, dequant_q6_k, dequant_q8_0, dequant_q8_k,
    };

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            dot / (na * nb)
        }
    }

    fn synth(n: usize, seed: u32) -> Vec<f32> {
        // Tiny deterministic LCG — avoids a `rand` dep in this crate.
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s as i32) as f32 / 2.147e9) * 2.0
            })
            .collect()
    }

    #[test]
    fn q8_0_roundtrip_cosine() {
        let x = synth(QK8_0 * 8, 42);
        let q = quantize_q8_0(&x).unwrap();
        let r = dequant_q8_0(&q, x.len()).unwrap();
        assert!(cosine(&x, &r) > 0.999, "cosine {}", cosine(&x, &r));
    }

    #[test]
    fn q4_0_roundtrip_cosine() {
        let x = synth(QK4_0 * 8, 7);
        let q = quantize_q4_0(&x).unwrap();
        let r = dequant_q4_0(&q, x.len()).unwrap();
        assert!(cosine(&x, &r) > 0.98, "cosine {}", cosine(&x, &r));
    }

    #[test]
    fn q4_1_roundtrip_cosine() {
        let x = synth(QK4_1 * 8, 11);
        let q = quantize_q4_1(&x).unwrap();
        let r = dequant_q4_1(&q, x.len()).unwrap();
        assert!(cosine(&x, &r) > 0.99, "cosine {}", cosine(&x, &r));
    }

    #[test]
    fn q5_0_roundtrip_cosine() {
        let x = synth(QK5_0 * 8, 13);
        let q = quantize_q5_0(&x).unwrap();
        let r = dequant_q5_0(&q, x.len()).unwrap();
        assert!(cosine(&x, &r) > 0.995, "cosine {}", cosine(&x, &r));
    }

    #[test]
    fn q5_1_roundtrip_cosine() {
        let x = synth(QK5_1 * 8, 15);
        let q = quantize_q5_1(&x).unwrap();
        let r = dequant_q5_1(&q, x.len()).unwrap();
        assert!(cosine(&x, &r) > 0.998, "cosine {}", cosine(&x, &r));
    }

    #[test]
    fn q8_k_roundtrip_cosine() {
        let x = synth(QK_K * 2, 23);
        let q = quantize(&x, GgmlType::Q8K).unwrap();
        let r = dequant_q8_k(&q, x.len()).unwrap();
        assert!(cosine(&x, &r) > 0.9999, "cosine {}", cosine(&x, &r));
    }

    #[test]
    fn q4_k_roundtrip_cosine() {
        let x = synth(QK_K * 2, 31);
        let q = quantize(&x, GgmlType::Q4K).unwrap();
        let r = dequant_q4_k(&q, x.len()).unwrap();
        assert!(cosine(&x, &r) > 0.99, "cosine {}", cosine(&x, &r));
    }

    #[test]
    fn q5_k_roundtrip_cosine() {
        let x = synth(QK_K * 2, 33);
        let q = quantize(&x, GgmlType::Q5K).unwrap();
        let r = dequant_q5_k(&q, x.len()).unwrap();
        assert!(cosine(&x, &r) > 0.995, "cosine {}", cosine(&x, &r));
    }

    #[test]
    fn q6_k_roundtrip_cosine() {
        let x = synth(QK_K * 2, 37);
        let q = quantize(&x, GgmlType::Q6K).unwrap();
        let r = dequant_q6_k(&q, x.len()).unwrap();
        assert!(cosine(&x, &r) > 0.998, "cosine {}", cosine(&x, &r));
    }

    #[test]
    fn q2_k_roundtrip_cosine() {
        let x = synth(QK_K * 2, 41);
        let q = quantize(&x, GgmlType::Q2K).unwrap();
        let r = dequant_q2_k(&q, x.len()).unwrap();
        assert!(cosine(&x, &r) > 0.9, "cosine {}", cosine(&x, &r));
    }

    #[test]
    fn q3_k_roundtrip_cosine() {
        let x = synth(QK_K * 2, 43);
        let q = quantize(&x, GgmlType::Q3K).unwrap();
        let r = dequant_q3_k(&q, x.len()).unwrap();
        assert!(cosine(&x, &r) > 0.95, "cosine {}", cosine(&x, &r));
    }
}
