// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Multi-level residual quantization** — the low-precision analog of
//! double-word extended precision.
//!
//! Double-word (a `hi+lo` pair of a wider float) can't extend FP4/NVF4: with
//! only 2 exponent bits the `lo` residual underflows. The working "deconstruct
//! into a sum of representable components" for low precision is instead
//! **residual quantization on top of MX block scaling**:
//!
//! ```text
//!   value ≈ s0·dequant(q0)                    (1 level — plain MXFP4/NVFP4)
//!   value ≈ s0·dequant(q0) + s1·dequant(q1)   (2 levels — this module)
//! ```
//!
//! Each level quantizes the *residual* of the previous one with its own block
//! scale. This reuses rlx's real FP-code codec ([`ScaledFormat`], e.g.
//! [`ScaledFormat::F4E2M1`]) and the same per-block scaling `ScaledMatMul`
//! already uses — a second level roughly doubles the effective mantissa at 2×
//! storage. `MxFp4x2` = `F4E2M1` with `levels = 2`.

use crate::quant::{QuantScheme, ScaledFormat};

/// A block quantized to `levels` residual FP-code levels: a per-level block
/// `scale` plus one code per element per level.
#[derive(Clone, Debug)]
pub struct ResidualBlock {
    /// One block scale per level (`scales[k]`).
    pub scales: Vec<f32>,
    /// `codes[level][element]` — the FP-code (e.g. E2M1 nibble) at each level.
    pub codes: Vec<Vec<u8>>,
    /// The per-element numeric format (shared across levels).
    pub format: ScaledFormat,
}

/// Block scale mapping the block's max-abs onto the format's max magnitude
/// (the MX / `ScaledMatMul` convention).
fn block_scale(block: &[f32], fmt: ScaledFormat) -> f32 {
    let amax = block.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let fmax = fmt
        .representable_values()
        .iter()
        .fold(0.0f32, |m, &x| m.max(x.abs()));
    (amax / fmax).max(f32::MIN_POSITIVE)
}

/// Quantize `block` to `levels` residual levels of `fmt` (`levels = 1` is plain
/// MXFP4/NVFP4; `MxFp4x2` is `ScaledFormat::F4E2M1` with `levels = 2`).
pub fn residual_quantize(block: &[f32], fmt: ScaledFormat, levels: usize) -> ResidualBlock {
    let mut resid: Vec<f32> = block.to_vec();
    let mut scales = Vec::with_capacity(levels);
    let mut codes = Vec::with_capacity(levels);
    for _ in 0..levels {
        let s = block_scale(&resid, fmt);
        let level_codes: Vec<u8> = resid.iter().map(|&v| fmt.encode(v / s)).collect();
        for (r, &c) in resid.iter_mut().zip(&level_codes) {
            *r -= s * fmt.decode(c);
        }
        scales.push(s);
        codes.push(level_codes);
    }
    ResidualBlock {
        scales,
        codes,
        format: fmt,
    }
}

/// Reconstruct `Σ_k s_k · dequant(q_k)` — the value the decode-GEMM would
/// accumulate.
pub fn residual_dequantize(rb: &ResidualBlock) -> Vec<f32> {
    let n = rb.codes.first().map(|c| c.len()).unwrap_or(0);
    let mut out = vec![0.0f32; n];
    for (level, codes) in rb.codes.iter().enumerate() {
        let s = rb.scales[level];
        for (o, &c) in out.iter_mut().zip(codes) {
            *o += s * rb.format.decode(c);
        }
    }
    out
}

/// Convenience: round-trip a block through `levels` residual levels of `fmt`.
pub fn residual_roundtrip(block: &[f32], fmt: ScaledFormat, levels: usize) -> Vec<f32> {
    residual_dequantize(&residual_quantize(block, fmt, levels))
}

/// Dequantize every `block`-length chunk of each `cols`-wide row through
/// `levels` residual levels — the operand a `MxFp4x2` `ScaledMatMul` decodes
/// (MX block scaling runs along the contraction dimension).
fn dequant_rows(data: &[f32], rows: usize, cols: usize, fmt: ScaledFormat, levels: usize, block: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let base = r * cols;
        let mut c0 = 0;
        while c0 < cols {
            let c1 = (c0 + block).min(cols);
            let de = residual_roundtrip(&data[base + c0..base + c1], fmt, levels);
            out[base + c0..base + c1].copy_from_slice(&de);
            c0 = c1;
        }
    }
    out
}

/// Round-trip each block of every row per a `QuantScheme::MxFp4x2Block`
/// (`group_size` = the K-block). Returns `None` for non-residual schemes — the
/// bridge from the scheme in the type system to this codec.
pub fn scheme_roundtrip_rows(
    scheme: QuantScheme,
    data: &[f32],
    rows: usize,
    cols: usize,
) -> Option<Vec<f32>> {
    let (fmt, levels, group) = scheme.mxfp4x2_config()?;
    Some(dequant_rows(data, rows, cols, fmt, levels as usize, group as usize))
}

/// Reference `MxFp4x2` decode-GEMM: `C[m,n] = A[m,k] · B[k,n]` with both
/// operands residual-quantized (K-blocked, `levels` levels of `fmt`) then
/// decoded — exactly what a two-level scaled-matmul kernel accumulates. This is
/// the CPU oracle; the GPU decode path (`scaled_lowp_general`) is a follow-up.
pub fn residual_matmul_ref(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    fmt: ScaledFormat,
    levels: usize,
    block: usize,
) -> Vec<f32> {
    // A rows are K-vectors; B columns are K-vectors → quantize Bᵀ rows.
    let aq = dequant_rows(a, m, k, fmt, levels, block);
    let mut bt = vec![0.0f32; n * k];
    for i in 0..k {
        for j in 0..n {
            bt[j * k + i] = b[i * n + j];
        }
    }
    let bq = dequant_rows(&bt, n, k, fmt, levels, block);
    let mut c = vec![0.0f32; m * n];
    for r in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for i in 0..k {
                s += aq[r * k + i] * bq[j * k + i];
            }
            c[r * n + j] = s;
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms_rel(orig: &[f32], approx: &[f32]) -> f64 {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (&o, &a) in orig.iter().zip(approx) {
            num += (o - a).powi(2) as f64;
            den += (o as f64).powi(2);
        }
        (num / den.max(1e-30)).sqrt()
    }

    /// A second FP4 residual level roughly doubles the effective mantissa —
    /// element reconstruction AND a dot product (the `ScaledMatMul` case).
    #[test]
    fn residual_fp4_doubles_precision() {
        let fmt = ScaledFormat::F4E2M1;
        const BLOCK: usize = 32;
        const BLOCKS: usize = 5_000;

        // Deterministic N(0,1) samples.
        let mut s = 0x1234_5678_9abc_def0u64;
        let mut u = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f32 / (1u64 << 53) as f32
        };
        let mut normal = || {
            let (a, b) = (u().max(1e-7), u());
            (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
        };

        let (mut e1, mut e2) = (0.0f64, 0.0f64);
        let (mut dot_ref, mut dot1, mut dot2) = (0.0f64, 0.0f64, 0.0f64);
        for _ in 0..BLOCKS {
            let a: Vec<f32> = (0..BLOCK).map(|_| normal()).collect();
            let b: Vec<f32> = (0..BLOCK).map(|_| normal()).collect();
            let a1 = residual_roundtrip(&a, fmt, 1);
            let a2 = residual_roundtrip(&a, fmt, 2);
            let b1 = residual_roundtrip(&b, fmt, 1);
            let b2 = residual_roundtrip(&b, fmt, 2);
            e1 += rms_rel(&a, &a1);
            e2 += rms_rel(&a, &a2);
            for i in 0..BLOCK {
                dot_ref += (a[i] * b[i]) as f64;
                dot1 += (a1[i] * b1[i]) as f64;
                dot2 += (a2[i] * b2[i]) as f64;
            }
        }
        e1 /= BLOCKS as f64;
        e2 /= BLOCKS as f64;
        let bits = |e: f64| -e.log2();
        eprintln!(
            "MXFP4 1-level RMS={e1:.3e} ({:.1} bits) | 2-level RMS={e2:.3e} ({:.1} bits) | {:.1}x",
            bits(e1),
            bits(e2),
            e1 / e2
        );
        eprintln!(
            "dot: 1-level err={:.2e}  2-level err={:.2e}",
            ((dot1 - dot_ref) / dot_ref).abs(),
            ((dot2 - dot_ref) / dot_ref).abs()
        );

        // 1 level is coarse FP4 (~3 bits); a residual level roughly doubles it.
        assert!(e1 > 5e-2, "1-level FP4 should be coarse, got {e1:e}");
        assert!(e2 < e1 / 4.0, "2-level should be >=4x better, got {e2:e} vs {e1:e}");
        assert!(bits(e2) > bits(e1) + 2.0, "2-level should add >2 mantissa bits");
    }

    /// Level 0 alone equals plain single-level MXFP4 (no regression to the
    /// existing path).
    #[test]
    fn one_level_is_plain_mxfp4() {
        let fmt = ScaledFormat::F4E2M1;
        let block = [0.0, 0.3, -1.7, 4.9, -6.2, 2.1, 0.05, -3.3];
        let rb = residual_quantize(&block, fmt, 1);
        let de = residual_dequantize(&rb);
        // Each element is scale · nearest-E2M1(value/scale).
        let s = rb.scales[0];
        for (i, &v) in block.iter().enumerate() {
            assert_eq!(de[i], s * fmt.quantize(v / s));
        }
    }

    /// A `MxFp4x2` (2-level) decode-GEMM is substantially closer to the f32
    /// matmul than plain single-level MXFP4.
    #[test]
    fn mxfp4x2_matmul_beats_single_level() {
        let fmt = ScaledFormat::F4E2M1;
        let (m, k, n, block) = (48usize, 96usize, 48usize, 32usize);
        let mut s = 0xdead_beef_cafe_babeu64;
        let mut u = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f32 / (1u64 << 53) as f32
        };
        let mut normal = || {
            let (a, b) = (u().max(1e-7), u());
            (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
        };
        let a: Vec<f32> = (0..m * k).map(|_| normal()).collect();
        let b: Vec<f32> = (0..k * n).map(|_| normal()).collect();
        let mut cf = vec![0.0f32; m * n];
        for r in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for i in 0..k {
                    acc += a[r * k + i] * b[i * n + j];
                }
                cf[r * n + j] = acc;
            }
        }
        let c1 = residual_matmul_ref(&a, &b, m, k, n, fmt, 1, block);
        let c2 = residual_matmul_ref(&a, &b, m, k, n, fmt, 2, block);
        let (e1, e2) = (rms_rel(&cf, &c1), rms_rel(&cf, &c2));
        eprintln!(
            "MxFp4 decode-GEMM: 1-level err={e1:.2e}  2-level err={e2:.2e}  ({:.1}x better)",
            e1 / e2
        );
        assert!(e2 < e1 / 2.0, "2-level matmul must beat 1-level: {e2:e} vs {e1:e}");
    }

    /// The first-class `QuantScheme::MxFp4x2Block` drives the residual codec and
    /// beats plain MXFP4 (single level) on the same data.
    #[test]
    fn quant_scheme_mxfp4x2_roundtrips() {
        let scheme = QuantScheme::MxFp4x2Block { group_size: 32 };
        assert_eq!(
            scheme.mxfp4x2_config(),
            Some((ScaledFormat::F4E2M1, 2, 32))
        );
        assert!(scheme.has_scale());
        assert_eq!(scheme.to_string(), "mxfp4x2/32");
        assert!(QuantScheme::MlxMxfp4 { group_size: 32 }.mxfp4x2_config().is_none());

        let (rows, cols) = (8usize, 64usize);
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.137).sin() * 3.0)
            .collect();
        let two = scheme_roundtrip_rows(scheme, &data, rows, cols).expect("mxfp4x2");
        let one = dequant_rows(&data, rows, cols, ScaledFormat::F4E2M1, 1, 32);
        let (e2, e1) = (rms_rel(&data, &two), rms_rel(&data, &one));
        eprintln!("QuantScheme::MxFp4x2Block: 1-level err={e1:.2e}  2-level err={e2:.2e}");
        assert!(e2 < e1 / 2.0, "MxFp4x2 must beat single-level MXFP4");
    }
}
