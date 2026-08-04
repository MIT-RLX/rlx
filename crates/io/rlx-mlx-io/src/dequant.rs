// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host dequant for MLX affine / mxfp packs (matches MLX CPU kernels).

use anyhow::{Result, bail};

/// How many values are packed into one storage unit for power-of-two bits
/// (MLX `get_pack_factor(bits, 8)` for bits ∈ {2,4,8}; odd bits use special packs).
pub fn pack_factor(bits: u32) -> Result<u32> {
    match bits {
        2 | 4 | 8 => Ok(8 / bits),
        3 | 5 => Ok(8),
        6 => Ok(4),
        other => bail!("unsupported MLX affine bits={other}"),
    }
}

/// Validate `k` / `group_size` / weight byte length before arena slicing.
pub fn validate_dequant_matmul_dims(
    scheme: rlx_ir::QuantScheme,
    k: usize,
    n: usize,
    w_len: Option<usize>,
) -> Result<()> {
    use rlx_ir::QuantScheme;
    let gs = scheme.mlx_group_size() as usize;
    if gs == 0 {
        bail!("not an MLX QuantScheme: {scheme}");
    }
    if !k.is_multiple_of(gs) {
        bail!("MLX DequantMatMul: k={k} not divisible by group_size={gs}");
    }
    let n_groups = k / gs;
    if let Some(w_len) = w_len {
        let need = match scheme {
            QuantScheme::MlxAffine { bits, .. } => {
                let pf = pack_factor(bits as u32)? as usize;
                let packs = gs / pf;
                let bpp = bytes_per_pack(bits as u32)?;
                n * n_groups * packs * bpp
            }
            QuantScheme::MlxMxfp4 { .. } => n * k / 2,
            QuantScheme::MlxMxfp8 { .. } => n * k,
            _ => bail!("not an MLX QuantScheme: {scheme}"),
        };
        if w_len < need {
            bail!("MLX DequantMatMul: weight bytes {w_len} < needed {need}");
        }
    }
    Ok(())
}

fn bytes_per_pack(bits: u32) -> Result<usize> {
    match bits {
        2 | 4 | 8 => Ok(1),
        3 => Ok(3),
        5 => Ok(5),
        6 => Ok(3),
        other => bail!("unsupported MLX affine bits={other}"),
    }
}

/// One quantized Linear / Embedding triple as stored by mlx-lm.
#[derive(Debug, Clone)]
pub struct QuantizedLayer {
    pub weight: Vec<u8>,
    pub weight_shape: Vec<usize>,
    pub scales: Vec<f32>,
    pub scales_shape: Vec<usize>,
    pub biases: Option<Vec<f32>>,
    pub biases_shape: Option<Vec<usize>>,
    pub bits: u32,
    pub group_size: u32,
}

fn extract_bits_3(w_in: &[u8], out: &mut [u8; 8]) {
    out[0] = w_in[0] & 0x7;
    out[1] = (w_in[0] & 0x38) >> 3;
    out[2] = ((w_in[0] & 0xc0) >> 6) + ((w_in[1] & 0x1) << 2);
    out[3] = (w_in[1] & 0xe) >> 1;
    out[4] = (w_in[1] & 0x70) >> 4;
    out[5] = ((w_in[1] & 0x80) >> 7) + ((w_in[2] & 0x3) << 1);
    out[6] = (w_in[2] & 0x1c) >> 2;
    out[7] = (w_in[2] & 0xe0) >> 5;
}

fn extract_bits_5(w_in: &[u8], out: &mut [u8; 8]) {
    out[0] = w_in[0] & 0x1f;
    out[1] = ((w_in[0] & 0xe0) >> 5) + ((w_in[1] & 0x3) << 3);
    out[2] = (w_in[1] & 0x7c) >> 2;
    out[3] = ((w_in[1] & 0x80) >> 7) + ((w_in[2] & 0xf) << 1);
    out[4] = ((w_in[2] & 0xf0) >> 4) + ((w_in[3] & 0x1) << 4);
    out[5] = (w_in[3] & 0x3e) >> 1;
    out[6] = ((w_in[3] & 0xc0) >> 6) + ((w_in[4] & 0x7) << 2);
    out[7] = (w_in[4] & 0xf8) >> 3;
}

fn extract_bits_6(w_in: &[u8], out: &mut [u8; 4]) {
    out[0] = w_in[0] & 0x3f;
    out[1] = ((w_in[0] >> 6) & 0x03) + ((w_in[1] & 0x0f) << 2);
    out[2] = ((w_in[1] >> 4) & 0x0f) + ((w_in[2] & 0x03) << 4);
    out[3] = (w_in[2] >> 2) & 0x3f;
}

/// Dequantize MLX affine packs to a dense `[rows, cols]` f32 matrix.
///
/// Layout matches MLX `_qmm_t` / `affine_dequantize`: rows = `scales_shape[0]`,
/// cols = `n_groups * group_size`, weights stored row-major packed along K.
pub fn dequant_affine_f32(
    w: &[u8],
    scales: &[f32],
    biases: &[f32],
    bits: u32,
    group_size: u32,
    rows: usize,
    n_groups: usize,
) -> Result<Vec<f32>> {
    let gs = group_size as usize;
    let cols = n_groups * gs;
    let pf = pack_factor(bits)? as usize;
    let bpp = bytes_per_pack(bits)?;
    let packs_in_group = gs / pf;
    let bitmask = (1u32 << bits) - 1;
    if scales.len() < rows * n_groups || biases.len() < rows * n_groups {
        bail!(
            "affine dequant: scales/biases too short (need {} each)",
            rows * n_groups
        );
    }
    let need_w = rows * n_groups * packs_in_group * bpp;
    if w.len() < need_w {
        bail!("affine dequant: weight bytes {} < needed {need_w}", w.len());
    }

    let mut out = vec![0f32; rows * cols];
    let mut w_off = 0usize;
    for r in 0..rows {
        for g in 0..n_groups {
            let scale = scales[r * n_groups + g];
            let bias = biases[r * n_groups + g];
            let base = r * cols + g * gs;
            let mut p = 0usize;
            for _ in 0..packs_in_group {
                match bits {
                    3 => {
                        let mut codes = [0u8; 8];
                        extract_bits_3(&w[w_off..w_off + 3], &mut codes);
                        for c in codes {
                            out[base + p] = scale * (c as f32) + bias;
                            p += 1;
                        }
                        w_off += 3;
                    }
                    5 => {
                        let mut codes = [0u8; 8];
                        extract_bits_5(&w[w_off..w_off + 5], &mut codes);
                        for c in codes {
                            out[base + p] = scale * (c as f32) + bias;
                            p += 1;
                        }
                        w_off += 5;
                    }
                    6 => {
                        let mut codes = [0u8; 4];
                        extract_bits_6(&w[w_off..w_off + 3], &mut codes);
                        for c in codes {
                            out[base + p] = scale * (c as f32) + bias;
                            p += 1;
                        }
                        w_off += 3;
                    }
                    2 | 4 | 8 => {
                        let mut wi = w[w_off];
                        w_off += 1;
                        for _ in 0..pf {
                            let code = (wi as u32) & bitmask;
                            out[base + p] = scale * (code as f32) + bias;
                            p += 1;
                            if bits != 8 {
                                wi >>= bits as u8;
                            }
                        }
                    }
                    other => bail!("unsupported bits {other}"),
                }
            }
        }
    }
    Ok(out)
}

/// Fused affine dequant-**matvec**: `y[0..n] = x[0..k] · dequant(w)^T` for a
/// SINGLE input row, WITHOUT materializing the f32 weight. The materialize path
/// ([`dequant_matmul_affine`]) writes an `n×k` f32 buffer (≈16× the packed 2-bit
/// size) and reads it straight back — pure memory traffic that dominates MoE
/// prefill, where every token dequantizes a whole expert weight. This reads the
/// packed codes once and accumulates in the SAME k-order (bit-exact with the
/// materialize path), parallelized across the `n` output features.
///
/// `w` is one expert's packed weight `[n, k]`; `scales`/`biases` are `[n,
/// n_groups]`. Use for `m == 1` (the per-token MoE expert dispatch).
#[allow(clippy::too_many_arguments)]
pub fn dequant_matvec_affine(
    x: &[f32],
    w: &[u8],
    scales: &[f32],
    biases: &[f32],
    bits: u32,
    group_size: u32,
    k: usize,
    n: usize,
) -> Result<Vec<f32>> {
    use rayon::prelude::*;
    let gs = group_size as usize;
    if !k.is_multiple_of(gs) {
        bail!("affine matvec: k={k} not divisible by group_size={gs}");
    }
    let n_groups = k / gs;
    let pf = pack_factor(bits)? as usize; // bails on unsupported bits
    let bpp = bytes_per_pack(bits)?;
    let packs_in_group = gs / pf;
    let bitmask = (1u32 << bits) - 1;
    let row_bytes = n_groups * packs_in_group * bpp;
    if w.len() < n * row_bytes {
        bail!(
            "affine matvec: weight bytes {} < needed {}",
            w.len(),
            n * row_bytes
        );
    }
    if scales.len() < n * n_groups || biases.len() < n * n_groups {
        bail!(
            "affine matvec: scales/biases too short (need {} each)",
            n * n_groups
        );
    }
    if x.len() < k {
        bail!("affine matvec: x len {} < k {k}", x.len());
    }
    let mut out = vec![0f32; n];
    out.par_iter_mut().enumerate().for_each(|(r, y)| {
        let wrow = &w[r * row_bytes..(r + 1) * row_bytes];
        let mut acc = 0f32;
        let mut w_off = 0usize;
        for grp in 0..n_groups {
            let scale = scales[r * n_groups + grp];
            let bias = biases[r * n_groups + grp];
            let xbase = grp * gs;
            let mut p = 0usize;
            for _ in 0..packs_in_group {
                match bits {
                    2 | 4 | 8 => {
                        let mut wi = wrow[w_off];
                        w_off += 1;
                        for _ in 0..pf {
                            let code = (wi as u32) & bitmask;
                            acc += x[xbase + p] * (scale * (code as f32) + bias);
                            p += 1;
                            if bits != 8 {
                                wi >>= bits as u8;
                            }
                        }
                    }
                    3 => {
                        let mut codes = [0u8; 8];
                        extract_bits_3(&wrow[w_off..w_off + 3], &mut codes);
                        for c in codes {
                            acc += x[xbase + p] * (scale * (c as f32) + bias);
                            p += 1;
                        }
                        w_off += 3;
                    }
                    5 => {
                        let mut codes = [0u8; 8];
                        extract_bits_5(&wrow[w_off..w_off + 5], &mut codes);
                        for c in codes {
                            acc += x[xbase + p] * (scale * (c as f32) + bias);
                            p += 1;
                        }
                        w_off += 5;
                    }
                    6 => {
                        let mut codes = [0u8; 4];
                        extract_bits_6(&wrow[w_off..w_off + 3], &mut codes);
                        for c in codes {
                            acc += x[xbase + p] * (scale * (c as f32) + bias);
                            p += 1;
                        }
                        w_off += 3;
                    }
                    _ => unreachable!("bits validated by pack_factor"),
                }
            }
        }
        *y = acc;
    });
    Ok(out)
}

/// Fused affine dequant + matmul: `x [m,k] @ w_dequant^T` when weights are
/// stored `[n, k]` (MLX Linear). Output `[m, n]`.
#[allow(clippy::too_many_arguments)]
pub fn dequant_matmul_affine(
    x: &[f32],
    w: &[u8],
    scales: &[f32],
    biases: &[f32],
    bits: u32,
    group_size: u32,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>> {
    let gs = group_size as usize;
    if !k.is_multiple_of(gs) {
        bail!("affine DequantMatMul: k={k} not divisible by group_size={gs}");
    }
    let n_groups = k / gs;
    let w_f = dequant_affine_f32(w, scales, biases, bits, group_size, n, n_groups)?;
    // w_f is [n, k]; compute x @ w_f^T → [m, n]. This is the CPU host-delegate
    // matmul for GPU backends (amd Vulkan / cuda / wgpu copy quantized weights back
    // and run it here) — it was a naive single-threaded triple loop, which made a
    // Vulkan stage's attention + shared-expert projections run ~16× slower than a
    // native CPU stage. Parallelize over output columns `j` (n is large; m is the
    // token count, small) into a transposed buffer so writes stay disjoint, then
    // transpose back. Bit-identical to the serial loop (same accumulation order).
    use rayon::prelude::*;
    let mut out_t = vec![0f32; n * m]; // [n, m], column j-major
    out_t.par_chunks_mut(m).enumerate().for_each(|(j, col)| {
        let wj = &w_f[j * k..(j + 1) * k];
        for (i, slot) in col.iter_mut().enumerate() {
            let xi = &x[i * k..(i + 1) * k];
            let mut acc = 0f32;
            for p in 0..k {
                acc += xi[p] * wj[p];
            }
            *slot = acc;
        }
    });
    let mut out = vec![0f32; m * n];
    for j in 0..n {
        for i in 0..m {
            out[i * n + j] = out_t[j * m + i];
        }
    }
    Ok(out)
}

/// MXFP4 E2M1 LUT (MLX CPU `FP4_LUT`).
const FP4_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// `x @ W^T` for one MXFP4-packed weight `W` (`[n, k]` nibbles) with **already
/// f32-decoded** per-group `scales` (`[n, n_groups]`). Mirrors
/// [`dequant_matmul_affine`] for the grouped-MoE op path: the E8M0/FP8 group
/// scales are decoded to f32 by the loader, so this stays scale-dtype-agnostic
/// and only differs from affine by the FP4 LUT code decode (and no zero-point).
/// `x`=`[m, k]`, returns `[m, n]`.
pub fn dequant_matmul_mxfp4(
    x: &[f32],
    w: &[u8],
    scales: &[f32],
    group_size: u32,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>> {
    let gs = group_size as usize;
    if gs == 0 || !k.is_multiple_of(gs) {
        bail!("mxfp4 DequantMatMul: k={k} not divisible by group_size={gs}");
    }
    let n_groups = k / gs;
    if w.len() < n * k / 2 {
        bail!(
            "mxfp4 DequantMatMul: weight bytes {} < {}",
            w.len(),
            n * k / 2
        );
    }
    if scales.len() < n * n_groups {
        bail!(
            "mxfp4 DequantMatMul: scales len {} < {}",
            scales.len(),
            n * n_groups
        );
    }
    // Dequantize W → [n, k] F32 (2 nibbles/byte, contiguous per row).
    let mut w_f = vec![0f32; n * k];
    let mut w_off = 0usize;
    for r in 0..n {
        for gidx in 0..n_groups {
            let scale = scales[r * n_groups + gidx];
            let base = r * k + gidx * gs;
            for p in (0..gs).step_by(2) {
                let b = w[w_off];
                w_off += 1;
                w_f[base + p] = FP4_LUT[(b & 0x0f) as usize] * scale;
                w_f[base + p + 1] = FP4_LUT[(b >> 4) as usize] * scale;
            }
        }
    }
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                acc += x[i * k + p] * w_f[j * k + p];
            }
            out[i * n + j] = acc;
        }
    }
    Ok(out)
}

/// **FUSED GROUPED MXFP4 matmul** — the MoE analogue of [`dequant_matvec_mxfp4`]:
/// `out[r] = x[r] @ dequant(W_{e(r)})ᵀ` for `m` routed rows, each picking expert
/// `e(r) = idx[r]`. Decodes the packed e2m1 codes inline into the accumulate — NO
/// per-row `[n,k]` f32 weight materialization (the old grouped path called
/// `dequant_matmul_mxfp4` with m=1 per row, allocating + decoding ~`n·k·4` bytes of
/// f32 weight FOR EVERY TOKEN). Parallel over ALL `m·n` outputs, so it saturates cores
/// even when `m` is tiny (decode) — the per-row-parallel form left most cores idle.
/// `scales` = decoded f32 `[num_experts, n, n_groups]`; `idx` = `[m]` f32 expert ids;
/// writes `out` = `[m, n]`. Same accumulation order as [`dequant_matvec_mxfp4`].
#[allow(clippy::too_many_arguments)]
pub fn grouped_matmul_mxfp4_bt(
    x: &[f32],
    w_bytes: &[u8],
    scales: &[f32],
    idx: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    num_experts: usize,
    group_size: usize,
) {
    use rayon::prelude::*;
    debug_assert_eq!(out.len(), m * n, "output buffer must be m×n");
    let gs = group_size.max(1);
    let n_groups = k / gs;
    let slab = w_bytes.len() / num_experts.max(1); // n·k/2 packed bytes / expert
    let sb = n * n_groups; // scale f32 / expert
    let bpr = k / 2; // packed bytes / output row (within a slab)
    out.par_iter_mut().enumerate().for_each(|(flat, o)| {
        let r = flat / n;
        let j = flat % n;
        let e = (idx[r] as usize).min(num_experts.saturating_sub(1));
        let w_slab = &w_bytes[e * slab..(e + 1) * slab];
        let s_base = e * sb + j * n_groups;
        let row = &x[r * k..(r + 1) * k];
        let mut acc = 0f32;
        let mut w_off = j * bpr;
        for g in 0..n_groups {
            let scale = scales[s_base + g];
            let base = g * gs;
            let mut p = 0;
            while p < gs {
                let b = w_slab[w_off];
                w_off += 1;
                acc += row[base + p] * FP4_LUT[(b & 0x0f) as usize] * scale;
                acc += row[base + p + 1] * FP4_LUT[(b >> 4) as usize] * scale;
                p += 2;
            }
        }
        *o = acc;
    });
}

/// **FUSED MXFP4 matvec**: `out[n] = x[k] @ dequant(W)ᵀ` reading the packed e2m1
/// codes ONCE and accumulating in-place — NO `[n,k]` f32 materialization (the
/// `dequant_matmul_mxfp4` path allocs + reads `n·k·4` bytes; this reads the packed
/// `n·k/2` bytes only, ~8× less traffic + no big alloc). Rows parallelized. `scales`
/// = decoded f32 `[n, k/group_size]`. Bit-exact with the materialize path.
pub fn dequant_matvec_mxfp4(
    x: &[f32],
    w: &[u8],
    scales: &[f32],
    group_size: u32,
    k: usize,
    n: usize,
) -> Result<Vec<f32>> {
    use rayon::prelude::*;
    let gs = group_size as usize;
    if gs == 0 || !k.is_multiple_of(gs) {
        bail!("mxfp4 matvec: k={k} not divisible by group_size={gs}");
    }
    let n_groups = k / gs;
    if w.len() < n * k / 2 {
        bail!("mxfp4 matvec: weight bytes {} < {}", w.len(), n * k / 2);
    }
    if scales.len() < n * n_groups {
        bail!("mxfp4 matvec: scales {} < {}", scales.len(), n * n_groups);
    }
    let bpr = k / 2; // packed bytes per output row
    let mut out = vec![0f32; n];
    out.par_iter_mut().enumerate().for_each(|(j, o)| {
        let mut acc = 0f32;
        let mut w_off = j * bpr;
        for g in 0..n_groups {
            let scale = scales[j * n_groups + g];
            let base = g * gs;
            for p in (0..gs).step_by(2) {
                let b = w[w_off];
                w_off += 1;
                acc += x[base + p] * FP4_LUT[(b & 0x0f) as usize] * scale;
                acc += x[base + p + 1] * FP4_LUT[(b >> 4) as usize] * scale;
            }
        }
        *o = acc;
    });
    Ok(out)
}

/// Decode an MLX E8M0 uint8 group scale to f32 (public for grouped-MoE loaders
/// that pre-convert MXFP4 scales so the matmul op stays f32-uniform).
pub fn mxfp4_scale_e8m0_to_f32(s: u8) -> f32 {
    dequant_scale_e8m0(s)
}

/// Decode MLX E8M0-style uint8 scale used for group_size != 16.
fn dequant_scale_e8m0(s: u8) -> f32 {
    if s == 0 {
        return half::bf16::from_bits(0x40).to_f32();
    }
    half::bf16::from_bits((s as u16) << 7).to_f32()
}

/// Decode FP8 E4M3 byte (OCP) to f32 — used when group_size == 16.
fn dequant_scale_fp8_e4m3(s: u8) -> f32 {
    // Match MLX FromFP8 for E4M3: reuse half conversion via softfloat-ish path.
    // Bias 7, no inf; NaN at 0x7f / 0xff.
    let sign = (s >> 7) as u32;
    let exp = ((s >> 3) & 0x0f) as i32;
    let mant = (s & 0x07) as u32;
    if exp == 0x0f && mant == 0x07 {
        return f32::NAN;
    }
    if exp == 0 {
        if mant == 0 {
            return if sign != 0 { -0.0 } else { 0.0 };
        }
        // subnormal
        let mut m = mant;
        let mut e = -6i32;
        while m & 0x8 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x7;
        let bits = (sign << 31) | (((e + 127) as u32) << 23) | (m << 20);
        return f32::from_bits(bits);
    }
    let bits = (sign << 31) | (((exp - 7 + 127) as u32) << 23) | (mant << 20);
    f32::from_bits(bits)
}

/// Dequantize a packed MXFP4 weight matrix (codes + **already-decoded f32** group
/// scales, e.g. `sc.to_f32()` in a backend lowering) to row-major `[n, k]` F32 — the
/// weight-only half of [`dequant_matmul_mxfp4`], for backends that build an f32
/// expert bank once and then gather + matmul on-device.
pub fn dequant_mxfp4_weights_f32(
    w: &[u8],
    scales: &[f32],
    group_size: u32,
    n: usize,
    k: usize,
) -> Result<Vec<f32>> {
    let gs = group_size as usize;
    if gs == 0 || !k.is_multiple_of(gs) {
        bail!("mxfp4 weights: k={k} not divisible by group_size={gs}");
    }
    let n_groups = k / gs;
    if w.len() < n * k / 2 {
        bail!("mxfp4 weights: bytes {} < {}", w.len(), n * k / 2);
    }
    if scales.len() < n * n_groups {
        bail!("mxfp4 weights: scales {} < {}", scales.len(), n * n_groups);
    }
    let mut w_f = vec![0f32; n * k];
    let mut w_off = 0usize;
    for r in 0..n {
        for gidx in 0..n_groups {
            let scale = scales[r * n_groups + gidx];
            let base = r * k + gidx * gs;
            for p in (0..gs).step_by(2) {
                let b = w[w_off];
                w_off += 1;
                w_f[base + p] = FP4_LUT[(b & 0x0f) as usize] * scale;
                w_f[base + p + 1] = FP4_LUT[(b >> 4) as usize] * scale;
            }
        }
    }
    Ok(w_f)
}

/// Dequantize MLX `mxfp4` packs: nibbles → FP4 LUT × per-group scale.
///
/// `scales_u8` length = `rows * n_groups`. `group_size` is typically 32
/// (E8M0 scales) or 16 (FP8 E4M3 scales).
pub fn dequant_mxfp4_f32(
    w: &[u8],
    scales_u8: &[u8],
    group_size: u32,
    rows: usize,
    n_groups: usize,
) -> Result<Vec<f32>> {
    let gs = group_size as usize;
    let cols = n_groups * gs;
    // 2 nibbles per byte
    let need_w = rows * cols / 2;
    if w.len() < need_w {
        bail!("mxfp4: weight bytes {} < {need_w}", w.len());
    }
    if scales_u8.len() < rows * n_groups {
        bail!("mxfp4: scales too short");
    }
    let mut out = vec![0f32; rows * cols];
    // Rows are independent (each reads its own `cols/2` packed bytes) → parallelize.
    let bpr = cols / 2; // packed bytes per row
    use rayon::prelude::*;
    out.par_chunks_mut(cols).enumerate().for_each(|(r, orow)| {
        let mut w_off = r * bpr;
        for g in 0..n_groups {
            let scale = if gs == 16 {
                dequant_scale_fp8_e4m3(scales_u8[r * n_groups + g])
            } else {
                dequant_scale_e8m0(scales_u8[r * n_groups + g])
            };
            let base = g * gs;
            for p in (0..gs).step_by(2) {
                let b = w[w_off];
                w_off += 1;
                orow[base + p] = FP4_LUT[(b & 0x0f) as usize] * scale;
                orow[base + p + 1] = FP4_LUT[(b >> 4) as usize] * scale;
            }
        }
    });
    Ok(out)
}

/// Dequantize MLX `mxfp8` packs: raw FP8 E4M3 codes × E8M0/FP8 group scales.
pub fn dequant_mxfp8_f32(
    w: &[u8],
    scales_u8: &[u8],
    group_size: u32,
    rows: usize,
    n_groups: usize,
) -> Result<Vec<f32>> {
    let gs = group_size as usize;
    let cols = n_groups * gs;
    if w.len() < rows * cols {
        bail!("mxfp8: weight bytes {} < {}", w.len(), rows * cols);
    }
    if scales_u8.len() < rows * n_groups {
        bail!("mxfp8: scales too short");
    }
    let mut out = vec![0f32; rows * cols];
    let mut w_off = 0usize;
    for r in 0..rows {
        for g in 0..n_groups {
            let scale = if gs == 16 {
                dequant_scale_fp8_e4m3(scales_u8[r * n_groups + g])
            } else {
                dequant_scale_e8m0(scales_u8[r * n_groups + g])
            };
            let base = r * cols + g * gs;
            for p in 0..gs {
                out[base + p] = dequant_scale_fp8_e4m3(w[w_off]) * scale;
                w_off += 1;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mxfp4_matmul_matches_full_dequant() {
        // n=2 out rows, k=64, group_size=32 → 2 groups; nibbles + E8M0 scales.
        let (n, k, gs) = (2usize, 64usize, 32u32);
        let n_groups = k / gs as usize;
        // Deterministic nibble pairs per byte, n*k/2 bytes.
        let w: Vec<u8> = (0..n * k / 2).map(|i| ((i * 7) % 256) as u8).collect();
        let scales_u8: Vec<u8> = (0..n * n_groups).map(|i| (120 + i * 3) as u8).collect();
        // Reference: full dequant (u8 scales) then x @ Wᵀ.
        let w_ref = dequant_mxfp4_f32(&w, &scales_u8, gs, n, n_groups).unwrap();
        let m = 3usize;
        let x: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.013).cos()).collect();
        let mut y_ref = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for p in 0..k {
                    acc += x[i * k + p] * w_ref[j * k + p];
                }
                y_ref[i * n + j] = acc;
            }
        }
        // New matmul path: E8M0 scales pre-decoded to f32 (as the loader does).
        let scales_f32: Vec<f32> = scales_u8
            .iter()
            .map(|&s| mxfp4_scale_e8m0_to_f32(s))
            .collect();
        let y = dequant_matmul_mxfp4(&x, &w, &scales_f32, gs, m, k, n).unwrap();
        assert_eq!(y.len(), y_ref.len());
        for (a, b) in y.iter().zip(&y_ref) {
            assert!((a - b).abs() < 1e-4, "mxfp4 matmul mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn affine_4bit_roundtrip_shape() {
        // 1 row, group_size 8, bits 4 → 1 group, 4 packs (8/2? pack_factor=2 for bits=4)
        // pack_factor(4)=2, packs_in_group = 8/2 = 4 bytes
        let bits = 4;
        let gs = 8u32;
        let rows = 1;
        let n_groups = 1;
        // codes 0..7 packed as nibbles: [0x10, 0x32, 0x54, 0x76]
        let w = vec![0x10, 0x32, 0x54, 0x76];
        let scales = vec![2.0f32];
        let biases = vec![-1.0f32];
        let out = dequant_affine_f32(&w, &scales, &biases, bits, gs, rows, n_groups).unwrap();
        assert_eq!(out.len(), 8);
        // first nibble = 0 → 2*0 + (-1) = -1; second = 1 → 2*1-1 = 1
        assert!((out[0] - (-1.0)).abs() < 1e-5);
        assert!((out[1] - 1.0).abs() < 1e-5);
        assert!((out[2] - 3.0).abs() < 1e-5); // 2*2-1
    }

    #[test]
    fn affine_3bit_dequant_runs() {
        // gs=8, bits=3 → 3 bytes/group, 8 values.
        let w = vec![0x01, 0x02, 0x03];
        let scales = vec![1.0f32];
        let biases = vec![0.0f32];
        let out = dequant_affine_f32(&w, &scales, &biases, 3, 8, 1, 1).unwrap();
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn affine_5bit_dequant_runs() {
        let w = vec![0u8; 5];
        let scales = vec![1.0f32];
        let biases = vec![0.0f32];
        let out = dequant_affine_f32(&w, &scales, &biases, 5, 8, 1, 1).unwrap();
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn affine_6bit_dequant_runs() {
        // gs=8, bits=6 → pack_factor 4, packs_in_group=2, bpp=3 → 6 bytes?
        // pack_factor(6)=4, packs = gs/pf = 2, bpp=3 → 6 bytes per group.
        let w = vec![0u8; 6];
        let scales = vec![1.0f32];
        let biases = vec![0.0f32];
        let out = dequant_affine_f32(&w, &scales, &biases, 6, 8, 1, 1).unwrap();
        assert_eq!(out.len(), 8);
    }
}
