// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tier 2 — host probes on a captured (deep-dumped) tensor, closing the gaps the
//! cheap CSV sketches can't see: **effective rank** (→ factored matmul),
//! **N:M structured-sparsity feasibility** (→ sparse tensor-core GEMM),
//! **quantization error** at a target bit-width (→ int kernel), and **value
//! cardinality** (→ LUT/palettized weights). Pure host arithmetic on `&[f32]`.

use std::collections::HashSet;

pub fn frob(x: &[f32]) -> f32 {
    x.iter().map(|v| v * v).sum::<f32>().sqrt()
}

/// Largest singular value of `X[rows,cols]` via power iteration on `XᵀX`.
pub fn sigma_max(x: &[f32], rows: usize, cols: usize, iters: usize) -> f32 {
    if rows == 0 || cols == 0 {
        return 0.0;
    }
    let mut v = vec![1.0f32 / (cols as f32).sqrt(); cols];
    let mut u = vec![0f32; rows];
    for _ in 0..iters {
        for i in 0..rows {
            let mut s = 0.0;
            for j in 0..cols {
                s += x[i * cols + j] * v[j];
            }
            u[i] = s;
        }
        let mut w = vec![0f32; cols];
        for i in 0..rows {
            let ui = u[i];
            for j in 0..cols {
                w[j] += x[i * cols + j] * ui;
            }
        }
        let n = frob(&w);
        if n < 1e-20 {
            break;
        }
        for j in 0..cols {
            v[j] = w[j] / n;
        }
    }
    for i in 0..rows {
        let mut s = 0.0;
        for j in 0..cols {
            s += x[i * cols + j] * v[j];
        }
        u[i] = s;
    }
    frob(&u)
}

/// Stable rank `‖X‖_F² / σ_max²` — an effective-rank proxy (≈ true rank when the
/// spectrum is flat). Much smaller than `min(rows,cols)` ⇒ low-rank ⇒ `W ≈ U·V`.
pub fn stable_rank(x: &[f32], rows: usize, cols: usize) -> f32 {
    let s = sigma_max(x, rows, cols, 30);
    if s < 1e-20 {
        0.0
    } else {
        let f = frob(x);
        (f * f) / (s * s)
    }
}

/// Relative reconstruction error of symmetric per-tensor `bits`-bit quant.
pub fn quant_error(x: &[f32], bits: u32) -> f32 {
    let amax = x.iter().fold(0f32, |a, &v| a.max(v.abs()));
    if amax < 1e-20 {
        return 0.0;
    }
    let levels = ((1u32 << (bits - 1)) - 1).max(1) as f32;
    let scale = amax / levels;
    let (mut num, mut den) = (0f32, 0f32);
    for &v in x {
        let q = (v / scale).round().clamp(-levels, levels) * scale;
        num += (v - q) * (v - q);
        den += v * v;
    }
    if den < 1e-20 { 0.0 } else { (num / den).sqrt() }
}

/// **Per-channel** (per-row) symmetric `bits`-bit quant error — each row gets its
/// own scale. Far lower than per-tensor when channels have different ranges, so
/// comparing the two says whether per-channel quant is worth it.
pub fn per_channel_quant_error(x: &[f32], rows: usize, cols: usize, bits: u32) -> f32 {
    let levels = ((1u32 << (bits - 1)) - 1).max(1) as f32;
    let (mut num, mut den) = (0f32, 0f32);
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let amax = row.iter().fold(0f32, |a, &v| a.max(v.abs()));
        let scale = if amax < 1e-20 { 1.0 } else { amax / levels };
        for &v in row {
            let q = (v / scale).round().clamp(-levels, levels) * scale;
            num += (v - q) * (v - q);
            den += v * v;
        }
    }
    if den < 1e-20 { 0.0 } else { (num / den).sqrt() }
}

/// Outlier input-channels (columns) whose `max|·|` ≫ the median channel's —
/// keep these in high precision (AWQ/SmoothQuant) and quantize the rest hard.
/// Returns `(count above `ratio`× median, max/median ratio)`.
pub fn outlier_channels(x: &[f32], rows: usize, cols: usize, ratio: f32) -> (usize, f32) {
    let mut cmax = vec![0f32; cols];
    for r in 0..rows {
        for c in 0..cols {
            cmax[c] = cmax[c].max(x[r * cols + c].abs());
        }
    }
    let mut sorted = cmax.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[cols / 2].max(1e-20);
    let maxc = *sorted.last().unwrap_or(&0.0);
    let count = cmax.iter().filter(|&&v| v > ratio * median).count();
    (count, maxc / median)
}

/// Smallest bit-width in {3,4,6,8} whose **per-channel** error is under `budget`.
pub fn best_bitwidth(x: &[f32], rows: usize, cols: usize, budget: f32) -> Option<u32> {
    [3u32, 4, 6, 8]
        .iter()
        .find(|&&b| per_channel_quant_error(x, rows, cols, b) < budget)
        .copied()
}

/// Relative error of N:M structured sparsity — keep the `n` largest-|·| of every
/// `m` consecutive elements along the last axis. Low ⇒ sparse tensor-core GEMM.
pub fn nm_sparsity_error(x: &[f32], rows: usize, cols: usize, n: usize, m: usize) -> f32 {
    let (mut num, mut den) = (0f32, 0f32);
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        for grp in row.chunks(m) {
            let mut idx: Vec<usize> = (0..grp.len()).collect();
            idx.sort_by(|&a, &b| grp[b].abs().partial_cmp(&grp[a].abs()).unwrap());
            let keep: HashSet<usize> = idx.into_iter().take(n).collect();
            for (i, &v) in grp.iter().enumerate() {
                den += v * v;
                if !keep.contains(&i) {
                    num += v * v;
                }
            }
        }
    }
    if den < 1e-20 { 0.0 } else { (num / den).sqrt() }
}

/// Relative error of **ternary** quantization `w → α·{-1,0,+1}` (TWN, Li & Liu
/// 2016): threshold `Δ = 0.7·mean|w|`, keep sign where `|w|>Δ`, scale `α =
/// mean|w| over the kept values. Low ⇒ the weight is effectively ternary — an
/// ~1.6-bit / adds-only matmul (5× smaller than int8, no multiplies). Only safe
/// where the data says so; most trained weights are NOT ternary (high error).
pub fn ternary_error(w: &[f32]) -> f32 {
    let n = w.len().max(1);
    let mean_abs = w.iter().map(|v| v.abs()).sum::<f32>() / n as f32;
    let delta = 0.7 * mean_abs;
    let (mut kept, mut cnt) = (0f32, 0usize);
    for &v in w {
        if v.abs() > delta {
            kept += v.abs();
            cnt += 1;
        }
    }
    let alpha = if cnt > 0 { kept / cnt as f32 } else { 0.0 };
    let (mut num, mut den) = (0f32, 0f32);
    for &v in w {
        let t = if v > delta {
            alpha
        } else if v < -delta {
            -alpha
        } else {
            0.0
        };
        num += (v - t) * (v - t);
        den += v * v;
    }
    if den < 1e-20 { 0.0 } else { (num / den).sqrt() }
}

/// Relative change a residual block makes to its input: `‖out−in‖/‖in‖`. Near 0
/// ⇒ the block is ~identity (adds almost nothing to the residual stream) → a
/// **layer-skip / minimization** candidate. The DATA-FLOW signal for dropping
/// whole layers (mirrors "some LLM layers are near-identity and prunable").
pub fn block_identity_gap(input: &[f32], output: &[f32]) -> f32 {
    let num: f32 = input
        .iter()
        .zip(output)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    let den: f32 = input.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-9;
    num / den
}

/// Distinct value count (values snapped to `1/round_to`), capped for reporting.
/// Few distinct values ⇒ LUT-GEMM / palettized weights.
pub fn cardinality(x: &[f32], round_to: f32) -> usize {
    let mut set = HashSet::new();
    for &v in x {
        set.insert((v * round_to).round() as i64);
        if set.len() > 100_000 {
            break;
        }
    }
    set.len()
}

/// Serialize a captured tensor: `rows cols` (u32 LE) then row-major f32 LE.
pub fn save_tensor(path: &str, rows: usize, cols: usize, data: &[f32]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(&(rows as u32).to_le_bytes())?;
    f.write_all(&(cols as u32).to_le_bytes())?;
    for &v in data {
        f.write_all(&v.to_le_bytes())?;
    }
    f.flush()
}

/// Load a tensor saved by [`save_tensor`].
pub fn load_tensor(path: &str) -> std::io::Result<(usize, usize, Vec<f32>)> {
    let b = std::fs::read(path)?;
    let rows = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
    let cols = u32::from_le_bytes([b[4], b[5], b[6], b[7]]) as usize;
    let data: Vec<f32> = b[8..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok((rows, cols, data))
}
