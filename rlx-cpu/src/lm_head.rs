// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
//! Greedy tied-LM-head argmax without materializing full vocab logits.

use rlx_gguf::QK_K;
use rlx_ir::quant::QuantScheme;
use std::cmp::Ordering;

use crate::gguf_matmul::dequant_block;

fn row_dot_f32(hidden: &[f32], row: &[f32]) -> f32 {
    hidden.iter().zip(row).map(|(a, b)| a * b).sum()
}

/// Dot product with per-prefix Cauchy bound early exit: returns `None`
/// when `partial + Σ|h[q]||row[q]|` for remaining `q` cannot reach `floor`.
fn row_dot_f32_early_exit(hidden: &[f32], row: &[f32], floor: f32) -> Option<f32> {
    let n = hidden.len();
    debug_assert_eq!(n, row.len());
    let mut dot = 0f32;
    for p in 0..n {
        dot += hidden[p] * row[p];
        let mut rem = 0f32;
        for q in (p + 1)..n {
            rem += hidden[q].abs() * row[q].abs();
        }
        if dot + rem < floor {
            return None;
        }
    }
    Some(dot)
}

fn row_dot_gguf(
    hidden: &[f32],
    w_bytes: &[u8],
    row: usize,
    n_embd: usize,
    scheme: QuantScheme,
) -> f32 {
    let k = n_embd;
    let row_start = row * k;
    let row_end = row_start + k;
    let block_elems = scheme.gguf_block_size() as usize;
    let block_bytes = scheme.gguf_block_bytes() as usize;
    let first_block = row_start / block_elems;
    let last_block = (row_end - 1) / block_elems;
    let mut block_f32 = [0f32; QK_K];
    let mut dot = 0f32;
    for bi in first_block..=last_block {
        let off = bi * block_bytes;
        dequant_block(scheme, &w_bytes[off..off + block_bytes], &mut block_f32);
        let idx0 = bi * block_elems;
        for t in 0..block_elems {
            let idx = idx0 + t;
            if idx < row_start || idx >= row_end {
                continue;
            }
            let p = idx - row_start;
            dot += hidden[p] * block_f32[t];
        }
    }
    dot
}

/// GGUF row dot with block-wise Cauchy bound early exit.
fn row_dot_gguf_early_exit(
    hidden: &[f32],
    w_bytes: &[u8],
    row: usize,
    n_embd: usize,
    scheme: QuantScheme,
    floor: f32,
) -> Option<f32> {
    let k = n_embd;
    let row_start = row * k;
    let row_end = row_start + k;
    let block_elems = scheme.gguf_block_size() as usize;
    let block_bytes = scheme.gguf_block_bytes() as usize;
    let first_block = row_start / block_elems;
    let last_block = (row_end - 1) / block_elems;
    let mut block_f32 = [0f32; QK_K];
    let mut dot = 0f32;
    for bi in first_block..=last_block {
        let off = bi * block_bytes;
        dequant_block(scheme, &w_bytes[off..off + block_bytes], &mut block_f32);
        let idx0 = bi * block_elems;
        for t in 0..block_elems {
            let idx = idx0 + t;
            if idx < row_start || idx >= row_end {
                continue;
            }
            let p = idx - row_start;
            dot += hidden[p] * block_f32[t];
        }
        // Remaining dims: bound with |h| × max dequant magnitude in each block.
        let mut rem = 0f32;
        for bj in (bi + 1)..=last_block {
            let offj = bj * block_bytes;
            dequant_block(scheme, &w_bytes[offj..offj + block_bytes], &mut block_f32);
            let j0 = bj * block_elems;
            for t in 0..block_elems {
                let idx = j0 + t;
                if idx < row_start || idx >= row_end {
                    continue;
                }
                let p = idx - row_start;
                rem += hidden[p].abs() * block_f32[t].abs();
            }
        }
        if dot + rem < floor {
            return None;
        }
    }
    Some(dot)
}

fn topk_from_scores(mut scores: Vec<(u32, f32)>, cap: usize) -> Vec<(u32, f32)> {
    let cap = cap.min(scores.len()).max(1);
    if scores.len() > cap {
        scores.select_nth_unstable_by(cap - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
        });
        scores.truncate(cap);
    }
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    scores
}

/// Top-`cap` logits as `(token_id, logit)` sorted descending.
pub fn f32_tied_lm_topk(
    hidden: &[f32],
    embed: &[f32],
    n_embd: usize,
    n_vocab: usize,
    cap: usize,
) -> Vec<(u32, f32)> {
    assert_eq!(hidden.len(), n_embd);
    let cap = cap.min(n_vocab).max(1);
    let mut scores: Vec<(u32, f32)> = Vec::with_capacity(cap + 1);
    let mut floor = f32::NEG_INFINITY;
    for j in 0..n_vocab {
        let row = &embed[j * n_embd..(j + 1) * n_embd];
        if let Some(score) = row_dot_f32_early_exit(hidden, row, floor) {
            scores.push((j as u32, score));
            if scores.len() > cap {
                scores.select_nth_unstable_by(cap - 1, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
                });
                scores.truncate(cap);
                scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                floor = scores[cap - 1].1;
            }
        }
    }
    topk_from_scores(scores, cap)
}

/// Top-`cap` logits from packed GGUF tied embedding.
pub fn gguf_tied_lm_topk(
    hidden: &[f32],
    w_bytes: &[u8],
    n_embd: usize,
    n_vocab: usize,
    scheme: QuantScheme,
    cap: usize,
) -> Vec<(u32, f32)> {
    assert_eq!(hidden.len(), n_embd);
    let cap = cap.min(n_vocab).max(1);
    let mut scores: Vec<(u32, f32)> = Vec::with_capacity(cap + 1);
    let mut floor = f32::NEG_INFINITY;
    for j in 0..n_vocab {
        if let Some(score) = row_dot_gguf_early_exit(hidden, w_bytes, j, n_embd, scheme, floor) {
            scores.push((j as u32, score));
            if scores.len() > cap {
                scores.select_nth_unstable_by(cap - 1, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
                });
                scores.truncate(cap);
                scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                floor = scores[cap - 1].1;
            }
        }
    }
    topk_from_scores(scores, cap)
}

/// Argmax over rows of `embed` `[n_vocab × n_embd]` row-major: `hidden @ embed^T`.
pub fn f32_tied_lm_argmax(hidden: &[f32], embed: &[f32], n_embd: usize, n_vocab: usize) -> (u32, f32) {
    assert_eq!(hidden.len(), n_embd);
    assert!(embed.len() >= n_vocab * n_embd);
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for j in 0..n_vocab {
        let row = &embed[j * n_embd..(j + 1) * n_embd];
        let dot = row_dot_f32(hidden, row);
        if dot > best_val {
            best_val = dot;
            best_idx = j as u32;
        }
    }
    (best_idx, best_val)
}

/// Packed GGUF tied embedding — row-streaming block dequant + argmax (O(1) scratch).
pub fn gguf_tied_lm_argmax(
    hidden: &[f32],
    w_bytes: &[u8],
    n_embd: usize,
    n_vocab: usize,
    scheme: QuantScheme,
) -> (u32, f32) {
    assert_eq!(hidden.len(), n_embd);
    let k = n_embd;
    let block_elems = scheme.gguf_block_size() as usize;
    let block_bytes = scheme.gguf_block_bytes() as usize;
    let num_blocks = (k * n_vocab) / block_elems;
    debug_assert_eq!(w_bytes.len(), num_blocks * block_bytes);

    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;

    for j in 0..n_vocab {
        let dot = row_dot_gguf(hidden, w_bytes, j, n_embd, scheme);
        if dot > best_val {
            best_val = dot;
            best_idx = j as u32;
        }
    }
    (best_idx, best_val)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf_matmul::gguf_matmul_bt;

    #[test]
    fn topk_matches_full_sort() {
        let k = 256;
        let n = 32;
        let hidden: Vec<f32> = (0..k).map(|i| 0.01 * i as f32).collect();
        let embed: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32) * 0.003 + (i % 11) as f32 * 1e-4))
            .collect();
        let top = f32_tied_lm_topk(&hidden, &embed, k, n, 8);
        let mut full: Vec<(u32, f32)> = (0..n)
            .map(|j| {
                let row = &embed[j * k..(j + 1) * k];
                (
                    j as u32,
                    hidden.iter().zip(row).map(|(a, b)| a * b).sum(),
                )
            })
            .collect();
        full.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        assert_eq!(top.len(), 8);
        let top_scores: Vec<f32> = top.iter().map(|(_, s)| *s).collect();
        let ref_scores: Vec<f32> = full.iter().take(8).map(|(_, s)| *s).collect();
        assert_eq!(top_scores, ref_scores);
    }

    #[test]
    fn topk_early_exit_matches_full_sort_gguf() {
        let k = 256;
        let n = 64;
        let cap = 8;
        let scale = 1.0f32;
        let mut packed = Vec::new();
        for j in 0..n {
            packed.extend_from_slice(&scale.to_le_bytes());
            for i in 0..QK_K {
                let q = ((j as i32 * 10 + i as i32) - 128).clamp(-128, 127) as i8;
                packed.push(q as u8);
            }
            for _ in 0..(QK_K / 16) {
                packed.extend_from_slice(&0i16.to_le_bytes());
            }
        }
        let hidden: Vec<f32> = (0..k).map(|i| 0.01 * i as f32).collect();
        let top = gguf_tied_lm_topk(&hidden, &packed, k, n, QuantScheme::GgufQ8K, cap);
        let ref_top = f32_tied_lm_topk(
            &hidden,
            &rlx_gguf::dequant_q8_k(&packed, k * n).unwrap(),
            k,
            n,
            cap,
        );
        assert_eq!(top, ref_top);
    }

    #[test]
    fn argmax_matches_matmul_on_q8k() {
        let k = 256;
        let n = 8;
        let scale = 1.0f32;
        let mut packed = Vec::new();
        for j in 0..n {
            packed.extend_from_slice(&scale.to_le_bytes());
            for i in 0..QK_K {
                let q = ((j as i32 * 10 + i as i32) - 128).clamp(-128, 127) as i8;
                packed.push(q as u8);
            }
            for _ in 0..(QK_K / 16) {
                packed.extend_from_slice(&0i16.to_le_bytes());
            }
        }
        let hidden: Vec<f32> = (0..k).map(|i| 0.01 * i as f32).collect();
        let mut logits = vec![0f32; n];
        gguf_matmul_bt(
            &hidden,
            &packed,
            &mut logits,
            1,
            k,
            n,
            QuantScheme::GgufQ8K,
        );
        let (idx, val) = gguf_tied_lm_argmax(&hidden, &packed, k, n, QuantScheme::GgufQ8K);
        let ref_idx = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        assert_eq!(idx, ref_idx);
        assert!((val - logits[ref_idx as usize]).abs() < 1e-3);
    }
}
