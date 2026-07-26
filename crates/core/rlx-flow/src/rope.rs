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

//! Rotary Position Embedding (RoPE) frequency + table builders.
//!
//! Until now every Llama-shaped model crate (`rlx-llama32`, `rlx-gemma`,
//! `rlx-qwen3`, `rlx-qwen35`, …) carried a near-identical `rope.rs`:
//! a copy of `default_inv_freq`, `build_rope_tables`, plus an arch-specific
//! `resolve_inv_freq` that layered Llama 3 / NTK / YaRN / MRoPE on top.
//! This module collapses the inverse-frequency primitives into one place
//! so model crates only carry the *config → which scaling* glue, not the
//! math.
//!
//! Pairs with [`crate::RopeTablesStage`] (which consumes the `(cos, sin)`
//! tables this module emits) and with [`rlx_ir::op::Rope`] (the IR op).

use std::f64::consts::PI;

/// Per-pair inverse frequencies for the canonical RoPE schedule
/// `inv_freq[i] = 1 / theta^(2i / head_dim)`.
///
/// `head_dim` is the **full** rotary dimension (typically equal to
/// the attention head dimension). The returned slice has length
/// `head_dim / 2`.
pub fn default_inv_freq(rope_theta: f64, head_dim: usize) -> Vec<f64> {
    (0..head_dim)
        .step_by(2)
        .map(|i| 1.0 / rope_theta.powf(i as f64 / head_dim as f64))
        .collect()
}

/// Apply baked GGUF `rope_freqs.weight` factors: `inv_freq[i] /= factors[i]`.
///
/// Mirrors llama.cpp's `ggml_rope` when `freq_factors` is supplied.
/// Panics on length mismatch — caller's contract.
pub fn inv_freq_with_factors(base: &[f64], factors: &[f32]) -> Vec<f64> {
    assert_eq!(
        base.len(),
        factors.len(),
        "rope_freqs.weight length must match head_dim/2"
    );
    base.iter()
        .zip(factors.iter())
        .map(|(f, ff)| f / *ff as f64)
        .collect()
}

/// Llama 3 RoPE scaling parameters (HF `rope_scaling`, `rope_type=llama3`).
#[derive(Debug, Clone, Copy)]
pub struct Llama3Scaling {
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: u32,
}

/// Llama 3 RoPE scaling (matches HuggingFace `transformers` and `candle`).
///
/// Wavelength-based piecewise scaling: low-frequency components are
/// divided by `factor`, high-frequency components are kept as-is, and
/// mid-band components are linearly interpolated between the two.
pub fn llama3_scaled_inv_freq(base: &[f64], s: &Llama3Scaling) -> Vec<f64> {
    let low_freq_wavelen = s.original_max_position_embeddings as f64 / s.low_freq_factor as f64;
    let high_freq_wavelen = s.original_max_position_embeddings as f64 / s.high_freq_factor as f64;

    base.iter()
        .map(|&freq| {
            let wavelen = 2.0 * PI / freq;
            if wavelen < high_freq_wavelen {
                freq
            } else if wavelen > low_freq_wavelen {
                freq / s.factor as f64
            } else {
                let smooth = (s.original_max_position_embeddings as f64 / wavelen
                    - s.low_freq_factor as f64)
                    / (s.high_freq_factor as f64 - s.low_freq_factor as f64);
                (1.0 - smooth) * freq / s.factor as f64 + smooth * freq
            }
        })
        .collect()
}

/// NTK-aware scaling (NTK-by-parts). `factor` extends the context window
/// by scaling theta as `theta * factor^(dim / (dim - 2))`.
pub fn ntk_scaled_inv_freq(rope_theta: f64, head_dim: usize, factor: f32) -> Vec<f64> {
    let alpha = (factor as f64).powf(head_dim as f64 / (head_dim as f64 - 2.0));
    default_inv_freq(rope_theta * alpha, head_dim)
}

/// YaRN scaling parameters (HF `rope_scaling`, `rope_type=yarn`).
#[derive(Debug, Clone, Copy)]
pub struct YarnScaling {
    pub factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub original_max_position_embeddings: u32,
}

/// YaRN scaling (Peng et al. 2023). Smooth ramp between extrapolated
/// and NTK-scaled regimes per inverse frequency.
pub fn yarn_scaled_inv_freq(base_theta: f64, head_dim: usize, s: &YarnScaling) -> Vec<f64> {
    let base = default_inv_freq(base_theta, head_dim);
    let ntk = ntk_scaled_inv_freq(base_theta, head_dim, s.factor);

    let low = yarn_correction_dim(
        s.beta_fast,
        head_dim,
        base_theta,
        s.original_max_position_embeddings as f64,
    );
    let high = yarn_correction_dim(
        s.beta_slow,
        head_dim,
        base_theta,
        s.original_max_position_embeddings as f64,
    );
    let (low, high) = if low > high { (high, low) } else { (low, high) };

    base.iter()
        .zip(ntk.iter())
        .enumerate()
        .map(|(i, (b, n))| {
            let mask = yarn_linear_ramp_mask(low, high, i as f64);
            *n * (1.0 - mask) + *b * mask
        })
        .collect()
}

fn yarn_correction_dim(num_rot: f32, dim: usize, base: f64, max_pos: f64) -> f64 {
    let num = (max_pos / (num_rot as f64 * 2.0 * PI)).ln();
    let den = (base.ln()) * 2.0;
    (dim as f64 * num / den)
        .floor()
        .max(0.0)
        .min(dim as f64 / 2.0 - 1.0)
}

fn yarn_linear_ramp_mask(low: f64, high: f64, i: f64) -> f64 {
    if (high - low).abs() < f64::EPSILON {
        return 1.0;
    }
    ((i - low) / (high - low)).clamp(0.0, 1.0)
}

/// Build `[max_pos, head_dim/2]` cos/sin tables from inverse frequencies.
///
/// Output layout: `cos[pos * half + i] = cos(pos * inv_freq[i])`. Suitable
/// for direct binding into [`crate::RopeTablesStage`] or for splitting
/// into per-section MRoPE tables.
pub fn build_tables(inv_freq: &[f64], max_pos: usize) -> (Vec<f32>, Vec<f32>) {
    let half = inv_freq.len();
    let mut cos = vec![0f32; max_pos * half];
    let mut sin = vec![0f32; max_pos * half];
    for pos in 0..max_pos {
        for (i, &freq) in inv_freq.iter().enumerate() {
            let angle = pos as f64 * freq;
            cos[pos * half + i] = angle.cos() as f32;
            sin[pos * half + i] = angle.sin() as f32;
        }
    }
    (cos, sin)
}

/// Convenience: build tables directly from `(theta, head_dim, max_pos)`
/// with no scaling. Equivalent to `build_tables(&default_inv_freq(...), max_pos)`.
pub fn build_default_tables(
    rope_theta: f64,
    head_dim: usize,
    max_pos: usize,
) -> (Vec<f32>, Vec<f32>) {
    build_tables(&default_inv_freq(rope_theta, head_dim), max_pos)
}

/// MRoPE section schedule (Qwen 2-VL / Qwen 3.5 / Qwen 3-VL).
///
/// `ggml_rope_multi` in llama.cpp always takes four ints; sections that
/// don't apply are encoded as 0. Each section gets its own local frequency
/// slice over its share of the first `n_rot` dims; remaining pairs use
/// identity rotation.
pub fn mrope_sections4(sections: &[usize]) -> [usize; 4] {
    let mut out = [0usize; 4];
    for (i, &v) in sections.iter().take(4).enumerate() {
        out[i] = v;
    }
    out
}

/// Map a global rotary-pair index to its MRoPE section (llama.cpp
/// `sector = (i0/2) % sect_dims` with `sect_dims = sum(sections)`).
///
/// Sections are encoded as **pair counts** (not dim counts) — they
/// describe how many `(cos, sin)` pairs each modality owns, summing to
/// `n_rot / 2`. Returns the section index (0..=3) that owns
/// `global_pair_j`.
pub fn mrope_section_for_pair(global_pair_j: usize, sections: [usize; 4]) -> usize {
    let mut acc = 0usize;
    for (sec_i, &sec_dim) in sections.iter().enumerate() {
        if sec_dim == 0 {
            continue;
        }
        if global_pair_j < acc + sec_dim {
            return sec_i;
        }
        acc += sec_dim;
    }
    3
}

/// Build one MRoPE cos/sin row from explicit per-section positions.
///
/// `sections` are pair counts in `[s0, s1, s2, s3]`; `section_pos` are
/// the per-modality positions `[p0, p1, p2, p3]`. Pairs beyond
/// `n_rot / 2` (up to `head_half`) stay at identity rotation — supports
/// partial-RoPE head dims where only the first `n_rot` dims rotate.
///
/// When `interleaved` is true (HF `mrope_interleaved`), pair ownership
/// cycles THWTHW… per Qwen3.5 / Qwen3-VL `apply_interleaved_mrope`
/// instead of contiguous TTT…HHH…WWW sections.
pub fn mrope_row_for_sections(
    rope_theta: f64,
    n_rot: usize,
    sections: [usize; 4],
    section_pos: [usize; 4],
    head_half: usize,
) -> (Vec<f32>, Vec<f32>) {
    mrope_row_for_sections_ex(rope_theta, n_rot, sections, section_pos, head_half, false)
}

/// Like [`mrope_row_for_sections`] with an explicit interleaved layout flag.
pub fn mrope_row_for_sections_ex(
    rope_theta: f64,
    n_rot: usize,
    sections: [usize; 4],
    section_pos: [usize; 4],
    head_half: usize,
    interleaved: bool,
) -> (Vec<f32>, Vec<f32>) {
    let half_rot = n_rot / 2;
    let mut cos = vec![0f32; head_half];
    let mut sin = vec![0f32; head_half];

    for global_j in 0..half_rot.min(head_half) {
        let sec_i = if interleaved {
            interleaved_mrope_section_for_pair(global_j, sections)
        } else {
            mrope_section_for_pair(global_j, sections)
        };
        let p = section_pos[sec_i] as f64;
        let freq = 1.0 / rope_theta.powf((2 * global_j) as f64 / n_rot as f64);
        let angle = p * freq;
        let (s, c) = angle.sin_cos();
        cos[global_j] = c as f32;
        sin[global_j] = s as f32;
    }
    for j in half_rot.min(head_half)..head_half {
        cos[j] = 1.0;
        sin[j] = 0.0;
    }
    (cos, sin)
}

/// HF interleaved layout: start as T for every pair, then overwrite
/// `slice(1, s_h*3, 3)` with H and `slice(2, s_w*3, 3)` with W.
fn interleaved_mrope_section_for_pair(global_pair_j: usize, sections: [usize; 4]) -> usize {
    let s_h = sections[1];
    let s_w = sections[2];
    match global_pair_j % 3 {
        1 if global_pair_j < s_h.saturating_mul(3) => 1,
        2 if global_pair_j < s_w.saturating_mul(3) => 2,
        _ => 0,
    }
}

/// Build MRoPE cos/sin tables for the text modality (positions repeat
/// across all four sections — see `llm_graph_input_pos::set_input` in
/// llama.cpp). When `sections` sum to less than `head_dim/2`, the
/// remaining pairs are filled with `(1, 0)` (identity rotation).
pub fn build_mrope_text_tables(
    rope_theta: f64,
    head_dim: usize,
    sections: [usize; 4],
    max_pos: usize,
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![1f32; max_pos * half];
    let mut sin = vec![0f32; max_pos * half];

    let inv = default_inv_freq(rope_theta, head_dim);
    let mut offset = 0usize;
    for &section in sections.iter() {
        if section == 0 {
            continue;
        }
        // The section operates on the first `section` rotary dim pairs
        // local to this segment of `inv`; for text MRoPE every section
        // uses the same position vector `p`.
        let section_half = section / 2;
        for pos in 0..max_pos {
            for i in 0..section_half {
                let idx = offset + i;
                if idx >= half || i >= inv.len() {
                    break;
                }
                let angle = pos as f64 * inv[idx];
                cos[pos * half + idx] = angle.cos() as f32;
                sin[pos * half + idx] = angle.sin() as f32;
            }
        }
        offset += section_half;
    }
    (cos, sin)
}

/// Build a **per-token** MRoPE cos/sin table of shape `[seq, head_dim/2]` from
/// explicit per-token 3-D (+optional 4th) positions.
///
/// `positions[t] = [pt, ph, pw, pe]` is token `t`'s position in each modality
/// section (temporal / height / width / extra). Text tokens set all entries to
/// the same running scalar; image tokens carry their grid `(t, h, w)`. Row `t`
/// of the returned tables holds the rotation angles for token `t`, so it drops
/// straight into the existing per-token [`rlx_ir::op::Op::Rope`] path (the
/// kernel indexes the table by global token when `rows == batch·seq`) — MRoPE
/// needs no dedicated Rope op.
///
/// `interleaved` selects HF's THWTHW… pair layout (Qwen3.5 / Qwen3-VL
/// `apply_interleaved_mrope`) over contiguous TTT…HHH…WWW sections.
pub fn build_mrope_tables(
    rope_theta: f64,
    head_dim: usize,
    n_rot: usize,
    sections: [usize; 4],
    positions: &[[usize; 4]],
    interleaved: bool,
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let seq = positions.len();
    let mut cos = vec![1f32; seq * half];
    let mut sin = vec![0f32; seq * half];
    for (t, pos) in positions.iter().enumerate() {
        let (crow, srow) =
            mrope_row_for_sections_ex(rope_theta, n_rot, sections, *pos, half, interleaved);
        cos[t * half..t * half + half].copy_from_slice(&crow);
        sin[t * half..t * half + half].copy_from_slice(&srow);
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mrope_reduces_to_standard_rope_for_text() {
        // A single section owning every pair + identical positions across all
        // modalities ⇒ MRoPE row == standard RoPE row (n_rot == head_dim).
        let (head_dim, n_rot) = (8usize, 8usize);
        let sections = [4, 0, 0, 0];
        let positions = [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]];
        let (mc, ms) = build_mrope_tables(10_000.0, head_dim, n_rot, sections, &positions, false);
        let (dc, ds) = build_default_tables(10_000.0, head_dim, positions.len());
        assert_eq!(mc.len(), dc.len());
        for i in 0..mc.len() {
            assert!(
                (mc[i] - dc[i]).abs() < 1e-6,
                "cos[{i}]: {} vs {}",
                mc[i],
                dc[i]
            );
            assert!(
                (ms[i] - ds[i]).abs() < 1e-6,
                "sin[{i}]: {} vs {}",
                ms[i],
                ds[i]
            );
        }
    }

    #[test]
    fn mrope_sections_isolate_modalities() {
        // sections = [T:1, H:1, W:2] over head_dim/2 = 4 pairs.
        let (head_dim, n_rot) = (8usize, 8usize);
        let sections = [1, 1, 2, 0];
        let half = head_dim / 2;
        // Two tokens sharing T but differing in H and W.
        let positions = [[5, 5, 5, 0], [5, 3, 7, 0]];
        let (cos, _) = build_mrope_tables(10_000.0, head_dim, n_rot, sections, &positions, false);
        let row0 = &cos[0..half];
        let row1 = &cos[half..2 * half];
        // Pair 0 (T section, pos 5 for both) is identical.
        assert!((row0[0] - row1[0]).abs() < 1e-7, "T pair should match");
        // Pair 1 (H section) differs (5 vs 3).
        assert!((row0[1] - row1[1]).abs() > 1e-4, "H pair should differ");
        // Pairs 2,3 (W section) differ (5 vs 7).
        assert!((row0[2] - row1[2]).abs() > 1e-4, "W pair should differ");
    }

    #[test]
    fn default_freq_lengths() {
        let f = default_inv_freq(10_000.0, 64);
        assert_eq!(f.len(), 32);
        assert!((f[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn tables_shape() {
        let (cos, sin) = build_default_tables(10_000.0, 64, 16);
        assert_eq!(cos.len(), 16 * 32);
        assert_eq!(sin.len(), 16 * 32);
        // pos=0 → cos=1, sin=0 over the entire row.
        for i in 0..32 {
            assert!((cos[i] - 1.0).abs() < 1e-6);
            assert!(sin[i].abs() < 1e-6);
        }
    }

    #[test]
    fn llama3_scaling_high_freq_passthrough() {
        // Very-high-frequency dims (wavelen < high_freq_wavelen) stay as-is.
        let base = default_inv_freq(500_000.0, 128);
        let scaling = Llama3Scaling {
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192,
        };
        let scaled = llama3_scaled_inv_freq(&base, &scaling);
        // First (highest-frequency) entry untouched.
        assert!((scaled[0] - base[0]).abs() < 1e-12);
    }

    #[test]
    fn mrope_sections_clamp() {
        assert_eq!(mrope_sections4(&[24, 20, 20, 0, 5]), [24, 20, 20, 0]);
        assert_eq!(mrope_sections4(&[8]), [8, 0, 0, 0]);
    }
}
