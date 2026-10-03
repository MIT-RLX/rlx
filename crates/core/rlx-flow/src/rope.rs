// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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
//! tables this module emits) and with `rlx_ir::op::Rope` (the IR op).

use rlx_ir::GraphExt;

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
    /// Explicit `attention_factor` from config; `None` ⇒ derive `0.1·ln(factor)+1`.
    pub attention_factor: Option<f32>,
}

/// YaRN "NTK-by-parts" scaling (Peng et al. 2023), matching HuggingFace
/// `_compute_yarn_parameters`. Per rotary pair `i`, blend between the
/// **extrapolated** frequency (original `1/θ^(2i/d)`, kept for high-frequency /
/// low-index dims so short-range detail survives) and the **interpolated**
/// frequency (`freq / factor`, linear position interpolation, for low-frequency /
/// high-index dims) across the correction range `[low, high]` set by
/// `beta_fast` / `beta_slow`.
///
/// (Note: YaRN interpolates with `freq/factor`, *not* an NTK theta rescale — an
/// earlier version of this fn used NTK and an inverted ramp; both were wrong.)
pub fn yarn_scaled_inv_freq(base_theta: f64, head_dim: usize, s: &YarnScaling) -> Vec<f64> {
    let base = default_inv_freq(base_theta, head_dim); // extrapolation
    let factor = s.factor as f64;
    let low = yarn_correction_dim(
        s.beta_fast,
        head_dim,
        base_theta,
        s.original_max_position_embeddings as f64,
    )
    .floor();
    let high = yarn_correction_dim(
        s.beta_slow,
        head_dim,
        base_theta,
        s.original_max_position_embeddings as f64,
    )
    .ceil();
    let (low, high) = (low.max(0.0), high.min(head_dim as f64 / 2.0 - 1.0));

    base.iter()
        .enumerate()
        .map(|(i, &ext)| {
            let interp = ext / factor;
            // ramp: 0 at `low` (extrapolate) → 1 at `high` (interpolate).
            let ramp = yarn_linear_ramp_mask(low, high, i as f64);
            interp * ramp + ext * (1.0 - ramp)
        })
        .collect()
}

/// YaRN attention temperature (`mscale`) baked into the cos/sin tables so the
/// attention logits are scaled by `mscale²` — the length-extrapolation
/// correction. Uses the explicit `attention_factor` if set, else HF's default
/// `0.1·ln(factor) + 1` (and `1.0` when `factor ≤ 1`).
pub fn yarn_mscale(s: &YarnScaling) -> f64 {
    if let Some(a) = s.attention_factor {
        return a as f64;
    }
    let f = s.factor as f64;
    if f <= 1.0 { 1.0 } else { 0.1 * f.ln() + 1.0 }
}

/// Correction dimension for `num_rot` rotations (HF `find_correction_dim`).
fn yarn_correction_dim(num_rot: f32, dim: usize, base: f64, max_pos: f64) -> f64 {
    let num = (max_pos / (num_rot as f64 * 2.0 * PI)).ln();
    let den = base.ln() * 2.0;
    dim as f64 * num / den
}

fn yarn_linear_ramp_mask(low: f64, high: f64, i: f64) -> f64 {
    let denom = if (high - low).abs() < 1e-9 {
        0.001
    } else {
        high - low
    };
    ((i - low) / denom).clamp(0.0, 1.0)
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

/// Build **YaRN-scaled** cos/sin tables: YaRN NTK-by-parts inv_freq with the
/// attention `mscale` baked into every entry (so the rotated q·k logits pick up
/// `mscale²`, YaRN's length-extrapolation temperature). `max_pos` should be the
/// *extended* window (e.g. `factor × original_max_position_embeddings`).
pub fn build_yarn_tables(
    base_theta: f64,
    head_dim: usize,
    s: &YarnScaling,
    max_pos: usize,
) -> (Vec<f32>, Vec<f32>) {
    let inv = yarn_scaled_inv_freq(base_theta, head_dim, s);
    let (mut cos, mut sin) = build_tables(&inv, max_pos);
    let m = yarn_mscale(s) as f32;
    if (m - 1.0).abs() > 1e-9 {
        for c in cos.iter_mut() {
            *c *= m;
        }
        for si in sin.iter_mut() {
            *si *= m;
        }
    }
    (cos, sin)
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
    // Row stride = n_rot/2 (the RoPE kernel indexes cos/sin with an n_rot/2 stride).
    // A head_dim/2 stride only rotates seq position 0 for partial rope
    // (n_rot < head_dim); later positions read the previous token's identity tail.
    // For full rope (n_rot == head_dim) this equals head_dim/2 — unchanged.
    let _ = head_dim; // kept for API stability; stride is n_rot-derived now.
    let half = n_rot / 2;
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

// ── continuous-time rotary (RotaryTimeEmbedding) ─────────────────────────────
//
// Everything above builds tables from **integer positions**: token 0, 1, 2, …
// A model over irregularly-sampled events has no such index — a spike happens at
// 1.037 s, and two spikes 3 ms apart must rotate by 3 ms' worth of phase, not by
// "one position". That is what `torch_brain`'s `RotaryTimeEmbedding` does, and
// what POYO's cross- and self-attention consume.
//
// Three differences from the integer-position family, all load-bearing:
//
//   * the phase is `t * omega[i]` for a **real** `t`, not `pos * inv_freq[i]`;
//   * `omega` comes from **log-spaced periods** between `t_min` and `t_max`, and
//     is **zero past `rotate_dim / 2`** — so the tail of each head is left
//     unrotated rather than rotated at some default frequency;
//   * the layout is **interleaved and duplicated**: entry `2i` and `2i+1` of a
//     head both carry the same `cos(t·omega[i])`. That is the GPT-J pairing
//     (`RopeStyle::GptJ`), which pairs adjacent channels; the NeoX layout pairs
//     `i` with `i + d/2` and would rotate the wrong channels together.

/// `generate_logspace_timeperiods(f, t_min, t_max)` — `f` periods log-spaced
/// inclusively between the two bounds.
///
/// `f32` throughout, matching the reference: computing the interpolation in `f64`
/// and narrowing at the end gives a different last bit for most `f`.
pub fn logspace_time_periods(f: usize, t_min: f32, t_max: f32) -> Vec<f32> {
    let (lmin, lmax) = (t_min.ln(), t_max.ln());
    (0..f)
        .map(|i| {
            // `torch.lerp(a, b, w)` is `a + w * (b - a)`.
            let w = if f <= 1 {
                0.0
            } else {
                i as f32 / (f - 1) as f32
            };
            (lmin + w * (lmax - lmin)).exp()
        })
        .collect()
}

/// `SinusoidalTimeEmbedding.omega` — `[dim / 2]`, one frequency per pair.
///
/// Unlike [`rotary_time_omega`] there is **no zero tail**: the sinusoidal embedding
/// uses all `dim / 2` frequencies, because it is *concatenated to* a token rather than
/// used to rotate one. `dim` must be even; an odd `dim` cannot be split into a sin half
/// and a cos half and the reference raises.
pub fn sinusoidal_time_omega(dim: usize, t_min: f32, t_max: f32) -> Vec<f32> {
    assert!(
        dim.is_multiple_of(2),
        "`dim` must be a multiple of 2, got {dim}"
    );
    let two_pi = 2.0 * std::f32::consts::PI;
    logspace_time_periods(dim / 2, t_min, t_max)
        .into_iter()
        .map(|p| two_pi / p)
        .collect()
}

/// `SinusoidalTimeEmbedding.forward` — `[N, dim]` row-major, **sin then cos**.
///
/// The concatenation order is the trap. `torch.cat((angles.sin(), angles.cos()), -1)`
/// puts every sine in the first half and every cosine in the second; the interleaved
/// `sin(0), cos(0), sin(1), …` layout used by the original Transformer paper — and by
/// [`rotary_time_tables`] above — is a **permutation** of the same numbers. A model
/// trained against one reads the other as noise, and nothing about the tensor's shape
/// or value range says which it is.
///
/// Returns `[timestamps.len() * dim]`.
pub fn sinusoidal_time_embedding(timestamps: &[f32], omega: &[f32]) -> Vec<f32> {
    let f = omega.len();
    let dim = 2 * f;
    let mut out = vec![0f32; timestamps.len() * dim];
    for (n, &t) in timestamps.iter().enumerate() {
        let row = n * dim;
        for (i, &w) in omega.iter().enumerate() {
            let angle = t * w;
            out[row + i] = angle.sin();
            out[row + f + i] = angle.cos();
        }
    }
    out
}

/// `RotaryTimeEmbedding.omega` — `[dim_head / 2]`, angular frequencies for the
/// first `rotate_dim / 2` pairs and **zero** for the rest.
///
/// The zero tail is not padding: a zero frequency means `cos = 1, sin = 0`, so
/// those channels pass through unrotated. Filling them with a default frequency
/// instead would rotate dimensions the reference leaves alone.
pub fn rotary_time_omega(dim_head: usize, rotate_dim: usize, t_min: f32, t_max: f32) -> Vec<f32> {
    let d = dim_head / 2;
    let f = (rotate_dim / 2).min(d);
    let periods = logspace_time_periods(f, t_min, t_max);
    let two_pi = 2.0 * std::f32::consts::PI;
    let mut omega = vec![0f32; d];
    for i in 0..f {
        omega[i] = two_pi / periods[i];
    }
    omega
}

/// Interleaved `cos` / `sin` / `-sin` tables for a flat timestamp array.
///
/// `timestamps` is `[B·N]`; each output is `[B·N·dim_head]`, row-major. Entries
/// `2i` and `2i+1` of every head share the same value — the GPT-J interleaving.
///
/// `-sin` is returned rather than left to the caller because POYO's
/// `rotate_value = True` path needs the **inverse** rotation to undo the value
/// rotation after attention, and deriving it by negating `sin` at three separate
/// call sites is how one of them ends up with the wrong sign.
pub fn rotary_time_tables(
    timestamps: &[f32],
    omega: &[f32],
    dim_head: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let d = omega.len();
    debug_assert_eq!(d * 2, dim_head, "omega must be dim_head/2 long");
    let n = timestamps.len();
    let mut cos = vec![0f32; n * dim_head];
    let mut sin = vec![0f32; n * dim_head];
    let mut nsin = vec![0f32; n * dim_head];
    for (t_idx, &t) in timestamps.iter().enumerate() {
        let base = t_idx * dim_head;
        for i in 0..d {
            let a = t * omega[i];
            let (c, s) = (a.cos(), a.sin());
            cos[base + 2 * i] = c;
            cos[base + 2 * i + 1] = c;
            sin[base + 2 * i] = s;
            sin[base + 2 * i + 1] = s;
            nsin[base + 2 * i] = -s;
            nsin[base + 2 * i + 1] = -s;
        }
    }
    (cos, sin, nsin)
}

/// Build the `rotate_half` half of a GptJ rotary rotation, in-graph.
///
/// `rotate_half([x₀, x₁, x₂, x₃, …]) = [-x₁, x₀, -x₃, x₂, …]` on a
/// `[B, H, N, dₕ]` tensor — adjacent-lane (GptJ) pairing.
///
/// ## Why this is built out of shape ops rather than `Op::Rope`
///
/// `Op::Rope` takes `(x, cos_cache, sin_cache)` and looks like the natural fit,
/// but it does not suit a **continuous-time** rotation:
///
///   * it wants **half-width** cache rows (`head_dim / 2` per position), while a
///     per-timestamp table is duplicated to full width;
///   * its per-token path triggers on `cos_rows == total_tokens`, and for a
///     `[B, H, N, dₕ]` input it counts `B·H·N` chunks — so a `[B, N]` timestamp
///     table does not line up;
///   * `RopeStyle::GptJ` is honoured in the fused *thunk* path
///     (`rlx-cpu/src/thunk/ops/attention.rs` sets `interleaved` from it), but the
///     standalone `Op::Rope` executor arm destructures
///     `Op::Rope { head_dim, n_rot, .. }` and implements only the NeoX pairing.
///     Whether a lowering pass routes GptJ away from that arm was not established.
///
/// So an irregularly-sampled model supplies its own tables and composes the
/// rotation from shape ops, which is what this and [`rotate_by_tables`] are for.
pub fn rotate_half_gptj(
    g: &mut rlx_ir::Graph,
    x: rlx_ir::NodeId,
    b: usize,
    h: usize,
    n: usize,
    dh: usize,
) -> rlx_ir::NodeId {
    let d2 = (dh / 2) as i64;
    let xr = g.reshape_(x, vec![b as i64, h as i64, n as i64, d2, 2]);
    let x1 = g.narrow_(xr, 4, 0, 1); // even lanes
    let x2 = g.narrow_(xr, 4, 1, 1); // odd lanes
    let nx2 = g.neg(x2);
    let stacked = g.concat_(vec![nx2, x1], 4); // [-x_odd, x_even]
    g.reshape_(stacked, vec![b as i64, h as i64, n as i64, dh as i64])
}

/// Apply a rotary rotation from supplied tables: `x·cos + rotate_half(x)·sin`.
///
/// `cos4` / `sin4` are `[B, 1, N, dₕ]`, broadcast over the head axis. Passing the
/// **negated** sine table performs the inverse rotation, which is how a
/// `rotate_value = True` attention undoes the value rotation after the attend —
/// see [`rotary_time_tables`], which returns `-sin` for exactly that reason.
#[allow(clippy::too_many_arguments)]
pub fn rotate_by_tables(
    g: &mut rlx_ir::Graph,
    x: rlx_ir::NodeId,
    cos4: rlx_ir::NodeId,
    sin4: rlx_ir::NodeId,
    b: usize,
    h: usize,
    n: usize,
    dh: usize,
) -> rlx_ir::NodeId {
    let xc = g.mul(x, cos4);
    let rh = rotate_half_gptj(g, x, b, h, n, dh);
    let rs = g.mul(rh, sin4);
    g.add(xc, rs)
}

#[cfg(test)]
mod time_rotary_tests {
    use super::*;

    #[test]
    fn periods_are_log_spaced_and_hit_both_bounds() {
        let p = logspace_time_periods(5, 1e-4, 2.0627);
        assert_eq!(p.len(), 5);
        assert!(
            (p[0] - 1e-4).abs() < 1e-9,
            "first period is t_min, got {}",
            p[0]
        );
        assert!(
            (p[4] - 2.0627).abs() < 1e-4,
            "last period is t_max, got {}",
            p[4]
        );
        // log-spaced: successive ratios are equal
        let r: Vec<f32> = (1..5).map(|i| p[i] / p[i - 1]).collect();
        for w in r.windows(2) {
            assert!(
                (w[0] - w[1]).abs() / w[0] < 1e-4,
                "ratios {r:?} are not constant"
            );
        }
    }

    #[test]
    fn a_single_period_sits_at_t_min() {
        // `f <= 1` makes the interpolation weight 0 rather than dividing by zero.
        //
        // The value is `t_min.ln().exp()`, which in f32 is 9.999999e-5 and not
        // exactly 1e-4 — a log/exp round trip is not the identity at this
        // precision. The reference computes it the same way and inherits the same
        // inexactness, so reproducing it is correct and asserting exact equality
        // was the wrong expectation.
        let p = logspace_time_periods(1, 1e-4, 2.0);
        assert_eq!(p.len(), 1);
        assert!(
            (p[0] - 1e-4).abs() < 1e-9,
            "{} should be t_min to within an f32 log/exp round trip",
            p[0]
        );
        assert_ne!(p[0], 1e-4, "premise: the round trip is NOT exact in f32");
        assert!(logspace_time_periods(0, 1e-4, 2.0).is_empty());
    }

    #[test]
    fn omega_zeroes_the_tail_past_rotate_dim() {
        // dim_head 8 -> 4 pairs; rotate_dim 4 -> only the first 2 rotate.
        let om = rotary_time_omega(8, 4, 1e-4, 2.0627);
        assert_eq!(om.len(), 4, "omega is dim_head/2 long");
        assert!(om[0] > 0.0 && om[1] > 0.0);
        assert_eq!(
            (om[2], om[3]),
            (0.0, 0.0),
            "channels past rotate_dim/2 must be left UNROTATED, not given a default frequency"
        );
    }

    #[test]
    fn a_zero_frequency_is_an_identity_rotation() {
        // Which is why the zero tail means "pass through": cos 1, sin 0.
        let om = vec![0.0f32, 0.0];
        let (cos, sin, nsin) = rotary_time_tables(&[0.7, 1.9], &om, 4);
        assert!(
            cos.iter().all(|c| *c == 1.0),
            "cos should be all ones: {cos:?}"
        );
        assert!(sin.iter().all(|s| *s == 0.0));
        assert!(nsin.iter().all(|s| *s == 0.0 || *s == -0.0));
    }

    #[test]
    fn the_layout_is_interleaved_and_duplicated() {
        // GPT-J pairing: entries 2i and 2i+1 of a head share a value. The NeoX
        // layout would pair i with i + d/2 and rotate different channels together.
        let om = rotary_time_omega(6, 6, 1e-2, 1.0);
        let (cos, sin, _) = rotary_time_tables(&[0.37], &om, 6);
        assert_eq!(cos.len(), 6);
        for i in 0..3 {
            assert_eq!(
                cos[2 * i],
                cos[2 * i + 1],
                "cos pair {i} must be duplicated"
            );
            assert_eq!(
                sin[2 * i],
                sin[2 * i + 1],
                "sin pair {i} must be duplicated"
            );
        }
        // and the three pairs are genuinely different from each other
        assert_ne!(cos[0], cos[2]);
        assert_ne!(cos[2], cos[4]);
    }

    #[test]
    fn the_phase_is_proportional_to_elapsed_time() {
        // The whole point: two events 3 ms apart rotate by 3 ms of phase, not by
        // "one position". Doubling the interval doubles the angle.
        let om = vec![1.0f32];
        let (cos, sin, _) = rotary_time_tables(&[0.0, 0.5, 1.0], &om, 2);
        assert_eq!((cos[0], sin[0]), (1.0, 0.0), "t=0 is no rotation");
        assert!((sin[2] - 0.5f32.sin()).abs() < 1e-6);
        assert!((sin[4] - 1.0f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn nsin_is_exactly_the_negation_of_sin() {
        // The inverse rotation `rotate_value=True` needs. Returned rather than
        // re-derived, so three call sites cannot disagree about the sign.
        let om = rotary_time_omega(8, 8, 1e-3, 2.0);
        let (_, sin, nsin) = rotary_time_tables(&[0.11, 1.7, 3.3], &om, 8);
        assert_eq!(sin.len(), nsin.len());
        for (s, n) in sin.iter().zip(&nsin) {
            assert_eq!(
                s.to_bits(),
                (-n).to_bits(),
                "-sin must be the exact negation"
            );
        }
    }
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

    #[test]
    fn yarn_endpoints_extrapolate_high_freq_interpolate_low_freq() {
        let (theta, head_dim) = (10_000.0f64, 128usize);
        let s = YarnScaling {
            factor: 4.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_max_position_embeddings: 8192,
            attention_factor: None,
        };
        let base = default_inv_freq(theta, head_dim);
        let yarn = yarn_scaled_inv_freq(theta, head_dim, &s);
        assert_eq!(yarn.len(), base.len());
        // Highest-frequency pair (i=0) is EXTRAPOLATED → unchanged.
        assert!(
            (yarn[0] - base[0]).abs() < 1e-9,
            "high-freq should be untouched"
        );
        // Lowest-frequency pair (last i) is INTERPOLATED → base / factor.
        let last = base.len() - 1;
        assert!(
            (yarn[last] - base[last] / 4.0).abs() < 1e-9,
            "low-freq should be base/factor: {} vs {}",
            yarn[last],
            base[last] / 4.0
        );
        // Monotone: every YaRN freq is between interp (base/factor) and extrap (base).
        for (i, (&y, &b)) in yarn.iter().zip(base.iter()).enumerate() {
            assert!(
                y <= b + 1e-9 && y >= b / 4.0 - 1e-9,
                "pair {i} out of [base/factor, base]"
            );
        }
    }

    #[test]
    fn yarn_mscale_matches_hf_default() {
        let s = YarnScaling {
            factor: 4.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_max_position_embeddings: 8192,
            attention_factor: None,
        };
        // HF default: 0.1·ln(4)+1 ≈ 1.13863.
        assert!((yarn_mscale(&s) - (0.1 * 4.0f64.ln() + 1.0)).abs() < 1e-12);
        // Explicit attention_factor overrides.
        let s2 = YarnScaling {
            attention_factor: Some(1.5),
            ..s
        };
        assert!((yarn_mscale(&s2) - 1.5).abs() < 1e-9);
        // factor ≤ 1 ⇒ no temperature change.
        let s3 = YarnScaling {
            factor: 1.0,
            attention_factor: None,
            ..s
        };
        assert!((yarn_mscale(&s3) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn yarn_tables_bake_mscale() {
        let (theta, head_dim, max_pos) = (10_000.0f64, 64usize, 8usize);
        let s = YarnScaling {
            factor: 4.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_max_position_embeddings: 4096,
            attention_factor: None,
        };
        let (cos, _sin) = build_yarn_tables(theta, head_dim, &s, max_pos);
        // pos=0 → angle 0 → cos = mscale (1·mscale), sin = 0.
        let m = yarn_mscale(&s) as f32;
        assert!(
            (cos[0] - m).abs() < 1e-5,
            "cos[0] should equal mscale {m}, got {}",
            cos[0]
        );
    }
}

#[cfg(test)]
mod sinusoidal_time_tests {
    use super::*;

    #[test]
    fn the_halves_are_sin_then_cos_not_interleaved() {
        // `torch.cat((sin, cos), -1)`: every sine first. The interleaved layout is a
        // permutation of the same numbers, and a model trained on one reads the other
        // as noise with nothing in the shape to say so.
        let omega = sinusoidal_time_omega(4, 1e-3, 1.0);
        assert_eq!(omega.len(), 2, "dim / 2 frequencies");
        let e = sinusoidal_time_embedding(&[0.25], &omega);
        assert_eq!(e.len(), 4);
        for i in 0..2 {
            let a = 0.25 * omega[i];
            assert!(
                (e[i] - a.sin()).abs() < 1e-6,
                "slot {i} is a SINE, got {}",
                e[i]
            );
            assert!(
                (e[2 + i] - a.cos()).abs() < 1e-6,
                "slot {} is a COSINE",
                2 + i
            );
        }
    }

    #[test]
    fn at_time_zero_the_embedding_is_zeros_then_ones() {
        // Every angle is 0, so sin is 0 and cos is 1 -- which is the cheapest way to
        // see the layout without computing anything.
        let omega = sinusoidal_time_omega(8, 1e-4, 2.0);
        let e = sinusoidal_time_embedding(&[0.0], &omega);
        assert_eq!(&e[..4], &[0.0, 0.0, 0.0, 0.0], "the sine half");
        assert_eq!(&e[4..], &[1.0, 1.0, 1.0, 1.0], "the cosine half");
    }

    #[test]
    fn there_is_no_zero_tail_unlike_the_rotary_family() {
        // `rotary_time_omega` zeroes past `rotate_dim / 2` so those channels pass
        // through unrotated. The sinusoidal embedding is CONCATENATED rather than
        // applied, so every frequency is used.
        let sin_omega = sinusoidal_time_omega(16, 1e-3, 1.0);
        assert!(
            sin_omega.iter().all(|w| *w > 0.0),
            "no zeros: {sin_omega:?}"
        );
        let rot = rotary_time_omega(16, 8, 1e-3, 1.0);
        assert!(
            rot.contains(&0.0),
            "the rotary family DOES have a zero tail"
        );
    }

    #[test]
    fn the_frequencies_span_the_requested_periods() {
        // omega = 2 pi / period, so the FIRST frequency is the largest (shortest
        // period) only if t_min < t_max -- and the periods are inclusive at both ends.
        let omega = sinusoidal_time_omega(6, 1e-3, 1.0);
        let two_pi = 2.0 * std::f32::consts::PI;
        assert!((omega[0] - two_pi / 1e-3).abs() / (two_pi / 1e-3) < 1e-5);
        assert!(
            (omega[2] - two_pi / 1.0).abs() < 1e-3,
            "the last period is t_max"
        );
        assert!(
            omega[0] > omega[2],
            "a shorter period is a higher frequency"
        );
    }

    #[test]
    #[should_panic(expected = "must be a multiple of 2")]
    fn an_odd_dimension_is_refused() {
        // It cannot be split into a sine half and a cosine half; the reference raises.
        sinusoidal_time_omega(7, 1e-3, 1.0);
    }

    #[test]
    fn several_timestamps_give_one_row_each() {
        let omega = sinusoidal_time_omega(4, 1e-2, 1.0);
        let e = sinusoidal_time_embedding(&[0.0, 0.5, 1.0], &omega);
        assert_eq!(e.len(), 3 * 4, "row-major [N, dim]");
        assert_eq!(&e[..2], &[0.0, 0.0], "row 0's sine half is zero");
        assert_eq!(&e[2..4], &[1.0, 1.0], "row 0's cosine half is one");
        assert_ne!(e[4], 0.0, "row 1 is not row 0");
    }
}
