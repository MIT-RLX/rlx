// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Minimal RoPE cos/sin tables for mlx-lm Llama-like graphs.
//! (Avoids a `rlx-flow` dependency from `rlx-mlx-io`.)

/// `inv_freq[i] = 1 / theta^(2i / head_dim)` — length `head_dim / 2`.
pub fn default_inv_freq(rope_theta: f64, head_dim: usize) -> Vec<f64> {
    (0..head_dim)
        .step_by(2)
        .map(|i| 1.0 / rope_theta.powf(i as f64 / head_dim as f64))
        .collect()
}

/// Build `[max_pos, head_dim/2]` cos/sin tables.
pub fn build_default_tables(
    rope_theta: f64,
    head_dim: usize,
    max_pos: usize,
) -> (Vec<f32>, Vec<f32>) {
    let inv = default_inv_freq(rope_theta, head_dim);
    let half = inv.len();
    let mut cos = vec![0f32; max_pos * half];
    let mut sin = vec![0f32; max_pos * half];
    for pos in 0..max_pos {
        for (i, &freq) in inv.iter().enumerate() {
            let angle = pos as f64 * freq;
            cos[pos * half + i] = angle.cos() as f32;
            sin[pos * half + i] = angle.sin() as f32;
        }
    }
    (cos, sin)
}

pub fn f32_le_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
