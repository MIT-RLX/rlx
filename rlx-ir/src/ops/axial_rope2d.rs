// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! SAM2-style axial 2-D RoPE on `[batch, seq, num_heads * head_dim]`.

use crate::{Graph, NodeId, Op, Shape};

/// Apply axial 2-D RoPE on flattened `[nh, n_tokens, head_dim]` layout.
pub fn apply_axial_rope2d(
    x: &[f32],
    num_heads: usize,
    n_tokens: usize,
    head_dim: usize,
    end_x: usize,
    end_y: usize,
    theta: f32,
    repeat_factor: usize,
) -> Vec<f32> {
    debug_assert!(head_dim.is_multiple_of(4));
    let half = head_dim / 2;
    let q4 = head_dim / 4;
    let spatial = end_x * end_y;
    let repeat = repeat_factor.max(1);
    debug_assert_eq!(n_tokens, spatial * repeat);

    let mut freqs = vec![0f32; q4];
    for i in 0..q4 {
        freqs[i] = 1.0 / theta.powf((4 * i) as f32 / head_dim as f32);
    }
    let mut cs_x = vec![0f32; spatial * q4];
    let mut sn_x = vec![0f32; spatial * q4];
    let mut cs_y = vec![0f32; spatial * q4];
    let mut sn_y = vec![0f32; spatial * q4];
    for pos in 0..spatial {
        let tx = (pos % end_x) as f32;
        let ty = (pos / end_x) as f32;
        for c in 0..q4 {
            let ax = tx * freqs[c];
            let ay = ty * freqs[c];
            cs_x[pos * q4 + c] = ax.cos();
            sn_x[pos * q4 + c] = ax.sin();
            cs_y[pos * q4 + c] = ay.cos();
            sn_y[pos * q4 + c] = ay.sin();
        }
    }
    let mut cos_x = vec![0f32; n_tokens * q4];
    let mut sin_x = vec![0f32; n_tokens * q4];
    let mut cos_y = vec![0f32; n_tokens * q4];
    let mut sin_y = vec![0f32; n_tokens * q4];
    for tok in 0..n_tokens {
        let pos = tok / repeat;
        for c in 0..q4 {
            cos_x[tok * q4 + c] = cs_x[pos * q4 + c];
            sin_x[tok * q4 + c] = sn_x[pos * q4 + c];
            cos_y[tok * q4 + c] = cs_y[pos * q4 + c];
            sin_y[tok * q4 + c] = sn_y[pos * q4 + c];
        }
    }

    // `[batch, seq, num_heads * head_dim]` layout (token-major, heads interleaved).
    let hs = num_heads * head_dim;
    let mut out = x.to_vec();
    for tok in 0..n_tokens {
        let pos = tok / repeat;
        for h in 0..num_heads {
            let base = tok * hs + h * head_dim;
            for c in 0..q4 {
                let ix0 = base + 2 * c;
                let ix1 = base + 2 * c + 1;
                let x0 = out[ix0];
                let x1 = out[ix1];
                let co = cos_x[tok * q4 + c];
                let si = sin_x[tok * q4 + c];
                out[ix0] = x0 * co - x1 * si;
                out[ix1] = x0 * si + x1 * co;
            }
            for c in 0..q4 {
                let ix0 = base + half + 2 * c;
                let ix1 = base + half + 2 * c + 1;
                let x0 = out[ix0];
                let x1 = out[ix1];
                let co = cos_y[tok * q4 + c];
                let si = sin_y[tok * q4 + c];
                out[ix0] = x0 * co - x1 * si;
                out[ix1] = x0 * si + x1 * co;
            }
        }
    }
    out
}

impl Graph {
    /// `x`: `[1, seq, num_heads * head_dim]` → same shape.
    pub fn axial_rope2d(
        &mut self,
        x: NodeId,
        end_x: usize,
        end_y: usize,
        head_dim: usize,
        num_heads: usize,
        theta: f32,
        repeat_factor: usize,
    ) -> NodeId {
        let s = crate::shape::unary_shape(self.shape(x));
        self.push(
            Op::AxialRope2d {
                end_x,
                end_y,
                head_dim,
                num_heads,
                theta,
                repeat_factor,
            },
            vec![x],
            s,
            None,
        )
    }
}
