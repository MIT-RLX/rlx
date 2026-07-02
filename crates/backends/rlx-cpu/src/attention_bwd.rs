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

//! Scaled dot-product attention backward (recomputes scores + softmax).

use rlx_ir::op::{AttentionBwdWrt, MaskKind};

/// Apply the same synthetic masks as forward [`Thunk::Attention`].
#[inline]
fn apply_synthetic_mask(scores: &mut [f32], q_seq: usize, k_seq: usize, kind: MaskKind) {
    let neg = crate::config::RuntimeConfig::global().attn_mask_neg_inf;
    let q_offset = k_seq.saturating_sub(q_seq);
    match kind {
        MaskKind::None | MaskKind::Custom | MaskKind::Bias => {}
        MaskKind::Causal => {
            for qi in 0..q_seq {
                let abs_q = q_offset + qi;
                for ki in (abs_q + 1)..k_seq {
                    scores[qi * k_seq + ki] = neg;
                }
            }
        }
        MaskKind::SlidingWindow(w) => {
            for qi in 0..q_seq {
                let abs_q = q_offset + qi;
                let lo = abs_q.saturating_sub(w);
                for ki in 0..k_seq {
                    if ki < lo || ki > abs_q {
                        scores[qi * k_seq + ki] = neg;
                    }
                }
            }
        }
    }
}

/// Dense per-head tile: `q`, `k`, `v`, `dy`, `out` are `[seq, head_dim]`.
#[inline]
fn backward_dense_head(
    wrt: AttentionBwdWrt,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    dy: &[f32],
    out: &mut [f32],
    q_seq: usize,
    k_seq: usize,
    head_dim: usize,
    mask_kind: MaskKind,
    mask_head: &[f32],
    mask_thr: f32,
    mask_neg: f32,
) {
    let scale = (head_dim as f32).sqrt().recip();
    let ss = q_seq * k_seq;
    let mut scores = vec![0f32; ss];
    let mut dp = vec![0f32; ss];
    let mut ds = vec![0f32; ss];

    for v in out.iter_mut() {
        *v = 0.0;
    }

    for qi in 0..q_seq {
        for ki in 0..k_seq {
            let mut dot = 0f32;
            for d in 0..head_dim {
                dot += q[qi * head_dim + d] * k[ki * head_dim + d];
            }
            scores[qi * k_seq + ki] = dot * scale;
        }
    }

    if matches!(mask_kind, MaskKind::Custom) && !mask_head.is_empty() {
        for qi in 0..q_seq {
            for ki in 0..k_seq {
                if mask_head[ki] < mask_thr {
                    scores[qi * k_seq + ki] = mask_neg;
                }
            }
        }
    }
    if matches!(mask_kind, MaskKind::Bias) && !mask_head.is_empty() {
        for i in 0..ss {
            scores[i] += mask_head[i];
        }
    }
    apply_synthetic_mask(&mut scores, q_seq, k_seq, mask_kind);
    crate::kernels::neon_softmax(&mut scores, q_seq, k_seq);

    match wrt {
        AttentionBwdWrt::Value => {
            for ki in 0..k_seq {
                for d in 0..head_dim {
                    let mut acc = 0f32;
                    for qi in 0..q_seq {
                        acc += scores[qi * k_seq + ki] * dy[qi * head_dim + d];
                    }
                    out[ki * head_dim + d] = acc;
                }
            }
        }
        AttentionBwdWrt::Query | AttentionBwdWrt::Key => {
            for qi in 0..q_seq {
                for ki in 0..k_seq {
                    let mut acc = 0f32;
                    for d in 0..head_dim {
                        acc += dy[qi * head_dim + d] * v[ki * head_dim + d];
                    }
                    dp[qi * k_seq + ki] = acc;
                }
            }
            for qi in 0..q_seq {
                let mut row_sum = 0f32;
                for ki in 0..k_seq {
                    row_sum += scores[qi * k_seq + ki] * dp[qi * k_seq + ki];
                }
                for ki in 0..k_seq {
                    let p = scores[qi * k_seq + ki];
                    ds[qi * k_seq + ki] = p * (dp[qi * k_seq + ki] - row_sum) * scale;
                }
            }
            match wrt {
                AttentionBwdWrt::Query => {
                    for qi in 0..q_seq {
                        for d in 0..head_dim {
                            let mut acc = 0f32;
                            for ki in 0..k_seq {
                                acc += ds[qi * k_seq + ki] * k[ki * head_dim + d];
                            }
                            out[qi * head_dim + d] = acc;
                        }
                    }
                }
                AttentionBwdWrt::Key => {
                    for ki in 0..k_seq {
                        for d in 0..head_dim {
                            let mut acc = 0f32;
                            for qi in 0..q_seq {
                                acc += ds[qi * k_seq + ki] * q[qi * head_dim + d];
                            }
                            out[ki * head_dim + d] = acc;
                        }
                    }
                }
                _ => unreachable!(),
            }
        }
    }
}

/// Full-tensor attention backward for `[B, S, H, D]` or `[B, H, S, D]`.
pub fn attention_backward(
    wrt: AttentionBwdWrt,
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
    dy_data: &[f32],
    out_data: &mut [f32],
    batch: usize,
    num_heads: usize,
    q_seq: usize,
    k_seq: usize,
    head_dim: usize,
    mask_kind: MaskKind,
    mask_data: &[f32],
    bhsd: bool,
) {
    for v in out_data.iter_mut() {
        *v = 0.0;
    }
    let cfg = crate::config::RuntimeConfig::global();
    let mask_thr = cfg.mask_binary_threshold;
    let mask_neg = cfg.attn_mask_neg_inf;
    let hs = num_heads * head_dim;
    let q_tile = q_seq * head_dim;
    let k_tile = k_seq * head_dim;
    let mut q_buf = vec![0f32; q_tile];
    let mut k_buf = vec![0f32; k_tile];
    let mut v_buf = vec![0f32; k_tile];
    let mut dy_buf = vec![0f32; q_tile];
    let mut out_buf = vec![0f32; q_tile.max(k_tile)];

    for bi in 0..batch {
        for hi in 0..num_heads {
            let mask_head: &[f32] = match mask_kind {
                MaskKind::Custom if !mask_data.is_empty() => {
                    &mask_data[bi * k_seq..(bi + 1) * k_seq]
                }
                MaskKind::Bias if !mask_data.is_empty() => {
                    let off = (bi * num_heads + hi) * q_seq * k_seq;
                    &mask_data[off..off + q_seq * k_seq]
                }
                _ => &[],
            };

            if bhsd {
                let q_base = bi * num_heads * q_seq * head_dim + hi * q_seq * head_dim;
                let k_base = bi * num_heads * k_seq * head_dim + hi * k_seq * head_dim;
                let (out_base, out_len) = match wrt {
                    AttentionBwdWrt::Key | AttentionBwdWrt::Value => (k_base, k_tile),
                    AttentionBwdWrt::Query => (q_base, q_tile),
                };
                backward_dense_head(
                    wrt,
                    &q_data[q_base..q_base + q_tile],
                    &k_data[k_base..k_base + k_tile],
                    &v_data[k_base..k_base + k_tile],
                    &dy_data[q_base..q_base + q_tile],
                    &mut out_data[out_base..out_base + out_len],
                    q_seq,
                    k_seq,
                    head_dim,
                    mask_kind,
                    mask_head,
                    mask_thr,
                    mask_neg,
                );
            } else {
                let q_batch = bi * q_seq * hs;
                let k_batch = bi * k_seq * hs;
                let h_off = hi * head_dim;
                for qi in 0..q_seq {
                    let src = q_batch + qi * hs + h_off;
                    let dst = qi * head_dim;
                    out_buf[dst..dst + head_dim].copy_from_slice(&q_data[src..src + head_dim]);
                }
                q_buf.copy_from_slice(&out_buf[..q_tile]);
                for ki in 0..k_seq {
                    let src = k_batch + ki * hs + h_off;
                    let dst = ki * head_dim;
                    k_buf[dst..dst + head_dim].copy_from_slice(&k_data[src..src + head_dim]);
                    v_buf[dst..dst + head_dim].copy_from_slice(&v_data[src..src + head_dim]);
                }
                for qi in 0..q_seq {
                    let src = q_batch + qi * hs + h_off;
                    let dst = qi * head_dim;
                    dy_buf[dst..dst + head_dim].copy_from_slice(&dy_data[src..src + head_dim]);
                }

                let out_len = match wrt {
                    AttentionBwdWrt::Key | AttentionBwdWrt::Value => k_tile,
                    AttentionBwdWrt::Query => q_tile,
                };
                backward_dense_head(
                    wrt,
                    &q_buf,
                    &k_buf,
                    &v_buf,
                    &dy_buf,
                    &mut out_buf[..out_len],
                    q_seq,
                    k_seq,
                    head_dim,
                    mask_kind,
                    mask_head,
                    mask_thr,
                    mask_neg,
                );

                if matches!(wrt, AttentionBwdWrt::Key | AttentionBwdWrt::Value) {
                    for ki in 0..k_seq {
                        let dst = k_batch + ki * hs + h_off;
                        let src = ki * head_dim;
                        out_data[dst..dst + head_dim]
                            .copy_from_slice(&out_buf[src..src + head_dim]);
                    }
                } else {
                    for qi in 0..q_seq {
                        let dst = q_batch + qi * hs + h_off;
                        let src = qi * head_dim;
                        out_data[dst..dst + head_dim]
                            .copy_from_slice(&out_buf[src..src + head_dim]);
                    }
                }
            }
        }
    }
}
