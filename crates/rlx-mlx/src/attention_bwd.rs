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
//! Compositional scaled dot-product attention backward for MLX (rank-4 BHSd).

use rlx_ir::DType;
use rlx_ir::op::{AttentionBwdWrt, MaskKind};

use crate::array::{Array, MlxError};
use crate::ffi::MlxReduce;
use crate::ops;

fn causal_additive_mask(sq: usize, sk: usize) -> Result<Array, MlxError> {
    let neg_inf = f32::NEG_INFINITY;
    let sq_u = sq;
    let sk_u = sk;
    let mut buf = vec![neg_inf; sq_u * sk_u];
    for qi in 0..sq_u {
        for ki in 0..sk_u {
            if ki <= qi {
                buf[qi * sk_u + ki] = 0.0;
            }
        }
    }
    Array::from_f32_slice(&buf, &[sq_u, sk_u], DType::F32)
}

fn apply_score_mask(
    scores: &Array,
    mask_kind: MaskKind,
    mask_additive: Option<&Array>,
    sq: usize,
    sk: usize,
    _window: usize,
) -> Result<Array, MlxError> {
    let mut s = scores.clone_handle()?;
    match mask_kind {
        MaskKind::None => {}
        MaskKind::Causal => {
            let m2 = causal_additive_mask(sq, sk)?;
            let m4 = ops::reshape(&m2, &[1, 1, sq as i32, sk as i32])?;
            s = ops::add(&s, &m4)?;
        }
        MaskKind::SlidingWindow(w) => {
            let neg_inf = f32::NEG_INFINITY;
            let sq_u = sq;
            let sk_u = sk;
            let win = w as i64;
            let mut buf = vec![neg_inf; sq_u * sk_u];
            for qi in 0..sq_u {
                for ki in 0..sk_u {
                    let q = qi as i64;
                    let k = ki as i64;
                    if k <= q && (q - k) <= win {
                        buf[qi * sk_u + ki] = 0.0;
                    }
                }
            }
            let m2 = Array::from_f32_slice(&buf, &[sq_u, sk_u], DType::F32)?;
            let m4 = ops::reshape(&m2, &[1, 1, sq as i32, sk as i32])?;
            s = ops::add(&s, &m4)?;
        }
        MaskKind::Custom | MaskKind::Bias => {
            if let Some(m) = mask_additive {
                s = ops::add(&s, m)?;
            }
        }
    }
    Ok(s)
}

/// Backward w.r.t. Q, K, or V. Inputs are rank-4 `[B, H, S, D]` (after
/// layout normalization in `lower.rs`).
pub fn attention_backward_bhsd(
    wrt: AttentionBwdWrt,
    q: &Array,
    k: &Array,
    v: &Array,
    dy: &Array,
    head_dim: i32,
    mask_kind: MaskKind,
    mask_additive: Option<&Array>,
    window: usize,
) -> Result<Array, MlxError> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let sh = q.shape()?;
    let sq = sh[2];
    let sk = k.shape()?[2];
    let dtype = DType::F32;
    let scale_a = Array::from_f32_slice(&[scale], &[1], dtype)?;

    let k_t = ops::transpose(k, &[0, 1, 3, 2])?;
    let mut scores = ops::matmul(q, &k_t)?;
    scores = ops::mul(&scores, &scale_a)?;
    scores = apply_score_mask(&scores, mask_kind, mask_additive, sq, sk, window)?;

    let p = ops::softmax(&scores, -1)?;
    let v_t = ops::transpose(v, &[0, 1, 3, 2])?;
    let p_t = ops::transpose(&p, &[0, 1, 3, 2])?;

    let dp = ops::matmul(dy, &v_t)?;
    let p_dp = ops::mul(&p, &dp)?;
    let sum = ops::reduce(&p_dp, MlxReduce::Sum, &[-1], true)?;
    let dscores = ops::sub(&p_dp, &ops::mul(&p, &sum)?)?;
    let dscores = ops::mul(&dscores, &scale_a)?;

    let out = match wrt {
        AttentionBwdWrt::Query => ops::matmul(&dscores, k)?,
        AttentionBwdWrt::Key => {
            let ds_t = ops::transpose(&dscores, &[0, 1, 3, 2])?;
            ops::matmul(&ds_t, q)?
        }
        AttentionBwdWrt::Value => ops::matmul(&p_t, dy)?,
    };
    Ok(out)
}
