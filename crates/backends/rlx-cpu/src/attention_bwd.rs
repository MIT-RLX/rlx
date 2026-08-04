// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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

    // scores = scale · (Q · Kᵀ)   [q_seq, k_seq]. BLAS sgemm (trans_b) instead
    // of the scalar triple loop — Accelerate runs single-threaded per small
    // per-head tile, composing safely under the batch×heads rayon parallelism.
    unsafe {
        crate::blas::sgemm_general(
            q.as_ptr(),
            k.as_ptr(),
            scores.as_mut_ptr(),
            q_seq,
            k_seq,
            head_dim,
            scale,
            0.0,
            head_dim,
            head_dim,
            k_seq,
            false,
            true,
        );
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
            // dV = Pᵀ · dy   [k_seq, head_dim]
            unsafe {
                crate::blas::sgemm_general(
                    scores.as_ptr(),
                    dy.as_ptr(),
                    out.as_mut_ptr(),
                    k_seq,
                    head_dim,
                    q_seq,
                    1.0,
                    0.0,
                    k_seq,
                    head_dim,
                    head_dim,
                    true,
                    false,
                );
            }
        }
        AttentionBwdWrt::Query | AttentionBwdWrt::Key => {
            // dP = dy · Vᵀ   [q_seq, k_seq]
            unsafe {
                crate::blas::sgemm_general(
                    dy.as_ptr(),
                    v.as_ptr(),
                    dp.as_mut_ptr(),
                    q_seq,
                    k_seq,
                    head_dim,
                    1.0,
                    0.0,
                    head_dim,
                    head_dim,
                    k_seq,
                    false,
                    true,
                );
            }
            // softmax backward (elementwise, memory-bound — stays scalar):
            //   ds = P ⊙ (dP − rowsum(P ⊙ dP)) · scale
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
                    // dQ = ds · K   [q_seq, head_dim]  (ds already carries `scale`)
                    unsafe {
                        crate::blas::sgemm_general(
                            ds.as_ptr(),
                            k.as_ptr(),
                            out.as_mut_ptr(),
                            q_seq,
                            head_dim,
                            k_seq,
                            1.0,
                            0.0,
                            k_seq,
                            head_dim,
                            head_dim,
                            false,
                            false,
                        );
                    }
                }
                AttentionBwdWrt::Key => {
                    // dK = dsᵀ · Q   [k_seq, head_dim]
                    unsafe {
                        crate::blas::sgemm_general(
                            ds.as_ptr(),
                            q.as_ptr(),
                            out.as_mut_ptr(),
                            k_seq,
                            head_dim,
                            q_seq,
                            1.0,
                            0.0,
                            k_seq,
                            head_dim,
                            head_dim,
                            true,
                            false,
                        );
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
    // Each (bi, hi) head is independent — it reads disjoint input slices and
    // writes a disjoint output region — so parallelize over batch×heads. This
    // kernel was single-threaded naive scalar and dominated CPU training time
    // (~110 ms/call, 92% of the CPU backward). Scratch is per-thread; output
    // writes go through a raw base pointer at disjoint offsets, so the result
    // is bit-identical to the serial version.
    let out_ptr = out_data.as_mut_ptr() as usize;
    // A/B toggle for measuring the parallelization win (grain = total → serial).
    let grain = if std::env::var_os("RLX_ATTN_BWD_SERIAL").is_some() {
        (batch * num_heads).max(1)
    } else {
        1
    };
    crate::pool::par_for(batch * num_heads, grain, &|off, cnt| {
        let mut q_buf = vec![0f32; q_tile];
        let mut k_buf = vec![0f32; k_tile];
        let mut v_buf = vec![0f32; k_tile];
        let mut dy_buf = vec![0f32; q_tile];
        let mut out_buf = vec![0f32; q_tile.max(k_tile)];
        for idx in off..off + cnt {
            let bi = idx / num_heads;
            let hi = idx % num_heads;
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
                    // SAFETY: disjoint per (bi,hi) — no aliasing across threads.
                    unsafe {
                        std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(out_base), out_len)
                    },
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

                // SAFETY: each write targets this head's disjoint column block
                // (`h_off..h_off+head_dim`) in its own batch row — no overlap
                // across the parallel (bi,hi) iterations.
                if matches!(wrt, AttentionBwdWrt::Key | AttentionBwdWrt::Value) {
                    for ki in 0..k_seq {
                        let dst = k_batch + ki * hs + h_off;
                        let src = ki * head_dim;
                        unsafe {
                            std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(dst), head_dim)
                        }
                        .copy_from_slice(&out_buf[src..src + head_dim]);
                    }
                } else {
                    for qi in 0..q_seq {
                        let dst = q_batch + qi * hs + h_off;
                        let src = qi * head_dim;
                        unsafe {
                            std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(dst), head_dim)
                        }
                        .copy_from_slice(&out_buf[src..src + head_dim]);
                    }
                }
            }
        }
    });
}

/// Dense per-head backward computing dQ, dK **and** dV from a **single** score +
/// softmax recompute (8 GEMMs → 5). Bit-comparable to the three `wrt` calls.
#[inline]
#[allow(clippy::too_many_arguments)]
fn backward_dense_head_all(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    dy: &[f32],
    out_q: &mut [f32],
    out_k: &mut [f32],
    out_v: &mut [f32],
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

    unsafe {
        crate::blas::sgemm_general(
            q.as_ptr(),
            k.as_ptr(),
            scores.as_mut_ptr(),
            q_seq,
            k_seq,
            head_dim,
            scale,
            0.0,
            head_dim,
            head_dim,
            k_seq,
            false,
            true,
        );
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

    // dV = Pᵀ·dy
    unsafe {
        crate::blas::sgemm_general(
            scores.as_ptr(),
            dy.as_ptr(),
            out_v.as_mut_ptr(),
            k_seq,
            head_dim,
            q_seq,
            1.0,
            0.0,
            k_seq,
            head_dim,
            head_dim,
            true,
            false,
        );
    }
    // dP = dy·Vᵀ
    unsafe {
        crate::blas::sgemm_general(
            dy.as_ptr(),
            v.as_ptr(),
            dp.as_mut_ptr(),
            q_seq,
            k_seq,
            head_dim,
            1.0,
            0.0,
            head_dim,
            head_dim,
            k_seq,
            false,
            true,
        );
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
    // dQ = ds·K
    unsafe {
        crate::blas::sgemm_general(
            ds.as_ptr(),
            k.as_ptr(),
            out_q.as_mut_ptr(),
            q_seq,
            head_dim,
            k_seq,
            1.0,
            0.0,
            k_seq,
            head_dim,
            head_dim,
            false,
            false,
        );
    }
    // dK = dsᵀ·Q
    unsafe {
        crate::blas::sgemm_general(
            ds.as_ptr(),
            q.as_ptr(),
            out_k.as_mut_ptr(),
            k_seq,
            head_dim,
            q_seq,
            1.0,
            0.0,
            k_seq,
            head_dim,
            head_dim,
            true,
            false,
        );
    }
}

/// Fused attention backward: dQ, dK, dV in one pass (one score+softmax per head).
/// `out_q`/`out_k`/`out_v` are disjoint sub-ranges of one packed output buffer.
#[allow(clippy::too_many_arguments)]
pub fn attention_backward_all(
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
    dy_data: &[f32],
    out_q: &mut [f32],
    out_k: &mut [f32],
    out_v: &mut [f32],
    batch: usize,
    num_heads: usize,
    q_seq: usize,
    k_seq: usize,
    head_dim: usize,
    mask_kind: MaskKind,
    mask_data: &[f32],
    bhsd: bool,
) {
    // NOTE: do NOT pre-zero the outputs. The non-bhsd path scatters every output
    // element (all batches × all heads × all rows), so zeroing is unnecessary —
    // and if the packed output happens to overlap the `v` input in the arena,
    // pre-zeroing would wipe `v` before `dP = dy·Vᵀ` reads it. (The bhsd path
    // also writes every element via the per-head GEMMs.)
    let cfg = crate::config::RuntimeConfig::global();
    let mask_thr = cfg.mask_binary_threshold;
    let mask_neg = cfg.attn_mask_neg_inf;
    let hs = num_heads * head_dim;
    let q_tile = q_seq * head_dim;
    let k_tile = k_seq * head_dim;
    let oq = out_q.as_mut_ptr() as usize;
    let ok = out_k.as_mut_ptr() as usize;
    let ov = out_v.as_mut_ptr() as usize;
    let grain = if std::env::var_os("RLX_ATTN_BWD_SERIAL").is_some() {
        (batch * num_heads).max(1)
    } else {
        1
    };
    crate::pool::par_for(batch * num_heads, grain, &|off, cnt| {
        let mut qb = vec![0f32; q_tile];
        let mut kb = vec![0f32; k_tile];
        let mut vb = vec![0f32; k_tile];
        let mut dyb = vec![0f32; q_tile];
        let mut oqb = vec![0f32; q_tile];
        let mut okb = vec![0f32; k_tile];
        let mut ovb = vec![0f32; k_tile];
        for idx in off..off + cnt {
            let bi = idx / num_heads;
            let hi = idx % num_heads;
            let mask_head: &[f32] = match mask_kind {
                MaskKind::Custom if !mask_data.is_empty() => {
                    &mask_data[bi * k_seq..(bi + 1) * k_seq]
                }
                MaskKind::Bias if !mask_data.is_empty() => {
                    let o = (bi * num_heads + hi) * q_seq * k_seq;
                    &mask_data[o..o + q_seq * k_seq]
                }
                _ => &[],
            };
            if bhsd {
                let qbase = bi * num_heads * q_seq * head_dim + hi * q_seq * head_dim;
                let kbase = bi * num_heads * k_seq * head_dim + hi * k_seq * head_dim;
                unsafe {
                    backward_dense_head_all(
                        &q_data[qbase..qbase + q_tile],
                        &k_data[kbase..kbase + k_tile],
                        &v_data[kbase..kbase + k_tile],
                        &dy_data[qbase..qbase + q_tile],
                        std::slice::from_raw_parts_mut((oq as *mut f32).add(qbase), q_tile),
                        std::slice::from_raw_parts_mut((ok as *mut f32).add(kbase), k_tile),
                        std::slice::from_raw_parts_mut((ov as *mut f32).add(kbase), k_tile),
                        q_seq,
                        k_seq,
                        head_dim,
                        mask_kind,
                        mask_head,
                        mask_thr,
                        mask_neg,
                    );
                }
            } else {
                let qb0 = bi * q_seq * hs;
                let kb0 = bi * k_seq * hs;
                let hoff = hi * head_dim;
                for qi in 0..q_seq {
                    let s = qb0 + qi * hs + hoff;
                    qb[qi * head_dim..qi * head_dim + head_dim]
                        .copy_from_slice(&q_data[s..s + head_dim]);
                    dyb[qi * head_dim..qi * head_dim + head_dim]
                        .copy_from_slice(&dy_data[s..s + head_dim]);
                }
                for ki in 0..k_seq {
                    let s = kb0 + ki * hs + hoff;
                    kb[ki * head_dim..ki * head_dim + head_dim]
                        .copy_from_slice(&k_data[s..s + head_dim]);
                    vb[ki * head_dim..ki * head_dim + head_dim]
                        .copy_from_slice(&v_data[s..s + head_dim]);
                }
                backward_dense_head_all(
                    &qb, &kb, &vb, &dyb, &mut oqb, &mut okb, &mut ovb, q_seq, k_seq, head_dim,
                    mask_kind, mask_head, mask_thr, mask_neg,
                );
                unsafe {
                    for qi in 0..q_seq {
                        let d = qb0 + qi * hs + hoff;
                        let s = qi * head_dim;
                        std::slice::from_raw_parts_mut((oq as *mut f32).add(d), head_dim)
                            .copy_from_slice(&oqb[s..s + head_dim]);
                    }
                    for ki in 0..k_seq {
                        let d = kb0 + ki * hs + hoff;
                        let s = ki * head_dim;
                        std::slice::from_raw_parts_mut((ok as *mut f32).add(d), head_dim)
                            .copy_from_slice(&okb[s..s + head_dim]);
                        std::slice::from_raw_parts_mut((ov as *mut f32).add(d), head_dim)
                            .copy_from_slice(&ovb[s..s + head_dim]);
                    }
                }
            }
        }
    });
}

#[cfg(test)]
mod fuse_tests {
    use super::*;
    use rlx_ir::op::{AttentionBwdWrt, MaskKind};

    #[test]
    fn all_matches_three_separate() {
        let (b, nh, s, dh) = (2usize, 2usize, 5usize, 4usize);
        let n = b * s * nh * dh;
        let mk = |seed: u64| -> Vec<f32> {
            (0..n)
                .map(|i| (((i as u64 * 2654435761 + seed) % 1000) as f32 / 500.0) - 1.0)
                .collect()
        };
        let (q, k, v, dy) = (mk(1), mk(2), mk(3), mk(4));
        // Reference: three separate wrt calls.
        let mut dq = vec![0f32; n];
        let mut dk = vec![0f32; n];
        let mut dv = vec![0f32; n];
        attention_backward(
            AttentionBwdWrt::Query,
            &q,
            &k,
            &v,
            &dy,
            &mut dq,
            b,
            nh,
            s,
            s,
            dh,
            MaskKind::Causal,
            &[],
            false,
        );
        attention_backward(
            AttentionBwdWrt::Key,
            &q,
            &k,
            &v,
            &dy,
            &mut dk,
            b,
            nh,
            s,
            s,
            dh,
            MaskKind::Causal,
            &[],
            false,
        );
        attention_backward(
            AttentionBwdWrt::Value,
            &q,
            &k,
            &v,
            &dy,
            &mut dv,
            b,
            nh,
            s,
            s,
            dh,
            MaskKind::Causal,
            &[],
            false,
        );
        // Fused.
        let mut fq = vec![0f32; n];
        let mut fk = vec![0f32; n];
        let mut fv = vec![0f32; n];
        attention_backward_all(
            &q,
            &k,
            &v,
            &dy,
            &mut fq,
            &mut fk,
            &mut fv,
            b,
            nh,
            s,
            s,
            dh,
            MaskKind::Causal,
            &[],
            false,
        );
        let maxdiff = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max)
        };
        let (ddq, ddk, ddv) = (maxdiff(&dq, &fq), maxdiff(&dk, &fk), maxdiff(&dv, &fv));
        let normq = dq.iter().map(|x| x.abs()).fold(0f32, f32::max);
        println!(
            "dQ maxdiff={ddq:.2e} (|dq|={normq:.2e})  dK maxdiff={ddk:.2e}  dV maxdiff={ddv:.2e}"
        );
        assert!(ddq < 1e-4 && ddk < 1e-4 && ddv < 1e-4, "fused != separate");
    }
}
