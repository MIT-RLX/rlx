// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! GPU scaled dot-product attention backward (recomputes scores + softmax).

use crate::kernels::Kernels;
use metal::{Buffer, ComputeCommandEncoderRef, MTLSize};
use rlx_ir::Graph;
use rlx_ir::op::{AttentionBwdWrt, Op};

pub fn scratch_bytes(graph: &Graph) -> usize {
    let mut max = 0usize;
    for node in graph.nodes() {
        // Both the per-`wrt` op and the IR-level fused `AttentionBackwardAll`
        // (from `FuseAttentionBackwardAll`) drive the same batched kernels and
        // so need the same scores+dp+ds scratch. Sizing off only `AttentionBackward`
        // would leave the fused path with a 0-sized scratch (writes land at arena
        // offset 0 → corruption), so match both.
        let num_heads = match &node.op {
            Op::AttentionBackward { num_heads, .. }
            | Op::AttentionBackwardAll { num_heads, .. } => *num_heads,
            _ => continue,
        };
        {
            let q_shape = &graph.node(node.inputs[0]).shape;
            let k_shape = &graph.node(node.inputs[1]).shape;
            let sq = q_shape.dim(1).unwrap_static();
            let sk = if k_shape.rank() >= 2 {
                k_shape.dim(1).unwrap_static()
            } else {
                k_shape.dim(0).unwrap_static()
            };
            let ss = sq.saturating_mul(sk);
            // scores + dp + ds, sized for `tile` heads processed in parallel.
            // (The fused batched path uses `tile·3·ss`; the per-wrt fallback uses
            // `3·ss` ≤ that since tile ≥ 1.) `batch·num_heads` bounds the tile.
            let batch = q_shape.dim(0).unwrap_static();
            let tile = attn_bwd_tile(batch.saturating_mul(num_heads), ss);
            max = max.max(tile * 3 * ss * std::mem::size_of::<f32>());
        }
    }
    max
}

pub fn use_gpu(mask_kind: u32, bhsd: u32, sq: usize, sk: usize, scratch_off: usize) -> bool {
    if scratch_off == 0 {
        return false;
    }
    if bhsd != 0 {
        return false;
    }
    if sq == 0 || sk == 0 {
        return false;
    }
    if rlx_ir::env::var("RLX_METAL_ATTN_BWD_GPU")
        .map(|v| v == "0")
        .unwrap_or(false)
    {
        return false;
    }
    // Custom / bias masks still use CPU reference for now.
    !matches!(mask_kind, 2 | 4)
}

#[allow(clippy::too_many_arguments)]
pub fn encode_attention_bwd(
    enc: &ComputeCommandEncoderRef,
    kk: &Kernels,
    buffer: &Buffer,
    q: usize,
    k: usize,
    v: usize,
    dy: usize,
    out: usize,
    scratch: usize,
    batch: u32,
    sq: u32,
    sk: u32,
    heads: u32,
    head_dim: u32,
    mask_kind: u32,
    window: u32,
    wrt: u32,
) {
    let ss = (sq as usize).saturating_mul(sk as usize);
    let scores = scratch;
    let dp = scratch + ss * 4;
    let ds = scratch + ss * 8;
    let wrt_ir = match wrt {
        0 => AttentionBwdWrt::Query,
        1 => AttentionBwdWrt::Key,
        _ => AttentionBwdWrt::Value,
    };
    let scale = (head_dim as f32).sqrt().recip();

    for bi in 0..batch {
        for hi in 0..heads {
            let hs = heads * head_dim;
            let q_off = ((bi * sq * hs + hi * head_dim) * 4) as usize;
            let k_off = ((bi * sk * hs + hi * head_dim) * 4) as usize;
            let out_off = match wrt_ir {
                AttentionBwdWrt::Key | AttentionBwdWrt::Value => k_off,
                AttentionBwdWrt::Query => q_off,
            };

            encode_scores(
                enc,
                kk,
                buffer,
                q + q_off,
                k + k_off,
                scores,
                sq,
                sk,
                heads,
                head_dim,
                scale,
                mask_kind,
                window,
            );
            encode_softmax_rows(enc, kk, buffer, scores, sq, sk, mask_kind, sq);
            match wrt_ir {
                AttentionBwdWrt::Value => encode_dv(
                    enc,
                    kk,
                    buffer,
                    scores,
                    dy + q_off,
                    out + out_off,
                    sq,
                    sk,
                    heads,
                    head_dim,
                ),
                AttentionBwdWrt::Query => {
                    encode_dp(
                        enc,
                        kk,
                        buffer,
                        dy + q_off,
                        v + k_off,
                        dp,
                        sq,
                        sk,
                        heads,
                        head_dim,
                    );
                    encode_ds(enc, kk, buffer, scores, dp, ds, sq, sk, scale);
                    encode_dq(
                        enc,
                        kk,
                        buffer,
                        ds,
                        k + k_off,
                        out + out_off,
                        sq,
                        sk,
                        heads,
                        head_dim,
                    );
                }
                AttentionBwdWrt::Key => {
                    encode_dp(
                        enc,
                        kk,
                        buffer,
                        dy + q_off,
                        v + k_off,
                        dp,
                        sq,
                        sk,
                        heads,
                        head_dim,
                    );
                    encode_ds(enc, kk, buffer, scores, dp, ds, sq, sk, scale);
                    encode_dk(
                        enc,
                        kk,
                        buffer,
                        ds,
                        q + q_off,
                        out + out_off,
                        sq,
                        sk,
                        heads,
                        head_dim,
                    );
                }
            }
        }
    }
}

/// Heads processed per parallel dispatch. Scratch is `tile · 3 · seq²` floats;
/// cap the tile to a ~384 MB budget so it stays bounded at large seq, and never
/// exceed the total (batch·heads) count.
fn attn_bwd_tile(total_bh: usize, ss: usize) -> usize {
    const BUDGET_FLOATS: usize = 96 * 1024 * 1024; // 3·ss·tile ≤ this ⇒ ≤384 MB
    (BUDGET_FLOATS / (3 * ss.max(1))).clamp(1, total_bh.max(1))
}

/// Fused **and parallel** backward for all three gradients. The earlier fused
/// version removed the 3×/2× recompute but still walked (batch,head) SERIALLY —
/// the shared scratch made Metal (which hazard-tracks at buffer granularity)
/// serialize every (b,h). Here each stage (scores → softmax → dv/dp → ds →
/// dq/dk) is ONE 3D dispatch whose `grid.z` runs `tile` heads **in parallel**,
/// so the GPU is filled the way the forward `fused_attn_block` fills it. Heads
/// run in tiles to bound scratch; scores/dp/ds are laid out `[tile][sq][sk]`.
#[allow(clippy::too_many_arguments)]
pub fn encode_attention_bwd_all(
    enc: &ComputeCommandEncoderRef,
    kk: &Kernels,
    buffer: &Buffer,
    q: usize,
    k: usize,
    v: usize,
    dy: usize,
    out_dq: usize,
    out_dk: usize,
    out_dv: usize,
    scratch: usize,
    batch: u32,
    sq: u32,
    sk: u32,
    heads: u32,
    head_dim: u32,
    mask_kind: u32,
    window: u32,
) {
    let ss = (sq as usize).saturating_mul(sk as usize);
    let total_bh = (batch as usize) * (heads as usize);
    let tile = attn_bwd_tile(total_bh, ss);
    let scale = (head_dim as f32).sqrt().recip();

    // Opt-in fused flash-backward: one kernel, S/P/dP/dS on-chip (no scores/dp/ds
    // device scratch), dQ direct + dK/dV atomic. Gated to the shapes the kernel's
    // threadgroup tiles support (sk≤256, head_dim≤64) and causal/none masks.
    if rlx_ir::env::flag("RLX_METAL_ATTN_BWD_FUSED")
        && mask_kind <= 1
        && sk <= 256
        && head_dim <= 64
        && sq == sk
    {
        let set_u = |enc: &ComputeCommandEncoderRef, idx: u64, v: u32| {
            enc.set_bytes(idx, 4, &v as *const u32 as *const _);
        };
        let hs = heads * head_dim;
        let dkdv_elems = batch * sk * hs;
        // Zero dK / dV (accumulated via atomics below).
        for off in [out_dk, out_dv] {
            enc.set_compute_pipeline_state(&kk.scatter_add_zero);
            enc.set_buffer(0, Some(buffer), off as u64);
            set_u(enc, 1, dkdv_elems);
            enc.dispatch_threads(
                MTLSize {
                    width: dkdv_elems as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256.min(dkdv_elems as u64).max(1),
                    height: 1,
                    depth: 1,
                },
            );
        }
        // Benchmark probe: stop after the dK/dV zeroing to measure its isolated cost.
        if rlx_ir::env::flag("RLX_METAL_ATTN_BWD_FUSED_ZEROONLY") {
            return;
        }
        enc.set_compute_pipeline_state(&kk.attn_bwd_fused_f32);
        enc.set_buffer(0, Some(buffer), q as u64);
        enc.set_buffer(1, Some(buffer), k as u64);
        enc.set_buffer(2, Some(buffer), v as u64);
        enc.set_buffer(3, Some(buffer), dy as u64);
        enc.set_buffer(4, Some(buffer), out_dq as u64);
        enc.set_buffer(5, Some(buffer), out_dk as u64);
        enc.set_buffer(6, Some(buffer), out_dv as u64);
        set_u(enc, 7, sq);
        set_u(enc, 8, sk);
        set_u(enc, 9, heads);
        set_u(enc, 10, head_dim);
        enc.set_bytes(11, 4, &scale as *const f32 as *const _);
        set_u(enc, 12, mask_kind);
        const BR: u32 = 8;
        let q_tiles = sq.div_ceil(BR);
        enc.dispatch_thread_groups(
            MTLSize {
                width: q_tiles as u64,
                height: heads as u64,
                depth: batch as u64,
            },
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
        return;
    }

    let scores = scratch;
    let dp = scratch + tile * ss * 4;
    let ds = scratch + 2 * tile * ss * 4;
    let set_u = |enc: &ComputeCommandEncoderRef, idx: u64, v: u32| {
        enc.set_bytes(idx, 4, &v as *const u32 as *const _);
    };

    let mut b = 0usize;
    while b < total_bh {
        let cur = tile.min(total_bh - b) as u64;
        let tb = b as u32;

        // scores = QKᵀ·scale (+ causal/window mask). grid [sk, sq, cur].
        enc.set_compute_pipeline_state(&kk.attn_bwd_scores_batched_f32);
        enc.set_buffer(0, Some(buffer), q as u64);
        enc.set_buffer(1, Some(buffer), k as u64);
        enc.set_buffer(2, Some(buffer), scores as u64);
        set_u(enc, 3, sq);
        set_u(enc, 4, sk);
        set_u(enc, 5, heads);
        set_u(enc, 6, head_dim);
        enc.set_bytes(7, 4, &scale as *const f32 as *const _);
        set_u(enc, 8, mask_kind);
        set_u(enc, 9, window);
        set_u(enc, 10, tb);
        enc.dispatch_threads(
            MTLSize {
                width: sk as u64,
                height: sq as u64,
                depth: cur,
            },
            MTLSize {
                width: 8.min(sk as u64),
                height: 8.min(sq as u64),
                depth: 1,
            },
        );

        // softmax over each row (cur·sq rows of sk, contiguous from `scores`).
        encode_softmax_rows(
            enc,
            kk,
            buffer,
            scores,
            (cur as u32) * sq,
            sk,
            mask_kind,
            sq,
        );

        // dv = scoresᵀ·dy. grid [head_dim, sk, cur].
        enc.set_compute_pipeline_state(&kk.attn_bwd_dv_batched_f32);
        enc.set_buffer(0, Some(buffer), scores as u64);
        enc.set_buffer(1, Some(buffer), dy as u64);
        enc.set_buffer(2, Some(buffer), out_dv as u64);
        set_u(enc, 3, sq);
        set_u(enc, 4, sk);
        set_u(enc, 5, heads);
        set_u(enc, 6, head_dim);
        set_u(enc, 7, tb);
        set_u(enc, 8, mask_kind);
        set_u(enc, 9, window);
        enc.dispatch_threads(
            MTLSize {
                width: head_dim as u64,
                height: sk as u64,
                depth: cur,
            },
            MTLSize {
                width: 8.min(head_dim as u64),
                height: 8.min(sk as u64),
                depth: 1,
            },
        );

        // dp = dy·vᵀ. grid [sk, sq, cur].
        enc.set_compute_pipeline_state(&kk.attn_bwd_dp_batched_f32);
        enc.set_buffer(0, Some(buffer), dy as u64);
        enc.set_buffer(1, Some(buffer), v as u64);
        enc.set_buffer(2, Some(buffer), dp as u64);
        set_u(enc, 3, sq);
        set_u(enc, 4, sk);
        set_u(enc, 5, heads);
        set_u(enc, 6, head_dim);
        set_u(enc, 7, tb);
        set_u(enc, 8, mask_kind);
        set_u(enc, 9, window);
        enc.dispatch_threads(
            MTLSize {
                width: sk as u64,
                height: sq as u64,
                depth: cur,
            },
            MTLSize {
                width: 8.min(sk as u64),
                height: 8.min(sq as u64),
                depth: 1,
            },
        );

        // ds = softmax-jacobian(scores, dp). One threadgroup per (slot,row).
        let mut tg_w: u64 = 1;
        while tg_w * 2 <= sk as u64 && tg_w * 2 <= 256 {
            tg_w *= 2;
        }
        enc.set_compute_pipeline_state(&kk.attn_bwd_ds_batched_f32);
        enc.set_buffer(0, Some(buffer), scores as u64);
        enc.set_buffer(1, Some(buffer), dp as u64);
        enc.set_buffer(2, Some(buffer), ds as u64);
        set_u(enc, 3, sq);
        set_u(enc, 4, sk);
        enc.set_bytes(5, 4, &scale as *const f32 as *const _);
        set_u(enc, 6, mask_kind);
        set_u(enc, 7, window);
        enc.dispatch_thread_groups(
            MTLSize {
                width: cur * sq as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_w,
                height: 1,
                depth: 1,
            },
        );

        // dq = ds·k. grid [head_dim, sq, cur].
        enc.set_compute_pipeline_state(&kk.attn_bwd_dq_batched_f32);
        enc.set_buffer(0, Some(buffer), ds as u64);
        enc.set_buffer(1, Some(buffer), k as u64);
        enc.set_buffer(2, Some(buffer), out_dq as u64);
        set_u(enc, 3, sq);
        set_u(enc, 4, sk);
        set_u(enc, 5, heads);
        set_u(enc, 6, head_dim);
        set_u(enc, 7, tb);
        set_u(enc, 8, mask_kind);
        set_u(enc, 9, window);
        enc.dispatch_threads(
            MTLSize {
                width: head_dim as u64,
                height: sq as u64,
                depth: cur,
            },
            MTLSize {
                width: 8.min(head_dim as u64),
                height: 8.min(sq as u64),
                depth: 1,
            },
        );

        // dk = dsᵀ·q. grid [head_dim, sk, cur].
        enc.set_compute_pipeline_state(&kk.attn_bwd_dk_batched_f32);
        enc.set_buffer(0, Some(buffer), ds as u64);
        enc.set_buffer(1, Some(buffer), q as u64);
        enc.set_buffer(2, Some(buffer), out_dk as u64);
        set_u(enc, 3, sq);
        set_u(enc, 4, sk);
        set_u(enc, 5, heads);
        set_u(enc, 6, head_dim);
        set_u(enc, 7, tb);
        set_u(enc, 8, mask_kind);
        set_u(enc, 9, window);
        enc.dispatch_threads(
            MTLSize {
                width: head_dim as u64,
                height: sk as u64,
                depth: cur,
            },
            MTLSize {
                width: 8.min(head_dim as u64),
                height: 8.min(sk as u64),
                depth: 1,
            },
        );

        b += cur as usize;
    }
}

fn encode_softmax_rows(
    enc: &ComputeCommandEncoderRef,
    kk: &Kernels,
    buffer: &Buffer,
    data: usize,
    rows: u32,
    cols: u32,
    mask_kind: u32,
    sq: u32,
) {
    let mut tg_w: u64 = 1;
    while tg_w * 2 <= cols as u64 && tg_w * 2 <= 256 {
        tg_w *= 2;
    }
    // Causal (mask_kind==1): each row `slot*sq + qi` only softmaxes over its band
    // [0, qi]; the masked sentinels underflow to exactly 0, so this is bit-identical
    // to the full softmax on the band that dv/ds read — ~2× less work.
    if mask_kind == 1 {
        enc.set_compute_pipeline_state(&kk.softmax_lastax_causal);
        enc.set_buffer(0, Some(buffer), data as u64);
        enc.set_bytes(1, 4, &cols as *const u32 as *const _);
        enc.set_bytes(2, 4, &sq as *const u32 as *const _);
    } else {
        enc.set_compute_pipeline_state(&kk.softmax_lastax);
        enc.set_buffer(0, Some(buffer), data as u64);
        enc.set_bytes(1, 4, &cols as *const u32 as *const _);
    }
    enc.dispatch_threads(
        MTLSize {
            width: tg_w * rows as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn encode_scores(
    enc: &ComputeCommandEncoderRef,
    kk: &Kernels,
    buffer: &Buffer,
    q: usize,
    k: usize,
    scores: usize,
    sq: u32,
    sk: u32,
    heads: u32,
    head_dim: u32,
    scale: f32,
    mask_kind: u32,
    window: u32,
) {
    let hs = heads * head_dim;
    enc.set_compute_pipeline_state(&kk.attn_bwd_scores_f32);
    enc.set_buffer(0, Some(buffer), q as u64);
    enc.set_buffer(1, Some(buffer), k as u64);
    enc.set_buffer(2, Some(buffer), scores as u64);
    enc.set_bytes(3, 4, &sq as *const u32 as *const _);
    enc.set_bytes(4, 4, &sk as *const u32 as *const _);
    enc.set_bytes(5, 4, &hs as *const u32 as *const _);
    enc.set_bytes(6, 4, &head_dim as *const u32 as *const _);
    enc.set_bytes(7, 4, &scale as *const f32 as *const _);
    enc.set_bytes(8, 4, &mask_kind as *const u32 as *const _);
    enc.set_bytes(9, 4, &window as *const u32 as *const _);
    enc.dispatch_threads(
        MTLSize {
            width: sk as u64,
            height: sq as u64,
            depth: 1,
        },
        MTLSize {
            width: 8.min(sk as u64),
            height: 8.min(sq as u64),
            depth: 1,
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn encode_dp(
    enc: &ComputeCommandEncoderRef,
    kk: &Kernels,
    buffer: &Buffer,
    dy: usize,
    v: usize,
    dp: usize,
    sq: u32,
    sk: u32,
    heads: u32,
    head_dim: u32,
) {
    let hs = heads * head_dim;
    enc.set_compute_pipeline_state(&kk.attn_bwd_dp_f32);
    enc.set_buffer(0, Some(buffer), dy as u64);
    enc.set_buffer(1, Some(buffer), v as u64);
    enc.set_buffer(2, Some(buffer), dp as u64);
    enc.set_bytes(3, 4, &sq as *const u32 as *const _);
    enc.set_bytes(4, 4, &sk as *const u32 as *const _);
    enc.set_bytes(5, 4, &hs as *const u32 as *const _);
    enc.set_bytes(6, 4, &head_dim as *const u32 as *const _);
    enc.dispatch_threads(
        MTLSize {
            width: sk as u64,
            height: sq as u64,
            depth: 1,
        },
        MTLSize {
            width: 8.min(sk as u64),
            height: 8.min(sq as u64),
            depth: 1,
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn encode_ds(
    enc: &ComputeCommandEncoderRef,
    kk: &Kernels,
    buffer: &Buffer,
    scores: usize,
    dp: usize,
    ds: usize,
    sq: u32,
    sk: u32,
    scale: f32,
) {
    let mut tg_w: u64 = 1;
    while tg_w * 2 <= sk as u64 && tg_w * 2 <= 256 {
        tg_w *= 2;
    }
    enc.set_compute_pipeline_state(&kk.attn_bwd_ds_f32);
    enc.set_buffer(0, Some(buffer), scores as u64);
    enc.set_buffer(1, Some(buffer), dp as u64);
    enc.set_buffer(2, Some(buffer), ds as u64);
    enc.set_bytes(3, 4, &sq as *const u32 as *const _);
    enc.set_bytes(4, 4, &sk as *const u32 as *const _);
    enc.set_bytes(5, 4, &scale as *const f32 as *const _);
    enc.dispatch_thread_groups(
        MTLSize {
            width: sq as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn encode_dv(
    enc: &ComputeCommandEncoderRef,
    kk: &Kernels,
    buffer: &Buffer,
    scores: usize,
    dy: usize,
    out: usize,
    sq: u32,
    sk: u32,
    heads: u32,
    head_dim: u32,
) {
    let hs = heads * head_dim;
    enc.set_compute_pipeline_state(&kk.attn_bwd_dv_f32);
    enc.set_buffer(0, Some(buffer), scores as u64);
    enc.set_buffer(1, Some(buffer), dy as u64);
    enc.set_buffer(2, Some(buffer), out as u64);
    enc.set_bytes(3, 4, &sq as *const u32 as *const _);
    enc.set_bytes(4, 4, &sk as *const u32 as *const _);
    enc.set_bytes(5, 4, &hs as *const u32 as *const _);
    enc.set_bytes(6, 4, &head_dim as *const u32 as *const _);
    enc.dispatch_threads(
        MTLSize {
            width: head_dim as u64,
            height: sk as u64,
            depth: 1,
        },
        MTLSize {
            width: 8.min(head_dim as u64),
            height: 8.min(sk as u64),
            depth: 1,
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn encode_dq(
    enc: &ComputeCommandEncoderRef,
    kk: &Kernels,
    buffer: &Buffer,
    ds: usize,
    k: usize,
    out: usize,
    sq: u32,
    sk: u32,
    heads: u32,
    head_dim: u32,
) {
    let hs = heads * head_dim;
    enc.set_compute_pipeline_state(&kk.attn_bwd_dq_f32);
    enc.set_buffer(0, Some(buffer), ds as u64);
    enc.set_buffer(1, Some(buffer), k as u64);
    enc.set_buffer(2, Some(buffer), out as u64);
    enc.set_bytes(3, 4, &sq as *const u32 as *const _);
    enc.set_bytes(4, 4, &sk as *const u32 as *const _);
    enc.set_bytes(5, 4, &hs as *const u32 as *const _);
    enc.set_bytes(6, 4, &head_dim as *const u32 as *const _);
    enc.dispatch_threads(
        MTLSize {
            width: head_dim as u64,
            height: sq as u64,
            depth: 1,
        },
        MTLSize {
            width: 8.min(head_dim as u64),
            height: 8.min(sq as u64),
            depth: 1,
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn encode_dk(
    enc: &ComputeCommandEncoderRef,
    kk: &Kernels,
    buffer: &Buffer,
    ds: usize,
    q: usize,
    out: usize,
    sq: u32,
    sk: u32,
    heads: u32,
    head_dim: u32,
) {
    let hs = heads * head_dim;
    enc.set_compute_pipeline_state(&kk.attn_bwd_dk_f32);
    enc.set_buffer(0, Some(buffer), ds as u64);
    enc.set_buffer(1, Some(buffer), q as u64);
    enc.set_buffer(2, Some(buffer), out as u64);
    enc.set_bytes(3, 4, &sq as *const u32 as *const _);
    enc.set_bytes(4, 4, &sk as *const u32 as *const _);
    enc.set_bytes(5, 4, &hs as *const u32 as *const _);
    enc.set_bytes(6, 4, &head_dim as *const u32 as *const _);
    enc.dispatch_threads(
        MTLSize {
            width: head_dim as u64,
            height: sk as u64,
            depth: 1,
        },
        MTLSize {
            width: 8.min(head_dim as u64),
            height: 8.min(sk as u64),
            depth: 1,
        },
    );
}
