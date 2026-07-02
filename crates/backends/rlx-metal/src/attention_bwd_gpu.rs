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

//! GPU scaled dot-product attention backward (recomputes scores + softmax).

use crate::kernels::Kernels;
use metal::{Buffer, ComputeCommandEncoderRef, MTLSize};
use rlx_ir::Graph;
use rlx_ir::op::{AttentionBwdWrt, Op};

pub fn scratch_bytes(graph: &Graph) -> usize {
    let mut max = 0usize;
    for node in graph.nodes() {
        if let Op::AttentionBackward { .. } = &node.op {
            let q_shape = &graph.node(node.inputs[0]).shape;
            let k_shape = &graph.node(node.inputs[1]).shape;
            let sq = q_shape.dim(1).unwrap_static();
            let sk = if k_shape.rank() >= 2 {
                k_shape.dim(1).unwrap_static()
            } else {
                k_shape.dim(0).unwrap_static()
            };
            let ss = sq.saturating_mul(sk);
            // scores + dp + ds
            max = max.max(ss * 3 * std::mem::size_of::<f32>());
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
            encode_softmax_rows(enc, kk, buffer, scores, sq, sk);
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

fn encode_softmax_rows(
    enc: &ComputeCommandEncoderRef,
    kk: &Kernels,
    buffer: &Buffer,
    data: usize,
    rows: u32,
    cols: u32,
) {
    let mut tg_w: u64 = 1;
    while tg_w * 2 <= cols as u64 && tg_w * 2 <= 256 {
        tg_w *= 2;
    }
    enc.set_compute_pipeline_state(&kk.softmax_lastax);
    enc.set_buffer(0, Some(buffer), data as u64);
    enc.set_bytes(1, 4, &cols as *const u32 as *const _);
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
