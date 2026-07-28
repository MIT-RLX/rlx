// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared attention Q/K/V layout detection for backends.
//!
//! Rank-4 tensors may be either `[B, H, S, D]` (BHSD, GPU default) or
//! `[B, S, H, D]` (BSHD, CPU / EEG-DINO convention). Disambiguate by
//! checking whether axis 1 equals `num_heads`.

use crate::graph::{Graph, NodeId};
use crate::op::Op;
use crate::shape::{Dim, Shape};

/// Resolved batch / sequence / head geometry for an attention op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionGeom {
    pub batch: usize,
    pub seq_q: usize,
    pub seq_k: usize,
    pub heads: usize,
    pub head_dim: usize,
    /// `true` when Q/K/V are `[B, H, S, D]`; `false` for `[B, S, H, D]`.
    pub bhsd: bool,
}

impl AttentionGeom {
    /// `true` when Q/K/V are `[B, S, H, D]` (EEG-DINO / CPU convention).
    pub const fn is_bshd(self) -> bool {
        !self.bhsd
    }
}

/// Max `head_dim` for tiled flash attention in `rlx-gpu-kernels` (`attention.cu`).
pub const ATTENTION_FLASH_MAX_HEAD_DIM: u32 = 128;

/// When `true`, GPU backends should use the row-wise `attention_row` kernel
/// instead of tiled flash. Flash supports both BHSD and BSHD via strides.
pub fn attention_dispatch_use_row(head_dim: u32, force_row_env: &str) -> bool {
    head_dim > ATTENTION_FLASH_MAX_HEAD_DIM || crate::env::flag(force_row_env)
}

fn dim_usize(d: Dim) -> usize {
    d.unwrap_static()
}

/// Infer attention geometry from Q and K shapes (rank 3 or 4).
pub fn attention_geom(
    q_shape: &Shape,
    k_shape: &Shape,
    num_heads: usize,
    head_dim: usize,
) -> AttentionGeom {
    let rank = q_shape.rank();
    let (batch, seq_q, seq_k, heads, bhsd) = if rank == 4 {
        let d1 = dim_usize(q_shape.dim(1));
        let d2 = dim_usize(q_shape.dim(2));
        if d1 == num_heads {
            (
                dim_usize(q_shape.dim(0)),
                d2,
                dim_usize(k_shape.dim(2)),
                num_heads,
                true,
            )
        } else {
            (
                dim_usize(q_shape.dim(0)),
                d1,
                dim_usize(k_shape.dim(1)),
                num_heads,
                false,
            )
        }
    } else if rank >= 3 {
        (
            dim_usize(q_shape.dim(0)),
            dim_usize(q_shape.dim(1)),
            dim_usize(k_shape.dim(1)),
            num_heads,
            false,
        )
    } else {
        (
            1,
            dim_usize(q_shape.dim(0)),
            dim_usize(k_shape.dim(0)),
            num_heads,
            false,
        )
    };
    AttentionGeom {
        batch,
        seq_q,
        seq_k,
        heads,
        head_dim,
        bhsd,
    }
}

/// Strides passed to GPU attention kernels (Q/K/V/out + mask).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionLaunchStrides {
    pub q_batch: u32,
    pub q_head: u32,
    pub q_seq: u32,
    pub k_batch: u32,
    pub k_head: u32,
    pub k_seq: u32,
    pub v_batch: u32,
    pub v_head: u32,
    pub v_seq: u32,
    pub o_batch: u32,
    pub o_head: u32,
    pub o_seq: u32,
    pub mask_batch: u32,
    pub mask_head: u32,
    pub mask_q: u32,
    pub mask_k: u32,
}

/// Canonical `[B, H, S_q, S_k]` mask strides (unused when mask_kind is causal/none).
pub fn mask_strides_bhsd(heads: u32, seq_q: u32, seq_k: u32) -> (u32, u32, u32, u32) {
    (heads * seq_q * seq_k, seq_q * seq_k, seq_k, 1)
}

/// Mask address strides from the mask tensor's IR shape (matches wgpu).
pub fn mask_strides_for_shape(
    m_dims: &[Dim],
    heads: u32,
    seq_q: u32,
    seq_k: u32,
) -> (u32, u32, u32, u32) {
    let dim = |i: usize| m_dims[i].unwrap_static() as u32;
    match m_dims.len() {
        2 => (dim(1), 0, 0, 1),
        3 => (dim(1) * dim(2), 0, dim(2), 1),
        4 => (dim(1) * dim(2) * dim(3), dim(2) * dim(3), dim(3), 1),
        _ => mask_strides_bhsd(heads, seq_q, seq_k),
    }
}

/// Q/K/V/out strides for a forward attention launch.
pub fn attention_launch_strides(
    geom: AttentionGeom,
    q_shape: &[Dim],
    k_shape: &[Dim],
    v_shape: &[Dim],
    out_shape: &[Dim],
    mask_shape: Option<&[Dim]>,
) -> AttentionLaunchStrides {
    let h = geom.heads as u32;
    let hd = geom.head_dim as u32;
    let sq = geom.seq_q as u32;
    let sk = geom.seq_k as u32;
    let stride = |shape: &[Dim], seq: u32| strides_for_shape(shape, h, hd, seq, geom.bhsd);
    let (qb, qh, qs) = stride(q_shape, sq);
    let (kb, kh, ks) = stride(k_shape, sk);
    let (vb, vh, vs) = stride(v_shape, sk);
    let (ob, oh, os) = stride(out_shape, sq);
    let (mb, mh, mq, mk) = mask_shape
        .map(|m| mask_strides_for_shape(m, h, sq, sk))
        .unwrap_or_else(|| mask_strides_bhsd(h, sq, sk));
    AttentionLaunchStrides {
        q_batch: qb,
        q_head: qh,
        q_seq: qs,
        k_batch: kb,
        k_head: kh,
        k_seq: ks,
        v_batch: vb,
        v_head: vh,
        v_seq: vs,
        o_batch: ob,
        o_head: oh,
        o_seq: os,
        mask_batch: mb,
        mask_head: mh,
        mask_q: mq,
        mask_k: mk,
    }
}

/// Per-axis strides (batch, head, seq) in f32 elements for `[B, H, S, D]`.
pub fn strides_bhsd(heads: u32, head_dim: u32, seq_extent: u32) -> (u32, u32, u32) {
    let hd = heads * head_dim;
    (hd * seq_extent, seq_extent * head_dim, head_dim)
}

/// Per-axis strides (batch, head, seq) in f32 elements for `[B, S, H, D]`.
pub fn strides_bshd(heads: u32, head_dim: u32, seq_extent: u32) -> (u32, u32, u32) {
    let hd = heads * head_dim;
    (seq_extent * hd, head_dim, hd)
}

/// Strides for a tensor with the given rank and layout flag.
pub fn strides_for_shape(
    shape: &[Dim],
    heads: u32,
    head_dim: u32,
    seq_extent: u32,
    bhsd: bool,
) -> (u32, u32, u32) {
    let last = shape[shape.len() - 1].unwrap_static() as u32;
    let hd_total = heads * head_dim;
    if (shape.len() == 3 && last == hd_total) || (shape.len() == 4 && !bhsd) {
        strides_bshd(heads, head_dim, seq_extent)
    } else {
        strides_bhsd(heads, head_dim, seq_extent)
    }
}

/// `FMB → [B,S,3,H,D] → Narrow×3 (axis 2) → Attention` packed QKV pattern.
/// Returns `(parent, head_width_elems, [q_narrow, k_narrow, v_narrow])`.
pub fn detect_packed_bshd_qkv_attention(
    graph: &Graph,
    q_id: NodeId,
    k_id: NodeId,
    v_id: NodeId,
) -> Option<(NodeId, usize, [NodeId; 3])> {
    fn narrow_on_axis_2(graph: &Graph, id: NodeId) -> Option<(NodeId, usize, usize, NodeId)> {
        let mut cur = id;
        loop {
            let node = graph.node(cur);
            match &node.op {
                Op::Reshape { .. } | Op::Cast { .. } => {
                    if node.inputs.is_empty() {
                        return None;
                    }
                    cur = node.inputs[0];
                }
                Op::Narrow {
                    axis: 2,
                    start,
                    len,
                } => {
                    if node.inputs.is_empty() {
                        return None;
                    }
                    return Some((node.inputs[0], *start, *len, cur));
                }
                _ => return None,
            }
        }
    }
    let (parent, qs, ql, q_n) = narrow_on_axis_2(graph, q_id)?;
    let (parent_k, ks, kl, k_n) = narrow_on_axis_2(graph, k_id)?;
    let (parent_v, vs, vl, v_n) = narrow_on_axis_2(graph, v_id)?;
    if parent != parent_k || parent != parent_v {
        return None;
    }
    if qs != 0 || ks != 1 || vs != 2 {
        return None;
    }
    if ql != kl || ql != vl {
        return None;
    }
    let p_shape = graph.node(parent).shape.dims();
    if p_shape.len() != 5 || p_shape[2].unwrap_static() != 3 {
        return None;
    }
    let head_width = ql * p_shape[3].unwrap_static() * p_shape[4].unwrap_static();
    Some((parent, head_width, [q_n, k_n, v_n]))
}

/// True when a packed axis-2 narrow may be skipped: attention reads the
/// parent directly and no other consumer needs the narrow's contiguous layout
/// (e.g. the narrow view is still a graph output).
pub fn packed_bshd_narrow_elidable(
    graph: &crate::Graph,
    narrow_id: NodeId,
    attn_id: NodeId,
) -> bool {
    use crate::op::Op;
    use std::collections::HashSet;

    let mut stack = vec![narrow_id];
    let mut seen = HashSet::new();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        if graph.outputs.contains(&id) {
            return false;
        }
        for user in graph.users(id) {
            if user == attn_id {
                continue;
            }
            match &graph.node(user).op {
                Op::Reshape { .. } | Op::Cast { .. } => stack.push(user),
                Op::Attention { .. } => {}
                _ => return false,
            }
        }
    }
    true
}

/// Strides for reading Q/K/V from a packed `[B,S,3,H,D]` parent (CPU fused_strided_attn).
pub fn packed_bshd_qkv_strides(head_width: usize, head_dim: u32, seq_q: u32) -> (u32, u32, u32) {
    let pack_seq = head_width as u32 * 3;
    (seq_q * pack_seq, head_dim, pack_seq)
}

/// Reference SDPA on separate `[B,S,H,D]` tensors (backend parity tests).
pub fn cpu_attention_bshd(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    seq: usize,
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let hs = num_heads * head_dim;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; batch * seq * hs];
    for bi in 0..batch {
        for qi in 0..seq {
            for hi in 0..num_heads {
                let q_base = bi * seq * hs + qi * hs + hi * head_dim;
                let mut m = f32::NEG_INFINITY;
                let mut l = 0.0f32;
                let mut acc = vec![0.0f32; head_dim];
                for ki in 0..seq {
                    let k_base = bi * seq * hs + ki * hs + hi * head_dim;
                    let mut score = 0.0f32;
                    for d in 0..head_dim {
                        score += q[q_base + d] * k[k_base + d];
                    }
                    score *= scale;
                    let m_new = m.max(score);
                    let e_old = (m - m_new).exp();
                    let e_cur = (score - m_new).exp();
                    l = e_old * l + e_cur;
                    let v_base = bi * seq * hs + ki * hs + hi * head_dim;
                    for d in 0..head_dim {
                        acc[d] = e_old * acc[d] + e_cur * v[v_base + d];
                    }
                    m = m_new;
                }
                let inv_l = 1.0 / l;
                let o_base = bi * seq * hs + qi * hs + hi * head_dim;
                for d in 0..head_dim {
                    out[o_base + d] = acc[d] * inv_l;
                }
            }
        }
    }
    out
}

/// Reference SDPA reading Q/K/V from packed `[B,S,3,H,D]` row-major storage.
pub fn cpu_attention_packed_bshd_qkv(
    packed: &[f32],
    batch: usize,
    seq: usize,
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let hs = num_heads * head_dim;
    let qrs = hs * 3;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; batch * seq * hs];
    for bi in 0..batch {
        for hi in 0..num_heads {
            let mut qh = vec![0.0f32; seq * head_dim];
            let mut kh = vec![0.0f32; seq * head_dim];
            let mut vh = vec![0.0f32; seq * head_dim];
            for si in 0..seq {
                let q_off = bi * seq * qrs + si * qrs + hi * head_dim;
                let k_off = q_off + hs;
                let v_off = q_off + 2 * hs;
                qh[si * head_dim..(si + 1) * head_dim]
                    .copy_from_slice(&packed[q_off..q_off + head_dim]);
                kh[si * head_dim..(si + 1) * head_dim]
                    .copy_from_slice(&packed[k_off..k_off + head_dim]);
                vh[si * head_dim..(si + 1) * head_dim]
                    .copy_from_slice(&packed[v_off..v_off + head_dim]);
            }
            for qi in 0..seq {
                let o_off = bi * seq * hs + qi * hs + hi * head_dim;
                let mut m = f32::NEG_INFINITY;
                let mut l = 0.0f32;
                let mut acc = vec![0.0f32; head_dim];
                for ki in 0..seq {
                    let mut score = 0.0f32;
                    for d in 0..head_dim {
                        score += qh[qi * head_dim + d] * kh[ki * head_dim + d];
                    }
                    score *= scale;
                    let m_new = m.max(score);
                    let e_old = (m - m_new).exp();
                    let e_cur = (score - m_new).exp();
                    l = e_old * l + e_cur;
                    for d in 0..head_dim {
                        acc[d] = e_old * acc[d] + e_cur * vh[ki * head_dim + d];
                    }
                    m = m_new;
                }
                let inv_l = 1.0 / l;
                for d in 0..head_dim {
                    out[o_off + d] = acc[d] * inv_l;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::{MaskKind, Op};
    use crate::{DType, Graph, GraphExt, Shape};

    #[test]
    fn packed_bshd_cpu_ref_runs() {
        let (b, s, nh, dh) = (1usize, 4, 2, 3);
        let hs = nh * dh;
        let qrs = hs * 3;
        let mut packed = vec![0.0f32; b * s * qrs];
        for i in 0..packed.len() {
            packed[i] = (i as f32 * 0.1).sin();
        }
        let out = cpu_attention_packed_bshd_qkv(&packed, b, s, nh, dh);
        assert_eq!(out.len(), b * s * hs);
    }

    #[test]
    fn eeg_bshd_geom_and_flash_dispatch() {
        let f = DType::F32;
        let shape = Shape::new(&[1, 191, 8, 25], f);
        let geom = attention_geom(&shape, &shape, 8, 25);
        assert!(!geom.bhsd);
        assert!(geom.is_bshd());
        assert_eq!(geom.seq_q, 191);
        assert!(!attention_dispatch_use_row(
            25,
            "RLX_CUDA_FORCE_ATTENTION_ROW"
        ));
        assert!(attention_dispatch_use_row(
            129,
            "RLX_CUDA_FORCE_ATTENTION_ROW"
        ));
    }

    #[test]
    fn bhsd_geom_detected_on_axis1_heads() {
        let f = DType::F32;
        let shape = Shape::new(&[2, 8, 64, 32], f);
        let geom = attention_geom(&shape, &shape, 8, 32);
        assert!(geom.bhsd);
        assert!(!geom.is_bshd());
        let (qb, qh, qs) = strides_bhsd(8, 32, 64);
        assert_eq!(
            (qb, qh, qs),
            strides_for_shape(shape.dims(), 8, 32, 64, true)
        );
    }

    #[test]
    fn narrow_elidable_only_when_not_graph_output() {
        let (b, s, nh, dh) = (1usize, 4, 2, 3);
        let hd = nh * dh;
        let f = DType::F32;
        let mut g = Graph::new("packed_qkv");
        let x = g.input("x", Shape::new(&[b, s, hd], f));
        let w = g.param("w", Shape::new(&[hd, 3 * hd], f));
        let qkv = g.add_node(
            Op::FusedMatMulBiasAct { activation: None },
            vec![x, w],
            Shape::new(&[b, s, 3 * hd], f),
        );
        let qkv4 = g.reshape_(qkv, vec![b as i64, s as i64, 3, nh as i64, dh as i64]);
        let q0 = g.narrow_(qkv4, 2, 0, 1);
        let k0 = g.narrow_(qkv4, 2, 1, 1);
        let v0 = g.narrow_(qkv4, 2, 2, 1);
        let q = g.reshape_(q0, vec![b as i64, s as i64, nh as i64, dh as i64]);
        let k = g.reshape_(k0, vec![b as i64, s as i64, nh as i64, dh as i64]);
        let v = g.reshape_(v0, vec![b as i64, s as i64, nh as i64, dh as i64]);
        let attn = g.add_node(
            Op::Attention {
                num_heads: nh,
                head_dim: dh,
                mask_kind: MaskKind::None,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q, k, v],
            Shape::new(&[b, s, nh, dh], f),
        );
        g.set_outputs(vec![q]);
        assert!(!packed_bshd_narrow_elidable(&g, q0, attn));
        assert!(packed_bshd_narrow_elidable(&g, k0, attn));
        assert!(packed_bshd_narrow_elidable(&g, v0, attn));
    }
}
