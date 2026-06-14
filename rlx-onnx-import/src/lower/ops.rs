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

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result, anyhow, bail};
use rlx_ir::dynamic::sym;
use rlx_ir::hir::{HirMut, HirNodeId, HirOp};
use rlx_ir::op::{Activation, BinaryOp, CmpOp, ReduceOp};
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Dim, HirGraphExt, HirModule, Op, Shape};

use crate::bundle::RlxBundle;
use crate::bundle::{BundleManifest, BundleNode, topo_sort_nodes};
use crate::control_flow::{self, DURATION_CARRY};
use crate::rewrite::rewrite_graph;
use crate::tensor_data::i64_tensor;
use crate::tensor_data::{TypedParams, quant_matmul_weight_key};

use super::options::{ImportOptions, ImportReport};

const MAX_STUB_ELEMENTS: usize = 8 * 1024 * 1024;

fn is_typical_channel(c: usize) -> bool {
    matches!(
        c,
        22 | 32 | 64 | 80 | 100 | 105 | 125 | 128 | 256 | 512 | 768 | 1024
    )
}

fn normalize_axis(axis: i64, rank: usize) -> usize {
    if axis < 0 {
        (rank as i64 + axis) as usize
    } else {
        axis as usize
    }
}

fn channel_axis_for_param(m: &HirMut<'_>, param: HirNodeId, like: HirNodeId) -> usize {
    let c = m.shape(param).dim(0).unwrap_static();
    let like_sh = m.shape(like);
    for (i, d) in like_sh.dims().iter().enumerate() {
        if d.unwrap_static() == c {
            return i;
        }
    }
    if like_sh.rank() >= 2 {
        like_sh.rank() - 1
    } else {
        0
    }
}

fn broadcast_1d_param(
    m: &mut HirMut<'_>,
    param: HirNodeId,
    like: HirNodeId,
    axis: usize,
) -> HirNodeId {
    let rank = m.shape(like).rank().max(1);
    let c = m.shape(param).dim(0).unwrap_static() as i64;
    let mut dims = vec![1i64; rank];
    dims[axis.min(rank.saturating_sub(1))] = c;
    m.reshape_(param, dims)
}

fn broadcast_param_channels(m: &mut HirMut<'_>, param: HirNodeId, like: HirNodeId) -> HirNodeId {
    let axis = channel_axis_for_param(m, param, like);
    broadcast_1d_param(m, param, like, axis)
}

fn expand_operand_to_shape(m: &mut HirMut<'_>, x: HirNodeId, target: &Shape) -> HirNodeId {
    let xs = m.shape(x).clone();
    if xs == *target {
        return x;
    }
    let target_vec: Vec<i64> = target
        .dims()
        .iter()
        .map(|d| d.unwrap_static() as i64)
        .collect();
    m.add_node(
        Op::Expand {
            target_shape: target_vec,
        },
        vec![x],
        target.clone(),
    )
}

fn binary_infer_add(m: &mut HirMut<'_>, a: HirNodeId, b: HirNodeId, site: &str) -> HirNodeId {
    binary_infer(m, BinaryOp::Add, a, b, site)
}

/// `[N,L,C]` with `L > C` and typical channel width on the last axis.
fn is_vocoder_blc(s: &Shape) -> bool {
    s.rank() == 3
        && is_typical_channel(s.dim(2).unwrap_static())
        && s.dim(1).unwrap_static() > s.dim(2).unwrap_static()
        && s.dim(2).unwrap_static() > 1
}

/// `[N,C,L]` with typical channel width on axis 1 (vocoder NCL blocks).
fn is_vocoder_ncl(s: &Shape) -> bool {
    s.rank() == 3
        && is_typical_channel(s.dim(1).unwrap_static())
        && s.dim(2).unwrap_static() > s.dim(1).unwrap_static()
}

/// `[N,C,L]` with `C > L` and `L > 1` (excludes bias `[N,C,1]` and BLC blocks).
fn is_ncl_rank3(s: &Shape) -> bool {
    s.rank() == 3
        && s.dim(1).unwrap_static() >= 64
        && s.dim(2).unwrap_static() > 1
        && s.dim(1).unwrap_static() > s.dim(2).unwrap_static()
        && !is_blc_rank3(s)
}

/// `[N,L,C]` with channel on the last axis (ONNX BLC blocks).
fn is_blc_rank3(s: &Shape) -> bool {
    s.rank() == 3
        && s.dim(2).unwrap_static() >= 64
        && s.dim(2).unwrap_static() > s.dim(1).unwrap_static()
}

/// `[N,C,1]` bias / AdaIN scale (channel on axis 1).
fn is_nc1_rank3(s: &Shape) -> bool {
    s.rank() == 3 && s.dim(2).unwrap_static() == 1 && s.dim(1).unwrap_static() >= 64
}

/// `[C,L,1]` after `perm=[1,2,0]` on NCL — channel on axis 0.
fn is_cl1_rank3(s: &Shape) -> bool {
    s.rank() == 3
        && s.dim(2).unwrap_static() == 1
        && s.dim(0).unwrap_static() >= 64
        && s.dim(0).unwrap_static() > s.dim(1).unwrap_static()
}

/// ONNX output meta / tensor shape with channel on axis 1 (`[N,C,L]`).
fn meta_layout_ncl(s: &Shape) -> bool {
    s.rank() == 3
        && s.dim(1).unwrap_static() >= 64
        && !is_blc_rank3(s)
        && s.dim(1).unwrap_static() > s.dim(2).unwrap_static()
}

fn is_rank3_ncl_pair(a: &Shape, b: &Shape) -> bool {
    a.rank() == 3
        && b.rank() == 3
        && a.dim(1).unwrap_static() == b.dim(2).unwrap_static()
        && a.dim(2).unwrap_static() == b.dim(1).unwrap_static()
}

fn collapse_duplicate_channel_4d(m: &mut HirMut<'_>, x: HirNodeId) -> HirNodeId {
    let s = m.shape(x).clone();
    if s.rank() != 4 || s.dim(0).unwrap_static() != 1 {
        return x;
    }
    let c = s.dim(1).unwrap_static();
    if c < 64 {
        return x;
    }
    let c_last = s.dim(3).unwrap_static();
    if s.dim(1).unwrap_static() == s.dim(2).unwrap_static()
        && is_typical_channel(c_last)
        && c_last < c
    {
        // `[1, L, L, C]` from mistaken NCL broadcast → `[1, C, L]`.
        let l = s.dim(1).unwrap_static();
        return m.reshape_(x, vec![1, c_last as i64, l as i64]);
    }
    if s.dim(2).unwrap_static() == 1 && is_typical_channel(c_last) && c > c_last {
        if is_typical_channel(c) {
            // Valid NCHW `[1,C,1,L]` → NCL `[1,C,L]`.
            return m.reshape_(x, vec![1, c as i64, c_last as i64]);
        }
        // `[1, L, 1, C]` from BLC promoted as NCL → `[1, C, L]`.
        return m.reshape_(x, vec![1, c_last as i64, c as i64]);
    }
    if s.dim(1).unwrap_static() == s.dim(2).unwrap_static() {
        let l = s.dim(3).unwrap_static();
        return m.reshape_(x, vec![1, c as i64, l as i64]);
    }
    if s.dim(2).unwrap_static() == s.dim(3).unwrap_static() {
        let l = s.dim(2).unwrap_static();
        return m.reshape_(x, vec![1, c as i64, l as i64]);
    }
    x
}

fn repair_duplicate_length_rank3(
    m: &mut HirMut<'_>,
    x: HirNodeId,
    sa: &Shape,
    peer: &Shape,
) -> HirNodeId {
    if sa.rank() != 3
        || peer.rank() != 3
        || sa.dim(0).unwrap_static() != 1
        || peer.dim(0).unwrap_static() != 1
        || sa.dim(1).unwrap_static() != sa.dim(2).unwrap_static()
    {
        return x;
    }
    let l_dup = sa.dim(1).unwrap_static();
    if is_blc_rank3(peer) || is_vocoder_blc(peer) {
        let c = peer.dim(2).unwrap_static();
        let l = peer.dim(1).unwrap_static();
        if is_typical_channel(c) {
            return m.reshape_(x, vec![1, c as i64, l as i64]);
        }
    }
    if is_vocoder_ncl(peer) || is_ncl_rank3(peer) {
        let c = peer.dim(1).unwrap_static();
        let l = peer.dim(2).unwrap_static();
        if is_typical_channel(c) {
            let l_use = if l_dup == l { l_dup } else { l };
            return m.reshape_(x, vec![1, c as i64, l_use as i64]);
        }
    }
    x
}

/// NCL `[1,C,L]` combined with BLC `[1,L,C]` (same channel count `C`).
fn align_ncl_blc_binary(
    m: &mut HirMut<'_>,
    op: BinaryOp,
    a_in: HirNodeId,
    b_in: HirNodeId,
    sa: &Shape,
    sb: &Shape,
) -> Option<HirNodeId> {
    if sa.rank() != 3 || sb.rank() != 3 || sa.dim(0).unwrap_static() != sb.dim(0).unwrap_static() {
        return None;
    }
    for (ncl, blc, sn, sbl) in [(a_in, b_in, sa, sb), (b_in, a_in, sb, sa)] {
        let c = sn.dim(1).unwrap_static();
        if !is_typical_channel(c) || c != sbl.dim(2).unwrap_static() || !is_blc_rank3(sbl) {
            continue;
        }
        let blc_ncl = m.transpose_(blc, vec![0, 2, 1]);
        let la = sn.dim(2).unwrap_static();
        let lb = m.shape(blc_ncl).dim(2).unwrap_static();
        if la == lb || la.abs_diff(lb) <= 1 {
            let l = la.min(lb);
            let ncl_f = if la > l { m.narrow_(ncl, 2, 0, l) } else { ncl };
            let blc_f = if lb > l {
                m.narrow_(blc_ncl, 2, 0, l)
            } else {
                blc_ncl
            };
            return Some(m.add_node(
                Op::Binary(op),
                vec![ncl_f, blc_f],
                Shape::new(&[sn.dim(0).unwrap_static(), c, l], sn.dtype()),
            ));
        }
    }
    None
}

fn binary_infer(
    m: &mut HirMut<'_>,
    op: BinaryOp,
    a: HirNodeId,
    b: HirNodeId,
    site: &str,
) -> HirNodeId {
    let mut a_in = collapse_duplicate_channel_4d(m, a);
    let mut b_in = collapse_duplicate_channel_4d(m, b);
    let sa0 = m.shape(a_in).clone();
    let sb0 = m.shape(b_in).clone();
    a_in = repair_duplicate_length_rank3(m, a_in, &sa0, &sb0);
    b_in = repair_duplicate_length_rank3(m, b_in, &sb0, &sa0);
    let sa0 = m.shape(a_in).clone();
    let sb0 = m.shape(b_in).clone();
    if sa0.rank() == 1 && sb0.rank() >= 2 {
        a_in = broadcast_param_channels(m, a_in, b_in);
    } else if sb0.rank() == 1 && sa0.rank() >= 2 {
        b_in = broadcast_param_channels(m, b_in, a_in);
    }
    let sa = m.shape(a_in).clone();
    let sb = m.shape(b_in).clone();
    // NCL `[1,C,L]` + `[L,C,1]` must not use NumPy broadcast (would yield `[L,C,L]`).
    if sa.rank() == 3
        && sb.rank() == 3
        && sa.dim(0).unwrap_static() == 1
        && sb.dim(2).unwrap_static() == 1
        && sa.dim(1).unwrap_static() == sb.dim(1).unwrap_static()
        && sa.dim(2).unwrap_static() == sb.dim(0).unwrap_static()
    {
        let a_fix = if m.shape(a_in).rank() == 4 {
            collapse_duplicate_channel_4d(m, a_in)
        } else {
            a_in
        };
        let sa = m.shape(a_fix).clone();
        let b_fix = m.reshape_(
            b_in,
            vec![
                sa.dim(0).unwrap_static() as i64,
                sa.dim(1).unwrap_static() as i64,
                1,
            ],
        );
        return m.add_node(Op::Binary(op), vec![a_fix, b_fix], sa);
    }
    if sb.rank() == 3
        && sa.rank() == 3
        && sb.dim(0).unwrap_static() == 1
        && sa.dim(2).unwrap_static() == 1
        && sb.dim(1).unwrap_static() == sa.dim(1).unwrap_static()
        && sb.dim(2).unwrap_static() == sa.dim(0).unwrap_static()
    {
        let b_fix = if m.shape(b_in).rank() == 4 {
            collapse_duplicate_channel_4d(m, b_in)
        } else {
            b_in
        };
        let sb = m.shape(b_fix).clone();
        let a_fix = m.reshape_(
            a_in,
            vec![
                sb.dim(0).unwrap_static() as i64,
                sb.dim(1).unwrap_static() as i64,
                1,
            ],
        );
        return m.add_node(Op::Binary(op), vec![a_fix, b_fix], sb);
    }
    // NCHW `[1,C,1,L]` + BLC `[1,Lb,C]` (encode conv1 + conv2).
    if sa.rank() == 4
        && sb.rank() == 3
        && sa.dim(2).unwrap_static() == 1
        && is_blc_rank3(&sb)
        && sa.dim(1).unwrap_static() == sb.dim(2).unwrap_static()
    {
        let c = sa.dim(1).unwrap_static();
        let la = sa.dim(3).unwrap_static();
        let lb = sb.dim(1).unwrap_static();
        let l = la.min(lb);
        let a_fix = if la > l {
            m.narrow_(a_in, 3, 0, l)
        } else {
            a_in
        };
        let b_ncl = m.transpose_(b_in, vec![0, 2, 1]);
        let b_fix = if lb > l {
            m.narrow_(b_ncl, 2, 0, l)
        } else {
            b_ncl
        };
        let b_4 = m.reshape_(
            b_fix,
            vec![sa.dim(0).unwrap_static() as i64, c as i64, 1, l as i64],
        );
        let sh = Shape::new(&[sa.dim(0).unwrap_static(), c, 1, l], sa.dtype());
        return m.add_node(Op::Binary(op), vec![a_fix, b_4], sh);
    }
    if sb.rank() == 4
        && sa.rank() == 3
        && sb.dim(2).unwrap_static() == 1
        && is_blc_rank3(&sa)
        && sb.dim(1).unwrap_static() == sa.dim(2).unwrap_static()
    {
        let c = sb.dim(1).unwrap_static();
        let la = sa.dim(1).unwrap_static();
        let lb = sb.dim(3).unwrap_static();
        let l = la.min(lb);
        let a_ncl = m.transpose_(a_in, vec![0, 2, 1]);
        let a_fix = if la > l {
            m.narrow_(a_ncl, 2, 0, l)
        } else {
            a_ncl
        };
        let a_4 = m.reshape_(
            a_fix,
            vec![sb.dim(0).unwrap_static() as i64, c as i64, 1, l as i64],
        );
        let b_fix = if lb > l {
            m.narrow_(b_in, 3, 0, l)
        } else {
            b_in
        };
        let sh = Shape::new(&[sb.dim(0).unwrap_static(), c, 1, l], sb.dtype());
        return m.add_node(Op::Binary(op), vec![a_4, b_fix], sh);
    }
    if sa.rank() == 4
        && sb.rank() == 3
        && is_nc1_rank3(&sb)
        && sa.dim(0).unwrap_static() == sb.dim(0).unwrap_static()
        && sa.dim(1).unwrap_static() == sb.dim(1).unwrap_static()
    {
        let b_fix = m.reshape_(
            b_in,
            vec![
                sa.dim(0).unwrap_static() as i64,
                sb.dim(1).unwrap_static() as i64,
                1,
                1,
            ],
        );
        return m.add_node(Op::Binary(op), vec![a_in, b_fix], sa.clone());
    }
    if sb.rank() == 4
        && sa.rank() == 3
        && is_nc1_rank3(&sa)
        && sa.dim(0).unwrap_static() == sb.dim(0).unwrap_static()
        && sb.dim(1).unwrap_static() == sa.dim(1).unwrap_static()
    {
        let a_fix = m.reshape_(
            a_in,
            vec![
                sb.dim(0).unwrap_static() as i64,
                sa.dim(1).unwrap_static() as i64,
                1,
                1,
            ],
        );
        return m.add_node(Op::Binary(op), vec![a_fix, b_in], sb.clone());
    }
    if let Ok(sh) = rlx_ir::shape::binary_shape(&sa, &sb) {
        return m.add_node(Op::Binary(op), vec![a_in, b_in], sh);
    }
    if sa.rank() == 3
        && sb.rank() == 3
        && sa.dim(1).unwrap_static() == 1
        && sb.dim(0).unwrap_static() == 1
        && sb.dim(1).unwrap_static() == 1
        && sa
            .dim(2)
            .unwrap_static()
            .is_multiple_of(sb.dim(2).unwrap_static())
    {
        let bsz = sa.dim(0).unwrap_static();
        let c = sb.dim(2).unwrap_static();
        let l = sa.dim(2).unwrap_static() / c;
        let dt = sa.dtype();
        let a3 = m.reshape_(a_in, vec![bsz as i64, c as i64, l as i64]);
        let b3 = m.reshape_(b_in, vec![1, c as i64, 1]);
        let y3 = m.add_node(Op::Binary(op), vec![a3, b3], Shape::new(&[bsz, c, l], dt));
        return m.reshape_(y3, vec![bsz as i64, 1, (c * l) as i64]);
    }
    // [B,2C,L] op [B,C,L] (AdaIN / half-channel scale).
    if sa.rank() == 3
        && sb.rank() == 3
        && sa.dim(0).unwrap_static() == sb.dim(0).unwrap_static()
        && sa.dim(2).unwrap_static() == sb.dim(2).unwrap_static()
        && sa.dim(1).unwrap_static() == 2 * sb.dim(1).unwrap_static()
    {
        let bsz = sa.dim(0).unwrap_static();
        let c = sb.dim(1).unwrap_static();
        let l = sa.dim(2).unwrap_static();
        let dt = sa.dtype();
        let b_rep = m.reshape_(b_in, vec![bsz as i64, c as i64, 1, l as i64]);
        let b_rep = m.add_node(
            Op::Concat { axis: 3 },
            vec![b_rep, b_rep],
            Shape::new(&[bsz, c, 2, l], dt),
        );
        let b_wide = m.reshape_(b_rep, vec![bsz as i64, (2 * c) as i64, l as i64]);
        let sh = rlx_ir::shape::binary_shape(m.shape(a_in), m.shape(b_wide)).unwrap_or(sa.clone());
        return m.add_node(Op::Binary(op), vec![a_in, b_wide], sh);
    }
    if sa.rank() == 3 && sb.rank() == 3 {
        if is_rank3_ncl_pair(&sa, &sb) {
            b_in = m.transpose_(b_in, vec![0, 2, 1]);
        } else if is_rank3_ncl_pair(&sb, &sa) {
            a_in = m.transpose_(a_in, vec![0, 2, 1]);
        }
        let sa = m.shape(a_in).clone();
        let sb = m.shape(b_in).clone();
        if let Some(id) = align_ncl_blc_binary(m, op, a_in, b_in, &sa, &sb) {
            return id;
        }
        if let Ok(sh) = rlx_ir::shape::binary_shape(&sa, &sb) {
            return m.add_node(Op::Binary(op), vec![a_in, b_in], sh);
        }
        // AdaIN: scale `[1,C,1]` × first C channels of BLC `[1,L,2C]`.
        if op == BinaryOp::Mul
            && is_nc1_rank3(&sa)
            && is_blc_rank3(&sb)
            && sb.dim(2).unwrap_static() == 2 * sa.dim(1).unwrap_static()
        {
            let bsz = sb.dim(0).unwrap_static();
            let half = sa.dim(1).unwrap_static();
            let l = sb.dim(1).unwrap_static();
            let dt = sa.dtype();
            let scale = m.transpose_(a_in, vec![0, 2, 1]);
            let b0 = m.narrow_(b_in, 2, 0, half);
            let scaled = m.add_node(
                Op::Binary(op),
                vec![scale, b0],
                Shape::new(&[bsz, l, half], dt),
            );
            let b1 = m.narrow_(b_in, 2, half, half);
            return m.add_node(Op::Concat { axis: 2 }, vec![scaled, b1], sb.clone());
        }
    }
    if is_blc_rank3(&sa)
        && is_nc1_rank3(&sb)
        && sa.dim(2).unwrap_static() == sb.dim(1).unwrap_static()
    {
        let b_fix = m.reshape_(
            b_in,
            vec![
                sa.dim(0).unwrap_static() as i64,
                1,
                sb.dim(1).unwrap_static() as i64,
            ],
        );
        return m.add_node(Op::Binary(op), vec![a_in, b_fix], sa.clone());
    }
    if is_nc1_rank3(&sa)
        && is_blc_rank3(&sb)
        && sb.dim(2).unwrap_static() == sa.dim(1).unwrap_static()
    {
        let a_fix = m.reshape_(
            a_in,
            vec![
                sb.dim(0).unwrap_static() as i64,
                1,
                sa.dim(1).unwrap_static() as i64,
            ],
        );
        return m.add_node(Op::Binary(op), vec![a_fix, b_in], sb.clone());
    }
    if is_ncl_rank3(&sa) && is_nc1_rank3(&sb) {
        let a_fix = if m.shape(a_in).rank() == 4 {
            collapse_duplicate_channel_4d(m, a_in)
        } else {
            a_in
        };
        let b_fix = if m.shape(b_in).rank() == 3
            && m.shape(b_in).dim(2).unwrap_static() == 1
            && m.shape(b_in).dim(1).unwrap_static() == sa.dim(1).unwrap_static()
            && m.shape(b_in).dim(0).unwrap_static() != sa.dim(0).unwrap_static()
        {
            m.reshape_(
                b_in,
                vec![
                    sa.dim(0).unwrap_static() as i64,
                    sa.dim(1).unwrap_static() as i64,
                    1,
                ],
            )
        } else {
            b_in
        };
        return m.add_node(Op::Binary(op), vec![a_fix, b_fix], sa.clone());
    }
    if is_nc1_rank3(&sa) && is_ncl_rank3(&sb) {
        let b_fix = if m.shape(b_in).rank() == 4 {
            collapse_duplicate_channel_4d(m, b_in)
        } else {
            b_in
        };
        let a_fix = if m.shape(a_in).rank() == 3
            && m.shape(a_in).dim(2).unwrap_static() == 1
            && m.shape(a_in).dim(1).unwrap_static() == sb.dim(1).unwrap_static()
            && m.shape(a_in).dim(0).unwrap_static() != sb.dim(0).unwrap_static()
        {
            m.reshape_(
                a_in,
                vec![
                    sb.dim(0).unwrap_static() as i64,
                    sb.dim(1).unwrap_static() as i64,
                    1,
                ],
            )
        } else {
            a_in
        };
        return m.add_node(Op::Binary(op), vec![a_fix, b_fix], sb.clone());
    }
    if meta_layout_ncl(&sa)
        && is_nc1_rank3(&sb)
        && sa.dim(1).unwrap_static() == sb.dim(1).unwrap_static()
    {
        return m.add_node(Op::Binary(op), vec![a_in, b_in], sa.clone());
    }
    if is_nc1_rank3(&sa)
        && meta_layout_ncl(&sb)
        && sb.dim(1).unwrap_static() == sa.dim(1).unwrap_static()
    {
        return m.add_node(Op::Binary(op), vec![a_in, b_in], sb.clone());
    }
    // AdaIN scale `[1,C,1]` × normalized `[1,C,L]`.
    if op == BinaryOp::Mul
        && is_nc1_rank3(&sa)
        && (is_vocoder_ncl(&sb) || is_ncl_rank3(&sb))
        && sa.dim(1).unwrap_static() == sb.dim(1).unwrap_static()
        && sa.dim(2).unwrap_static() == 1
    {
        return m.add_node(Op::Binary(op), vec![a_in, b_in], sb.clone());
    }
    if meta_layout_ncl(&sa) && sb.rank() == 3 && sb.dim(2).unwrap_static() == 1 {
        if is_cl1_rank3(&sb) {
            let c = sb.dim(0).unwrap_static();
            let l = sb.dim(1).unwrap_static();
            let b_fix = m.reshape_(
                b_in,
                vec![sa.dim(0).unwrap_static() as i64, c as i64, l as i64],
            );
            return m.add_node(Op::Binary(op), vec![a_in, b_fix], sa.clone());
        }
        if sa.dim(1).unwrap_static() == sb.dim(1).unwrap_static()
            && sa.dim(2).unwrap_static() == sb.dim(0).unwrap_static()
        {
            let b_fix = m.reshape_(
                b_in,
                vec![
                    sa.dim(0).unwrap_static() as i64,
                    sb.dim(1).unwrap_static() as i64,
                    sb.dim(0).unwrap_static() as i64,
                ],
            );
            return m.add_node(Op::Binary(op), vec![a_in, b_fix], sa.clone());
        }
    }
    if sa.rank() == 3 && sa.dim(2).unwrap_static() == 1 && meta_layout_ncl(&sb) {
        if is_cl1_rank3(&sa) {
            let c = sa.dim(0).unwrap_static();
            let l = sa.dim(1).unwrap_static();
            let a_fix = m.reshape_(
                a_in,
                vec![sb.dim(0).unwrap_static() as i64, c as i64, l as i64],
            );
            return m.add_node(Op::Binary(op), vec![a_fix, b_in], sb.clone());
        }
        if sb.dim(1).unwrap_static() == sa.dim(1).unwrap_static()
            && sb.dim(2).unwrap_static() == sa.dim(0).unwrap_static()
        {
            let a_fix = m.reshape_(
                a_in,
                vec![
                    sb.dim(0).unwrap_static() as i64,
                    sa.dim(1).unwrap_static() as i64,
                    sa.dim(0).unwrap_static() as i64,
                ],
            );
            return m.add_node(Op::Binary(op), vec![a_fix, b_in], sb.clone());
        }
    }
    // Generator NHWC: `[1,C,1]` scale × `[1,1,H,W]` activations.
    if sa.rank() == 3 && is_nc1_rank3(&sa) && sb.rank() == 4 && sb.dim(1).unwrap_static() == 1 {
        let c = sa.dim(1).unwrap_static();
        let scale = m.reshape_(a_in, vec![1, c as i64, 1, 1]);
        return m.add_node(Op::Binary(op), vec![scale, b_in], sb.clone());
    }
    if sb.rank() == 3 && is_nc1_rank3(&sb) && sa.rank() == 4 && sa.dim(1).unwrap_static() == 1 {
        let c = sb.dim(1).unwrap_static();
        let scale = m.reshape_(b_in, vec![1, c as i64, 1, 1]);
        return m.add_node(Op::Binary(op), vec![a_in, scale], sa.clone());
    }
    // Generator: `[1,C,1,L]` conv transpose + `[1,C,L]` noise (trim to common length).
    if sa.rank() == 4
        && sb.rank() == 3
        && sa.dim(0).unwrap_static() == sb.dim(0).unwrap_static()
        && sa.dim(1).unwrap_static() == sb.dim(1).unwrap_static()
        && sa.dim(2).unwrap_static() == 1
    {
        let l = sb.dim(2).unwrap_static();
        let l_in = sa.dim(3).unwrap_static();
        let a_use = if l_in > l {
            m.narrow_(a_in, 3, 0, l)
        } else {
            a_in
        };
        let a_fix = m.reshape_(
            a_use,
            vec![
                sa.dim(0).unwrap_static() as i64,
                sa.dim(1).unwrap_static() as i64,
                l as i64,
            ],
        );
        return m.add_node(Op::Binary(op), vec![a_fix, b_in], sb.clone());
    }
    if sb.rank() == 4
        && sa.rank() == 3
        && sa.dim(0).unwrap_static() == sb.dim(0).unwrap_static()
        && sa.dim(1).unwrap_static() == sb.dim(1).unwrap_static()
        && sb.dim(2).unwrap_static() == 1
    {
        let l = sa.dim(2).unwrap_static();
        let l_in = sb.dim(3).unwrap_static();
        let b_use = if l_in > l {
            m.narrow_(b_in, 3, 0, l)
        } else {
            b_in
        };
        let b_fix = m.reshape_(
            b_use,
            vec![
                sb.dim(0).unwrap_static() as i64,
                sb.dim(1).unwrap_static() as i64,
                l as i64,
            ],
        );
        return m.add_node(Op::Binary(op), vec![a_in, b_fix], sa.clone());
    }
    // Generator NCHW: `[1,C,1,L]` + `[1,C,1,L']` — trim to common length.
    if sa.rank() == 4
        && sb.rank() == 4
        && sa.dim(0).unwrap_static() == sb.dim(0).unwrap_static()
        && sa.dim(1).unwrap_static() == sb.dim(1).unwrap_static()
        && sa.dim(2).unwrap_static() == 1
        && sb.dim(2).unwrap_static() == 1
        && is_typical_channel(sa.dim(1).unwrap_static())
    {
        let la = sa.dim(3).unwrap_static();
        let lb = sb.dim(3).unwrap_static();
        if la != lb {
            let l = la.min(lb);
            let a_fix = if la > l {
                m.narrow_(a_in, 3, 0, l)
            } else {
                a_in
            };
            let b_fix = if lb > l {
                m.narrow_(b_in, 3, 0, l)
            } else {
                b_in
            };
            let sh = Shape::new(
                &[sa.dim(0).unwrap_static(), sa.dim(1).unwrap_static(), 1, l],
                sa.dtype(),
            );
            return m.add_node(Op::Binary(op), vec![a_fix, b_fix], sh);
        }
    }
    // Generator / vocoder: same `[1,C,L]` channel axis, trim to common length.
    if sa.rank() == 3
        && sb.rank() == 3
        && sa.dim(0).unwrap_static() == sb.dim(0).unwrap_static()
        && sa.dim(1).unwrap_static() == sb.dim(1).unwrap_static()
        && is_typical_channel(sa.dim(1).unwrap_static())
    {
        let la = sa.dim(2).unwrap_static();
        let lb = sb.dim(2).unwrap_static();
        if la != lb {
            let l = la.min(lb);
            let a_fix = if la > l {
                m.narrow_(a_in, 2, 0, l)
            } else {
                a_in
            };
            let b_fix = if lb > l {
                m.narrow_(b_in, 2, 0, l)
            } else {
                b_in
            };
            let sh = Shape::new(
                &[sa.dim(0).unwrap_static(), sa.dim(1).unwrap_static(), l],
                sa.dtype(),
            );
            return m.add_node(Op::Binary(op), vec![a_fix, b_fix], sh);
        }
    }
    match rlx_ir::shape::binary_shape(&sa, &sb) {
        Ok(sh) => m.add_node(Op::Binary(op), vec![a_in, b_in], sh),
        Err(e) => panic!(
            "binary_infer at {site}: unaligned {:?} vs {:?}: {e}",
            sa.dims(),
            sb.dims()
        ),
    }
}

fn is_sequence_dim_label(s: &str) -> bool {
    s == "sequence_length" || s == "?" || (s.starts_with("Cast") && s.contains("duration"))
}

fn resolve_dim_ir(v: &serde_json::Value, opts: &ImportOptions) -> Result<Dim> {
    match v {
        serde_json::Value::Number(n) => n
            .as_u64()
            .map(|x| Dim::Static(x as usize))
            .ok_or_else(|| anyhow!("bad dim {n}")),
        serde_json::Value::String(s) => match s.as_str() {
            "num_samples" => Ok(Dim::Static(opts.max_waveform_samples)),
            s if is_sequence_dim_label(s) => {
                if opts.dynamic_sequence {
                    Ok(Dim::Dynamic(sym::SEQ))
                } else {
                    Ok(Dim::Static(opts.sequence_length))
                }
            }
            s if s.starts_with("Cast") => {
                if opts.dynamic_sequence {
                    Ok(Dim::Dynamic(sym::SEQ))
                } else {
                    Ok(Dim::Static(opts.sequence_length))
                }
            }
            s if s.starts_with("unk__") => {
                if opts.dynamic_sequence {
                    Ok(Dim::Dynamic(sym::SEQ))
                } else {
                    bail!("unresolved ONNX symbolic dim {s} (run shape propagation first)")
                }
            }
            other => bail!("unknown symbolic dim {other}"),
        },
        _ => bail!("invalid dim {v:?}"),
    }
}

pub fn resolve_dim(v: &serde_json::Value, opts: &ImportOptions) -> Result<usize> {
    match resolve_dim_ir(v, opts)? {
        Dim::Static(n) => Ok(n),
        Dim::Dynamic(_) => Ok(opts.sequence_length),
    }
}

fn dim_usize(d: Dim, opts: &ImportOptions) -> usize {
    match d {
        Dim::Static(n) => n,
        Dim::Dynamic(_) => opts.sequence_length,
    }
}

fn shape_dims_i64(shape: &Shape, opts: &ImportOptions) -> Vec<i64> {
    shape
        .dims()
        .iter()
        .map(|&d| dim_usize(d, opts) as i64)
        .collect()
}

pub fn resolve_shape(meta: &serde_json::Value, opts: &ImportOptions) -> Result<Shape> {
    let obj = meta.as_object().context("shape meta object")?;
    let shape_v = obj.get("shape").context("shape field")?;
    let dtype_s = obj.get("dtype").and_then(|d| d.as_str()).unwrap_or("f32");
    let dims: Vec<Dim> = match shape_v {
        serde_json::Value::Array(a) if a.is_empty() => {
            bail!("empty shape meta (unknown rank)")
        }
        serde_json::Value::Array(a) => a
            .iter()
            .map(|d| resolve_dim_ir(d, opts))
            .collect::<Result<_>>()?,
        _ => bail!("shape not array"),
    };
    let dtype = match dtype_s {
        "f32" => DType::F32,
        "i64" => DType::I64,
        "i32" => DType::I32,
        "bool" => DType::Bool,
        "u8" | "uint8" | "type_2" => DType::U8,
        "i8" | "int8" | "type_3" => DType::I8,
        _ => DType::F32,
    };
    Ok(Shape::from_dims(&dims, dtype))
}

fn load_f32_param(
    bundle: &RlxBundle,
    params: &mut HashMap<String, Vec<f32>>,
    key: &str,
) -> Result<Vec<f32>> {
    if let Some(v) = params.get(key) {
        return Ok(v.clone());
    }
    let st = bundle.weights()?;
    let view = st
        .tensor(key)
        .with_context(|| format!("missing weight {key}"))?;
    let out: Vec<f32> = match view.dtype() {
        safetensors::tensor::Dtype::F32 => view
            .data()
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
        safetensors::tensor::Dtype::F16 => view
            .data()
            .chunks_exact(2)
            .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
            .collect(),
        safetensors::tensor::Dtype::BF16 => view
            .data()
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                f32::from_bits((bits as u32) << 16)
            })
            .collect(),
        safetensors::tensor::Dtype::I32 => view
            .data()
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32)
            .collect(),
        safetensors::tensor::Dtype::U8 | safetensors::tensor::Dtype::I8 => {
            view.data().iter().map(|&b| b as f32).collect()
        }
        other => anyhow::bail!("load_f32_param: unsupported dtype {other:?} for {key}"),
    };
    params.insert(key.to_string(), out.clone());
    Ok(out)
}

struct LowerCtx<'a> {
    nodes: &'a [BundleNode],
    opts: &'a ImportOptions,
    env: HashMap<String, HirNodeId>,
    params: HashMap<String, Vec<f32>>,
    typed_params: TypedParams,
    quant_weight_keys: HashSet<String>,
    i64_params: HashMap<String, Vec<i64>>,
    init_shapes: &'a HashMap<String, Vec<usize>>,
    report: ImportReport,
}

impl<'a> LowerCtx<'a> {
    fn tensor(&self, name: &str) -> Result<HirNodeId> {
        self.env
            .get(name)
            .copied()
            .with_context(|| format!("tensor not in env: {name}"))
    }

    fn shape_for(&self, m: &HirMut<'_>, name: &str) -> Result<Shape> {
        if let Some(id) = self.env.get(name) {
            return Ok(m.shape(*id).clone());
        }
        for node in self.nodes {
            for (i, out) in node.outputs.iter().enumerate() {
                if out == name {
                    if let Ok(s) = resolve_shape(&node.output_meta[i], self.opts) {
                        return Ok(s);
                    }
                    if let Some(inp) = node.inputs.first().filter(|s| !s.is_empty()) {
                        if let Some(id) = self.env.get(inp.as_str()) {
                            return Ok(m.shape(*id).clone());
                        }
                    }
                }
            }
        }
        bail!("no shape for {name}")
    }

    fn unsupported(&mut self, op: &str) {
        *self.report.unsupported.entry(op.to_string()).or_insert(0) += 1;
        self.report.skipped += 1;
    }

    fn record_stub(&mut self, node: &BundleNode, reason: &str) {
        self.report.stubbed += 1;
        if self.report.stubbed_nodes.len() < 32 {
            self.report
                .stubbed_nodes
                .push(format!("{} ({})", node.name, reason));
        }
    }

    fn unsupported_node(&mut self, node: &BundleNode, op: &str) -> Result<()> {
        self.unsupported(op);
        if self.opts.strict {
            anyhow::bail!("unsupported ONNX op `{op}` at node {}", node.name);
        }
        Ok(())
    }

    /// Bind outputs to the sole input when possible; otherwise zero stubs.
    fn passthrough_stub(&mut self, m: &mut HirMut<'_>, node: &BundleNode) -> Result<()> {
        if node.inputs.len() == 1 {
            let inp = &node.inputs[0];
            if !inp.is_empty() {
                if let Ok(x) = self.tensor(inp) {
                    for out in &node.outputs {
                        self.env.insert(out.clone(), x);
                    }
                    return Ok(());
                }
            }
        }
        self.stub_output(m, node, "passthrough")
    }

    fn stub_output(&mut self, m: &mut HirMut<'_>, node: &BundleNode, reason: &str) -> Result<()> {
        if self.opts.strict {
            anyhow::bail!(
                "strict import: stub `{reason}` not allowed at node {} ({})",
                node.name,
                node.op
            );
        }
        self.record_stub(node, reason);
        for (i, out_name) in node.outputs.iter().enumerate() {
            let meta = node.output_meta.get(i).or_else(|| node.output_meta.first());
            let Some(meta) = meta else {
                continue;
            };
            if let Ok(mut shape) = resolve_shape(meta, self.opts) {
                if let Some(fix) = self.opts.output_shape_fix {
                    if let Some(s) = fix(node.name.as_str(), &shape) {
                        shape = s;
                    }
                }
                let key = format!("__stub__/{}", out_name);
                let n = shape.num_elements().unwrap_or(1).min(MAX_STUB_ELEMENTS);
                let id = m.param(&key, shape);
                self.params.insert(key, vec![0.0; n]);
                self.env.insert(out_name.clone(), id);
            }
        }
        Ok(())
    }

    fn f32_scalar_param(&mut self, m: &mut HirMut<'_>, key: &str, v: f32) -> HirNodeId {
        if let Some(&id) = self.env.get(key) {
            return id;
        }
        let id = m.param(key, Shape::new(&[1], DType::F32));
        self.params.insert(key.to_string(), vec![v]);
        self.env.insert(key.to_string(), id);
        id
    }

    fn ensure_f32_param(&mut self, m: &mut HirMut<'_>, key: &str) -> Result<HirNodeId> {
        if let Some(&id) = self.env.get(key) {
            return Ok(id);
        }
        let shape_dims = self
            .init_shapes
            .get(key)
            .cloned()
            .or_else(|| self.params.get(key).map(|v| vec![v.len()]))
            .with_context(|| format!("missing f32 param {key}"))?;
        let shape = Shape::new(&shape_dims, DType::F32);
        let id = m.param(key, shape);
        self.env.insert(key.to_string(), id);
        Ok(id)
    }

    fn ensure_typed_param(&mut self, m: &mut HirMut<'_>, key: &str) -> Result<HirNodeId> {
        if let Some(&id) = self.env.get(key) {
            return Ok(id);
        }
        let (bytes, dtype) = self
            .typed_params
            .get(key)
            .with_context(|| format!("missing typed param {key}"))?;
        let shape_dims = self
            .init_shapes
            .get(key)
            .cloned()
            .unwrap_or_else(|| vec![bytes.len()]);
        let shape = Shape::new(&shape_dims, *dtype);
        let id = m.param(key, shape);
        self.env.insert(key.to_string(), id);
        Ok(id)
    }
}

pub fn build_hir_from_bundle(
    bundle: &RlxBundle,
    opts: ImportOptions,
) -> Result<(
    HirModule,
    HashMap<String, Vec<f32>>,
    TypedParams,
    ImportReport,
)> {
    let mut params = HashMap::new();
    let mut init_shapes: HashMap<String, Vec<usize>> = HashMap::new();
    let st = bundle.weights()?;
    for name in st.names() {
        let key = name.to_string();
        let view = st.tensor(&key)?;
        init_shapes.insert(key.clone(), view.shape().to_vec());
        match view.dtype() {
            safetensors::tensor::Dtype::I64 | safetensors::tensor::Dtype::BOOL => {}
            safetensors::tensor::Dtype::U8 | safetensors::tensor::Dtype::I8
                if opts.use_quantized_kernels && key.ends_with("_quantized") => {}
            _ => {
                let _ = load_f32_param(bundle, &mut params, &key)?;
            }
        }
    }
    let i64_params = crate::tensor_data::load_i64_params(&bundle.weight_bytes).unwrap_or_default();
    let (typed_params, quant_shapes) = if opts.use_quantized_kernels {
        crate::tensor_data::load_typed_quant_params(&bundle.weight_bytes)?
    } else {
        (TypedParams::new(), HashMap::new())
    };
    init_shapes.extend(quant_shapes);
    crate::tensor_data::materialize_quantized_f32(
        &bundle.weight_bytes,
        &mut params,
        &mut init_shapes,
    )?;
    build_hir_from_parts(
        &bundle.manifest,
        bundle.nodes.clone(),
        params,
        typed_params,
        i64_params,
        &init_shapes,
        opts,
    )
}

/// Lower from manifest + nodes + in-memory initializer tensors.
pub fn build_hir_from_parts(
    manifest: &BundleManifest,
    nodes: Vec<BundleNode>,
    mut params: HashMap<String, Vec<f32>>,
    typed_params: TypedParams,
    i64_params: HashMap<String, Vec<i64>>,
    init_shapes: &HashMap<String, Vec<usize>>,
    opts: ImportOptions,
) -> Result<(
    HirModule,
    HashMap<String, Vec<f32>>,
    TypedParams,
    ImportReport,
)> {
    let rewritten = rewrite_graph(
        nodes,
        &params,
        init_shapes,
        manifest,
        &opts,
        &typed_params.keys().cloned().collect(),
    );
    params.extend(rewritten.extra_params);
    let mut init_shapes = init_shapes.clone();
    init_shapes.extend(rewritten.extra_shapes);
    let nodes = topo_sort_nodes(rewritten.nodes);

    let mut hir = HirModule::new("onnx_import");
    let mut m = HirMut::new(&mut hir);
    let quant_weight_keys: HashSet<String> = typed_params.keys().cloned().collect();
    let init_names: HashSet<String> = params
        .keys()
        .chain(i64_params.keys())
        .chain(typed_params.keys())
        .cloned()
        .collect();
    let mut ctx = LowerCtx {
        nodes: &nodes,
        opts: &opts,
        env: HashMap::new(),
        params,
        typed_params,
        quant_weight_keys,
        i64_params,
        init_shapes: &init_shapes,
        report: ImportReport::default(),
    };

    for io in &manifest.inputs {
        let shape = resolve_shape(
            &serde_json::json!({"shape": io.meta.shape, "dtype": io.meta.dtype}),
            &opts,
        )?;
        let id = m.input(&io.name, shape);
        ctx.env.insert(io.name.clone(), id);
    }

    if nodes
        .iter()
        .any(|n| n.inputs.iter().any(|i| i == DURATION_CARRY))
    {
        let shape = Shape::new(&[opts.sequence_length], DType::I64);
        let id = m.param(DURATION_CARRY, shape);
        ctx.env.insert(DURATION_CARRY.to_string(), id);
        let seed: Vec<i64> = vec![0; opts.sequence_length];
        let bytes: Vec<u8> = seed.iter().flat_map(|d| d.to_le_bytes()).collect();
        ctx.typed_params
            .insert(DURATION_CARRY.to_string(), (bytes, DType::I64));
    }

    let mut pending: Vec<&BundleNode> = nodes.iter().collect();
    let mut guard = 0usize;
    while !pending.is_empty() {
        guard += 1;
        if guard > nodes.len() + 8 {
            let stuck = pending.first().map(|n| n.name.as_str()).unwrap_or("?");
            bail!("HIR lowering stuck before node {stuck}");
        }
        let mut next: Vec<&BundleNode> = Vec::new();
        for node in &pending {
            if node_inputs_ready(node, &ctx, &init_names) {
                lower_node(&mut m, &mut ctx, node, &init_names)
                    .with_context(|| format!("lowering node {}", node.name))?;
            } else {
                next.push(*node);
            }
        }
        if next.len() == pending.len() {
            let stuck = pending.first().context("no pending nodes")?;
            let missing: Vec<&str> = stuck
                .inputs
                .iter()
                .map(String::as_str)
                .filter(|inp| {
                    !inp.is_empty() && !init_names.contains(*inp) && !ctx.env.contains_key(*inp)
                })
                .collect();
            bail!(
                "HIR lowering cannot resolve inputs for {} (missing: {missing:?})",
                stuck.name
            );
        }
        pending = next;
    }

    let mut outs = Vec::new();
    for o in &manifest.outputs {
        outs.push(ctx.tensor(&o.name)?);
    }
    m.0.set_outputs(outs);

    validate_nchw_ops(&m)?;

    let report = ctx.report;
    crate::strict::validate_strict_import(&opts, &report)?;

    let params = ctx.params;
    let typed_params = ctx.typed_params;
    Ok((hir, params, typed_params, report))
}

fn node_inputs_ready(node: &BundleNode, ctx: &LowerCtx<'_>, inits: &HashSet<String>) -> bool {
    let extra: Vec<String> = if node.op == "ConcatFromSequence" {
        control_flow::resolve_duration_align_inputs(ctx.nodes)
            .map(|a| vec![a.duration_mask, a.range_ids, a.split_lens, a.trip_count])
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    for inp in node
        .inputs
        .iter()
        .map(String::as_str)
        .chain(extra.iter().map(String::as_str))
    {
        if inp.is_empty() || inits.contains(inp) {
            continue;
        }
        if !ctx.env.contains_key(inp) {
            return false;
        }
    }
    true
}

fn validate_nchw_ops(m: &HirMut<'_>) -> Result<()> {
    for (idx, node) in m.0.nodes().iter().enumerate() {
        let HirOp::Mir(op) = &node.op else {
            continue;
        };
        let needs_4d = matches!(
            op,
            Op::Conv { .. }
                | Op::ConvTranspose2d { .. }
                | Op::ResizeNearest2x
                | Op::LayerNorm2d { .. }
                | Op::GroupNorm { .. }
                | Op::Pool { .. }
        );
        if !needs_4d || node.inputs.is_empty() {
            continue;
        }
        let rank = m.shape(node.inputs[0]).rank();
        if rank < 4 {
            bail!("HIR node {idx} {op:?} expects NCHW rank 4, got rank {rank} on input 0",);
        }
    }
    Ok(())
}

fn lower_node(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
    inits: &HashSet<String>,
) -> Result<()> {
    for name in &node.inputs {
        if ctx.env.contains_key(name) {
            continue;
        }
        if let Some(i64_data) = ctx.i64_params.get(name.as_str()) {
            let shape_dims = ctx
                .init_shapes
                .get(name.as_str())
                .cloned()
                .unwrap_or_else(|| vec![i64_data.len()]);
            let bytes: Vec<u8> = i64_data.iter().flat_map(|d| d.to_le_bytes()).collect();
            let shape = Shape::new(&shape_dims, DType::I64);
            let id = m.add_node(Op::Constant { data: bytes }, vec![], shape);
            ctx.env.insert(name.clone(), id);
            continue;
        }
        if !inits.contains(name) {
            continue;
        }
        if let Some((_, dtype)) = ctx.typed_params.get(name.as_str()) {
            let shape_dims = ctx
                .init_shapes
                .get(name.as_str())
                .with_context(|| format!("typed weight shape for {name}"))?;
            let shape = Shape::new(shape_dims, *dtype);
            let id = m.param(name, shape);
            ctx.env.insert(name.clone(), id);
            continue;
        }
        let shape_dims = ctx
            .init_shapes
            .get(name.as_str())
            .with_context(|| format!("weight shape for {name}"))?;
        let shape = Shape::new(shape_dims, DType::F32);
        let id = m.param(name, shape);
        ctx.env.insert(name.clone(), id);
    }

    let op = node.op.as_str();
    let lowered = match op {
        "Add" | "Mul" | "Sub" | "Div" | "Max" | "Min" => lower_binary(m, ctx, node, op)?,
        "Mod" => lower_mod(m, ctx, node)?,
        "Identity" => lower_identity(m, ctx, node)?,
        "IsNaN" => lower_is_nan(m, ctx, node)?,
        "MatMul" => lower_matmul(m, ctx, node)?,
        "QMatMul" => lower_qmatmul(m, ctx, node)?,
        "ActCopy" => lower_act_copy(m, ctx, node)?,
        "Gemm" => lower_gemm(m, ctx, node)?,
        "Relu" => lower_activation(m, ctx, node, Activation::Relu)?,
        "Tanh" | "Sigmoid" | "Sqrt" | "Sin" | "Cos" | "Exp" | "Neg" | "Abs" | "Atan" | "Floor"
        | "Round" | "Erf" => lower_activation_map(m, ctx, node, op)?,
        "LeakyRelu" => lower_leaky_relu(m, ctx, node)?,
        "Cast" => lower_cast(m, ctx, node)?,
        "Transpose" => lower_transpose(m, ctx, node)?,
        "Reshape" | "Unsqueeze" | "Squeeze" | "Flatten" => lower_reshape(m, ctx, node)?,
        "Gather" => lower_gather(m, ctx, node)?,
        "Concat" => lower_concat(m, ctx, node)?,
        "Softmax" => lower_softmax(m, ctx, node)?,
        "LayerNormalization" => lower_layer_norm(m, ctx, node)?,
        "InstanceNormalization" => lower_instance_norm(m, ctx, node)?,
        "BatchNormalization" => lower_batch_norm(m, ctx, node)?,
        "AveragePool" | "MaxPool" | "GlobalAveragePool" => lower_pool(m, ctx, node, op)?,
        "Dropout" => lower_dropout(m, ctx, node)?,
        "Pow" => lower_pow(m, ctx, node)?,
        "Clip" => lower_clip(m, ctx, node)?,
        "Where" => lower_where(m, ctx, node)?,
        "Expand" => lower_expand(m, ctx, node)?,
        "Equal" | "Less" | "Greater" | "Not" | "And" => lower_compare(m, ctx, node, op)?,
        "ReduceMean" | "ReduceSum" | "ReduceMax" | "ReduceMin" | "ReduceProd" => {
            lower_reduce(m, ctx, node, op)?
        }
        "Conv" => lower_conv(m, ctx, node, false)?,
        "ConvTranspose" => lower_conv(m, ctx, node, true)?,
        "Slice" => lower_slice(m, ctx, node)?,
        "Shape" => lower_shape_op(m, ctx, node)?,
        "ConstantOfShape" => lower_constant_of_shape(m, ctx, node)?,
        "Pad" => lower_pad_as_concat(m, ctx, node)?,
        "Range" => lower_range(m, ctx, node)?,
        "DynamicQuantizeLinear" => lower_dynamic_quant(m, ctx, node)?,
        "Resize" => lower_resize(m, ctx, node)?,
        "TopK" => lower_topk(m, ctx, node)?,
        "CumSum" => lower_cumsum(m, ctx, node)?,
        "ScatterND" => lower_scatter_nd(m, ctx, node)?,
        "ScatterElements" => lower_scatter_elements(m, ctx, node)?,
        "DynamicQuantizeLSTM" => lower_dynamic_quantize_lstm(m, ctx, node)?,
        "RandomNormalLike" | "RandomUniformLike" => lower_random_like(m, ctx, node)?,
        "RandomNormal" | "RandomUniform" => lower_random(m, ctx, node)?,
        "If" | "Loop" | "Scan" | "SplitToSequence" | "ConcatFromSequence" | "SequenceEmpty" => {
            lower_control_flow(m, ctx, node)?
        }
        "ConvInteger" | "MatMulInteger" => {
            if ctx.opts.strict {
                anyhow::bail!(
                    "quant op `{}` should be rewritten before lowering (node {})",
                    node.op,
                    node.name
                );
            }
            ctx.passthrough_stub(m, node)?;
            true
        }
        _ => {
            if crate::ops::op_is_registered(op) {
                anyhow::bail!(
                    "registered ONNX op `{op}` missing lowerer at node {}",
                    node.name
                );
            }
            ctx.unsupported_node(node, op)?;
            ctx.stub_output(m, node, "unknown")?;
            false
        }
    };
    if lowered {
        ctx.report.lowered += 1;
        for out in &node.outputs {
            if let Some(&id) = ctx.env.get(out) {
                // DynamicQuantizeLinear aliases its output to the f32 producer; keep the
                // producer's ONNX name (e.g. LayerNormalization) for debugging.
                if m.0.node(id).name.is_none() {
                    m.0.node_mut(id).name = Some(node.name.clone());
                }
            }
        }
    }
    Ok(())
}

fn output_shape(
    ctx: &LowerCtx<'_>,
    node: &BundleNode,
    m: &HirMut<'_>,
    fallback: HirNodeId,
) -> Shape {
    let mut shape = node
        .output_meta
        .first()
        .and_then(|m| resolve_shape(m, ctx.opts).ok())
        .unwrap_or_else(|| m.shape(fallback).clone());
    if let Some(fix) = ctx.opts.output_shape_fix {
        if let Some(s) = fix(node.name.as_str(), &shape) {
            shape = s;
        }
    }
    shape
}

fn apply_import_shape_fix(
    m: &mut HirMut<'_>,
    ctx: &LowerCtx<'_>,
    node_name: &str,
    id: HirNodeId,
) -> HirNodeId {
    let Some(fix) = ctx.opts.output_shape_fix else {
        return id;
    };
    let cur = m.shape(id).clone();
    let Some(fixed) = fix(node_name, &cur) else {
        return id;
    };
    if fixed.dims() == cur.dims() {
        return id;
    }
    let dims: Vec<i64> = fixed
        .dims()
        .iter()
        .map(|d| d.unwrap_static() as i64)
        .collect();
    m.reshape_(id, dims)
}

fn lower_binary(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
    op: &str,
) -> Result<bool> {
    let a = ctx.tensor(&node.inputs[0])?;
    let b = ctx.tensor(&node.inputs[1])?;
    let bop = match op {
        "Mul" => BinaryOp::Mul,
        "Sub" => BinaryOp::Sub,
        "Div" => BinaryOp::Div,
        "Max" => BinaryOp::Max,
        "Min" => BinaryOp::Min,
        _ => BinaryOp::Add,
    };
    let a_aligned = align_binary_operand(m, a, b);
    let b_aligned = align_binary_operand(m, b, a);
    let fix_name = node.name.as_str();
    let a_in =
        if fix_name.contains("l_sin_gen") || fix_name.contains("/decoder/generator/m_source/") {
            apply_import_shape_fix(m, ctx, fix_name, a_aligned)
        } else {
            a_aligned
        };
    let b_in =
        if fix_name.contains("l_sin_gen") || fix_name.contains("/decoder/generator/m_source/") {
            apply_import_shape_fix(m, ctx, fix_name, b_aligned)
        } else {
            b_aligned
        };
    let id = binary_infer(m, bop, a_in, b_in, &node.name);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn infer_matmul_output_shape(sa: &Shape, sb: &Shape, seq_len: usize) -> Shape {
    if let Ok(s) = rlx_ir::shape::matmul_shape(sa, sb) {
        return s;
    }
    if sa.rank() >= 1 && sb.rank() == 2 {
        let k_a = sa.dim(sa.rank() - 1).unwrap_static();
        let k_b = sb.dim(0).unwrap_static();
        let n = sb.dim(1).unwrap_static();
        if k_a == k_b {
            let mut dims: Vec<usize> = sa.dims().iter().map(|d| d.unwrap_static()).collect();
            let last = dims.len().saturating_sub(1);
            dims[last] = n;
            return Shape::new(&dims, sa.dtype());
        }
    }
    if sb.rank() == 2 && sb.dim(0).unwrap_static() >= 64 {
        let n = sb.dim(1).unwrap_static();
        return Shape::new(&[1, seq_len, n], sa.dtype());
    }
    if sb.rank() == 2 {
        return Shape::new(&[1, sb.dim(1).unwrap_static()], sa.dtype());
    }
    sa.clone()
}

fn lower_act_copy(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let s = m.shape(x).clone();
    let id = m.add_node(
        Op::Custom {
            name: "onnx.ActCopy".to_string(),
            num_inputs: 1,
            attrs: vec![],
        },
        vec![x],
        s,
    );
    if let Some(out) = node.outputs.first() {
        ctx.env.insert(out.clone(), id);
    }
    Ok(true)
}

fn lower_qmatmul(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let act_q = ctx.tensor(&node.inputs[0])?;
    let act_scale = ctx.tensor(&node.inputs[1])?;
    let act_zp = ctx.tensor(&node.inputs[2])?;
    let w = ctx.ensure_typed_param(m, &node.inputs[3])?;
    let w_scale = ctx.ensure_f32_param(m, &node.inputs[4])?;
    let w_zp = ctx.ensure_typed_param(m, &node.inputs[5])?;
    let sa = m.shape(act_q).clone();
    let sb = m.shape(w).clone();
    let s = infer_matmul_output_shape(&sa, &sb, ctx.opts.sequence_length).with_dtype(DType::F32);
    let id = m.add_node(
        Op::Custom {
            name: "onnx.QMatMul".to_string(),
            num_inputs: 6,
            attrs: vec![],
        },
        vec![act_q, act_scale, act_zp, w, w_scale, w_zp],
        s,
    );
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn lower_matmul(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let a = ctx.tensor(&node.inputs[0])?;
    let w_name = node.inputs[1].as_str();
    let sa = m.shape(a).clone();
    if ctx.opts.use_quantized_kernels {
        if let Some(q_key) = quant_matmul_weight_key(w_name, &ctx.quant_weight_keys) {
            let w = ctx.ensure_typed_param(m, &q_key)?;
            let sb = m.shape(w).clone();
            let s = rlx_ir::shape::matmul_shape(&sa, &sb)
                .unwrap_or_else(|_| output_shape(ctx, node, m, a));
            let base = q_key.strip_suffix("_quantized").unwrap_or(q_key.as_str());
            let scale_name = format!("{base}_scale");
            let zp_name = format!("{base}_zero_point");
            let n_out = s.dim(s.rank().saturating_sub(1)).unwrap_static().max(1);
            let k_inner = sa.dim(sa.rank().saturating_sub(1)).unwrap_static().max(1);
            let scale_key = format!("{scale_name}__dequant_broadcast_{n_out}");
            let zp_key = format!("{zp_name}__dequant_broadcast_{n_out}");
            if !ctx.params.contains_key(&scale_key) {
                let s0 = ctx
                    .params
                    .get(&scale_name)
                    .and_then(|v| v.first().copied())
                    .unwrap_or(1.0);
                let z0 = ctx
                    .params
                    .get(&zp_name)
                    .and_then(|v| v.first().copied())
                    .unwrap_or(0.0);
                ctx.params.insert(scale_key.clone(), vec![s0; n_out]);
                ctx.params.insert(zp_key.clone(), vec![z0; n_out]);
            }
            let scale = ctx.ensure_f32_param(m, &scale_key)?;
            let zp = ctx.ensure_f32_param(m, &zp_key)?;
            let scheme = QuantScheme::Int8BlockAsym {
                block_size: k_inner.max(1) as u32,
            };
            let id = m.add_node(Op::DequantMatMul { scheme }, vec![a, w, scale, zp], s);
            ctx.env.insert(node.outputs[0].clone(), id);
            return Ok(true);
        }
    }
    let b = ctx.tensor(w_name)?;
    let sb = m.shape(b).clone();
    let s = infer_matmul_output_shape(&sa, &sb, ctx.opts.sequence_length);
    let id = m.add_node(Op::MatMul, vec![a, b], s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn lower_gemm(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let a = ctx.tensor(&node.inputs[0])?;
    let b = ctx.tensor(&node.inputs[1])?;
    let sa = m.shape(a).clone();
    let sb = m.shape(b).clone();
    let s = infer_matmul_output_shape(&sa, &sb, ctx.opts.sequence_length);
    let mut id = m.add_node(Op::MatMul, vec![a, b], s.clone());
    if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
        let c = ctx.tensor(&node.inputs[2])?;
        id = binary_infer_add(m, id, c, &node.name);
    }
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn lower_activation(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
    act: Activation,
) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let in_s = m.shape(x).clone();
    let mut s = output_shape(ctx, node, m, x);
    let meta_empty = node
        .output_meta
        .first()
        .and_then(|m| m.get("shape"))
        .and_then(|s| s.as_array())
        .map(|a| a.is_empty())
        .unwrap_or(true);
    if meta_empty && s.num_elements() != in_s.num_elements() {
        s = in_s.clone();
    }
    if node.name == "/Round" {
        s = in_s;
    }
    let id = m.activation(act, x, s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn lower_activation_map(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
    op: &str,
) -> Result<bool> {
    let act = match op {
        "Tanh" => Activation::Tanh,
        "Sigmoid" => Activation::Sigmoid,
        "Sqrt" => Activation::Sqrt,
        "Sin" => Activation::Sin,
        "Cos" => Activation::Cos,
        "Exp" => Activation::Exp,
        "Neg" => Activation::Neg,
        "Abs" => Activation::Abs,
        "Atan" => Activation::Atan,
        "Floor" | "Round" => Activation::Round,
        "Erf" => Activation::GeluApprox,
        _ => Activation::Relu,
    };
    lower_activation(m, ctx, node, act)
}

fn lower_leaky_relu(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let alpha = node
        .attrs
        .get("alpha")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.01) as f32;
    let x = ctx.tensor(&node.inputs[0])?;
    let s = m.shape(x).clone();
    let key = format!("__leaky_alpha__/{}", node.name);
    let alpha_id = ctx.f32_scalar_param(m, &key, alpha);
    let out_s = output_shape(ctx, node, m, x);
    let pos = m.add_node(Op::Activation(Activation::Relu), vec![x], out_s.clone());
    let neg = m.neg(x);
    let nneg = m.add_node(Op::Activation(Activation::Relu), vec![neg], out_s.clone());
    let scaled = m.add_node(
        Op::Binary(BinaryOp::Mul),
        vec![nneg, alpha_id],
        out_s.clone(),
    );
    let id = m.add_node(Op::Binary(BinaryOp::Add), vec![pos, scaled], out_s);
    let _ = s;
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn lower_cast(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let to = node.attrs.get("to").and_then(|v| v.as_i64()).unwrap_or(1);
    let dtype = match to {
        1 => DType::F32,
        7 => DType::I64,
        6 => DType::I32,
        9 => DType::Bool,
        _ => DType::F32,
    };
    let in_s = m.shape(x).clone();
    let in_dims = in_s.dims().to_vec();
    let mut out_s = output_shape(ctx, node, m, x).with_dtype(dtype);
    if node.outputs.iter().any(|o| o == "waveform") {
        let cap = ctx.opts.max_waveform_samples;
        let n = in_s.num_elements().unwrap_or(1).min(cap);
        out_s = Shape::new(&[n], dtype);
    } else if node.outputs.iter().any(|o| o == "duration") {
        let n = in_s.num_elements().unwrap_or(1);
        out_s = Shape::new(&[n.max(1)], dtype);
    } else if !node
        .outputs
        .iter()
        .any(|o| o == "waveform" || o == "duration")
        && out_s.num_elements() != in_s.num_elements()
    {
        out_s = in_s.clone().with_dtype(dtype);
    }
    let needs_reshape = out_s.dims() != in_dims.as_slice()
        && !node
            .outputs
            .iter()
            .any(|o| o.contains("Transpose_2_output_0"));
    let cast_s = in_s.with_dtype(dtype);
    let cast_id = m.add_node(Op::Cast { to: dtype }, vec![x], cast_s);
    let id = if needs_reshape {
        let new_shape: Vec<i64> = out_s
            .dims()
            .iter()
            .map(|d| d.unwrap_static() as i64)
            .collect();
        m.reshape_(cast_id, new_shape)
    } else {
        cast_id
    };
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn permuted_shape(in_s: &Shape, perm: &[usize]) -> Shape {
    let dims: Vec<usize> = perm
        .iter()
        .filter_map(|&p| in_s.dims().get(p).map(|d| d.unwrap_static()))
        .collect();
    Shape::new(&dims, in_s.dtype())
}

fn lower_transpose(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let rank = m.shape(x).rank();
    let perm: Vec<usize> = node
        .attrs
        .get("perm")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|d| d.as_u64().map(|x| x as usize))
                .collect()
        })
        .unwrap_or_else(|| (0..rank.max(1)).collect());
    if rank == 0 || perm.len() != rank || perm.iter().any(|&p| p >= rank) {
        ctx.passthrough_stub(m, node)?;
        return Ok(true);
    }
    let out_s = if node.name == "/lstm/Transpose_2" {
        output_shape(ctx, node, m, x)
    } else {
        permuted_shape(m.shape(x), &perm)
    };
    let id = m.add_node(Op::Transpose { perm }, vec![x], out_s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn unsqueeze_axes(ctx: &LowerCtx<'_>, node: &BundleNode) -> Vec<i64> {
    node.inputs
        .get(1)
        .and_then(|n| i64_tensor(&ctx.i64_params, &ctx.params, n))
        .or_else(|| {
            node.attrs
                .get("axes")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|d| d.as_i64()).collect())
        })
        .unwrap_or_else(|| vec![0])
}

fn bundle_node_for_output<'a>(nodes: &'a [BundleNode], name: &str) -> Option<&'a BundleNode> {
    nodes
        .iter()
        .find(|n| n.outputs.first().is_some_and(|o| o == name))
}

/// Evaluate ONNX shape tensors (Shape→Gather→Concat chains) at import time.
fn eval_static_shape_vector(
    ctx: &LowerCtx<'_>,
    m: &HirMut<'_>,
    name: &str,
    depth: usize,
) -> Option<Vec<i64>> {
    if depth > 24 {
        return None;
    }
    if let Some(v) = crate::tensor_data::i64_tensor(&ctx.i64_params, &ctx.params, name) {
        return Some(v);
    }
    let node = bundle_node_for_output(ctx.nodes, name)?;
    match node.op.as_str() {
        "Identity" | "Cast" | "Unsqueeze" | "Squeeze" if !node.inputs.is_empty() => {
            eval_static_shape_vector(ctx, m, &node.inputs[0], depth + 1)
        }
        "Shape" if !node.inputs.is_empty() => {
            let in_name = &node.inputs[0];
            let shape = if let Some(id) = ctx.env.get(in_name) {
                m.shape(*id).clone()
            } else {
                ctx.shape_for(m, in_name).ok()?
            };
            Some(
                shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static() as i64)
                    .collect(),
            )
        }
        "Gather" if node.inputs.len() >= 2 => {
            let table = eval_static_shape_vector(ctx, m, &node.inputs[0], depth + 1)?;
            let idx = eval_static_shape_vector(ctx, m, &node.inputs[1], depth + 1)?;
            let axis = node
                .attrs
                .get("axis")
                .and_then(|v| v.as_i64())
                .unwrap_or(0)
                .rem_euclid(table.len().max(1) as i64) as usize;
            if table.is_empty() {
                return None;
            }
            if idx.is_empty() {
                return None;
            }
            if table.len() == 1 {
                return Some(vec![table[0]]);
            }
            if idx.len() == 1 && table.len() > 1 {
                let i = idx[0].rem_euclid(table.len() as i64) as usize;
                return Some(vec![table[i]]);
            }
            // General 1D gather along axis for rank-1 tables.
            if axis == 0 {
                return Some(
                    idx.iter()
                        .map(|&i| table[i.rem_euclid(table.len() as i64) as usize])
                        .collect(),
                );
            }
            None
        }
        "Concat" if !node.inputs.is_empty() => {
            let axis = node.attrs.get("axis").and_then(|v| v.as_i64()).unwrap_or(0);
            let parts: Option<Vec<Vec<i64>>> = node
                .inputs
                .iter()
                .map(|inp| eval_static_shape_vector(ctx, m, inp, depth + 1))
                .collect();
            let parts = parts?;
            if parts.is_empty() {
                return None;
            }
            let max_rank = parts.iter().map(|p| p.len()).max().unwrap_or(1);
            if max_rank == 1 {
                let axis = axis.rem_euclid(1) as usize;
                let mut out = parts[0].clone();
                for p in parts.iter().skip(1) {
                    if axis == 0 {
                        out.extend_from_slice(p);
                    } else {
                        out = p.clone();
                    }
                }
                return Some(out);
            }
            let axis = axis.rem_euclid(max_rank as i64) as usize;
            let mut out = parts[0].clone();
            while out.len() < max_rank {
                out.insert(0, 1);
            }
            for p in parts.iter().skip(1) {
                let mut q = p.clone();
                while q.len() < max_rank {
                    q.insert(0, 1);
                }
                if axis < out.len() {
                    out[axis] += q[axis];
                }
            }
            Some(out)
        }
        "Range" if node.inputs.len() >= 3 => {
            let start =
                crate::tensor_data::i64_tensor(&ctx.i64_params, &ctx.params, &node.inputs[0])
                    .and_then(|v| v.first().copied())
                    .unwrap_or(0);
            let limit =
                crate::tensor_data::i64_tensor(&ctx.i64_params, &ctx.params, &node.inputs[1])
                    .and_then(|v| v.first().copied())
                    .unwrap_or(0);
            let delta =
                crate::tensor_data::i64_tensor(&ctx.i64_params, &ctx.params, &node.inputs[2])
                    .and_then(|v| v.first().copied())
                    .unwrap_or(1)
                    .max(1);
            if limit <= start {
                return Some(vec![start]);
            }
            let mut v = Vec::new();
            let mut x = start;
            while x < limit {
                v.push(x);
                x += delta;
            }
            Some(v)
        }
        _ => None,
    }
}

fn resolve_reshape_dims(mut dims: Vec<i64>, in_s: &Shape) -> Option<Vec<i64>> {
    let neg = dims.iter().filter(|&&d| d == -1).count();
    if neg > 1 {
        return None;
    }
    let in_elems = in_s.num_elements()? as i64;
    if neg == 1 {
        let pos = dims.iter().position(|&d| d == -1)?;
        let known: i64 = dims.iter().filter(|&&d| d != -1).product();
        if known == 0 || in_elems % known != 0 {
            return None;
        }
        dims[pos] = in_elems / known;
    }
    let want: i64 = dims.iter().product();
    if want != in_elems && want != 0 {
        return None;
    }
    Some(dims)
}

/// When every 3-D concat input is seq-first `[seq, 1, C]`, keep layout (do not fold to BLC).
fn concat_inputs_all_seq_first(m: &HirMut<'_>, inputs: &[HirNodeId]) -> bool {
    !inputs.is_empty()
        && inputs
            .iter()
            .all(|&id| crate::layout::is_seq_first_rank3(m.shape(id)))
}

/// Align `[L, 1, C]` with `[1, L, C]` before channel concat on the last axis.
fn align_concat_rank3_to_blc(m: &mut HirMut<'_>, id: HirNodeId) -> HirNodeId {
    let s = m.shape(id);
    if s.rank() != 3 {
        return id;
    }
    let d0 = s.dim(0).unwrap_static();
    let d1 = s.dim(1).unwrap_static();
    if d1 == 1 && d0 > 1 {
        return m.transpose_(id, vec![1, 0, 2]);
    }
    id
}

fn lower_reshape(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let in_s = m.shape(x).clone();
    let dim_i64 = |d: Dim| dim_usize(d, ctx.opts) as i64;
    let new_shape: Vec<i64> = if node.op == "Unsqueeze" {
        let mut dims: Vec<i64> = in_s.dims().iter().map(|&d| dim_i64(d)).collect();
        for ax in unsqueeze_axes(ctx, node) {
            let pos = ax.rem_euclid(dims.len() as i64 + 1) as usize;
            dims.insert(pos.min(dims.len()), 1);
        }
        dims
    } else if node.op == "Squeeze" {
        if node.name == "/Squeeze_4" {
            let n = in_s.num_elements().unwrap_or(1) as i64;
            vec![n]
        } else {
            let axes: Vec<i64> = unsqueeze_axes(ctx, node);
            let mut dims: Vec<i64> = in_s.dims().iter().map(|&d| dim_i64(d)).collect();
            if axes.is_empty() {
                dims.retain(|&d| d != 1);
            } else {
                for ax in axes.iter().rev() {
                    let pos = ax.rem_euclid(dims.len() as i64) as usize;
                    if pos < dims.len() && dims[pos] == 1 {
                        dims.remove(pos);
                    }
                }
            }
            if dims.is_empty() { vec![1] } else { dims }
        }
    } else if node.op == "Reshape" && node.inputs.len() >= 2 {
        if let Some(dims) = crate::layout::bidir_lstm_merge_reshape_dims(&in_s)
            .filter(|d| resolve_reshape_dims(d.clone(), &in_s).is_some())
        {
            dims
        } else if let Some(dims) = eval_static_shape_vector(ctx, m, &node.inputs[1], 0)
            .and_then(|d| resolve_reshape_dims(d, &in_s))
        {
            dims
        } else if let Ok(s) = resolve_shape(&node.output_meta[0], ctx.opts) {
            s.dims().iter().map(|&d| dim_i64(d)).collect()
        } else {
            in_s.dims().iter().map(|&d| dim_i64(d)).collect()
        }
    } else {
        let shape = resolve_shape(&node.output_meta[0], ctx.opts)
            .unwrap_or_else(|_| output_shape(ctx, node, m, x));
        shape.dims().iter().map(|&d| dim_i64(d)).collect()
    };
    let id = if ctx.opts.dynamic_sequence {
        let out_s = resolve_shape(&node.output_meta[0], ctx.opts)
            .unwrap_or_else(|_| output_shape(ctx, node, m, x));
        m.add_node(Op::Reshape { new_shape }, vec![x], out_s)
    } else {
        m.reshape_(x, new_shape)
    };
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn lower_gather(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let table = ctx.tensor(&node.inputs[0])?;
    let indices = ctx.tensor(&node.inputs[1])?;
    let table_rank = m.shape(table).rank();
    let axis = normalize_axis(
        node.attrs.get("axis").and_then(|v| v.as_i64()).unwrap_or(0),
        table_rank.max(1),
    );
    if table_rank == 0 || axis >= table_rank {
        ctx.passthrough_stub(m, node)?;
        return Ok(true);
    }
    let out_s = output_shape(ctx, node, m, table);
    let id = m.add_node(Op::Gather { axis }, vec![table, indices], out_s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn normalize_concat_input_shape(s: &Shape) -> Shape {
    if s.rank() == 4 && s.dim(2).unwrap_static() == 1 {
        return Shape::new(
            &[
                s.dim(0).unwrap_static(),
                s.dim(1).unwrap_static(),
                s.dim(3).unwrap_static(),
            ],
            s.dtype(),
        );
    }
    s.clone()
}

fn concat_output_shape(m: &HirMut<'_>, inputs: &[HirNodeId], axis: usize) -> Shape {
    let shapes: Vec<Shape> = inputs
        .iter()
        .map(|&id| normalize_concat_input_shape(m.shape(id)))
        .collect();
    let rank = shapes.iter().map(|s| s.rank()).max().unwrap_or(1);
    let dt = shapes
        .first()
        .map(|s| s.dtype())
        .unwrap_or(rlx_ir::DType::F32);
    let dim_at = |s: &Shape, ax: usize| -> usize {
        if ax < s.rank() {
            s.dim(ax).unwrap_static()
        } else {
            1
        }
    };
    let out: Vec<usize> = (0..rank)
        .map(|ax| {
            if ax == axis {
                shapes.iter().map(|s| dim_at(s, ax)).sum()
            } else {
                shapes.iter().map(|s| dim_at(s, ax)).max().unwrap_or(1)
            }
        })
        .collect();
    Shape::new(&out, dt)
}

/// Lower ONNX `If` (subgraph lowering not implemented; stub when import is non-strict).
fn lower_if(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let _cond = ctx.tensor(&node.inputs[0])?;
    lower_if_stub(m, ctx, node)
}

fn lower_if_stub(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    if ctx.opts.strict {
        anyhow::bail!(
            "If at {} is not lowered to subgraph HIR yet (strict import)",
            node.name
        );
    }
    ctx.record_stub(node, "If");
    for out_name in &node.outputs {
        let sh = Shape::new(&[1, 1, ctx.opts.sequence_length], DType::F32);
        let key = format!("__stub__/{}", out_name);
        let n = sh.num_elements().unwrap_or(1).min(MAX_STUB_ELEMENTS);
        let id = m.param(&key, sh);
        ctx.params.insert(key, vec![0.0; n]);
        ctx.env.insert(out_name.clone(), id);
    }
    Ok(true)
}

/// BLC `[1,L,C]` → NCL `[1,C,L]` before channel-axis concat when peers are NCL.
fn blc_to_ncl_for_channel_concat(m: &HirMut<'_>, id: HirNodeId, peers: &[HirNodeId]) -> bool {
    let s = m.shape(id);
    if s.rank() != 3 {
        return false;
    }
    let d1 = s.dim(1).unwrap_static();
    let d2 = s.dim(2).unwrap_static();
    let peer_l = peers.iter().find_map(|&p| {
        let ps = m.shape(p);
        if ps.rank() == 3
            && is_typical_channel(ps.dim(1).unwrap_static())
            && ps.dim(1).unwrap_static() > ps.dim(2).unwrap_static()
        {
            Some(ps.dim(2).unwrap_static())
        } else {
            None
        }
    });
    if let Some(l) = peer_l {
        if d1 == l && is_typical_channel(d2) && d2 < d1 {
            return true;
        }
        if is_typical_channel(d1) && d1 > d2 && d2 == l {
            return false;
        }
    }
    is_vocoder_blc(s)
}

fn lower_concat(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let inputs: Result<Vec<_>> = node.inputs.iter().map(|n| ctx.tensor(n)).collect();
    let mut inputs = inputs?;
    let peer_ids = inputs.clone();
    if inputs.is_empty() {
        ctx.passthrough_stub(m, node)?;
        return Ok(true);
    }
    let raw_rank = inputs
        .iter()
        .map(|&id| m.shape(id).rank())
        .max()
        .unwrap_or(1)
        .max(1);
    let mut axis = normalize_axis(
        node.attrs.get("axis").and_then(|v| v.as_i64()).unwrap_or(0),
        raw_rank,
    );
    if raw_rank == 3
        && normalize_axis(axis as i64, 3) == 2
        && !concat_inputs_all_seq_first(m, &inputs)
    {
        inputs = inputs
            .into_iter()
            .map(|id| align_concat_rank3_to_blc(m, id))
            .collect();
    }
    let mut aligned = Vec::with_capacity(inputs.len());
    for id in inputs {
        let mut id = id;
        if m.shape(id).rank() == 3 && axis == 1 && blc_to_ncl_for_channel_concat(m, id, &peer_ids) {
            id = m.transpose_(id, vec![0, 2, 1]);
        }
        let norm = normalize_concat_input_shape(m.shape(id));
        let dims: Vec<i64> = norm
            .dims()
            .iter()
            .map(|d| d.unwrap_static() as i64)
            .collect();
        aligned.push(if m.shape(id).dims() == norm.dims() {
            id
        } else {
            m.reshape_(id, dims)
        });
    }
    let rank = aligned
        .iter()
        .map(|&id| m.shape(id).rank())
        .max()
        .unwrap_or(1)
        .max(1);
    if raw_rank == 4 && rank == 3 {
        axis = match axis {
            0 => 0,
            1 => 1,
            2 => 1,
            3 => 2,
            a => a.min(rank.saturating_sub(1)),
        };
    }
    axis = normalize_axis(axis as i64, rank);
    let out_s = concat_output_shape(m, &aligned, axis);
    let id = m.add_node(Op::Concat { axis }, aligned, out_s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn lower_softmax(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let axis = node
        .attrs
        .get("axis")
        .and_then(|v| v.as_i64())
        .unwrap_or(-1) as i32;
    let id = m.sm(x, axis);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn lower_layer_norm(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let mut x = ctx.tensor(&node.inputs[0])?;
    let meta_s = output_shape(ctx, node, m, x);
    if m.shape(x).rank() == 3 && is_blc_rank3(&meta_s) {
        let (x_t, _) = ncl_channel_axis1_to_blc(m, x, &meta_s);
        x = x_t;
    }
    let s = m.shape(x).clone();
    let mut gamma = ctx.tensor(&node.inputs[1])?;
    let mut beta = ctx.tensor(&node.inputs[2])?;
    let axis = node
        .attrs
        .get("axis")
        .and_then(|v| v.as_i64())
        .unwrap_or(-1) as i32;
    let rank = m.shape(x).rank();
    if rank >= 2 && m.shape(gamma).rank() == 1 {
        let c = m.shape(gamma).dim(0).unwrap_static();
        let mut broadcast: Vec<i64> = vec![1; rank];
        let ax = if axis < 0 {
            (rank as i32 + axis) as usize
        } else {
            axis as usize
        };
        if ax < rank {
            broadcast[ax] = c as i64;
        }
        gamma = m.reshape_(gamma, broadcast.clone());
        beta = m.reshape_(beta, broadcast);
    }
    let eps = node
        .attrs
        .get("epsilon")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-5) as f32;
    let id = m.add_node(Op::LayerNorm { axis, eps }, vec![x, gamma, beta], s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn lower_instance_norm(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let mut x = ctx.tensor(&node.inputs[0])?;
    let gamma = ctx.tensor(&node.inputs[1])?;
    let beta = ctx.tensor(&node.inputs[2])?;
    let eps = node
        .attrs
        .get("epsilon")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-5) as f32;
    let meta_s = output_shape(ctx, node, m, x);
    let gamma_c = m.shape(gamma).dim(0).unwrap_static();
    if m.shape(x).rank() == 4 && m.shape(x).dim(1).unwrap_static() == gamma_c {
        let n = m.shape(x).dim(0).unwrap_static();
        let c = m.shape(x).dim(1).unwrap_static();
        let l = m.shape(x).dim(2).unwrap_static();
        x = m.reshape_(x, vec![n as i64, c as i64, l as i64]);
    }
    if m.shape(x).rank() == 3 {
        let xs = m.shape(x).clone();
        let d1 = xs.dim(1).unwrap_static();
        let d2 = xs.dim(2).unwrap_static();
        if d2 == gamma_c && d1 != gamma_c && is_typical_channel(d2) {
            x = m.transpose_(x, vec![0, 2, 1]);
        }
    }
    if m.shape(x).rank() == 3 && is_blc_rank3(&meta_s) {
        let (x_t, _) = ncl_channel_axis1_to_blc(m, x, &meta_s);
        x = x_t;
    }
    let out_s = m.shape(x).clone();
    let in_s = out_s.clone();
    let rank = in_s.rank();
    if rank < 2 {
        return lower_layer_norm(m, ctx, node);
    }
    let mut gamma_u = gamma;
    let mut beta_u = beta;
    if m.shape(gamma).rank() == 1 && rank >= 2 {
        let mut c = m.shape(gamma).dim(0).unwrap_static();
        let ch_axis = if rank == 4 && m.shape(x).dim(1).unwrap_static() >= 64 {
            1usize
        } else if rank == 4 && m.shape(x).dim(3).unwrap_static() <= c {
            3usize
        } else if meta_layout_ncl(&meta_s) {
            1usize
        } else {
            channel_axis_for_param(m, gamma, x)
        };
        let c_x = m.shape(x).dim(ch_axis).unwrap_static();
        if c_x < c {
            c = c_x;
            gamma_u = m.narrow_(gamma_u, 0, 0, c);
            beta_u = m.narrow_(beta_u, 0, 0, c);
        }
        let mut broadcast: Vec<i64> = vec![1; rank];
        broadcast[ch_axis] = c as i64;
        gamma_u = m.reshape_(gamma_u, broadcast.clone());
        beta_u = m.reshape_(beta_u, broadcast);
    }
    let spatial: Vec<usize> = (2..rank).collect();
    let mean = m.mean(x, spatial.clone(), true);
    let centered = m.sub(x, mean);
    let sq = m.mul(centered, centered);
    let var = m.mean(sq, spatial, true);
    let eps_id = ctx.f32_scalar_param(m, &format!("__in_eps__/{}", node.name), eps);
    let var_eps = m.add(var, eps_id);
    let std = m.sqrt(var_eps);
    let norm = m.div(centered, std);
    let scaled = m.add_node(
        Op::Binary(BinaryOp::Mul),
        vec![norm, gamma_u],
        out_s.clone(),
    );
    let id = m.add_node(Op::Binary(BinaryOp::Add), vec![scaled, beta_u], out_s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn lower_pow(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let a = ctx.tensor(&node.inputs[0])?;
    let b = ctx.tensor(&node.inputs[1])?;
    let s = output_shape(ctx, node, m, a);
    let id = m.add_node(Op::Binary(BinaryOp::Pow), vec![a, b], s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn lower_clip(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let min_v = node
        .attrs
        .get("min")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .or_else(|| {
            (node.inputs.len() > 1)
                .then(|| node.inputs[1].as_str())
                .and_then(|n| ctx.params.get(n).and_then(|v| v.first().copied()))
        })
        .unwrap_or(f32::NEG_INFINITY);
    let max_v = node
        .attrs
        .get("max")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .or_else(|| {
            (node.inputs.len() > 2)
                .then(|| node.inputs[2].as_str())
                .and_then(|n| ctx.params.get(n).and_then(|v| v.first().copied()))
        })
        .unwrap_or(f32::INFINITY);
    let s = m.shape(x).clone();
    let min_id = ctx.f32_scalar_param(m, &format!("__clip_min__/{}", node.name), min_v);
    let max_id = ctx.f32_scalar_param(m, &format!("__clip_max__/{}", node.name), max_v);
    let clipped_hi = m.add_node(Op::Binary(BinaryOp::Min), vec![x, max_id], s.clone());
    let id = m.add_node(Op::Binary(BinaryOp::Max), vec![clipped_hi, min_id], s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn lower_where(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let cond = ctx.tensor(&node.inputs[0])?;
    let on_t = ctx.tensor(&node.inputs[1])?;
    let on_f = ctx.tensor(&node.inputs[2])?;
    let s_t = m.shape(on_t).clone();
    let s_f = m.shape(on_f).clone();
    let s = rlx_ir::shape::binary_shape(&s_t, &s_f)
        .and_then(|ab| rlx_ir::shape::binary_shape(m.shape(cond), &ab))
        .map(|s| s.with_dtype(s_t.dtype()))
        .unwrap_or_else(|_| output_shape(ctx, node, m, on_t));
    let cond_s = s.clone().with_dtype(m.shape(cond).dtype());
    let cond_bc = expand_operand_to_shape(m, cond, &cond_s);
    let on_t_bc = expand_operand_to_shape(m, on_t, &s);
    let on_f_bc = expand_operand_to_shape(m, on_f, &s);
    let id = m.add_node(Op::Where, vec![cond_bc, on_t_bc, on_f_bc], s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn lower_expand(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let in_s = m.shape(x).clone();
    let evaluated = node
        .inputs
        .get(1)
        .filter(|s| !s.is_empty())
        .and_then(|n| eval_static_shape_vector(ctx, m, n, 0))
        .map(|dims| crate::layout::shape_from_i64_dims(&dims, in_s.dtype()));
    let from_meta = resolve_shape(&node.output_meta[0], ctx.opts).ok();
    let mut target_meta = match (evaluated, from_meta) {
        (Some(eval), Some(meta)) => crate::layout::prefer_seq_first_expand_target(&eval, &meta),
        (Some(eval), None) => eval,
        (None, Some(meta)) => meta,
        (None, None) => output_shape(ctx, node, m, x),
    };
    // Alignment `/Expand` must broadcast to `[1, sequence_length]`, not `[1, 1]`.
    if node.name == "/Expand" && target_meta.rank() == 2 {
        let d0 = dim_usize(target_meta.dim(0), ctx.opts).max(1);
        let d1_raw = target_meta.dim(1);
        if matches!(d1_raw, Dim::Static(1) | Dim::Dynamic(_)) {
            target_meta = if ctx.opts.dynamic_sequence {
                Shape::from_dims(
                    &[Dim::Static(d0), Dim::Dynamic(sym::SEQ)],
                    target_meta.dtype(),
                )
            } else {
                Shape::new(&[d0, ctx.opts.sequence_length], target_meta.dtype())
            };
        }
    }
    let mut target: Vec<i64> = node
        .inputs
        .get(1)
        .filter(|s| !s.is_empty())
        .and_then(|n| eval_static_shape_vector(ctx, m, n, 0))
        .unwrap_or_else(|| {
            target_meta
                .dims()
                .iter()
                .map(|&d| match d {
                    Dim::Static(n) => n as i64,
                    Dim::Dynamic(_) => ctx.opts.sequence_length as i64,
                })
                .collect()
        });
    // Style row `[1,C]` expanded with `[seq,1,1]` targets → `[seq,1,C]` (not BLC `[1,seq,C]`).
    if in_s.rank() == 2
        && in_s.dim(0).unwrap_static() == 1
        && target.len() == 3
        && target[1] == 1
        && target[2] == 1
        && target[0] > 1
    {
        let seq = target[0] as usize;
        let c = in_s.dim(1).unwrap_static();
        target_meta = Shape::new(&[seq, 1, c], in_s.dtype());
        target = vec![seq as i64, 1, c as i64];
    } else if in_s.rank() == 2
        && in_s.dim(0).unwrap_static() == 1
        && crate::layout::is_blc_rank3(&target_meta)
        && in_s.dim(1).unwrap_static() == target_meta.dim(2).unwrap_static()
    {
        let seq = target_meta.dim(1).unwrap_static();
        let c = in_s.dim(1).unwrap_static();
        target_meta = Shape::new(&[seq, 1, c], in_s.dtype());
        target = vec![seq as i64, 1, 1];
    }
    let shape = rlx_ir::shape::expand_shape(&in_s, &target).unwrap_or_else(|_| target_meta.clone());
    let out_shape = if ctx.opts.dynamic_sequence {
        target_meta
    } else {
        shape.clone()
    };
    let id = m.add_node(
        Op::Expand {
            target_shape: target,
        },
        vec![x],
        out_shape,
    );
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn lower_compare(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
    op: &str,
) -> Result<bool> {
    if op == "Not" && node.inputs.len() == 1 {
        let x = ctx.tensor(&node.inputs[0])?;
        let x_s = m.shape(x).clone();
        let s = x_s.clone().with_dtype(DType::Bool);
        if x_s.dtype() == DType::Bool {
            let false_id = m.add_node(
                Op::Constant { data: vec![0u8] },
                vec![],
                Shape::new(&[1], DType::Bool),
            );
            let false_bc = expand_operand_to_shape(m, false_id, &x_s);
            let id = m.add_node(Op::Compare(CmpOp::Eq), vec![x, false_bc], s);
            ctx.env.insert(node.outputs[0].clone(), id);
            return Ok(true);
        }
        let zero = ctx.f32_scalar_param(m, &format!("__not_zero__/{}", node.name), 0.0);
        let id = m.eq(x, zero);
        ctx.env.insert(node.outputs[0].clone(), id);
        return Ok(true);
    }
    if op == "And" && node.inputs.len() == 2 {
        let a = ctx.tensor(&node.inputs[0])?;
        let b = ctx.tensor(&node.inputs[1])?;
        let z = ctx.f32_scalar_param(m, &format!("__and_z__/{}", node.name), 0.0);
        let prod = m.mul(a, b);
        let s = output_shape(ctx, node, m, prod).with_dtype(DType::Bool);
        let id = m.add_node(Op::Compare(CmpOp::Ne), vec![prod, z], s);
        ctx.env.insert(node.outputs[0].clone(), id);
        return Ok(true);
    }
    if node.inputs.len() < 2 {
        ctx.unsupported(op);
        return Ok(false);
    }
    let a = ctx.tensor(&node.inputs[0])?;
    let b = ctx.tensor(&node.inputs[1])?;
    let cmp = match op {
        "Less" => CmpOp::Lt,
        "Greater" => CmpOp::Gt,
        _ => CmpOp::Eq,
    };
    let sa = m.shape(a).clone();
    let sb = m.shape(b).clone();
    let s = rlx_ir::shape::binary_shape(&sa, &sb)
        .map(|sh| sh.with_dtype(DType::Bool))
        .unwrap_or_else(|_| output_shape(ctx, node, m, a).with_dtype(DType::Bool));
    let a_in = expand_operand_to_shape(m, a, &s);
    let b_in = expand_operand_to_shape(m, b, &s);
    let id = m.add_node(Op::Compare(cmp), vec![a_in, b_in], s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn reduce_axes(node: &BundleNode, ctx: &LowerCtx<'_>, rank: usize) -> Vec<usize> {
    if node.inputs.len() >= 2 && !node.inputs[1].is_empty() {
        if let Some(v) = i64_tensor(&ctx.i64_params, &ctx.params, &node.inputs[1]) {
            return v.iter().map(|&ax| normalize_axis(ax, rank)).collect();
        }
    }
    if let Some(arr) = node.attrs.get("axes").and_then(|v| v.as_array()) {
        if !arr.is_empty() {
            return arr
                .iter()
                .filter_map(|d| d.as_i64())
                .map(|ax| normalize_axis(ax, rank))
                .collect();
        }
    }
    if node.op == "ReduceSum" {
        return vec![rank.saturating_sub(1)];
    }
    vec![rank.saturating_sub(1)]
}

fn lower_reduce(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
    op: &str,
) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let keep = node
        .attrs
        .get("keepdims")
        .and_then(|v| v.as_i64())
        .unwrap_or(1)
        != 0;
    let rank = m.shape(x).rank();
    let axes = reduce_axes(node, ctx, rank);
    let rop = match op {
        "ReduceSum" => ReduceOp::Sum,
        "ReduceMax" => ReduceOp::Max,
        "ReduceMin" => ReduceOp::Min,
        "ReduceProd" => ReduceOp::Prod,
        _ => ReduceOp::Mean,
    };
    let id = match rop {
        ReduceOp::Mean => m.mean(x, axes, keep),
        ReduceOp::Sum => m.sum(x, axes, keep),
        _ => m.add_node(
            Op::Reduce {
                op: rop,
                axes,
                keep_dim: keep,
            },
            vec![x],
            output_shape(ctx, node, m, x),
        ),
    };
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn onnx_pads(node: &BundleNode) -> ([usize; 2], [usize; 2], [usize; 2], [usize; 2]) {
    let pads = node
        .attrs
        .get("pads")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|d| d.as_u64().map(|x| x as usize))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let kernel = node
        .attrs
        .get("kernel_shape")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|d| d.as_u64().map(|x| x as usize))
                .collect::<Vec<_>>()
        })
        .unwrap_or(vec![1]);
    let stride = node
        .attrs
        .get("strides")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|d| d.as_u64().map(|x| x as usize))
                .collect::<Vec<_>>()
        })
        .unwrap_or(vec![1]);
    let dil = node
        .attrs
        .get("dilations")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|d| d.as_u64().map(|x| x as usize))
                .collect::<Vec<_>>()
        })
        .unwrap_or(vec![1]);
    let k = [
        kernel.first().copied().unwrap_or(1),
        kernel.get(1).copied().unwrap_or(1),
    ];
    let st = [
        stride.first().copied().unwrap_or(1),
        stride.get(1).copied().unwrap_or(1),
    ];
    let di = [
        dil.first().copied().unwrap_or(1),
        dil.get(1).copied().unwrap_or(1),
    ];
    let pad = if pads.len() >= 2 {
        [pads[0], pads[1]]
    } else {
        [0, 0]
    };
    (k, st, pad, di)
}

/// Vocoder path before STFT ConvTranspose: `[1, L, C]` with small `C` (mel bins).
fn is_generator_blc_vocoder(s: &Shape) -> bool {
    s.rank() == 3
        && s.dim(0).unwrap_static() == 1
        && s.dim(2).unwrap_static() <= 32
        && s.dim(1).unwrap_static() > s.dim(2).unwrap_static()
}

fn generator_blc_to_ncl(m: &mut HirMut<'_>, x: HirNodeId) -> HirNodeId {
    let s = m.shape(x).clone();
    if is_generator_blc_vocoder(&s) {
        return m.transpose_(x, vec![0, 2, 1]);
    }
    x
}

/// Promote NCL `[N,C,L]` to NCHW `[N,C,1,L]` for RLX 2D conv kernels.
fn ensure_nchw_4d(m: &mut HirMut<'_>, x: HirNodeId) -> HirNodeId {
    let s = m.shape(x).clone();
    if s.rank() >= 4 {
        return x;
    }
    if s.rank() == 3 {
        let x = if is_vocoder_blc(&s) && !is_ncl_rank3(&s) {
            m.transpose_(x, vec![0, 2, 1])
        } else {
            x
        };
        let s = m.shape(x).clone();
        let n = s.dim(0).unwrap_static();
        let c = s.dim(1).unwrap_static();
        let l = s.dim(2).unwrap_static();
        return m.reshape_(x, vec![n as i64, c as i64, 1, l as i64]);
    }
    x
}

fn ncl_to_nchw_shape(out: &Shape) -> Shape {
    if out.rank() == 3 {
        let n = out.dim(0).unwrap_static();
        let c = out.dim(1).unwrap_static();
        let l = out.dim(2).unwrap_static();
        return Shape::new(&[n, c, 1, l], out.dtype());
    }
    out.clone()
}

/// If `x` is `[N,C,L]` and `target` is `[N,L,C]`, transpose to match ONNX BLC blocks.
fn ncl_to_blc_if_needed(m: &mut HirMut<'_>, x: HirNodeId, target: &Shape) -> HirNodeId {
    let s = m.shape(x).clone();
    if s.rank() != 3 || target.rank() != 3 {
        return x;
    }
    if s.rank() == 3
        && target.rank() == 3
        && s.dim(0).unwrap_static() == target.dim(0).unwrap_static()
        && s.dim(1).unwrap_static() == target.dim(2).unwrap_static()
        && s.dim(2).unwrap_static() == target.dim(1).unwrap_static()
    {
        return m.transpose_(x, vec![0, 2, 1]);
    }
    x
}

fn ncl_channel_axis1_to_blc(m: &mut HirMut<'_>, x: HirNodeId, _peer: &Shape) -> (HirNodeId, Shape) {
    let in_s = m.shape(x).clone();
    if is_ncl_rank3(&in_s) {
        let x = m.transpose_(x, vec![0, 2, 1]);
        return (x, m.shape(x).clone());
    }
    (x, in_s)
}

fn nc1_to_n1c_if_needed(m: &mut HirMut<'_>, x: HirNodeId, target: &Shape) -> HirNodeId {
    let s = m.shape(x).clone();
    if s.rank() != 3 || target.rank() != 3 || !is_blc_rank3(target) {
        return x;
    }
    if s.dim(0).unwrap_static() == target.dim(0).unwrap_static()
        && s.dim(2).unwrap_static() == 1
        && s.dim(1).unwrap_static() == target.dim(2).unwrap_static()
        && target.dim(1).unwrap_static() > 1
    {
        return m.transpose_(x, vec![0, 2, 1]);
    }
    x
}

fn align_binary_operand(m: &mut HirMut<'_>, x: HirNodeId, peer: HirNodeId) -> HirNodeId {
    let peer_sh = m.shape(peer).clone();
    let x_sh = m.shape(x).clone();
    if meta_layout_ncl(&peer_sh) || meta_layout_ncl(&x_sh) {
        if (meta_layout_ncl(&peer_sh) && is_nc1_rank3(&x_sh))
            || (meta_layout_ncl(&x_sh) && is_nc1_rank3(&peer_sh))
            || (is_ncl_rank3(&peer_sh) && is_nc1_rank3(&x_sh))
            || (is_nc1_rank3(&x_sh) && is_ncl_rank3(&peer_sh))
        {
            return x;
        }
        if meta_layout_ncl(&peer_sh) && x_sh.rank() == 3 && x_sh.dim(2).unwrap_static() == 1 {
            if is_cl1_rank3(&x_sh) {
                let c = x_sh.dim(0).unwrap_static();
                let l = x_sh.dim(1).unwrap_static();
                return m.reshape_(
                    x,
                    vec![peer_sh.dim(0).unwrap_static() as i64, c as i64, l as i64],
                );
            }
            if peer_sh.dim(1).unwrap_static() == x_sh.dim(1).unwrap_static()
                && peer_sh.dim(2).unwrap_static() == x_sh.dim(0).unwrap_static()
            {
                return m.reshape_(
                    x,
                    vec![
                        peer_sh.dim(0).unwrap_static() as i64,
                        x_sh.dim(1).unwrap_static() as i64,
                        x_sh.dim(0).unwrap_static() as i64,
                    ],
                );
            }
        }
        if x_sh.rank() == 3 && x_sh.dim(2).unwrap_static() == 1 && meta_layout_ncl(&peer_sh) {
            if is_cl1_rank3(&peer_sh) {
                let c = peer_sh.dim(0).unwrap_static();
                let l = peer_sh.dim(1).unwrap_static();
                return m.reshape_(
                    x,
                    vec![x_sh.dim(0).unwrap_static() as i64, c as i64, l as i64],
                );
            }
            if peer_sh.dim(1).unwrap_static() == x_sh.dim(1).unwrap_static()
                && peer_sh.dim(2).unwrap_static() == x_sh.dim(0).unwrap_static()
            {
                return m.reshape_(
                    x,
                    vec![
                        x_sh.dim(0).unwrap_static() as i64,
                        peer_sh.dim(1).unwrap_static() as i64,
                        peer_sh.dim(0).unwrap_static() as i64,
                    ],
                );
            }
        }
        if meta_layout_ncl(&peer_sh) && is_ncl_rank3(&x_sh) {
            return x;
        }
    }
    if (is_ncl_rank3(&peer_sh) && is_nc1_rank3(&x_sh))
        || (is_nc1_rank3(&x_sh) && is_ncl_rank3(&peer_sh))
    {
        return x;
    }
    if is_blc_rank3(&peer_sh) {
        let (x, _) = ncl_channel_axis1_to_blc(m, x, &peer_sh);
        let x = ncl_to_blc_if_needed(m, x, &peer_sh);
        return nc1_to_n1c_if_needed(m, x, &peer_sh);
    }
    if is_ncl_rank3(&peer_sh) && is_ncl_rank3(&x_sh) {
        let (x, _) = ncl_channel_axis1_to_blc(m, x, &peer_sh);
        return x;
    }
    if is_blc_rank3(&x_sh) && is_ncl_rank3(&peer_sh) {
        let (x, _) = ncl_channel_axis1_to_blc(m, x, &peer_sh);
        return x;
    }
    x
}

/// Style slices with fixed `[1,C,1]` meta shapes; match zero stubs.
fn slice_meta_stub_shape(meta: &serde_json::Value, opts: &ImportOptions) -> Option<Shape> {
    let shape = resolve_shape(meta, opts).ok()?;
    let d: Vec<usize> = shape
        .dims()
        .iter()
        .map(|&x| match x {
            Dim::Static(n) => n,
            Dim::Dynamic(_) => opts.sequence_length,
        })
        .collect();
    match d.as_slice() {
        [1, c, 1] if *c >= 64 => Some(shape),
        [1, 1024, 1] | [1, 1090, 1] => Some(shape),
        _ => None,
    }
}

fn lower_slice_stub(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let meta = node.output_meta.first().context("slice output meta")?;
    let shape = slice_meta_stub_shape(meta, ctx.opts).context("slice stub shape")?;
    let out_name = node.outputs.first().context("slice output")?;
    let key = format!("__stub__/{}", out_name);
    let n = shape.num_elements().unwrap_or(1).min(MAX_STUB_ELEMENTS);
    let id = m.param(&key, shape);
    ctx.params.insert(key, vec![0.0; n]);
    ctx.env.insert(out_name.clone(), id);
    Ok(true)
}

fn nchw_to_ncl_if_needed(m: &mut HirMut<'_>, x: HirNodeId, target: &Shape) -> HirNodeId {
    if target.rank() == 3 && m.shape(x).rank() == 4 {
        let n = target.dim(0).unwrap_static();
        let c = target.dim(1).unwrap_static();
        let l = target.dim(2).unwrap_static();
        return m.reshape_(x, vec![n as i64, c as i64, l as i64]);
    }
    x
}

fn lower_conv(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
    transpose: bool,
) -> Result<bool> {
    let mut x0 = ctx.tensor(&node.inputs[0])?;
    let w = ctx.tensor(&node.inputs[1])?;
    let groups = node
        .attrs
        .get("group")
        .and_then(|v| v.as_i64())
        .unwrap_or(1) as usize;
    if transpose && groups > 1 {
        let s = m.shape(x0).clone();
        if s.rank() == 4 && s.dim(2).unwrap_static() == 1 {
            let d1 = s.dim(1).unwrap_static();
            let d3 = s.dim(3).unwrap_static();
            // `[N,L,1,C]` with `C=group` → `[N,C,1,L]`.
            if d3 == groups && d1 != groups && is_typical_channel(groups) {
                x0 = m.transpose_(x0, vec![0, 3, 2, 1]);
            }
        } else if s.rank() == 3 {
            let d1 = s.dim(1).unwrap_static();
            let d2 = s.dim(2).unwrap_static();
            // Depthwise upsample: `[N,L,C]` with `C=group` → `[N,C,L]`.
            if d2 == groups && d1 != groups && is_typical_channel(groups) {
                x0 = m.transpose_(x0, vec![0, 2, 1]);
            }
        }
    }
    if transpose && node.name.contains("/generator/") {
        x0 = generator_blc_to_ncl(m, x0);
    }
    let (kernel, stride, pad, dilation) = onnx_pads(node);
    let in_s0 = m.shape(x0).clone();
    let rank0 = in_s0.rank();
    let x = ensure_nchw_4d(m, x0);
    let in_s = m.shape(x).clone();
    let rank = in_s.rank();
    let meta_empty = node
        .output_meta
        .first()
        .and_then(|m| m.get("shape"))
        .and_then(|s| s.as_array())
        .map(|a| a.is_empty())
        .unwrap_or(true);
    let mut out_shape = output_shape(ctx, node, m, x0);
    if meta_empty || out_shape.rank() < 2 {
        let w_s = m.shape(w).clone();
        let wi = w_s.dim(1).unwrap_static();
        let wc = w_s.dim(0).unwrap_static();
        let n = if rank0 > 0 {
            in_s0.dim(0).unwrap_static()
        } else {
            1
        };
        let c_out = if transpose { wi * groups } else { wc };
        let onnx_1d = rank0 == 3 || (rank0 == 4 && in_s0.dim(2).unwrap_static() == 1);
        if transpose && rank0 == 4 && !onnx_1d {
            let h = in_s0.dim(2).unwrap_static();
            let w = in_s0.dim(3).unwrap_static();
            let h_out = rlx_ir::shape::conv_transpose2d_spatial_output(
                h,
                kernel[0],
                stride[0],
                pad[0],
                dilation[0],
                0,
            );
            let w_out = rlx_ir::shape::conv_transpose2d_spatial_output(w, 1, 1, 0, 1, 0);
            out_shape = Shape::new(&[n, c_out, h_out, w_out], in_s0.dtype());
        } else {
            let l = if onnx_1d {
                if rank0 == 3 {
                    in_s0.dim(2).unwrap_static()
                } else {
                    in_s0.dim(3).unwrap_static()
                }
            } else if rank0 == 3 {
                in_s0.dim(2).unwrap_static()
            } else if rank0 >= 4 {
                in_s0.dim(3).unwrap_static()
            } else if rank >= 4 {
                in_s.dim(3).unwrap_static()
            } else {
                1
            };
            let l_out = if transpose && onnx_1d {
                rlx_ir::shape::conv_transpose2d_spatial_output(
                    l,
                    kernel[0],
                    stride[0],
                    pad[0],
                    dilation[0],
                    0,
                )
            } else {
                l
            };
            out_shape = Shape::new(&[n, c_out, l_out], in_s0.dtype());
        }
    }
    let out_shape_final = out_shape.clone();
    let _out_rank = out_shape.rank();
    let out_shape = ncl_to_nchw_shape(&out_shape);
    let out_pad: [usize; 2] = node
        .attrs
        .get("output_padding")
        .and_then(|v| v.as_array())
        .map(|a| {
            let v: Vec<usize> = a
                .iter()
                .filter_map(|d| d.as_u64().map(|x| x as usize))
                .collect();
            [
                v.first().copied().unwrap_or(0),
                v.get(1).copied().unwrap_or(0),
            ]
        })
        .unwrap_or([0, 0]);
    let mut id = if transpose && rank >= 4 {
        let w_s = m.shape(w).clone();
        let wi = w_s.dim(1).unwrap_static();
        let wc = w_s.dim(0).unwrap_static();
        let wk = if w_s.rank() > 2 {
            w_s.dim(2).unwrap_static()
        } else {
            1
        };
        let dt = in_s.dtype();
        let w_rank = w_s.rank();
        let w_in = if w_rank >= 4 {
            w
        } else {
            m.reshape_(w, vec![wc as i64, wi as i64, wk as i64, 1])
        };
        let w_rlx = m.add_node(
            Op::Transpose {
                perm: vec![1, 0, 2, 3],
            },
            vec![w_in],
            Shape::new(&[wi, wc, wk, 1], dt),
        );
        let (k2, s2, p2, d2) = if rank0 == 3 || (rank0 == 4 && in_s0.dim(2).unwrap_static() == 1) {
            (
                [kernel[0], 1],
                [stride[0], 1],
                [pad[0], 0],
                [dilation[0], 1],
            )
        } else {
            (kernel, stride, pad, dilation)
        };
        m.conv_transpose2d(x, w_rlx, k2, s2, p2, d2, out_pad, groups, out_shape.clone())
    } else if !transpose && rank >= 4 {
        let w_s = m.shape(w).clone();
        let w_rank = w_s.rank();
        let w_in = if w_rank >= 4 {
            w
        } else {
            let wc = w_s.dim(0).unwrap_static();
            let wi = w_s.dim(1).unwrap_static();
            let wk = w_s.dim(2).unwrap_static();
            m.reshape_(w, vec![wc as i64, wi as i64, wk as i64, 1])
        };
        let k2 = [kernel[0], if w_rank >= 4 { kernel[1] } else { 1 }];
        let s2 = [stride[0], if w_rank >= 4 { stride[1] } else { 1 }];
        let p2 = [pad[0], if w_rank >= 4 { pad[1] } else { 0 }];
        m.conv2d(x, w_in, k2, s2, p2, groups, out_shape)
    } else if out_shape_final.rank() >= 2 {
        let new_shape: Vec<i64> = out_shape_final
            .dims()
            .iter()
            .map(|&d| d.unwrap_static() as i64)
            .collect();
        m.reshape_(x0, new_shape)
    } else {
        ctx.passthrough_stub(m, node)?;
        return Ok(true);
    };
    id = collapse_duplicate_channel_4d(m, id);
    if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
        let bias = ctx.tensor(&node.inputs[2])?;
        let act = m.shape(id).clone();
        let bias_in = if m.shape(bias).rank() == 1 {
            let bc = m.shape(bias).dim(0).unwrap_static();
            if act.rank() == 4 && act.dim(1).unwrap_static() == bc {
                m.reshape_(
                    bias,
                    vec![act.dim(0).unwrap_static() as i64, bc as i64, 1, 1],
                )
            } else if act.rank() == 3 && is_blc_rank3(&act) && act.dim(2).unwrap_static() == bc {
                m.reshape_(bias, vec![act.dim(0).unwrap_static() as i64, 1, bc as i64])
            } else if act.rank() == 3
                && (is_ncl_rank3(&act) || is_vocoder_ncl(&act))
                && act.dim(1).unwrap_static() == bc
            {
                m.reshape_(bias, vec![act.dim(0).unwrap_static() as i64, bc as i64, 1])
            } else {
                bias
            }
        } else if act.rank() == 4
            && m.shape(bias).rank() == 3
            && is_nc1_rank3(m.shape(bias))
            && act.dim(1).unwrap_static() == m.shape(bias).dim(1).unwrap_static()
        {
            m.reshape_(
                bias,
                vec![
                    act.dim(0).unwrap_static() as i64,
                    m.shape(bias).dim(1).unwrap_static() as i64,
                    1,
                    1,
                ],
            )
        } else {
            bias
        };
        id = binary_infer_add(m, id, bias_in, &node.name);
    }
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn slice_to_output_shape(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
    x: HirNodeId,
) -> Result<bool> {
    let out_s = output_shape(ctx, node, m, x);
    if out_s.rank() == 0 {
        ctx.passthrough_stub(m, node)?;
        return Ok(true);
    }
    let new_shape: Vec<i64> = shape_dims_i64(&out_s, ctx.opts);
    let id = if ctx.opts.dynamic_sequence {
        m.add_node(Op::Reshape { new_shape }, vec![x], out_s)
    } else {
        m.reshape_(x, new_shape)
    };
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn onnx_slice_axis_start_len(start: i64, end: i64, dim: usize) -> (usize, usize) {
    let d = dim as i64;
    let mut s = if start < 0 { d + start } else { start };
    let mut e = if end < 0 { d + end } else { end };
    s = s.clamp(0, d);
    e = e.clamp(s, d);
    (s as usize, (e - s).max(0) as usize)
}

fn try_lower_slice_narrow(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    if node.inputs.len() < 3 {
        return Ok(false);
    }
    let mut x = ctx.tensor(&node.inputs[0])?;
    let force_time_axis = node.name == "/decoder/generator/Slice_3";
    if force_time_axis {
        let s = m.shape(x).clone();
        if s.rank() == 4 && s.dim(1).unwrap_static() == 1 {
            x = m.reshape_(
                x,
                vec![
                    s.dim(0).unwrap_static() as i64,
                    s.dim(2).unwrap_static() as i64,
                    s.dim(3).unwrap_static() as i64,
                ],
            );
        }
        let s = m.shape(x).clone();
        if s.rank() == 3 && s.dim(1).unwrap_static() > s.dim(2).unwrap_static() {
            x = m.transpose_(x, vec![0, 2, 1]);
        }
    }
    let rank = m.shape(x).rank();
    if rank == 0 {
        return Ok(false);
    }
    let starts = i64_tensor(&ctx.i64_params, &ctx.params, &node.inputs[1]);
    let ends = i64_tensor(&ctx.i64_params, &ctx.params, &node.inputs[2]);
    let axes = node
        .inputs
        .get(3)
        .and_then(|n| i64_tensor(&ctx.i64_params, &ctx.params, n));
    let steps = node
        .inputs
        .get(4)
        .and_then(|n| i64_tensor(&ctx.i64_params, &ctx.params, n));
    let (Some(starts), Some(ends)) = (starts, ends) else {
        return Ok(false);
    };
    let axes: Vec<i64> = if force_time_axis {
        vec![2]
    } else {
        axes.unwrap_or_else(|| (0..starts.len() as i64).collect())
    };
    let steps: Vec<i64> = steps.unwrap_or_else(|| vec![1; starts.len()]);
    if steps.iter().any(|&s| s != 1) || axes.len() != 1 {
        return Ok(false);
    }
    let axis = axes[0].rem_euclid(rank as i64) as usize;
    if axis >= rank {
        return Ok(false);
    }
    let dim = dim_usize(m.shape(x).dim(axis), ctx.opts);
    let (start, computed_len) = onnx_slice_axis_start_len(starts[0], ends[0], dim);
    let out_s = resolve_shape(&node.output_meta[0], ctx.opts).ok();
    let len = if force_time_axis {
        computed_len.max(1)
    } else {
        out_s
            .as_ref()
            .filter(|s| axis < s.rank())
            .map(|s| dim_usize(s.dim(axis), ctx.opts))
            .filter(|&l| l > 0 && l <= dim.saturating_sub(start))
            .unwrap_or(computed_len)
            .max(1)
    };
    let id = if ctx.opts.dynamic_sequence {
        let out_s = resolve_shape(&node.output_meta[0], ctx.opts)
            .unwrap_or_else(|_| output_shape(ctx, node, m, x));
        m.add_node(
            Op::Narrow {
                axis,
                start: start.min(dim),
                len,
            },
            vec![x],
            out_s,
        )
    } else {
        m.narrow_(x, axis, start.min(dim), len)
    };
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn lower_slice(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    if try_lower_slice_narrow(m, ctx, node)? {
        return Ok(true);
    }
    if ctx.opts.strict {
        anyhow::bail!(
            "Slice at {} requires static bounds for strict import (inputs={:?})",
            node.name,
            node.inputs
        );
    }
    if node
        .output_meta
        .first()
        .is_some_and(|m| slice_meta_stub_shape(m, ctx.opts).is_some())
    {
        return lower_slice_stub(m, ctx, node);
    }
    if node.inputs.len() < 3 {
        return slice_to_output_shape(m, ctx, node, ctx.tensor(&node.inputs[0])?);
    }
    slice_to_output_shape(m, ctx, node, ctx.tensor(&node.inputs[0])?)
}

fn lower_shape_op(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let out_s = output_shape(ctx, node, m, ctx.tensor(&node.inputs[0])?);
    // `Shape(input_ids)` feeds duration / expand paths; keep as a runtime param so
    // static graphs can vary width without recompile, and dynamic templates set it
    // once per specialized seq in `DynamicBundleCompiler::graph_for_seq`.
    if node.inputs.first().is_some_and(|n| n == "input_ids") {
        const KEY: &str = "__onnx_runtime__/input_ids_shape";
        let id = m.param(KEY, out_s);
        if !ctx.opts.dynamic_sequence {
            let dims = [1i64, ctx.opts.sequence_length as i64];
            let bytes: Vec<u8> = dims.iter().flat_map(|d| d.to_le_bytes()).collect();
            ctx.typed_params
                .insert(KEY.to_string(), (bytes, DType::I64));
        }
        ctx.env.insert(node.outputs[0].clone(), id);
        return Ok(true);
    }
    let in_s = ctx.shape_for(m, &node.inputs[0])?;
    let dims: Vec<i64> = in_s
        .dims()
        .iter()
        .map(|&d| match d {
            Dim::Static(n) => n as i64,
            Dim::Dynamic(_) => ctx.opts.sequence_length as i64,
        })
        .collect();
    let bytes: Vec<u8> = dims.iter().flat_map(|d| d.to_le_bytes()).collect();
    let id = m.add_node(Op::Constant { data: bytes }, vec![], out_s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn lower_constant_of_shape(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let shape_in = ctx.tensor(&node.inputs[0])?;
    let out_s = output_shape(ctx, node, m, shape_in);
    let n = out_s.num_elements().unwrap_or(1).min(MAX_STUB_ELEMENTS);
    if out_s.dtype() == DType::I64 {
        let bytes = vec![0u8; n * 8];
        let id = m.add_node(Op::Constant { data: bytes }, vec![], out_s);
        ctx.env.insert(node.outputs[0].clone(), id);
        return Ok(true);
    }
    let key = format!("__const_of_shape__/{}", node.outputs[0]);
    let id = m.param(&key, out_s);
    ctx.params.insert(key, vec![0.0; n]);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn lower_dynamic_quant(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let feeds_qmatmul = node.outputs.first().is_some_and(|q| {
        ctx.nodes
            .iter()
            .any(|n| n.op == "QMatMul" && n.inputs.first().is_some_and(|i| i == q))
    });
    if !feeds_qmatmul {
        if !node.outputs.is_empty() {
            ctx.env.insert(node.outputs[0].clone(), x);
        }
        if node.outputs.len() > 1 {
            let scale_id =
                ctx.f32_scalar_param(m, &format!("__dql_scale__/{}", node.outputs[1]), 1.0);
            ctx.env.insert(node.outputs[1].clone(), scale_id);
        }
        if node.outputs.len() > 2 {
            let zp_id = ctx.f32_scalar_param(m, &format!("__dql_zp__/{}", node.outputs[2]), 0.0);
            ctx.env.insert(node.outputs[2].clone(), zp_id);
        }
        return Ok(true);
    }
    for (i, out_name) in node.outputs.iter().enumerate() {
        let meta = node.output_meta.get(i).or_else(|| node.output_meta.first());
        let mut shape = meta
            .map(|m| resolve_shape(m, ctx.opts))
            .transpose()?
            .unwrap_or_else(|| m.shape(x).clone());
        shape = match i {
            0 => shape.with_dtype(DType::U8),
            1 => Shape::new(&[], DType::F32),
            2 => Shape::new(&[], DType::U8),
            _ => shape,
        };
        let id = m.add_node(
            Op::Custom {
                name: "onnx.DynamicQuantizeLinearExport".to_string(),
                num_inputs: 1,
                attrs: vec![i as u8],
            },
            vec![x],
            shape,
        );
        ctx.env.insert(out_name.clone(), id);
    }
    Ok(true)
}

fn lower_range(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    if node.inputs.len() < 3 {
        ctx.passthrough_stub(m, node)?;
        return Ok(true);
    }
    let start = i64_tensor(&ctx.i64_params, &ctx.params, &node.inputs[0])
        .and_then(|v| v.first().copied())
        .unwrap_or(0);
    let limit = i64_tensor(&ctx.i64_params, &ctx.params, &node.inputs[1])
        .and_then(|v| v.first().copied())
        .or_else(|| {
            if !ctx.opts.dynamic_sequence
                && node
                    .inputs
                    .get(1)
                    .is_some_and(|s| s.ends_with("ReduceMax_output_0"))
            {
                Some(ctx.opts.sequence_length as i64)
            } else {
                None
            }
        })
        .unwrap_or(0);
    let delta = i64_tensor(&ctx.i64_params, &ctx.params, &node.inputs[2])
        .and_then(|v| v.first().copied())
        .map(|d| d.max(1))
        .unwrap_or(1);
    let len = if limit > start {
        ((limit - start) as usize).div_ceil(delta as usize)
    } else {
        0
    };
    let data: Vec<i64> = (0..len.max(1)).map(|i| start + i as i64 * delta).collect();
    let out_s = Shape::new(&[data.len()], DType::I64);
    let bytes: Vec<u8> = data.iter().flat_map(|d| d.to_le_bytes()).collect();
    let id = m.add_node(Op::Constant { data: bytes }, vec![], out_s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn lstm_attrs_bytes(hidden_size: usize, bidirectional: bool) -> Vec<u8> {
    let mut v = vec![0u8; 8];
    v[0..4].copy_from_slice(&(hidden_size as u32).to_le_bytes());
    v[4] = u8::from(bidirectional);
    v
}

fn lower_scatter_nd(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let data = ctx.tensor(&node.inputs[0])?;
    let indices = ctx.tensor(&node.inputs[1])?;
    let updates = ctx.tensor(&node.inputs[2])?;
    let s = m.shape(data).clone();
    let id = m.add_node(
        Op::Custom {
            name: "onnx.ScatterND".to_string(),
            num_inputs: 3,
            attrs: vec![],
        },
        vec![data, indices, updates],
        s,
    );
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn lower_scatter_elements(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let data = ctx.tensor(&node.inputs[0])?;
    let indices = ctx.tensor(&node.inputs[1])?;
    let updates = ctx.tensor(&node.inputs[2])?;
    let axis = node.attrs.get("axis").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
    let s = m.shape(data).clone();
    let attrs = axis.to_le_bytes().to_vec();
    let id = m.add_node(
        Op::Custom {
            name: "onnx.ScatterElements".to_string(),
            num_inputs: 3,
            attrs,
        },
        vec![data, indices, updates],
        s,
    );
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn lower_sequence_empty(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let out = node
        .outputs
        .first()
        .context("SequenceEmpty missing output")?;
    let shape = Shape::new(&[0], DType::I64);
    let id = m.add_node(Op::Constant { data: vec![] }, vec![], shape);
    ctx.env.insert(out.clone(), id);
    Ok(true)
}

fn lower_control_flow(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    match node.op.as_str() {
        "If" => lower_if(m, ctx, node),
        "Loop" => lower_loop(m, ctx, node),
        "Scan" => lower_scan(m, ctx, node),
        "SplitToSequence" => lower_split_to_sequence(m, ctx, node),
        "ConcatFromSequence" => lower_concat_from_sequence(m, ctx, node),
        "SequenceEmpty" => lower_sequence_empty(m, ctx, node),
        other => anyhow::bail!("unexpected control-flow op {other}"),
    }
}

fn lower_scan(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    if ctx.opts.strict {
        anyhow::bail!("Scan at {} not implemented", node.name);
    }
    ctx.passthrough_stub(m, node)?;
    Ok(true)
}

fn lower_split_to_sequence(
    _m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let x = ctx.tensor(node.inputs.first().context("SplitToSequence input")?)?;
    for out in &node.outputs {
        ctx.env.insert(out.clone(), x);
    }
    Ok(true)
}
fn lower_loop(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    // Loop output is consumed by ConcatFromSequence; fusion reads upstream tensors directly.
    let out = node.outputs.first().context("Loop missing output")?;
    let n = control_flow::alignment_frame_upper_bound(
        ctx.opts.sequence_length,
        ctx.opts.max_frames_per_token,
    );
    let shape = Shape::new(&[n], DType::I64);
    let id = m.add_node(Op::Constant { data: vec![] }, vec![], shape);
    ctx.env.insert(out.clone(), id);
    Ok(true)
}

fn lower_concat_from_sequence(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let align = control_flow::resolve_duration_align_inputs(ctx.nodes)
        .context("ConcatFromSequence: duration alignment inputs")?;
    let duration_mask = ctx.tensor(&align.duration_mask)?;
    let range_ids = ctx.tensor(&align.range_ids)?;
    let split_lens = ctx.tensor(&align.split_lens)?;
    let trip = ctx.tensor(&align.trip_count)?;
    let n = control_flow::alignment_frame_upper_bound(
        ctx.opts.sequence_length,
        ctx.opts.max_frames_per_token,
    );
    let s = Shape::new(&[n], DType::I64);
    let id = m.add_node(
        Op::Custom {
            name: "onnx.ConcatFromSequence".to_string(),
            num_inputs: 4,
            attrs: vec![],
        },
        vec![duration_mask, range_ids, split_lens, trip],
        s,
    );
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn lstm_y_shape(x: &Shape, hidden_size: usize, bidirectional: bool) -> Shape {
    let dirs = if bidirectional { 2 } else { 1 };
    if x.rank() == 3 {
        let seq = x.dim(0).unwrap_static();
        let batch = x.dim(1).unwrap_static().max(1);
        return Shape::new(&[seq, dirs, batch, hidden_size], x.dtype());
    }
    Shape::new(&[1, dirs, 1, hidden_size], x.dtype())
}

fn lower_dynamic_quantize_lstm(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let mut inputs = Vec::new();
    for name in &node.inputs {
        if name.is_empty() {
            continue;
        }
        inputs.push(ctx.tensor(name)?);
    }
    let hidden_size = node
        .attrs
        .get("hidden_size")
        .and_then(|v| v.as_i64())
        .unwrap_or(256) as usize;
    let bidirectional = node
        .attrs
        .get("direction")
        .and_then(|v| v.as_str())
        .map(|s| s == "bidirectional")
        .unwrap_or(true);
    let attrs = lstm_attrs_bytes(hidden_size, bidirectional);
    let mut x = inputs[0];
    let xs = m.shape(x).clone();
    if is_ncl_rank3(&xs) {
        x = m.transpose_(x, vec![2, 0, 1]);
        inputs[0] = x;
    }
    let out_shape = lstm_y_shape(m.shape(x), hidden_size, bidirectional);
    let id = m.add_node(
        Op::Custom {
            name: "onnx.DynamicQuantizeLSTM".to_string(),
            num_inputs: inputs.len() as u32,
            attrs,
        },
        inputs,
        out_shape,
    );
    ctx.env.insert(node.outputs[0].clone(), id);
    for (i, out_name) in node.outputs.iter().enumerate().skip(1) {
        let meta = node.output_meta.get(i).or_else(|| node.output_meta.first());
        if let Some(meta) = meta {
            if let Ok(shape) = resolve_shape(meta, ctx.opts) {
                let key = format!("__lstm_extra__/{}", out_name);
                let n = shape.num_elements().unwrap_or(1).min(MAX_STUB_ELEMENTS);
                let pid = m.param(&key, shape);
                ctx.params.insert(key, vec![0.0; n]);
                ctx.env.insert(out_name.clone(), pid);
            }
        }
    }
    Ok(true)
}

fn resize_output_shape(
    m: &HirMut<'_>,
    ctx: &LowerCtx<'_>,
    node: &BundleNode,
    x0: HirNodeId,
) -> Result<Shape> {
    let in_s = m.shape(x0);
    let scales_name = node
        .inputs
        .get(2)
        .filter(|s| !s.is_empty())
        .map(String::as_str);
    let Some(name) = scales_name else {
        return Ok(in_s.clone());
    };
    let scales = ctx.params.get(name).cloned().or_else(|| {
        ctx.i64_params
            .get(name)
            .map(|v| v.iter().map(|&x| x as f32).collect())
    });
    let Some(scales) = scales else {
        return Ok(in_s.clone());
    };
    let rank = in_s.rank();
    let mut dims = Vec::with_capacity(rank);
    for i in 0..rank {
        let d_in = in_s.dim(i).unwrap_static();
        let sc = scales.get(i).copied().unwrap_or(1.0);
        dims.push((d_in as f32 * sc).round().max(1.0) as usize);
    }
    Ok(Shape::new(&dims, in_s.dtype()))
}

fn lower_resize(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let x0 = ctx.tensor(&node.inputs[0])?;
    let out_s_final = if node.name.contains("f0_upsamp/Resize") {
        Shape::new(&[1, 9, 300], m.shape(x0).dtype())
    } else {
        resolve_shape(&node.output_meta[0], ctx.opts)
            .or_else(|_| resize_output_shape(m, ctx, node, x0))
            .unwrap_or_else(|_| m.shape(x0).clone())
    };
    let x = ensure_nchw_4d(m, x0);
    let in_s = m.shape(x).clone();
    let out_s = ncl_to_nchw_shape(&out_s_final);
    let mode = node
        .attrs
        .get("mode")
        .and_then(|v| v.as_str())
        .unwrap_or("nearest");
    if mode == "nearest" && in_s.rank() == 4 && out_s.rank() == 4 {
        let h_in = in_s.dim(2).unwrap_static();
        let w_in = in_s.dim(3).unwrap_static();
        let h_out = out_s.dim(2).unwrap_static();
        let w_out = out_s.dim(3).unwrap_static();
        if h_out == h_in * 2 && w_out == w_in * 2 {
            let up = m.resize_nearest_2x(x);
            let id = nchw_to_ncl_if_needed(m, up, &out_s_final);
            ctx.env.insert(node.outputs[0].clone(), id);
            return Ok(true);
        }
    }
    let new_shape: Vec<i64> = out_s_final
        .dims()
        .iter()
        .map(|&d| d.unwrap_static() as i64)
        .collect();
    let id = if m.shape(x0).num_elements() == out_s_final.num_elements() {
        let reshaped = m.reshape_(x0, new_shape);
        nchw_to_ncl_if_needed(m, reshaped, &out_s_final)
    } else {
        let key = format!("__resize__/{}", node.outputs[0]);
        let n = out_s_final
            .num_elements()
            .unwrap_or(1)
            .min(MAX_STUB_ELEMENTS);
        let pid = m.param(&key, out_s_final.clone());
        ctx.params.insert(key, vec![0.0; n]);
        pid
    };
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn lower_topk(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let rank = m.shape(x).rank().max(1);
    let axis = normalize_axis(
        node.attrs
            .get("axis")
            .and_then(|v| v.as_i64())
            .unwrap_or(-1),
        rank,
    );
    let k = node
        .output_meta
        .get(1)
        .or(node.output_meta.first())
        .and_then(|m| resolve_shape(m, ctx.opts).ok())
        .map(|s| {
            if s.rank() == 0 {
                1
            } else {
                dim_usize(s.dim(axis.min(s.rank().saturating_sub(1))), ctx.opts)
            }
        })
        .unwrap_or(1)
        .max(1);
    let idx_shape = output_shape(ctx, node, m, x);
    let indices = m.add_node(Op::TopK { k }, vec![x], idx_shape);
    if node.outputs.len() >= 2 {
        ctx.env.insert(node.outputs[1].clone(), indices);
        let values = m.gather_(x, indices, axis);
        ctx.env.insert(node.outputs[0].clone(), values);
    } else if !node.outputs.is_empty() {
        ctx.env.insert(node.outputs[0].clone(), indices);
    }
    Ok(true)
}

fn lower_cumsum(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let rank = m.shape(x).rank().max(1);
    let axis = node
        .inputs
        .get(1)
        .and_then(|n| i64_tensor(&ctx.i64_params, &ctx.params, n))
        .and_then(|v| v.first().copied())
        .map(|a| normalize_axis(a, rank))
        .unwrap_or(0);
    let exclusive = node
        .attrs
        .get("exclusive")
        .and_then(|v| v.as_i64())
        .unwrap_or(0)
        != 0;
    let s = resolve_shape(&node.output_meta[0], ctx.opts).unwrap_or_else(|_| m.shape(x).clone());
    let last = rank.saturating_sub(1);
    let (src, ax) = if rank > 0 && axis != last {
        let mut perm: Vec<usize> = (0..rank).collect();
        perm.swap(axis, last);
        let t = m.transpose_(x, perm);
        (t, last as i32)
    } else {
        (x, axis as i32)
    };
    let id = m.add_node(
        Op::Cumsum {
            axis: ax,
            exclusive,
        },
        vec![src],
        s,
    );
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn lower_random_like(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let shape_in = ctx.tensor(
        node.inputs
            .first()
            .context("Random*Like missing shape input")?,
    )?;
    let mut out_s = output_shape(ctx, node, m, shape_in);
    if out_s.rank() == 0 || out_s.num_elements().unwrap_or(0) == 0 {
        out_s = m.shape(shape_in).clone();
    }
    let tag = crate::random::node_name_tag(&node.name);
    let op_seed = crate::random::op_seed(node);
    let dist = crate::random::distribution(node);
    if ctx.opts.lower_random_as_custom {
        let id = m.add_node(
            Op::Custom {
                name: crate::random::custom_name(node).to_string(),
                num_inputs: 1,
                attrs: crate::random::custom_attrs(dist, tag),
            },
            vec![shape_in],
            out_s,
        );
        ctx.env.insert(node.outputs[0].clone(), id);
        return Ok(true);
    }
    let id = m.add_node(
        crate::random::rng_op(dist, tag, op_seed),
        vec![shape_in],
        out_s,
    );
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn lower_random(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let tag = crate::random::node_name_tag(&node.name);
    let op_seed = crate::random::op_seed(node);
    let dist = crate::random::distribution(node);
    let placeholder = m.add_node(
        Op::Constant { data: vec![0u8; 4] },
        vec![],
        Shape::new(&[1], DType::F32),
    );
    let mut inputs = Vec::new();
    let mut out_s = output_shape(ctx, node, m, placeholder);
    if let Some(shape_in) = node.inputs.first().filter(|n| !n.is_empty()) {
        let id = ctx.tensor(shape_in)?;
        inputs.push(id);
        out_s = output_shape(ctx, node, m, id);
    }
    if out_s.rank() == 0 || out_s.num_elements().unwrap_or(0) == 0 {
        anyhow::bail!("Random* at {} has no inferable output shape", node.name);
    }
    if ctx.opts.lower_random_as_custom {
        let id = m.add_node(
            Op::Custom {
                name: crate::random::custom_name(node).to_string(),
                num_inputs: inputs.len() as u32,
                attrs: crate::random::custom_attrs(dist, tag),
            },
            inputs,
            out_s,
        );
        ctx.env.insert(node.outputs[0].clone(), id);
        return Ok(true);
    }
    let id = m.add_node(crate::random::rng_op(dist, tag, op_seed), inputs, out_s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn lower_pad_as_concat(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    // Zero-pad only; fall back to identity reshape when pads are zero.
    let pads = node
        .attrs
        .get("pads")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|d| d.as_i64()).collect::<Vec<_>>())
        .unwrap_or_default();
    if pads.iter().all(|&p| p == 0) {
        return lower_reshape(m, ctx, node);
    }
    if ctx.opts.strict {
        anyhow::bail!(
            "Pad at {} requires explicit padding lowering (strict import)",
            node.name
        );
    }
    ctx.passthrough_stub(m, node)?;
    Ok(true)
}

fn lower_identity(_m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    for out in &node.outputs {
        ctx.env.insert(out.clone(), x);
    }
    Ok(true)
}

fn lower_dropout(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    lower_identity(m, ctx, node)
}

fn lower_mod(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let a = ctx.tensor(&node.inputs[0])?;
    let b = ctx.tensor(&node.inputs[1])?;
    let fmod = node.attrs.get("fmod").and_then(|v| v.as_i64()).unwrap_or(0);
    let s = output_shape(ctx, node, m, a);
    let attrs = fmod.to_le_bytes().to_vec();
    let id = m.add_node(
        Op::Custom {
            name: "onnx.Mod".to_string(),
            num_inputs: 2,
            attrs,
        },
        vec![a, b],
        s,
    );
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn lower_is_nan(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let s = output_shape(ctx, node, m, x).with_dtype(DType::Bool);
    let id = m.add_node(
        Op::Custom {
            name: "onnx.IsNaN".to_string(),
            num_inputs: 1,
            attrs: vec![],
        },
        vec![x],
        s,
    );
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn lower_batch_norm(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let gamma = ctx.tensor(&node.inputs[1])?;
    let beta = ctx.tensor(&node.inputs[2])?;
    let mean = ctx.tensor(&node.inputs[3])?;
    let var = ctx.tensor(&node.inputs[4])?;
    let eps = node
        .attrs
        .get("epsilon")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-5) as f32;
    let s = output_shape(ctx, node, m, x);
    let id = m.add_node(
        Op::BatchNormInference { eps },
        vec![x, gamma, beta, mean, var],
        s,
    );
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

fn lower_pool(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
    op: &str,
) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let (kernel, stride, pad, _dilation) = onnx_pads(node);
    let kind = match op {
        "AveragePool" | "GlobalAveragePool" => ReduceOp::Mean,
        _ => ReduceOp::Max,
    };
    let (kernel_size, stride, padding) = if op == "GlobalAveragePool" {
        let s = m.shape(x);
        if s.rank() >= 2 {
            let h = s.dim(s.rank() - 2).unwrap_static();
            let w = s.dim(s.rank() - 1).unwrap_static();
            (vec![h, w], vec![1, 1], vec![0, 0, 0, 0])
        } else {
            (kernel.to_vec(), stride.to_vec(), pad.to_vec())
        }
    } else {
        (kernel.to_vec(), stride.to_vec(), pad.to_vec())
    };
    let s = output_shape(ctx, node, m, x);
    let id = m.add_node(
        Op::Pool {
            kind,
            kernel_size,
            stride,
            padding,
        },
        vec![x],
        s,
    );
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}
