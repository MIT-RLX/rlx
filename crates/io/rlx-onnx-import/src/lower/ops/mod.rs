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
use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Dim, HirGraphExt, HirModule, Op, Shape};

use crate::bundle::RlxBundle;
use crate::bundle::{BundleManifest, BundleNode, topo_sort_nodes};
use crate::control_flow::{self, DURATION_CARRY};
use crate::rewrite::rewrite_graph;
use crate::tensor_data::TypedParams;
use crate::tensor_data::i64_tensor;

use super::options::{ImportOptions, ImportReport};

mod activation;
mod attention;
mod binary;
mod cast_quant;
mod control;
mod conv_pool;
mod gather_scatter;
mod generators;
mod matmul;
mod norm;
mod reduce;
mod rnn;
mod shape_ops;

use activation::*;
use attention::*;
use binary::*;
use cast_quant::*;
use control::*;
use conv_pool::*;
use gather_scatter::*;
use generators::*;
use matmul::*;
use norm::*;
use reduce::*;
use rnn::*;
use shape_ops::*;

const MAX_STUB_ELEMENTS: usize = 8 * 1024 * 1024;

fn is_typical_channel(c: usize) -> bool {
    // Keep an *upper* bound: sequence lengths (LuxTTS/F5 T≫C) must not look like
    // channels, or `[1, T, C]` is misread as NCL with C=T (breaks long CFM).
    matches!(
        c,
        16 | 22
            | 24
            | 32
            | 48
            | 64
            | 80
            | 96
            | 100
            | 105
            | 125
            | 128
            | 160
            | 192
            | 256
            | 272
            | 320
            | 384
            | 512
            | 640
            | 768
            | 1024
            | 1152
            | 1280
            | 1536
            | 2048
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

/// Kitten generator ups + noise: batch-1 NCL/NCHW path broadcast to `[B,C,L]`.
fn try_vocoder_ncl_batch_binary(
    m: &mut HirMut<'_>,
    op: BinaryOp,
    a_in: HirNodeId,
    b_in: HirNodeId,
    sa: &Shape,
    sb: &Shape,
    site: &str,
) -> Option<HirNodeId> {
    if site != "/decoder/generator/Add_3" {
        return None;
    }
    // `[1,C,La]` (NCL) + `[B,Lb,C]` (BLC batch) — transpose BLC, trim; batch broadcast at LIR.
    for (ncl, blc, sn, sbl) in [(a_in, b_in, sa, sb), (b_in, a_in, sb, sa)] {
        if sn.rank() != 3 || sbl.rank() != 3 {
            continue;
        }
        if sn.dim(0).unwrap_static() != 1 || sbl.dim(0).unwrap_static() <= 1 {
            continue;
        }
        let c = sn.dim(1).unwrap_static();
        if !is_typical_channel(c) || c != sbl.dim(2).unwrap_static() || !is_blc_rank3(sbl) {
            continue;
        }
        let batch = sbl.dim(0).unwrap_static();
        let blc_ncl = m.transpose_(blc, vec![0, 2, 1]);
        let la = sn.dim(2).unwrap_static();
        let lb = m.shape(blc_ncl).dim(2).unwrap_static();
        let l = la.min(lb);
        let ncl_use = if la > l { m.narrow_(ncl, 2, 0, l) } else { ncl };
        let blc_use = if lb > l {
            m.narrow_(blc_ncl, 2, 0, l)
        } else {
            blc_ncl
        };
        let target = Shape::new(&[batch, c, l], sn.dtype());
        return Some(m.add_node(Op::Binary(op), vec![ncl_use, blc_use], target));
    }
    None
}

fn binary_infer_add(m: &mut HirMut<'_>, a: HirNodeId, b: HirNodeId, site: &str) -> HirNodeId {
    binary_infer(m, BinaryOp::Add, a, b, site)
}

/// All dims of `s` are statically known. Layout heuristics that call
/// `unwrap_static` must return false for symbolic dims under `dynamic_sequence`.
fn dims_all_static(s: &Shape) -> bool {
    s.dims().iter().all(|d| matches!(d, Dim::Static(_)))
}

/// ONNX `com.microsoft` fused activations `BiasGelu(x, bias) = Gelu(x + bias)`
/// and the tanh-approx `FastGelu` — decomposed to a broadcast bias-add + the
/// native Gelu activation. (F5-TTS's ConvNeXt-v2 pointwise blocks emit BiasGelu;
/// without this its output orphans and the whole GRN chain fails to import.)
fn lower_bias_gelu(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
    approx: bool,
) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let inp = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
        let bias = ctx.tensor(&node.inputs[1])?;
        binary_infer_add(m, x, bias, &node.name)
    } else {
        x
    };
    let act = if approx {
        Activation::GeluApprox
    } else {
        Activation::Gelu
    };
    let s = m.shape(inp).clone();
    let id = m.add_node(Op::Activation(act), vec![inp], s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
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
        && !is_typical_channel(s.dim(2).unwrap_static())
        && !is_blc_rank3(s)
}

/// `[N,L,C]` with channel on the last axis (ONNX BLC blocks). Requires `L > 1`:
/// a `[N,1,C]`/`[1,1,L]` operand is a pure broadcast (mask / per-channel bias),
/// layout-agnostic, and must not be mistaken for a transposed data block — that
/// otherwise transposes the *peer* data tensor when masking long sequences.
fn is_blc_rank3(s: &Shape) -> bool {
    s.rank() == 3
        && s.dim(1).unwrap_static() > 1
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
        && !is_typical_channel(s.dim(2).unwrap_static())
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
        // A genuine duplicate-length block is `[1,L,L]` with L>1. A `[1,1,1]`
        // (or `[1,d,d]` unit) is a scalar/broadcast operand — reshaping its single
        // element to the peer's `[1,C,L]` reads far past the buffer → garbage/inf.
        // (Seen as the supertonic CFM Euler step `dt=1/total_step` `[1,1,1]` × the
        // `[1,144,100]` guidance field exploding to amax ~5.9e3 with inf.)
        || sa.dim(1).unwrap_static() <= 1
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
    None
}

fn binary_infer(
    m: &mut HirMut<'_>,
    op: BinaryOp,
    a: HirNodeId,
    b: HirNodeId,
    site: &str,
) -> HirNodeId {
    // Clean same-rank NumPy broadcast, checked BEFORE any layout heuristic. When
    // the two operands already broadcast to one of their own shapes there is no
    // NCL/BLC ambiguity to fix, so `collapse_duplicate_channel_4d` (which would
    // reshape a RoPE cos `[1,seq,1,head_dim]` to `[1,head_dim,seq]`, mistaking it
    // for a mis-promoted BLC and breaking `Q[1,seq,H,hd] × cos` → head broadcast)
    // must not run. The collapse only exists to REPAIR shapes that DON'T broadcast.
    {
        let ra = m.shape(a).clone();
        let rb = m.shape(b).clone();
        if ra.rank() == rb.rank() && ra.rank() >= 2 {
            if let Ok(out) = rlx_ir::shape::binary_shape(&ra, &rb) {
                if out.dims() == ra.dims() || out.dims() == rb.dims() {
                    return m.add_node(Op::Binary(op), vec![a, b], out);
                }
            }
        }
    }
    // A scalar operand (numel 1) broadcasts UNAMBIGUOUSLY against any shape, so it
    // must skip every NCL/BLC/channel heuristic below — those exist to repair
    // conv-bias `[C]` and VITS-mask layout ambiguity, which a scalar never has.
    {
        let ra = m.shape(a).clone();
        let rb = m.shape(b).clone();
        let na = ra.num_elements().unwrap_or(0);
        let nb = rb.num_elements().unwrap_or(0);
        if (na == 1 || nb == 1) && dims_all_static(&ra) && dims_all_static(&rb) {
            if let Ok(out) = rlx_ir::shape::binary_shape(&ra, &rb) {
                return m.add_node(Op::Binary(op), vec![a, b], out);
            }
        }
    }
    // Dynamic-length operands (a `dynamic_sequence` import — e.g. the ChatterBox
    // S3Gen decoder whose U-Net lengths change via lookahead/up/downsampling)
    // carry symbolic dims that the static NCL/BLC layout heuristics below cannot
    // inspect (`unwrap_static` panics). Symbolic broadcast is unambiguous, so
    // route straight to the generic path; `Dim::Dynamic` operands unify at
    // runtime specialization.
    {
        let ra = m.shape(a).clone();
        let rb = m.shape(b).clone();
        if !dims_all_static(&ra) || !dims_all_static(&rb) {
            return match rlx_ir::shape::binary_shape(&ra, &rb) {
                Ok(sh) => m.add_node(Op::Binary(op), vec![a, b], sh),
                Err(_) => {
                    // Symbolic dims that don't statically broadcast: keep the
                    // higher-rank operand's shape (the extra axes broadcast; the
                    // symbolic lengths resolve identically at runtime).
                    let sh = if ra.rank() >= rb.rank() { ra } else { rb };
                    m.add_node(Op::Binary(op), vec![a, b], sh)
                }
            };
        }
    }
    let mut a_in = collapse_duplicate_channel_4d(m, a);
    let mut b_in = collapse_duplicate_channel_4d(m, b);
    let sa0 = m.shape(a_in).clone();
    let sb0 = m.shape(b_in).clone();
    a_in = repair_duplicate_length_rank3(m, a_in, &sa0, &sb0);
    b_in = repair_duplicate_length_rank3(m, b_in, &sb0, &sa0);
    let sa0 = m.shape(a_in).clone();
    let sb0 = m.shape(b_in).clone();
    // A rank-1 operand is channel-aligned (conv bias `[C]`→`[N,C,H,W]`). Skip that
    // ONLY for the RoPE outer-product signature — a rank-≥3 peer whose LAST dim is
    // 1 and whose SECOND-TO-LAST dim matches the param size: `inv_freq[32] ×
    // positions[..,32,1]` must NumPy-broadcast to `[..,32,32]` (channel-aligning
    // puts `32` on dim -2 and collapses the freq axis → `[..,32,1]` → wrong
    // Cos/Sin). Everything else keeps the (proven) channel heuristic.
    let is_rope_outer = |param: &Shape, peer: &Shape| -> bool {
        let r = peer.rank();
        r >= 3
            && peer.dim(r - 1).unwrap_static() == 1
            && peer.dim(r - 2).unwrap_static() == param.dim(0).unwrap_static()
    };
    if sa0.rank() == 1 && sb0.rank() >= 2 && !is_rope_outer(&sa0, &sb0) {
        a_in = broadcast_param_channels(m, a_in, b_in);
    } else if sb0.rank() == 1 && sa0.rank() >= 2 && !is_rope_outer(&sb0, &sa0) {
        b_in = broadcast_param_channels(m, b_in, a_in);
    }
    let sa = m.shape(a_in).clone();
    let sb = m.shape(b_in).clone();
    // Fast path: when one operand broadcasts cleanly into the other (the NumPy
    // result equals the larger operand's shape), there is no layout ambiguity, so
    // emit the binary directly. Avoids the channel-vs-length heuristics misfiring
    // when a *length* dim is ≥ 64 — e.g. masking `[1,C,L]*[1,1,L]` for long
    // sequences, where `[1,C,L]` is otherwise misread as channel-last.
    if sa.rank() == sb.rank() && sa.rank() >= 2 {
        if let Ok(out) = rlx_ir::shape::binary_shape(&sa, &sb) {
            if out.dims() == sa.dims() || out.dims() == sb.dims() {
                return m.add_node(Op::Binary(op), vec![a_in, b_in], out);
            }
        }
    }
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
    // BLC tensor × length-last mask: `[1, L, C] * [1, 1, L]` → reshape the mask to
    // `[1, L, 1]` so it broadcasts over channels. VITS `enc_q`/`enc_p` apply
    // `x * x_mask` where the importer holds `x` seq-first (BLC) while the mask
    // stays `[b, 1, t]` (length last). Only fires when the mask's last dim equals
    // the tensor's length dim and the channel dims genuinely differ (so the plain
    // NumPy broadcast — already handled by the fast path — does not apply).
    if sa.rank() == 3
        && sb.rank() == 3
        && sa.dim(0).unwrap_static() == sb.dim(0).unwrap_static()
        && sb.dim(1).unwrap_static() == 1
        && sb.dim(2).unwrap_static() == sa.dim(1).unwrap_static()
        && sa.dim(2).unwrap_static() != sb.dim(2).unwrap_static()
    {
        let b_fix = m.reshape_(
            b_in,
            vec![
                sa.dim(0).unwrap_static() as i64,
                sa.dim(1).unwrap_static() as i64,
                1,
            ],
        );
        return m.add_node(Op::Binary(op), vec![a_in, b_fix], sa.clone());
    }
    if sa.rank() == 3
        && sb.rank() == 3
        && sa.dim(0).unwrap_static() == sb.dim(0).unwrap_static()
        && sa.dim(1).unwrap_static() == 1
        && sa.dim(2).unwrap_static() == sb.dim(1).unwrap_static()
        && sb.dim(2).unwrap_static() != sa.dim(2).unwrap_static()
    {
        let a_fix = m.reshape_(
            a_in,
            vec![
                sb.dim(0).unwrap_static() as i64,
                sb.dim(1).unwrap_static() as i64,
                1,
            ],
        );
        return m.add_node(Op::Binary(op), vec![a_fix, b_in], sb.clone());
    }
    if let Some(id) = try_vocoder_ncl_batch_binary(m, op, a_in, b_in, &sa, &sb, site) {
        return id;
    }
    match rlx_ir::shape::binary_shape(&sa, &sb) {
        Ok(sh) => m.add_node(Op::Binary(op), vec![a_in, b_in], sh),
        Err(e) => {
            if std::env::var("RLX_DBG_BINF").is_ok() {
                eprintln!(
                    "[binf] {site}: unaligned {:?} vs {:?}: {e}",
                    sa.dims(),
                    sb.dims()
                );
                // best-effort: use the higher-rank operand's shape so import continues
                let sh = if sa.rank() >= sb.rank() {
                    sa.clone()
                } else {
                    sb.clone()
                };
                return m.add_node(Op::Binary(op), vec![a_in, b_in], sh);
            }
            panic!(
                "binary_infer at {site}: unaligned {:?} vs {:?}: {e}",
                sa.dims(),
                sb.dims()
            )
        }
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
        // Explicit per-name length override (`named_lengths`) ALWAYS wins — even
        // for `unk__*`/`Cast*`/`sequence_length` names — so a caller that binds a
        // graph-split boundary dim (e.g. Kokoro decoder `unk__368`=total_frames,
        // `unk__357`=batch) is honored before any heuristic (or the `unk__` bail).
        serde_json::Value::String(s) if opts.named_lengths.contains_key(s.as_str()) => {
            Ok(Dim::Static(opts.named_lengths[s.as_str()]))
        }
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
            // (Explicit `named_lengths` overrides are handled by the outer guard
            // above — they win over every heuristic here, including the `unk__` bail.)
            // Named dynamic dims models export (`batch_size`, `text_length`,
            // `latent_length`, `source_audio_len`, `max_duration`, …). Batch-like
            // → 1; length/time-like → the compile sequence length. Mirrors the
            // same logic in `shape_propagate::resolve_dims` so graph INPUTS
            // (resolved via this path) agree with propagated intermediate shapes.
            other => {
                let low = other.to_ascii_lowercase();
                let is_batch = low == "n" || low == "b" || low.contains("batch");
                let is_len = low.contains("length")
                    || low.contains("_len")
                    || low.ends_with("len")
                    || low.contains("seq")
                    || low.contains("duration")
                    || low.contains("frame")
                    || low.contains("audio");
                let _ = is_len;
                if is_batch {
                    Ok(Dim::Static(1))
                } else if opts.dynamic_sequence {
                    // Length-like or otherwise-unknown named dim → the sequence
                    // axis (matches the old `"?"` → seq_len behavior; only batch
                    // dims are special-cased to 1).
                    Ok(Dim::Dynamic(sym::SEQ))
                } else {
                    Ok(Dim::Static(opts.sequence_length))
                }
            }
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

/// ONNX `Trilu` (keep the upper/lower triangle of the last two dims, zero the
/// rest) — the causal-mask generator in transformer decoders (`Trilu(ones, k=1,
/// upper=1) * -inf`). rlx-ir has no triangular op, but the shape is static, so
/// bake the `[rows, cols]` keep-mask (1 in the retained triangle) as a Constant
/// and lower `Trilu(x)` to `x * mask` (mask broadcasts over x's leading dims).
/// Works on every backend. `k` = diagonal offset (input 1, default 0); `upper`
/// attribute defaults to 1. Without this, Trilu fell through to a stub that
/// produced a garbage shape → downstream Transpose OOB crash (Parler decoder).
fn lower_trilu(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let x_shape = m.shape(x).clone();
    let rank = x_shape.rank();
    if rank < 2 {
        return Ok(false);
    }
    let rows = dim_usize(x_shape.dim(rank - 2), ctx.opts);
    let cols = dim_usize(x_shape.dim(rank - 1), ctx.opts);
    if rows == 0 || cols == 0 {
        return Ok(false);
    }
    let k = node
        .inputs
        .get(1)
        .and_then(|n| eval_static_shape_vector(ctx, m, n, 0))
        .and_then(|v| v.first().copied())
        .unwrap_or(0);
    let upper = node
        .attrs
        .get("upper")
        .and_then(|v| v.as_i64())
        .unwrap_or(1)
        != 0;
    let dt = x_shape.dtype();
    let mut keep = vec![0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let retained = if upper {
                (j as i64) >= (i as i64) + k
            } else {
                (j as i64) <= (i as i64) + k
            };
            if retained {
                keep[i * cols + j] = 1.0;
            }
        }
    }
    let bytes: Vec<u8> = match dt {
        DType::F32 => keep.iter().flat_map(|&v| v.to_le_bytes()).collect(),
        DType::I64 => keep
            .iter()
            .flat_map(|&v| (v as i64).to_le_bytes())
            .collect(),
        DType::Bool | DType::U8 => keep.iter().map(|&v| v as u8).collect(),
        _ => return Ok(false),
    };
    let mask_id = m.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(&[rows, cols], dt),
    );
    let out = m.add_node(Op::Binary(BinaryOp::Mul), vec![x, mask_id], x_shape);
    ctx.env.insert(node.outputs[0].clone(), out);
    Ok(true)
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
            // An empty shape array is a rank-0 scalar (ONNX encodes scalars as a
            // present `shape` with zero dims — e.g. `speed`, `t`, `guidance_scale`,
            // `time_step` inputs of luxtts/f5). rlx has no rank-0 tensor, so
            // represent it as `[1]`; broadcasting treats it as a scalar. Genuine
            // unknown rank is encoded as `["?"]`, handled below — NOT an error.
            vec![Dim::Static(1)]
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
    /// Tensors that were rank-0 scalars in ONNX (padded to `[1]` by `resolve_shape`,
    /// since rlx has no rank-0). Tracked so `Gather` with a scalar index removes the
    /// gathered axis (ONNX semantics) instead of leaving a spurious size-1 dim —
    /// e.g. F5-TTS `Gather(time_embed, time_step_scalar, axis=1)`.
    scalars: HashSet<String>,
    /// `If`-node name → (then, else) branch subgraph nodes, so `lower_if` can inline
    /// a branch whose output is COMPUTED (not a folded constant). See `onnx_file`.
    if_branches: HashMap<String, (Vec<BundleNode>, Vec<BundleNode>)>,
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
        scalars: HashSet::new(),
        if_branches: crate::onnx_file::take_if_branches(),
        report: ImportReport::default(),
    };

    // Folded CONSTANT tensors that were rank-0 scalars — a scalar Gather index must
    // drop its axis, same as a scalar input (see `onnx_file::take_scalar_consts`).
    ctx.scalars.extend(crate::onnx_file::take_scalar_consts());

    for io in &manifest.inputs {
        // A present-but-empty ONNX shape = rank-0 scalar (resolve_shape pads to [1]).
        if io.meta.shape.is_empty() {
            ctx.scalars.insert(io.name.clone());
        }
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

    eval_caches_reset();
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
            if std::env::var("RLX_IMP_DBG").is_ok() {
                // A tensor is "missing" but is it produced by ANY pending node?
                // If not, its producer was lowered yet inserted nothing (the true
                // root); if yes, walk toward that pending producer. Dump the whole
                // stuck frontier so the deepest root is visible, not just the first.
                let pending_outs: std::collections::HashSet<&str> = pending
                    .iter()
                    .flat_map(|n| n.inputs.iter())
                    .map(String::as_str)
                    .collect();
                let _ = pending_outs;
                eprintln!("[imp-dbg] stuck frontier: {} pending nodes", pending.len());
                for n in pending.iter().take(24) {
                    let miss: Vec<&str> = n
                        .inputs
                        .iter()
                        .map(String::as_str)
                        .filter(|i| {
                            !i.is_empty() && !init_names.contains(*i) && !ctx.env.contains_key(*i)
                        })
                        .collect();
                    // Is each missing input produced by some other pending node?
                    let orphan: Vec<&str> = miss
                        .iter()
                        .copied()
                        .filter(|mi| !pending.iter().any(|p| p.outputs.iter().any(|o| o == mi)))
                        .collect();
                    eprintln!(
                        "[imp-dbg]   {} [{}] missing={:?} ORPHAN(no producer)={:?}",
                        n.name, n.op, miss, orphan
                    );
                }
            }
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
    // Debug tap: append named intermediate tensors as extra graph outputs, so a
    // compiled + run graph exposes them on ANY backend — used to bisect a native
    // result against a reference runtime (onnxruntime) tensor-by-tensor. Set
    // `RLX_ONNX_TAP=name1,name2` to the ONNX tensor names of interest. Tapped
    // tensors are emitted after the real outputs, in the order listed; names that
    // were not lowered (folded/stubbed) are skipped with a warning.
    if let Ok(tap) = std::env::var("RLX_ONNX_TAP") {
        for name in tap.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            if let Some(&id) = ctx.env.get(name) {
                eprintln!("[onnx-tap] + output {name} shape={:?}", m.shape(id).dims());
                outs.push(id);
            } else {
                eprintln!("[onnx-tap] warn: tensor '{name}' not found (not lowered)");
            }
        }
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
            let mut shape_dims = ctx
                .init_shapes
                .get(name.as_str())
                .cloned()
                .unwrap_or_else(|| vec![i64_data.len()]);
            // ONNX rank-0 scalars arrive as `dims=[]`; promote to `[1]` so
            // broadcast/BiasAdd heuristics see a real trailing dim of size 1.
            if shape_dims.is_empty() {
                shape_dims.push(1);
            }
            let bytes: Vec<u8> = i64_data.iter().flat_map(|d| d.to_le_bytes()).collect();
            let shape = Shape::new(&shape_dims, DType::I64);
            let id = m.add_node(Op::Constant { data: bytes }, vec![], shape);
            ctx.env.insert(name.clone(), id);
            continue;
        }
        // Folded f32 Constants (incl. inside `If` subgraphs) live in `ctx.params`
        // even when the caller passes a narrow `inits` set — materialize them.
        if ctx.params.contains_key(name.as_str()) || ctx.typed_params.contains_key(name.as_str()) {
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

    // Remember where this ONNX node's HIR nodes start, so we can stamp all of
    // them (not just the output) with the source op's name below.
    let first_new = m.0.len();
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
        "Tanh" | "Sigmoid" | "Sqrt" | "Sin" | "Cos" | "Exp" | "Log" | "Neg" | "Abs" | "Atan"
        | "Round" => lower_activation_map(m, ctx, node, op)?,
        "Sign" => lower_sign(m, ctx, node)?,
        "Erf" => lower_erf(m, ctx, node)?,
        // Standalone ONNX `Gelu` (opset 20) + Microsoft fused `BiasGelu`/`FastGelu`.
        // `lower_bias_gelu` handles the no-bias (single-input) case too.
        "Gelu" => {
            let approx = node.attrs.get("approximate").and_then(|v| v.as_str()) == Some("tanh");
            lower_bias_gelu(m, ctx, node, approx)?
        }
        "BiasGelu" => lower_bias_gelu(m, ctx, node, false)?,
        "FastGelu" => lower_bias_gelu(m, ctx, node, true)?,
        "Ceil" => lower_round_dir(m, ctx, node, true)?,
        "Floor" => lower_round_dir(m, ctx, node, false)?,
        "LeakyRelu" => lower_leaky_relu(m, ctx, node)?,
        "Elu" => lower_elu(m, ctx, node)?,
        "PRelu" => lower_prelu(m, ctx, node)?,
        "Reciprocal" => lower_reciprocal(m, ctx, node)?,
        "Softplus" => lower_softplus(m, ctx, node)?,
        "Tile" => lower_tile(m, ctx, node)?,
        "Cast" => lower_cast(m, ctx, node)?,
        "Transpose" => lower_transpose(m, ctx, node)?,
        "Reshape" | "Unsqueeze" | "Squeeze" | "Flatten" => lower_reshape(m, ctx, node)?,
        "Gather" => lower_gather(m, ctx, node)?,
        "Concat" => lower_concat(m, ctx, node)?,
        "Softmax" => lower_softmax(m, ctx, node)?,
        "LayerNormalization" => lower_layer_norm(m, ctx, node)?,
        "SimplifiedLayerNormalization" => lower_simplified_layer_norm(m, ctx, node)?,
        "SkipSimplifiedLayerNormalization" => lower_skip_simplified_layer_norm(m, ctx, node)?,
        "GroupQueryAttention" => lower_group_query_attention(m, ctx, node)?,
        "InstanceNormalization" => lower_instance_norm(m, ctx, node)?,
        "BatchNormalization" => lower_batch_norm(m, ctx, node)?,
        "AveragePool" | "MaxPool" | "GlobalAveragePool" => lower_pool(m, ctx, node, op)?,
        "Dropout" => lower_dropout(m, ctx, node)?,
        "Pow" => lower_pow(m, ctx, node)?,
        "Clip" => lower_clip(m, ctx, node)?,
        "Where" => lower_where(m, ctx, node)?,
        "Expand" => lower_expand(m, ctx, node)?,
        "Equal" | "Less" | "Greater" | "LessOrEqual" | "GreaterOrEqual" | "Not" | "And" | "Or"
        | "Xor" => lower_compare(m, ctx, node, op)?,
        "ReduceMean" | "ReduceSum" | "ReduceMax" | "ReduceMin" | "ReduceProd" | "ReduceL2"
        | "ReduceL1" | "ReduceSumSquare" | "ReduceLogSum" | "ReduceLogSumExp" => {
            lower_reduce(m, ctx, node, op)?
        }
        "Conv" => lower_conv(m, ctx, node, false)?,
        "ConvTranspose" => lower_conv(m, ctx, node, true)?,
        "Slice" => lower_slice(m, ctx, node)?,
        "Shape" => lower_shape_op(m, ctx, node)?,
        "ConstantOfShape" => lower_constant_of_shape(m, ctx, node)?,
        "Trilu" => lower_trilu(m, ctx, node)?,
        "Pad" => lower_pad_as_concat(m, ctx, node)?,
        "Range" => lower_range(m, ctx, node)?,
        "STFT" => lower_stft(m, ctx, node)?,
        "DFT" => lower_dft(m, ctx, node)?,
        "DynamicQuantizeLinear" => lower_dynamic_quant(m, ctx, node)?,
        "Resize" => lower_resize(m, ctx, node)?,
        "TopK" => lower_topk(m, ctx, node)?,
        "ArgMax" => lower_arg_reduce(m, ctx, node, true)?,
        "ArgMin" => lower_arg_reduce(m, ctx, node, false)?,
        "CumSum" => lower_cumsum(m, ctx, node)?,
        "ScatterND" => lower_scatter_nd(m, ctx, node)?,
        "ScatterElements" => lower_scatter_elements(m, ctx, node)?,
        "GatherND" => lower_gather_nd(m, ctx, node)?,
        "GatherElements" => lower_gather_elements(m, ctx, node)?,
        "OneHot" => lower_one_hot(m, ctx, node)?,
        "NonZero" => lower_non_zero(m, ctx, node)?,
        "CumProd" => lower_cumprod(m, ctx, node)?,
        "Einsum" => lower_einsum(m, ctx, node)?,
        "LSTM" => lower_lstm(m, ctx, node)?,
        "GRU" => lower_gru(m, ctx, node)?,
        "Split" => lower_split(m, ctx, node)?,
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
        // Source-op provenance label: prefer the ONNX node name, falling back
        // to its first output tensor name — `node.name` is frequently empty in
        // real graphs, whereas output tensor names (`/…/Softmax_output_0`) are
        // present and descriptive.
        let label = if !node.name.is_empty() {
            node.name.clone()
        } else {
            node.outputs
                .iter()
                .find(|o| !o.is_empty())
                .cloned()
                .unwrap_or_default()
        };
        // Stamp every HIR node this lowering produced so intermediates
        // (Softmax → sub/exp/reduce/div, LayerNorm, attention, …) carry the
        // source op's identity instead of a generic "mir" — a NaN in any of
        // them then localizes back to this ONNX node.
        m.0.label_nodes_since(first_new, &label);
        for out in &node.outputs {
            if let Some(&id) = ctx.env.get(out) {
                // DynamicQuantizeLinear aliases its output to the f32 producer; keep the
                // producer's own (already-stamped) name for debugging.
                if m.0.node(id).name.is_none() && !label.is_empty() {
                    m.0.node_mut(id).name = Some(label.clone());
                }
            }
        }
    }
    if std::env::var("RLX_DBG_SHAPES").is_ok() {
        for out in &node.outputs {
            if let Some(&id) = ctx.env.get(out) {
                eprintln!("[shape] {} = {:?} ({})", out, m.shape(id).dims(), node.op);
            }
        }
    }
    Ok(())
}

/// Output shape for a strictly shape-PRESERVING (unary elementwise) op — Erf,
/// Softplus, Reciprocal, LeakyRelu, Ceil/Floor, PRelu, Random*Like, … Their
/// output shape IS the input shape, so the input is authoritative when fully
/// static; the ONNX `output_meta` can carry a symbolic dim that
/// shape-propagation defaulted to a wrong concrete length (the ChatterBox S3Gen
/// decoder: a length-64 tensor whose declared meta resolved to 128, cascading
/// the whole U-Net). Only consult `output_meta` when the input is under-inferred.
fn unary_out_shape(ctx: &LowerCtx<'_>, node: &BundleNode, m: &HirMut<'_>, x: HirNodeId) -> Shape {
    let in_s = m.shape(x).clone();
    if in_s.is_static() {
        in_s
    } else {
        output_shape(ctx, node, m, x)
    }
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

/// Expand the batch (leading) dims of two matmul operands to the broadcasted batch
/// from the output shape `s`, leaving the trailing `[M,K]`/`[K,N]` intact.
fn broadcast_matmul_batch(
    m: &mut HirMut<'_>,
    a: HirNodeId,
    b: HirNodeId,
    s: &Shape,
) -> (HirNodeId, HirNodeId) {
    let sa = m.shape(a).clone();
    let sb = m.shape(b).clone();
    let rs = s.rank();
    if rs < 3 || sa.rank() != rs || sb.rank() != rs {
        return (a, b);
    }
    let batch: Vec<usize> = (0..rs - 2).map(|i| s.dim(i).unwrap_static()).collect();
    let expand = |m: &mut HirMut<'_>, x: HirNodeId, xs: &Shape| -> HirNodeId {
        let mut tgt = batch.clone();
        tgt.push(xs.dim(rs - 2).unwrap_static());
        tgt.push(xs.dim(rs - 1).unwrap_static());
        let cur: Vec<usize> = xs.dims().iter().map(|d| d.unwrap_static()).collect();
        if tgt == cur {
            x
        } else {
            expand_operand_to_shape(m, x, &Shape::new(&tgt, xs.dtype()))
        }
    };
    let a2 = expand(m, a, &sa);
    let b2 = expand(m, b, &sb);
    (a2, b2)
}

fn permuted_shape(in_s: &Shape, perm: &[usize]) -> Shape {
    // Preserve `Dim::Dynamic` — a transpose only reorders axes; unwrapping to
    // usize would panic on a `dynamic_sequence` import's symbolic length.
    let dims: Vec<Dim> = perm
        .iter()
        .filter_map(|&p| in_s.dims().get(p).copied())
        .collect();
    Shape::from_dims(&dims, in_s.dtype())
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
    nodes.iter().find(|n| n.outputs.iter().any(|o| o == name))
}

/// Like [`bundle_node_for_output`], but also searches `If` then/else subgraphs.
/// Needed so shape-arithmetic folds (`ConstantOfShape(Concat(…))`, etc.) work
/// when the Zipformer `encoder_pos` else branch is inlined for `2T−1 > max_len`.
fn bundle_node_for_output_ctx<'a>(ctx: &'a LowerCtx<'_>, name: &str) -> Option<&'a BundleNode> {
    if let Some(n) = bundle_node_for_output(ctx.nodes, name) {
        return Some(n);
    }
    for (then_n, else_n) in ctx.if_branches.values() {
        if let Some(n) =
            bundle_node_for_output(then_n, name).or_else(|| bundle_node_for_output(else_n, name))
        {
            return Some(n);
        }
    }
    None
}

// Per-compile memoization of the shape evaluators. Extending the ops these fold
// (control-flow / dynamic-pad / downsample shape plumbing) made deep
// Shape→Gather→arith→Concat→Reshape chains foldable — but the recursive
// evaluators re-walk them once per dependent Reshape/Concat, which is O(N²) on a
// big graph (the Zipformer fm_decoder). Cache SUCCESSFUL folds by tensor name:
// within one compile the bindings are fixed, so a name's fold is deterministic.
// `None` (unfoldable) is cheap and left uncached to avoid depth-limit poisoning.
// `eval_caches_reset()` is called at the start of each `build_hir_from_parts`.
thread_local! {
    static EVAL_VEC_CACHE: std::cell::RefCell<HashMap<String, Vec<i64>>> =
        std::cell::RefCell::new(HashMap::new());
    static EVAL_I64_CACHE: std::cell::RefCell<HashMap<String, (Vec<i64>, Vec<usize>)>> =
        std::cell::RefCell::new(HashMap::new());
}

fn eval_caches_reset() {
    EVAL_VEC_CACHE.with(|c| c.borrow_mut().clear());
    EVAL_I64_CACHE.with(|c| c.borrow_mut().clear());
}

/// Evaluate ONNX shape tensors (Shape→Gather→Concat chains) at import time.
/// Memoizing wrapper — see [`eval_static_shape_vector_uncached`].
fn eval_static_shape_vector(
    ctx: &LowerCtx<'_>,
    m: &HirMut<'_>,
    name: &str,
    depth: usize,
) -> Option<Vec<i64>> {
    if let Some(v) = EVAL_VEC_CACHE.with(|c| c.borrow().get(name).cloned()) {
        return Some(v);
    }
    let r = eval_static_shape_vector_uncached(ctx, m, name, depth);
    if let Some(ref v) = r {
        EVAL_VEC_CACHE.with(|c| c.borrow_mut().insert(name.to_string(), v.clone()));
    }
    r
}

fn eval_static_shape_vector_uncached(
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
    let node = bundle_node_for_output_ctx(ctx, name)?;
    match node.op.as_str() {
        // Identity for the flat value vector: rank-only ops don't change which
        // integers a shape-arithmetic tensor carries, only their layout (which
        // this evaluator ignores). `Reshape` here is the pad/downsample shape
        // plumbing (`Reshape → … → Concat` reshape targets).
        "Identity" | "Cast" | "Unsqueeze" | "Squeeze" | "Reshape" | "Flatten"
            if !node.inputs.is_empty() =>
        {
            eval_static_shape_vector(ctx, m, &node.inputs[0], depth + 1)
        }
        "Expand" if !node.inputs.is_empty() => {
            let data = eval_static_shape_vector(ctx, m, &node.inputs[0], depth + 1)?;
            // Expand to a target shape: broadcast a scalar to `prod(shape)`;
            // otherwise keep the values (a same-size expand is a no-op here).
            let tgt = node
                .inputs
                .get(1)
                .filter(|s| !s.is_empty())
                .and_then(|n| eval_static_shape_vector(ctx, m, n, depth + 1));
            // Cap the fold size: shape-arithmetic vectors are tiny. A large expand
            // is real data — return None so it stays a runtime op.
            const MAX: usize = 4096;
            match tgt {
                Some(t) if data.len() == 1 => {
                    let n: usize = t.iter().map(|&d| d.max(0) as usize).product();
                    if n > MAX { None } else { Some(vec![data[0]; n.max(1)]) }
                }
                _ => Some(data),
            }
        }
        "Tile" if node.inputs.len() >= 2 => {
            let data = eval_static_shape_vector(ctx, m, &node.inputs[0], depth + 1)?;
            let reps = eval_static_shape_vector(ctx, m, &node.inputs[1], depth + 1)?;
            let r = reps.first().copied().unwrap_or(1).max(1) as usize;
            if data.len().saturating_mul(r) > 4096 {
                return None;
            }
            Some(data.iter().cloned().cycle().take(data.len() * r).collect())
        }
        "Shape" if !node.inputs.is_empty() => {
            let in_name = &node.inputs[0];
            let shape = if let Some(id) = ctx.env.get(in_name) {
                m.shape(*id).clone()
            } else {
                ctx.shape_for(m, in_name).ok()?
            };
            // Resolve any residual dynamic dim to the compile sequence length rather
            // than panicking (`unwrap_static`) — a graph input can still carry a
            // symbolic seq dim here even when the compile length is fixed.
            Some(
                shape
                    .dims()
                    .iter()
                    .map(|d| dim_usize(*d, ctx.opts) as i64)
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
        "Add" | "Sub" | "Mul" | "Div" | "Pow" if node.inputs.len() >= 2 => {
            // Scalar/vector arithmetic on shape values (`2*length-1`,
            // `length-(window+1)`, `length^2` in VITS relative-position attention).
            let a = eval_static_shape_vector(ctx, m, &node.inputs[0], depth + 1)?;
            let b = eval_static_shape_vector(ctx, m, &node.inputs[1], depth + 1)?;
            if a.is_empty() || b.is_empty() {
                return None;
            }
            let n = a.len().max(b.len());
            let at = |v: &[i64], i: usize| if v.len() == 1 { v[0] } else { v[i] };
            if a.len() != 1 && b.len() != 1 && a.len() != b.len() {
                return None;
            }
            let op = node.op.as_str();
            Some(
                (0..n)
                    .map(|i| {
                        let (x, y) = (at(&a, i), at(&b, i));
                        match op {
                            "Add" => x + y,
                            "Sub" => x - y,
                            "Mul" => x * y,
                            "Pow" => {
                                if y >= 0 {
                                    x.pow(y.min(31) as u32)
                                } else {
                                    0
                                }
                            }
                            _ => {
                                if y != 0 {
                                    x / y
                                } else {
                                    0
                                }
                            }
                        }
                    })
                    .collect(),
            )
        }
        "Max" | "Min" if !node.inputs.is_empty() => {
            let parts: Option<Vec<Vec<i64>>> = node
                .inputs
                .iter()
                .map(|inp| eval_static_shape_vector(ctx, m, inp, depth + 1))
                .collect();
            let parts = parts?;
            let n = parts.iter().map(|p| p.len()).max().unwrap_or(0);
            if n == 0 {
                return None;
            }
            let at = |v: &[i64], i: usize| {
                if v.len() == 1 {
                    v[0]
                } else {
                    v[i.min(v.len() - 1)]
                }
            };
            let is_max = node.op == "Max";
            Some(
                (0..n)
                    .map(|i| {
                        parts
                            .iter()
                            .map(|p| at(p, i))
                            .reduce(|a, b| if is_max { a.max(b) } else { a.min(b) })
                            .unwrap_or(0)
                    })
                    .collect(),
            )
        }
        // Reductions over a (tiny) shape-arithmetic vector. The S3Gen decoder's
        // sequence mask builds its length as `Range(0, ReduceMax(Add(pad_offsets,
        // lengths)), 1)`; without folding `ReduceMax` the Range limit can't
        // resolve and the mask length falls back to a default, mismatching the
        // token length. Layout/axis is ignored (these vectors are 1-D).
        "ReduceMax" | "ReduceMin" | "ReduceSum" | "ReduceProd"
            // Only fold a FULL reduce-to-scalar (no `axes` input/attr). An
            // axis-specific reduce over a multi-element shape vector would need
            // layout the flat evaluator discards — folding it produces a wrong
            // scalar (observed: a spurious `264` deep in the S3Gen decoder).
            if !node.inputs.is_empty()
                && node.inputs.len() < 2
                && !node.attrs.contains_key("axes") =>
        {
            let data = eval_static_shape_vector(ctx, m, &node.inputs[0], depth + 1)?;
            if data.is_empty() {
                return None;
            }
            let v = match node.op.as_str() {
                "ReduceMax" => data.iter().copied().max().unwrap_or(0),
                "ReduceMin" => data.iter().copied().min().unwrap_or(0),
                "ReduceSum" => data.iter().sum(),
                _ => data.iter().product(),
            };
            Some(vec![v])
        }
        "Relu" if !node.inputs.is_empty() => {
            eval_static_shape_vector(ctx, m, &node.inputs[0], depth + 1)
                .map(|v| v.into_iter().map(|x| x.max(0)).collect())
        }
        // `Ceil(a/b)` — the ChatterBox speech_encoder's resampler computes its
        // output length as `Ceil(num_samples·2/3)` (feeding an audio Slice's
        // `ends`). The flat i64 `Div` arm FLOORS, so a plain Ceil would keep the
        // floored value (`ceil(32000/3)=10667` vs floor `10666`). When the input
        // is a `Div`, recompute it as ceil-division `(a+b−1)/b`; otherwise pass
        // the (already-integer) value through. `Floor` is the plain floor `Div`.
        "Ceil" | "Floor" if !node.inputs.is_empty() => {
            if node.op == "Ceil" {
                if let Some(div) = bundle_node_for_output_ctx(ctx, &node.inputs[0])
                    .filter(|n| n.op == "Div" && n.inputs.len() >= 2)
                {
                    let a = eval_static_shape_vector(ctx, m, &div.inputs[0], depth + 1)?;
                    let b = eval_static_shape_vector(ctx, m, &div.inputs[1], depth + 1)?;
                    if !a.is_empty() && !b.is_empty() {
                        let n = a.len().max(b.len());
                        let at = |v: &[i64], i: usize| if v.len() == 1 { v[0] } else { v[i] };
                        return Some(
                            (0..n)
                                .map(|i| {
                                    let (x, y) = (at(&a, i), at(&b, i));
                                    if y != 0 { (x + y - 1).div_euclid(y) } else { 0 }
                                })
                                .collect(),
                        );
                    }
                }
            }
            eval_static_shape_vector(ctx, m, &node.inputs[0], depth + 1)
        }
        "Clip" if !node.inputs.is_empty() => {
            let v = eval_static_shape_vector(ctx, m, &node.inputs[0], depth + 1)?;
            let lo = node
                .inputs
                .get(1)
                .filter(|s| !s.is_empty())
                .and_then(|n| eval_static_shape_vector(ctx, m, n, depth + 1))
                .and_then(|v| v.first().copied());
            let hi = node
                .inputs
                .get(2)
                .filter(|s| !s.is_empty())
                .and_then(|n| eval_static_shape_vector(ctx, m, n, depth + 1))
                .and_then(|v| v.first().copied());
            Some(
                v.into_iter()
                    .map(|x| {
                        let x = lo.map(|l| x.max(l)).unwrap_or(x);
                        hi.map(|h| x.min(h)).unwrap_or(x)
                    })
                    .collect(),
            )
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
        // `Shape[a:b]` — a 1-D slice over a shape vector (e.g. supertonic vocoder's
        // de-quant reshape target `Concat([batch, -1, 6, Shape[2:3]])`). Only axis-0,
        // step-1 (the only meaningful case for a rank-1 shape vector).
        "Slice" if node.inputs.len() >= 3 => {
            let data = eval_static_shape_vector(ctx, m, &node.inputs[0], depth + 1)?;
            let ev = |i: usize| {
                node.inputs
                    .get(i)
                    .filter(|s| !s.is_empty())
                    .and_then(|n| eval_static_shape_vector(ctx, m, n, depth + 1))
            };
            let starts = ev(1)?;
            let ends = ev(2)?;
            let ax = ev(3).and_then(|a| a.first().copied()).unwrap_or(0);
            let step = ev(4).and_then(|s| s.first().copied()).unwrap_or(1);
            if (ax != 0 && ax != -1) || step != 1 {
                return None;
            }
            let len = data.len() as i64;
            let clamp = |x: i64| (if x < 0 { len + x } else { x }).clamp(0, len) as usize;
            let s = clamp(starts.first().copied().unwrap_or(0));
            let e = clamp(ends.first().copied().unwrap_or(len));
            Some(if s < e { data[s..e].to_vec() } else { Vec::new() })
        }
        // Shape-arithmetic control/broadcast ops emitted by dynamic padding (e.g.
        // the Zipformer downsample's pad-to-multiple: `ConstantOfShape → Equal →
        // Where`) and ceil-division (`Mod`). Folding these lets the dependent
        // Reshape/Expand targets resolve statically at the compile length.
        "ConstantOfShape" if !node.inputs.is_empty() => {
            let dims = eval_static_shape_vector(ctx, m, &node.inputs[0], depth + 1)?;
            let val = node
                .attrs
                .get("value")
                .and_then(|v| {
                    v.get("tensor")
                        .and_then(|t| t.get("scalar"))
                        .and_then(|s| s.as_f64())
                        .map(|x| x as i64)
                        .or_else(|| v.as_array().and_then(|a| a.first()).and_then(|x| x.as_i64()))
                        .or_else(|| v.as_i64())
                })
                .unwrap_or(0);
            // Empty shape → scalar; product of empty = 1 in ONNX ConstantOfShape.
            let n: usize = if dims.is_empty() {
                1
            } else {
                dims.iter().map(|&d| d.max(0) as usize).product()
            };
            Some(vec![val; n])
        }
        "Where" if node.inputs.len() >= 3 => {
            let c = eval_static_shape_vector(ctx, m, &node.inputs[0], depth + 1)?;
            let x = eval_static_shape_vector(ctx, m, &node.inputs[1], depth + 1)?;
            let y = eval_static_shape_vector(ctx, m, &node.inputs[2], depth + 1)?;
            let n = c.len().max(x.len()).max(y.len());
            let at = |v: &[i64], i: usize| if v.len() == 1 { v[0] } else { *v.get(i).unwrap_or(&0) };
            Some((0..n).map(|i| if at(&c, i) != 0 { at(&x, i) } else { at(&y, i) }).collect())
        }
        "Equal" | "Mod" if node.inputs.len() >= 2 => {
            let a = eval_static_shape_vector(ctx, m, &node.inputs[0], depth + 1)?;
            let b = eval_static_shape_vector(ctx, m, &node.inputs[1], depth + 1)?;
            if a.is_empty() || b.is_empty() {
                return None;
            }
            let n = a.len().max(b.len());
            let at = |v: &[i64], i: usize| if v.len() == 1 { v[0] } else { *v.get(i).unwrap_or(&0) };
            let is_eq = node.op == "Equal";
            Some(
                (0..n)
                    .map(|i| {
                        let (x, y) = (at(&a, i), at(&b, i));
                        if is_eq {
                            (x == y) as i64
                        } else if y != 0 {
                            x.rem_euclid(y)
                        } else {
                            0
                        }
                    })
                    .collect(),
            )
        }
        _ => None,
    }
}

/// Evaluate a FLOAT scalar shape-expression at import time. The ONNX length
/// regulator computes frame counts in floating point
/// (`num_frames = ceil(prompt_feat_len / prompt_len × (…) / speed)`), which the
/// integer evaluators can't fold. Integer leaves (`Gather(Shape(x))`, bound dims)
/// are lifted to f64. Returns None if any leaf is a genuine runtime value.
fn eval_f64_scalar(ctx: &LowerCtx<'_>, m: &HirMut<'_>, name: &str, depth: usize) -> Option<f64> {
    if depth > 40 {
        return None;
    }
    // Folded params (Constant / initializer scalars).
    if let Some(v) = ctx.params.get(name) {
        return v.first().map(|&x| x as f64);
    }
    if let Some(v) = ctx.i64_params.get(name) {
        return v.first().map(|&x| x as f64);
    }
    let node = bundle_node_for_output_ctx(ctx, name)?;
    let arg = |i: usize| {
        node.inputs
            .get(i)
            .and_then(|n| eval_f64_scalar(ctx, m, n, depth + 1))
    };
    match node.op.as_str() {
        "Ceil" => arg(0).map(f64::ceil),
        "Floor" => arg(0).map(f64::floor),
        "Round" => arg(0).map(f64::round),
        "Neg" => arg(0).map(|x| -x),
        "Cast" | "Identity" => arg(0),
        "Div" | "Mul" | "Add" | "Sub" | "Pow" if node.inputs.len() >= 2 => {
            let (a, b) = (arg(0)?, arg(1)?);
            Some(match node.op.as_str() {
                "Div" => a / b,
                "Mul" => a * b,
                "Add" => a + b,
                "Sub" => a - b,
                _ => a.powf(b),
            })
        }
        // Integer shape-arithmetic leaf (e.g. `Gather(Shape(x))`) — lift to f64.
        _ => {
            eval_i64_shaped(ctx, m, name, depth + 1).and_then(|(d, _)| d.first().map(|&x| x as f64))
        }
    }
}

/// Evaluate a 1-D COMPILE-TIME-CONSTANT numeric vector (values, as f64). Walks a
/// constant initializer/param through element-wise `Mul/Add/Sub/Div`, a scalar
/// `Greater/Less/…/Equal` threshold (→ 0/1 mask), an axis-0 step-1 `Slice`
/// (unresolvable `ends` clamp to the data length — ONNX Slice semantics), and
/// `Cast/Identity`. Enables folding e.g. the ISTFT window-sum `NonZero`
/// normalization mask, whose values depend only on a fixed `window_sum`
/// initializer (not the audio), even though the graph routes it past a
/// data-dependent signal length. Returns `None` (no fold) if anything is
/// non-constant. Capped to keep import cheap.
fn eval_const_f64_vec(
    ctx: &LowerCtx<'_>,
    m: &HirMut<'_>,
    name: &str,
    depth: usize,
) -> Option<Vec<f64>> {
    const CAP: usize = 1 << 16;
    if depth > 40 {
        return None;
    }
    if let Some(v) = ctx.params.get(name) {
        return (v.len() <= CAP).then(|| v.iter().map(|&x| x as f64).collect());
    }
    if let Some(v) = ctx.i64_params.get(name) {
        return (v.len() <= CAP).then(|| v.iter().map(|&x| x as f64).collect());
    }
    let node = bundle_node_for_output_ctx(ctx, name)?;
    let vec_arg = |i: usize| {
        node.inputs
            .get(i)
            .filter(|s| !s.is_empty())
            .and_then(|n| eval_const_f64_vec(ctx, m, n, depth + 1))
    };
    let scalar_arg = |i: usize| {
        node.inputs
            .get(i)
            .filter(|s| !s.is_empty())
            .and_then(|n| eval_f64_scalar(ctx, m, n, depth + 1))
    };
    match node.op.as_str() {
        "Cast" | "Identity" => vec_arg(0),
        "Greater" | "GreaterOrEqual" | "Less" | "LessOrEqual" | "Equal"
            if node.inputs.len() >= 2 =>
        {
            let a = vec_arg(0)?;
            let t = scalar_arg(1)?;
            Some(
                a.iter()
                    .map(|&x| {
                        let hit = match node.op.as_str() {
                            "Greater" => x > t,
                            "GreaterOrEqual" => x >= t,
                            "Less" => x < t,
                            "LessOrEqual" => x <= t,
                            _ => x == t,
                        };
                        if hit { 1.0 } else { 0.0 }
                    })
                    .collect(),
            )
        }
        "Mul" | "Add" | "Sub" | "Div" if node.inputs.len() >= 2 => {
            let a = vec_arg(0)?;
            let b = vec_arg(1).or_else(|| scalar_arg(1).map(|s| vec![s; a.len()]))?;
            if b.len() != a.len() && b.len() != 1 {
                return None;
            }
            Some(
                a.iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        let y = if b.len() == 1 { b[0] } else { b[i] };
                        match node.op.as_str() {
                            "Mul" => x * y,
                            "Add" => x + y,
                            "Sub" => x - y,
                            _ => x / y,
                        }
                    })
                    .collect(),
            )
        }
        "Slice" if !node.inputs.is_empty() => {
            let data = vec_arg(0)?;
            let len = data.len() as i64;
            let start = scalar_arg(1).map(|s| s as i64).unwrap_or(0);
            let end = scalar_arg(2).map(|s| s as i64).unwrap_or(len);
            let step = scalar_arg(4).map(|s| s as i64).unwrap_or(1);
            if step != 1 {
                return None;
            }
            let s = start.clamp(0, len) as usize;
            let e = end.clamp(0, len) as usize;
            (s <= e).then(|| data[s..e].to_vec())
        }
        _ => None,
    }
}

/// Shaped int64 mini-interpreter for the small index/shape tensors torch emits
/// for dynamic ops (notably `F.pad`'s pad-spec construction in VITS relative-
/// position attention: `ConstantOfShape → Concat → Reshape → Transpose → Slice`).
/// Returns `(data, shape)` row-major. Memoizing wrapper (per-compile cache).
fn eval_i64_shaped(
    ctx: &LowerCtx<'_>,
    m: &HirMut<'_>,
    name: &str,
    depth: usize,
) -> Option<(Vec<i64>, Vec<usize>)> {
    if let Some(v) = EVAL_I64_CACHE.with(|c| c.borrow().get(name).cloned()) {
        return Some(v);
    }
    let r = eval_i64_shaped_uncached(ctx, m, name, depth);
    if let Some(ref v) = r {
        EVAL_I64_CACHE.with(|c| c.borrow_mut().insert(name.to_string(), v.clone()));
    }
    r
}

fn eval_i64_shaped_uncached(
    ctx: &LowerCtx<'_>,
    m: &HirMut<'_>,
    name: &str,
    depth: usize,
) -> Option<(Vec<i64>, Vec<usize>)> {
    if depth > 32 {
        return None;
    }
    // Genuine i64 params are shape/index tensors — use directly.
    if let Some(v) = ctx.i64_params.get(name) {
        let shape = ctx
            .init_shapes
            .get(name)
            .cloned()
            .unwrap_or_else(|| vec![v.len()]);
        return Some((v.clone(), shape));
    }
    // An f32 param is only shape/index data if it is SMALL and INTEGER-VALUED
    // (some exporters store shapes as f32, e.g. Reshape targets `[1, 192, -1]`).
    // A large or fractional f32 tensor is real data — e.g. OpenVoice's baked
    // `[1,192,32768]` noise constant — and must NOT be reinterpreted as an i64
    // shape (doing so folds a data Slice into integer garbage → NaN). Reject it
    // so the Slice falls through to a proper Narrow.
    if let Some(v) = ctx.params.get(name) {
        const MAX_SHAPE_ELEMS: usize = 4096;
        if v.len() <= MAX_SHAPE_ELEMS
            && v.iter()
                .all(|&x| x.is_finite() && (x - x.round()).abs() < 1e-3 && x.abs() < 9.0e18)
        {
            let iv: Vec<i64> = v.iter().map(|&x| x.round() as i64).collect();
            let shape = ctx
                .init_shapes
                .get(name)
                .cloned()
                .unwrap_or_else(|| vec![iv.len()]);
            return Some((iv, shape));
        }
    }
    let node = bundle_node_for_output_ctx(ctx, name)?;
    let numel = |s: &[usize]| s.iter().product::<usize>().max(1);
    match node.op.as_str() {
        "Identity" if !node.inputs.is_empty() => {
            eval_i64_shaped(ctx, m, &node.inputs[0], depth + 1)
        }
        "Cast" if !node.inputs.is_empty() => {
            // A `Cast` to an integer type of a FLOAT scalar expression must be
            // evaluated in floating point — the ONNX length regulator computes
            // `num_frames = Cast(Ceil(feat_len/plen * (…) / speed), int)`, and
            // integer eval would truncate `40/15` to 2 instead of 2.667. ONNX
            // `Cast` to int truncates toward zero.
            let to = node.attrs.get("to").and_then(|v| v.as_i64()).unwrap_or(0);
            let src_float = bundle_node_for_output_ctx(ctx, &node.inputs[0])
                .map(|s| {
                    matches!(s.op.as_str(), "Ceil" | "Floor" | "Round" | "Div")
                        || s.output_meta
                            .first()
                            .and_then(|meta| meta.get("dtype"))
                            .and_then(|d| d.as_str())
                            .is_some_and(|d| d == "f32" || d == "f64")
                })
                .unwrap_or(false);
            // The float-scalar path collapses its operand to a single value, so it
            // is ONLY valid for a genuinely scalar length expression. A Cast of a
            // multi-element vector (e.g. VITS `convert_pad_shape`'s
            // `Cast(Concat([0,l-1,0,…]))`, 2*rank pad amounts) must preserve every
            // element — route those through the integer vector eval instead.
            let int_eval = eval_i64_shaped(ctx, m, &node.inputs[0], depth + 1);
            if let Some((data, _)) = &int_eval {
                if data.len() > 1 {
                    return int_eval;
                }
            }
            if matches!(to, 6 | 7 | 9) && src_float {
                if let Some(v) = eval_f64_scalar(ctx, m, &node.inputs[0], depth + 1) {
                    return Some((vec![v.trunc() as i64], vec![]));
                }
            }
            int_eval
        }
        "Shape" if !node.inputs.is_empty() => {
            if let Some((_, shape)) = eval_i64_shaped(ctx, m, &node.inputs[0], depth + 1) {
                let dims: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
                let n = dims.len();
                return Some((dims, vec![n]));
            }
            let v = eval_static_shape_vector(ctx, m, name, depth + 1)?;
            let n = v.len();
            Some((v, vec![n]))
        }
        "Gather" if node.inputs.len() >= 2 => {
            // Shape-arithmetic gather over a 1-D shape vector, axis 0 — e.g.
            // `Gather(Shape(x), idx)` selecting dims. Output values = table[idx],
            // output shape = idx shape (so a scalar index yields a scalar dim, not
            // a vector). Only this case; real data gathers fall through to None.
            let (table, tshape) = eval_i64_shaped(ctx, m, &node.inputs[0], depth + 1)?;
            let (idx, ishape) = eval_i64_shaped(ctx, m, &node.inputs[1], depth + 1)?;
            let axis = node
                .attrs
                .get("axis")
                .and_then(|v| v.as_i64())
                .unwrap_or(0)
                .rem_euclid(tshape.len().max(1) as i64) as usize;
            if tshape.len() != 1 || axis != 0 || table.is_empty() {
                return None;
            }
            let out: Vec<i64> = idx
                .iter()
                .map(|&i| table[i.rem_euclid(table.len() as i64) as usize])
                .collect();
            let out_shape = if ishape.is_empty() {
                vec![out.len()]
            } else {
                ishape
            };
            Some((out, out_shape))
        }
        "ConstantOfShape" if !node.inputs.is_empty() => {
            let (shape_data, _) = eval_i64_shaped(ctx, m, &node.inputs[0], depth + 1)?;
            let dims: Vec<usize> = shape_data.iter().map(|&d| d.max(0) as usize).collect();
            let val = node
                .attrs
                .get("value")
                .and_then(|v| v.as_array())
                .and_then(|a| a.first())
                .and_then(|v| v.as_i64())
                .or_else(|| node.attrs.get("value").and_then(|v| v.as_i64()))
                .unwrap_or(0);
            let n: usize = dims.iter().product();
            Some((vec![val; n], dims))
        }
        "Unsqueeze" if !node.inputs.is_empty() => {
            let (data, mut shape) = eval_i64_shaped(ctx, m, &node.inputs[0], depth + 1)?;
            for ax in unsqueeze_axes(ctx, node) {
                let pos = ax.rem_euclid(shape.len() as i64 + 1) as usize;
                shape.insert(pos.min(shape.len()), 1);
            }
            Some((data, shape))
        }
        "Squeeze" if !node.inputs.is_empty() => {
            let (data, shape) = eval_i64_shaped(ctx, m, &node.inputs[0], depth + 1)?;
            let new: Vec<usize> = shape.into_iter().filter(|&d| d != 1).collect();
            Some((data, if new.is_empty() { vec![1] } else { new }))
        }
        "Reshape" if node.inputs.len() >= 2 => {
            let (data, _) = eval_i64_shaped(ctx, m, &node.inputs[0], depth + 1)?;
            let (mut tgt, _) = eval_i64_shaped(ctx, m, &node.inputs[1], depth + 1)?;
            let total = data.len() as i64;
            let known: i64 = tgt.iter().filter(|&&d| d > 0).product();
            for d in tgt.iter_mut() {
                if *d == -1 {
                    *d = if known > 0 { total / known } else { 0 };
                } else if *d == 0 {
                    *d = if known > 0 { total / known.max(1) } else { 0 };
                }
            }
            let dims: Vec<usize> = tgt.iter().map(|&d| d.max(0) as usize).collect();
            if numel(&dims) != data.len() {
                return None;
            }
            Some((data, dims))
        }
        "Transpose" => {
            let (data, shape) = eval_i64_shaped(ctx, m, &node.inputs[0], depth + 1)?;
            let rank = shape.len();
            let perm: Vec<usize> = node
                .attrs
                .get("perm")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|d| d.as_u64().map(|x| x as usize))
                        .collect()
                })
                .unwrap_or_else(|| (0..rank).rev().collect());
            if perm.len() != rank {
                return None;
            }
            let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
            let in_strides = row_major_strides(&shape);
            let out_strides = row_major_strides(&out_shape);
            let mut out = vec![0i64; data.len()];
            for (oi, slot) in out.iter_mut().enumerate() {
                let mut rem = oi;
                let mut in_off = 0usize;
                for (k, &os) in out_strides.iter().enumerate() {
                    if os == 0 {
                        continue;
                    }
                    let coord = rem / os;
                    rem %= os;
                    in_off += coord * in_strides[perm[k]];
                }
                *slot = data.get(in_off).copied().unwrap_or(0);
            }
            Some((out, out_shape))
        }
        "Slice" if !node.inputs.is_empty() => {
            let (data, shape) = eval_i64_shaped(ctx, m, &node.inputs[0], depth + 1)?;
            let rank = shape.len();
            let get = |i: usize| eval_static_shape_vector(ctx, m, node.inputs.get(i)?, depth + 1);
            let starts = get(1)?;
            let ends = get(2)?;
            let axes = get(3).unwrap_or_else(|| (0..rank as i64).collect());
            let steps = get(4).unwrap_or_else(|| vec![1; axes.len()]);
            let mut idx: Vec<Vec<usize>> = shape.iter().map(|&d| (0..d).collect()).collect();
            for (k, &ax) in axes.iter().enumerate() {
                let a = ax.rem_euclid(rank as i64) as usize;
                let d = shape[a] as i64;
                // Empty dim → empty selection (guards clamp(0, d-1) panic when d==0).
                if d == 0 {
                    idx[a] = Vec::new();
                    continue;
                }
                let step = steps.get(k).copied().unwrap_or(1);
                let mut s = starts[k];
                let mut e = ends[k];
                if s < 0 {
                    s += d;
                }
                if e < 0 {
                    e += d;
                }
                let mut sel = Vec::new();
                if step > 0 {
                    let s = s.clamp(0, d);
                    let e = e.clamp(0, d);
                    let mut i = s;
                    while i < e {
                        sel.push(i as usize);
                        i += step;
                    }
                } else if step < 0 {
                    let s = s.clamp(0, d - 1);
                    let e = e.max(-1).min(d - 1);
                    let mut i = s;
                    while i > e {
                        sel.push(i as usize);
                        i += step;
                    }
                }
                idx[a] = sel;
            }
            let out_shape: Vec<usize> = idx.iter().map(|v| v.len()).collect();
            let in_strides = row_major_strides(&shape);
            let out_strides = row_major_strides(&out_shape);
            let mut out = vec![0i64; numel(&out_shape)];
            for (oi, slot) in out.iter_mut().enumerate() {
                let mut rem = oi;
                let mut in_off = 0usize;
                for a in 0..rank {
                    let coord = rem.checked_div(out_strides[a]).unwrap_or(0);
                    if out_strides[a] != 0 {
                        rem %= out_strides[a];
                    }
                    let src = idx[a].get(coord).copied().unwrap_or(0);
                    in_off += src * in_strides[a];
                }
                *slot = data.get(in_off).copied().unwrap_or(0);
            }
            Some((out, out_shape))
        }
        "Concat" if !node.inputs.is_empty() => {
            let axis = node.attrs.get("axis").and_then(|v| v.as_i64()).unwrap_or(0);
            let parts: Option<Vec<(Vec<i64>, Vec<usize>)>> = node
                .inputs
                .iter()
                .map(|inp| eval_i64_shaped(ctx, m, inp, depth + 1))
                .collect();
            let parts = parts?;
            let rank = parts.iter().map(|(_, s)| s.len()).max().unwrap_or(1);
            if rank <= 1 {
                let mut out = Vec::new();
                for (d, _) in &parts {
                    out.extend_from_slice(d);
                }
                let n = out.len();
                return Some((out, vec![n]));
            }
            let ax = axis.rem_euclid(rank as i64) as usize;
            if ax != rank - 1 {
                let flat = eval_static_shape_vector(ctx, m, name, depth + 1)?;
                let n = flat.len();
                return Some((flat, vec![n]));
            }
            let outer = parts[0].1[..rank - 1].iter().product::<usize>().max(1);
            let mut out = Vec::new();
            for o in 0..outer {
                for (d, s) in &parts {
                    let last = *s.last().unwrap_or(&1);
                    out.extend_from_slice(&d[o * last..(o + 1) * last]);
                }
            }
            let mut out_shape = parts[0].1.clone();
            let last_sum: usize = parts.iter().map(|(_, s)| *s.last().unwrap_or(&1)).sum();
            *out_shape.last_mut().unwrap() = last_sum;
            Some((out, out_shape))
        }
        _ => {
            let v = eval_static_shape_vector(ctx, m, name, depth + 1)?;
            let n = v.len();
            Some((v, vec![n]))
        }
    }
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

fn resolve_reshape_dims(mut dims: Vec<i64>, in_s: &Shape) -> Option<Vec<i64>> {
    // ONNX Reshape (default allowzero=0): a 0 in the target copies the corresponding
    // input dimension. Treating 0 as a literal zero made `Reshape([0,32,-1])` on
    // `[1,512,200]` fail (`known` product 0) and fall back to the input shape, so
    // MioTTS InstanceNorm still saw `[1,512,200]` instead of `[1,32,3200]`.
    for (i, d) in dims.iter_mut().enumerate() {
        if *d == 0 {
            if i >= in_s.rank() {
                return None;
            }
            *d = in_s.dim(i).unwrap_static() as i64;
        }
    }
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
    // Symmetric channel-concat of identical NCL tensors — the VITS flow coupling
    // `cat([x0, x1_transformed], dim=1)` where both operands are `[1, C, L]`.
    // Concatenate channels directly; do NOT transpose. `is_vocoder_blc` misfires
    // here because the length dim (e.g. 64) coincidentally matches a "typical
    // channel" value while the true channel dim (e.g. 96) does not — flipping the
    // operands to `[1, L, C]` yields `[1, 2L, C]` instead of `[1, 2C, L]`.
    if is_ncl_rank3(s) && peers.iter().all(|&p| m.shape(p).dims() == s.dims()) {
        return false;
    }
    // Sequence(middle)-axis concat: all operands share the same LAST dim but differ
    // in the MIDDLE dim → the concat EXTENDS the middle axis with the channels
    // preserved (F5-TTS mel-condition: `[ref_frames,100] ++ [pad_frames,100]` fills
    // the duration canvas). A shared last dim ⇒ channels unchanged ⇒ this is a
    // sequence concat, not a channel concat, so do NOT transpose to NCL. Without
    // this, `is_vocoder_blc` decides purely on `dim1 > dim2`: at short references
    // (frames < 100 mels) it guessed NCL and skipped, but at long references
    // (frames > 100) it flipped to BLC and transposed → the mel-condition came out
    // `[1, 2·mels, frames]` instead of `[1, md, mels]` and corrupted conditioning.
    let same_last = peers.iter().all(|&p| {
        let ps = m.shape(p);
        ps.rank() == 3 && ps.dim(2).unwrap_static() == d2
    });
    let differs_mid = peers
        .iter()
        .any(|&p| m.shape(p).dim(1).unwrap_static() != d1);
    if same_last && differs_mid {
        return false;
    }
    is_vocoder_blc(s)
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

fn nchw_to_ncl_if_needed(m: &mut HirMut<'_>, x: HirNodeId, target: &Shape) -> HirNodeId {
    if target.rank() == 3 && m.shape(x).rank() == 4 {
        let n = target.dim(0).unwrap_static();
        let c = target.dim(1).unwrap_static();
        let l = target.dim(2).unwrap_static();
        return m.reshape_(x, vec![n as i64, c as i64, l as i64]);
    }
    x
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
    // This is the fallback for a Slice whose bounds/axes we couldn't fold (e.g. the
    // T5 encoder's position-bias slice `bias[:, :, -q:, :]` with a dynamic `-q`
    // start and an `axes` constant not in the param set). If the declared output
    // meta disagrees with the input element count, a Reshape here would CORRUPT the
    // data — the bias `[1,16,128,128]` (262144) got a stale meta of `[128]`, so it
    // was flattened to 128 values (head-0/query-0 only) and broadcast to every
    // query → garbled attention. At a fixed compile length these unresolvable
    // slices are full-length identities, so pass the input through unchanged.
    let in_numel = m.shape(x).num_elements();
    let out_numel = out_s.num_elements();
    if in_numel.is_some() && out_numel != in_numel {
        ctx.env.insert(node.outputs[0].clone(), x);
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
    // Starts/ends/axes/steps may be dynamically computed (`2*length-1` in VITS
    // relative attention), so fall back to the shape evaluator when not plain
    // initializers.
    let eval_vec = |ctx: &LowerCtx<'_>, m: &HirMut<'_>, n: &str| {
        i64_tensor(&ctx.i64_params, &ctx.params, n)
            .or_else(|| eval_static_shape_vector(ctx, m, n, 0))
    };
    let starts = eval_vec(ctx, m, &node.inputs[1]);
    let ends_opt = eval_vec(ctx, m, &node.inputs[2]);
    // An `axes` input that is present but non-empty; `""` marks an omitted optional.
    let axes_input = node.inputs.get(3).filter(|n| !n.is_empty());
    let axes_eval = axes_input.and_then(|n| eval_vec(ctx, m, n));
    let steps = node.inputs.get(4).and_then(|n| eval_vec(ctx, m, n));
    let Some(starts) = starts else {
        return Ok(false);
    };
    let axes: Vec<i64> = if force_time_axis {
        vec![2]
    } else if let Some(ax) = axes_eval {
        ax
    } else if axes_input.is_some() {
        // The Slice carries an explicit `axes` input we couldn't fold (e.g. the
        // T5 encoder's padding-mask / relative-position-bias slices whose `ends`
        // is a dynamic `2*len-1` and whose `axes` constant isn't in the param
        // set). Defaulting to `[0..len(starts)]` would silently narrow axis 0 —
        // for a `[1,1,1,128]` key mask that collapses the 128 keys onto the batch
        // axis, so real queries attend to *padding* keys (cosine 0.47, garbled
        // audio). Bail to `slice_to_output_shape` (a shape-correct Reshape) which
        // is exact for the identity slices these turn out to be.
        return Ok(false);
    } else {
        (0..starts.len() as i64).collect()
    };
    let steps: Vec<i64> = steps.unwrap_or_else(|| vec![1; starts.len()]);
    // When `ends` is a *dynamic* runtime length (e.g. a Slice `[:, :, 0:t]` whose
    // end is `Shape(audio)`), the shape evaluator can't fold it. Rather than drop
    // to the data-corrupting Reshape fallback, derive `ends` from the declared
    // output shape at the compile length: `end[i] = start[i] + out_meta.dim(ax)`.
    // This keeps it a real Narrow (crop) — critical for slices over baked-in data
    // (e.g. OpenVoice's `[1,192,32768]` noise constant, cropped to `[1,192,t]`).
    let ends: Vec<i64> = match ends_opt {
        Some(e) => e,
        None => {
            if steps.iter().any(|&s| s != 1) {
                return Ok(false);
            }
            let Some(out_s) = node
                .output_meta
                .first()
                .and_then(|mm| resolve_shape(mm, ctx.opts).ok())
            else {
                return Ok(false);
            };
            let mut e = Vec::with_capacity(axes.len());
            for (i, &ax) in axes.iter().enumerate() {
                let axis = ax.rem_euclid(rank as i64) as usize;
                if axis >= out_s.rank() {
                    return Ok(false);
                }
                let len = dim_usize(out_s.dim(axis), ctx.opts) as i64;
                e.push(starts.get(i).copied().unwrap_or(0) + len);
            }
            e
        }
    };
    // Negative-step (reverse) slicing. rlx-ir has no flip/reverse op; `Gather`
    // only supports axis 0 (and reads f32 indices), and a concat-of-unit-narrows
    // decomposition explodes the MLX compile time. So express the reversal as a
    // constant selection matmul: move the sliced axis to the front, left-multiply
    // by a constant `[n, dim]` selection matrix `P` (`P[r, idxs[r]] = 1`), then move
    // the axis back. This stays entirely in f32 and works on every backend. VITS
    // residual-coupling flips channels with `x[:, ::-1, :]` (starts=[-1],
    // ends=[INT_MIN], steps=[-1]); without this they silently became no-ops,
    // corrupting the whole flow graph.
    if axes.len() == 1 && steps.len() == 1 && steps[0] < 0 {
        let axis = axes[0].rem_euclid(rank as i64) as usize;
        if axis >= rank {
            return Ok(false);
        }
        let dim = dim_usize(m.shape(x).dim(axis), ctx.opts) as i64;
        let step = steps[0];
        // ONNX clamps negative-step start to [0, dim-1] and end to [-1, dim-1],
        // where end == -1 means "stop before index 0" (i.e. include index 0).
        let start = if starts[0] < 0 {
            (starts[0] + dim).clamp(0, dim - 1)
        } else {
            starts[0].min(dim - 1)
        };
        let end = if ends[0] < 0 {
            match ends[0].checked_add(dim) {
                Some(v) if v >= 0 => v,
                _ => -1,
            }
        } else {
            ends[0].min(dim - 1)
        };
        let mut idxs: Vec<usize> = Vec::new();
        let mut i = start;
        while i > end && i >= 0 {
            idxs.push(i as usize);
            i += step;
        }
        if idxs.is_empty() {
            return Ok(false);
        }
        let dim = dim as usize;
        let n = idxs.len();
        // Selection matrix P [n, dim]: row r picks source index idxs[r].
        let mut p_data = vec![0.0f32; n * dim];
        for (r, &src) in idxs.iter().enumerate() {
            p_data[r * dim + src] = 1.0;
        }
        let key = format!("__rev_perm__/{}", node.outputs[0]);
        ctx.params.insert(key.clone(), p_data);
        let p = m.param(&key, Shape::new(&[n, dim], DType::F32));

        // Move `axis` to the front, flatten the rest, matmul, restore.
        let dims: Vec<usize> = (0..rank)
            .map(|d| dim_usize(m.shape(x).dim(d), ctx.opts))
            .collect();
        let mut perm: Vec<usize> = vec![axis];
        perm.extend((0..rank).filter(|&d| d != axis));
        let rest: Vec<usize> = perm[1..].iter().map(|&d| dims[d]).collect();
        let mcols: usize = rest.iter().product::<usize>().max(1);

        let xf = m.transpose_(x, perm.clone());
        let x2 = m.reshape_(xf, vec![dim as i64, mcols as i64]);
        let y2 = m.mm(p, x2); // [n, mcols]
        let mut back_dims = vec![n as i64];
        back_dims.extend(rest.iter().map(|&d| d as i64));
        let yf = m.reshape_(y2, back_dims);
        // Inverse permutation: inv[perm[i]] = i.
        let mut inv = vec![0usize; rank];
        for (i, &p) in perm.iter().enumerate() {
            inv[p] = i;
        }
        let id = m.transpose_(yf, inv);
        ctx.env.insert(node.outputs[0].clone(), id);
        return Ok(true);
    }
    // Multi-axis positive-step slice — e.g. the VITS relative-position rel-shift
    // final index `[:, :, :l, l-1:]` (axes `[2,3]`, starts `[0, l-1]`, ends
    // `[l, 2l-1]`). The single-axis path below already resolves these dynamic
    // bounds; the only gap was >1 axis. Apply an independent `Narrow` per axis.
    // `output_meta` is unusable here because `Pad` propagates as a passthrough
    // upstream, so it never reduces `[l+1, 2l-1]` back to `[l, l]`.
    if axes.len() > 1 && steps.iter().all(|&s| s == 1) {
        if starts.len() != axes.len() || ends.len() != axes.len() {
            return Ok(false);
        }
        let mut cur = x;
        for (i, &ax_raw) in axes.iter().enumerate() {
            let axis = ax_raw.rem_euclid(rank as i64) as usize;
            if axis >= rank {
                return Ok(false);
            }
            let dim = dim_usize(m.shape(cur).dim(axis), ctx.opts);
            let (start, len) = onnx_slice_axis_start_len(starts[i], ends[i], dim);
            let len = len.max(1);
            cur = if ctx.opts.dynamic_sequence {
                let cur_s = m.shape(cur);
                let mut out_dims: Vec<Dim> = (0..cur_s.rank()).map(|d| cur_s.dim(d)).collect();
                out_dims[axis] = Dim::Static(len);
                let out_s = Shape::from_dims(&out_dims, cur_s.dtype());
                m.add_node(
                    Op::Narrow {
                        axis,
                        start: start.min(dim),
                        len,
                    },
                    vec![cur],
                    out_s,
                )
            } else {
                m.narrow_(cur, axis, start.min(dim), len)
            };
        }
        ctx.env.insert(node.outputs[0].clone(), cur);
        return Ok(true);
    }
    // Single-axis positive STRIDED slice (`step > 1`) — e.g. the RoPE interleaved
    // even/odd head-dim split `x[..., 0::2]` / `x[..., 1::2]` (starts 0/1, step 2,
    // halving 64→32). rlx-ir has no strided-narrow, so gather the strided indices
    // along the axis. Every interleaved-RoPE LM (moss-nano / maya1 / chatterbox)
    // needs this; the old blanket bail kept the FULL dim, doubling every downstream
    // shape (the rope Mul then failed to broadcast).
    if axes.len() == 1 && steps.len() == 1 && steps[0] > 1 {
        let axis = axes[0].rem_euclid(rank as i64) as usize;
        if axis < rank {
            let dim = dim_usize(m.shape(x).dim(axis), ctx.opts);
            let (start, clen) = onnx_slice_axis_start_len(starts[0], ends[0], dim);
            let step = steps[0] as usize;
            let idxs: Vec<i64> = (start..start + clen)
                .step_by(step)
                .map(|v| v as i64)
                .collect();
            if !idxs.is_empty() {
                let n = idxs.len();
                let bytes: Vec<u8> = idxs.iter().flat_map(|v| v.to_le_bytes()).collect();
                let idx_id = m.add_node(
                    Op::Constant { data: bytes },
                    vec![],
                    Shape::new(&[n], DType::I64),
                );
                let base = m.shape(x).clone();
                let mut out_dims: Vec<Dim> = (0..base.rank()).map(|d| base.dim(d)).collect();
                out_dims[axis] = Dim::Static(n);
                let out_s = Shape::from_dims(&out_dims, base.dtype());
                let id = m.add_node(Op::Gather { axis }, vec![x, idx_id], out_s);
                ctx.env.insert(node.outputs[0].clone(), id);
                return Ok(true);
            }
        }
    }
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
    // `starts`/`ends` were resolved concretely, so `computed_len` is authoritative.
    // Only fall back to the declared output meta when it is degenerate — the meta
    // is unreliable here because shape propagation treats `Pad` as a passthrough,
    // which would otherwise truncate a slice over a padded axis (rel embeddings).
    let len = if force_time_axis {
        computed_len.max(1)
    } else if computed_len > 0 && computed_len <= dim.saturating_sub(start) {
        computed_len
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

/// ONNX `Split` — divide `x` along `axis` into N contiguous chunks (one per
/// output). Split sizes come from the `split` input tensor (opset 13+), the
/// `split` attribute (opset <13), or an equal division. Lowered to one `Narrow`
/// per non-empty output. VITS `enc_p` uses this to split `stats` into `m, logs`.
fn lower_split(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let rank = m.shape(x).rank();
    if rank == 0 {
        ctx.passthrough_stub(m, node)?;
        return Ok(true);
    }
    let axis = normalize_axis(
        node.attrs.get("axis").and_then(|v| v.as_i64()).unwrap_or(0),
        rank,
    );
    let dim = dim_usize(m.shape(x).dim(axis), ctx.opts);
    let n_out = node.outputs.len().max(1);
    let sizes: Vec<usize> = node
        .inputs
        .get(1)
        .filter(|s| !s.is_empty())
        .and_then(|n| {
            i64_tensor(&ctx.i64_params, &ctx.params, n)
                .or_else(|| eval_static_shape_vector(ctx, m, n, 0))
        })
        .or_else(|| {
            node.attrs
                .get("split")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_i64()).collect())
        })
        .map(|v: Vec<i64>| v.iter().map(|&s| s.max(0) as usize).collect())
        .unwrap_or_else(|| {
            let base = (dim / n_out).max(1);
            let mut v = vec![base; n_out];
            if let Some(last) = v.last_mut() {
                *last = dim.saturating_sub(base * (n_out - 1)).max(1);
            }
            v
        });
    let mut offset = 0usize;
    for (i, out_name) in node.outputs.iter().enumerate() {
        let size_i = sizes.get(i).copied().unwrap_or(0);
        if !out_name.is_empty() {
            let start = offset.min(dim);
            let len = size_i.min(dim.saturating_sub(start)).max(1);
            let id = if ctx.opts.dynamic_sequence {
                let s = m.shape(x);
                let mut out_dims: Vec<Dim> = (0..s.rank()).map(|d| s.dim(d)).collect();
                out_dims[axis] = Dim::Static(len);
                let out_s = Shape::from_dims(&out_dims, s.dtype());
                m.add_node(Op::Narrow { axis, start, len }, vec![x], out_s)
            } else {
                m.narrow_(x, axis, start, len)
            };
            ctx.env.insert(out_name.clone(), id);
        }
        offset += size_i;
    }
    Ok(true)
}

fn lstm_attrs_bytes(hidden_size: usize, bidirectional: bool) -> Vec<u8> {
    let mut v = vec![0u8; 8];
    v[0..4].copy_from_slice(&(hidden_size as u32).to_le_bytes());
    v[4] = u8::from(bidirectional);
    v
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
    // Err (not `Ok(in_s)`) when there are no readable scales, so the caller can
    // fall back to the declared `output_meta` — an `Ok(in_s)` here masqueraded as
    // a real result and pinned some resizes to their INPUT shape.
    let Some(name) = scales_name else {
        return Err(anyhow!("resize: no scales input"));
    };
    let scales = ctx.params.get(name).cloned().or_else(|| {
        ctx.i64_params
            .get(name)
            .map(|v| v.iter().map(|&x| x as f32).collect())
    });
    let Some(scales) = scales else {
        return Err(anyhow!("resize: scales not a readable constant"));
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
