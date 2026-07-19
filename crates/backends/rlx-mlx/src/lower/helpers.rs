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

//! Dequant cache, GDN, reshape/narrow/broadcast helpers.

#![allow(unused_imports)]

use std::collections::{HashMap, HashSet};

use rlx_ir::RegionPrologue;
use rlx_ir::op::{
    Activation, AdaNormKind, BinaryOp, ChainOperand, ChainStep, CmpOp, MaskKind, ReduceOp,
    RopeStyle, ScaleMode, SteKind, TransformStep,
};
use rlx_ir::shape::{Dim, DimBinding, Shape};
use rlx_ir::{DType, Graph, NodeId, Op};

use crate::array::{Array, MlxError, async_eval, eval};
use crate::ffi::{MlxMask, MlxReduce, MlxUnary};
use crate::ops;

use super::*;

pub(crate) fn build_sliding_window_mask(
    s_q: i32,
    s_k: i32,
    window: i32,
) -> Result<Array, MlxError> {
    let neg_inf = f32::NEG_INFINITY;
    let s_q = s_q as usize;
    let s_k = s_k as usize;
    let w = window as i64;
    let mut buf = vec![neg_inf; s_q * s_k];
    for qi in 0..s_q {
        for ki in 0..s_k {
            let q = qi as i64;
            let k = ki as i64;
            // Causal + bounded distance.
            if k <= q && (q - k) <= w {
                buf[qi * s_k + ki] = 0.0;
            }
        }
    }
    Array::from_f32_slice(&buf, &[s_q, s_k], DType::F32)
}

pub(crate) fn quant_scheme_to_mlx(scheme: &rlx_ir::QuantScheme) -> Result<(i32, i32), MlxError> {
    use rlx_ir::QuantScheme as Q;
    let bits = scheme.bits_per_element() as i32;
    let gs = match scheme {
        Q::Int8Block { block_size } => *block_size as i32,
        Q::Int8BlockAsym { block_size } => *block_size as i32,
        Q::Int4Block { block_size } => *block_size as i32,
        other => {
            return Err(MlxError(format!(
                "MLX quantized_matmul: unsupported scheme {other:?}"
            )));
        }
    };
    Ok((bits, gs))
}

// ── GGUF dequant cache ──────────────────────────────────────────────────
//
// Each generate() call drives a fresh lower_and_run_typed traversal, so
// the Op::DequantMatMul branch above would otherwise re-dequant the full
// Q4K weight tensor on every dispatch. We key the cache by Param name —
// stable across compilations of the same model — and store the already-
// transposed `[k, n]` Array so the matmul gets MLX's tuned f32 matmul
// without per-call setup.
//
// The cache lives for the process lifetime; Arrays are reference-counted
// (cloning is free). Off-switch: `RLX_MLX_DEQUANT_CACHE_DISABLE=1`.
use std::sync::Mutex;
use std::sync::OnceLock;

pub(crate) fn dequant_cache_disabled() -> bool {
    crate::config::runtime_config().dequant_cache_disable
}

/// Byte budget for the dequanted-weight cache. Default 6 GiB.
pub(crate) fn dequant_cache_budget_bytes() -> usize {
    crate::config::runtime_config()
        .dequant_cache_bytes
        .unwrap_or(6usize * 1024 * 1024 * 1024)
}

#[derive(Default)]
pub(crate) struct DequantCache {
    /// key → (dequanted f32 weight, byte size)
    map: HashMap<String, (Array, usize)>,
    /// insertion order for FIFO eviction
    order: std::collections::VecDeque<String>,
    bytes: usize,
}

pub(crate) fn dequant_cache() -> &'static Mutex<DequantCache> {
    static CACHE: OnceLock<Mutex<DequantCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(DequantCache::default()))
}

pub(crate) fn mlx_dequant_cache_key(
    name: &str,
    k: usize,
    n: usize,
    scheme: &rlx_ir::QuantScheme,
    w_bytes: &[u8],
) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    w_bytes.hash(&mut hasher);
    format!("{name}#kn:{k}x{n}:{scheme:?}:{}", hasher.finish())
}

pub(crate) fn mlx_dequant_cache_get(key: &str) -> Result<Option<Array>, MlxError> {
    if dequant_cache_disabled() {
        return Ok(None);
    }
    let guard = match dequant_cache().lock() {
        Ok(g) => g,
        Err(_) => return Ok(None),
    };
    match guard.map.get(key) {
        Some((a, _)) => Ok(Some(a.clone_handle()?)),
        None => Ok(None),
    }
}

pub(crate) fn mlx_dequant_cache_put(key: String, arr: Array, nbytes: usize) {
    if dequant_cache_disabled() {
        return;
    }
    let budget = dequant_cache_budget_bytes();
    // A single weight larger than the whole budget can never be cached — skip
    // it (it'll be re-dequanted on demand) rather than evicting everything.
    if nbytes > budget {
        return;
    }
    if let Ok(mut c) = dequant_cache().lock() {
        if c.map.contains_key(&key) {
            return;
        }
        // FIFO-evict until the new entry fits under budget.
        while c.bytes + nbytes > budget {
            let Some(old) = c.order.pop_front() else {
                break;
            };
            if let Some((_, old_bytes)) = c.map.remove(&old) {
                c.bytes = c.bytes.saturating_sub(old_bytes);
            }
        }
        c.map.insert(key.clone(), (arr, nbytes));
        c.order.push_back(key);
        c.bytes += nbytes;
    }
}

pub(crate) fn build_dequanted_kn(
    w_bytes: &[u8],
    k: usize,
    n: usize,
    scheme: &rlx_ir::QuantScheme,
) -> Result<Array, MlxError> {
    let block_bytes = scheme.gguf_block_bytes() as usize;
    let block_elems = scheme.gguf_block_size() as usize;
    let blocks_actual = w_bytes.len() / block_bytes;
    let elems_actual = blocks_actual * block_elems;
    let elems_required = k * n;
    let elems_for_dequant = elems_required.min(elems_actual);
    let mut w_f32 = match scheme {
        rlx_ir::QuantScheme::GgufQ4K => rlx_gguf::dequant_q4_k(w_bytes, elems_for_dequant)
            .map_err(|e| MlxError(format!("GGUF Q4_K dequant: {e}")))?,
        _ => dequant_gguf_weight(w_bytes, k, n, scheme)?,
    };
    if w_f32.len() < elems_required {
        w_f32.resize(elems_required, 0.0);
    }
    let w_nk = Array::from_f32_slice(&w_f32, &[n, k], DType::F32)?;
    ops::transpose(&w_nk, &[1, 0])
}

pub(crate) fn dequant_gguf_weight(
    w_bytes: &[u8],
    k: usize,
    n: usize,
    scheme: &rlx_ir::QuantScheme,
) -> Result<Vec<f32>, MlxError> {
    use rlx_ir::QuantScheme as Q;
    let elems = k * n;
    match scheme {
        Q::GgufQ4K => rlx_gguf::dequant_q4_k(w_bytes, elems)
            .map_err(|e| MlxError(format!("GGUF Q4_K dequant: {e}"))),
        Q::GgufQ5K => rlx_gguf::dequant_q5_k(w_bytes, elems)
            .map_err(|e| MlxError(format!("GGUF Q5_K dequant: {e}"))),
        Q::GgufQ6K => rlx_gguf::dequant_q6_k(w_bytes, elems)
            .map_err(|e| MlxError(format!("GGUF Q6_K dequant: {e}"))),
        Q::GgufQ8K => rlx_gguf::dequant_q8_k(w_bytes, elems)
            .map_err(|e| MlxError(format!("GGUF Q8_K dequant: {e}"))),
        Q::GgufQ2K => rlx_gguf::dequant_q2_k(w_bytes, elems)
            .map_err(|e| MlxError(format!("GGUF Q2_K dequant: {e}"))),
        Q::GgufQ3K => rlx_gguf::dequant_q3_k(w_bytes, elems)
            .map_err(|e| MlxError(format!("GGUF Q3_K dequant: {e}"))),
        Q::GgufQ4_0 => rlx_gguf::dequant_q4_0(w_bytes, elems)
            .map_err(|e| MlxError(format!("GGUF Q4_0 dequant: {e}"))),
        Q::GgufQ4_1 => rlx_gguf::dequant_q4_1(w_bytes, elems)
            .map_err(|e| MlxError(format!("GGUF Q4_1 dequant: {e}"))),
        Q::GgufQ5_0 => rlx_gguf::dequant_q5_0(w_bytes, elems)
            .map_err(|e| MlxError(format!("GGUF Q5_0 dequant: {e}"))),
        Q::GgufQ5_1 => rlx_gguf::dequant_q5_1(w_bytes, elems)
            .map_err(|e| MlxError(format!("GGUF Q5_1 dequant: {e}"))),
        Q::GgufQ8_0 => rlx_gguf::dequant_q8_0(w_bytes, elems)
            .map_err(|e| MlxError(format!("GGUF Q8_0 dequant: {e}"))),
        Q::GgufIQ4NL => rlx_gguf::iq_dequant::dequant_iq4_nl(w_bytes, elems)
            .map_err(|e| MlxError(format!("GGUF IQ4_NL dequant: {e}"))),
        Q::GgufIQ4XS => rlx_gguf::iq_dequant::dequant_iq4_xs(w_bytes, elems)
            .map_err(|e| MlxError(format!("GGUF IQ4_XS dequant: {e}"))),
        Q::GgufIQ2XXS => rlx_gguf::iq_dequant::dequant_iq2_xxs(w_bytes, elems)
            .map_err(|e| MlxError(format!("GGUF IQ2_XXS dequant: {e}"))),
        Q::GgufIQ2XS => rlx_gguf::iq_dequant::dequant_iq2_xs(w_bytes, elems)
            .map_err(|e| MlxError(format!("GGUF IQ2_XS dequant: {e}"))),
        Q::GgufIQ2S => rlx_gguf::iq_dequant::dequant_iq2_s(w_bytes, elems)
            .map_err(|e| MlxError(format!("GGUF IQ2_S dequant: {e}"))),
        Q::GgufIQ3XXS => rlx_gguf::iq_dequant::dequant_iq3_xxs(w_bytes, elems)
            .map_err(|e| MlxError(format!("GGUF IQ3_XXS dequant: {e}"))),
        Q::GgufIQ3S => rlx_gguf::iq_dequant::dequant_iq3_s(w_bytes, elems)
            .map_err(|e| MlxError(format!("GGUF IQ3_S dequant: {e}"))),
        Q::GgufIQ1S => rlx_gguf::iq_dequant::dequant_iq1_s(w_bytes, elems)
            .map_err(|e| MlxError(format!("GGUF IQ1_S dequant: {e}"))),
        Q::GgufIQ1M => rlx_gguf::iq_dequant::dequant_iq1_m(w_bytes, elems)
            .map_err(|e| MlxError(format!("GGUF IQ1_M dequant: {e}"))),
        Q::GgufTQ1_0 => rlx_gguf::tq_dequant::dequant_tq1_0(w_bytes, elems)
            .map_err(|e| MlxError(format!("GGUF TQ1_0 dequant: {e}"))),
        Q::GgufTQ2_0 => rlx_gguf::tq_dequant::dequant_tq2_0(w_bytes, elems)
            .map_err(|e| MlxError(format!("GGUF TQ2_0 dequant: {e}"))),
        Q::GgufMXFP4 => rlx_gguf::mx_dequant::dequant_mxfp4(w_bytes, elems)
            .map_err(|e| MlxError(format!("GGUF MXFP4 dequant: {e}"))),
        Q::GgufNVFP4 => rlx_gguf::mx_dequant::dequant_nvfp4(w_bytes, elems)
            .map_err(|e| MlxError(format!("GGUF NVFP4 dequant: {e}"))),
        Q::GgufQ1_0 => rlx_gguf::q1_dequant::dequant_q1_0(w_bytes, elems)
            .map_err(|e| MlxError(format!("GGUF Q1_0 dequant: {e}"))),
        Q::GgufQ2_0 => rlx_gguf::q2_dequant::dequant_q2_0(w_bytes, elems)
            .map_err(|e| MlxError(format!("GGUF Q2_0 dequant: {e}"))),
        other => Err(MlxError(format!(
            "MLX DequantMatMul: unsupported GGUF scheme {other:?}"
        ))),
    }
}

/// Lower `Op::GatedDeltaNet` by unrolling the time loop into MLX
/// primitives (same strategy as [`Op::SelectiveScan`]).
///
/// When `state_in` is `Some`, threads recurrent state in/out (written
/// back by the caller to the state input node).
pub(crate) fn lower_gated_delta_net(
    q: &Array,
    k: &Array,
    v: &Array,
    g_in: &Array,
    beta: &Array,
    state_size: usize,
    state_in: Option<&Array>,
    q_shape: Vec<i32>,
) -> Result<(Array, Option<Array>), MlxError> {
    if q_shape.len() != 4 {
        return Err(MlxError(format!(
            "GatedDeltaNet: q must be rank-4 [B, S, H, N], got rank {}",
            q_shape.len()
        )));
    }
    let batch = q_shape[0];
    let seq = q_shape[1];
    let heads = q_shape[2];
    let n = state_size as i32;
    if n != q_shape[3] {
        return Err(MlxError(format!(
            "GatedDeltaNet: state_size={state_size} != q last dim {}",
            q_shape[3]
        )));
    }
    let bh = batch * heads;

    let mut state = if let Some(s0) = state_in {
        s0.clone_handle()?
    } else {
        let zero = Array::from_f32_slice(&[0.0], &[1], DType::F32)?;
        ops::broadcast_to(&zero, &[batch, heads, n, n])?
    };

    let scale = 1.0f32 / (n as f32).sqrt();
    let scale_arr = Array::from_f32_slice(&[scale], &[1], DType::F32)?;

    let mut ys: Vec<Array> = Vec::with_capacity(seq as usize);
    for t in 0..seq {
        let qt = ops::slice(q, &[0, t, 0, 0], &[batch, t + 1, heads, n])?;
        let kt = ops::slice(k, &[0, t, 0, 0], &[batch, t + 1, heads, n])?;
        let vt = ops::slice(v, &[0, t, 0, 0], &[batch, t + 1, heads, n])?;
        let gt = ops::slice(g_in, &[0, t, 0], &[batch, t + 1, heads])?;
        let beta_t = ops::slice(beta, &[0, t, 0], &[batch, t + 1, heads])?;

        let gt = ops::reshape(&gt, &[batch, heads, 1, 1])?;
        let beta_bh = ops::reshape(&beta_t, &[bh, 1, 1])?;
        let exp_g = ops::unary(&gt, MlxUnary::Exp)?;
        state = ops::mul(&state, &exp_g)?;

        let state_bh = ops::reshape(&state, &[bh, n, n])?;
        let kt_bh = ops::reshape(&kt, &[bh, 1, n])?;
        let vt_bh = ops::reshape(&vt, &[bh, 1, n])?;

        let mut sk = ops::matmul(&kt_bh, &state_bh)?;
        sk = ops::sub(&vt_bh, &sk)?;
        sk = ops::mul(&sk, &beta_bh)?;

        let kt_col = ops::reshape(&kt, &[bh, n, 1])?;
        let sk_row = ops::reshape(&sk, &[bh, 1, n])?;
        let outer = ops::mul(&kt_col, &sk_row)?;
        state = ops::add(&state, &ops::reshape(&outer, &[batch, heads, n, n])?)?;

        let state_bh = ops::reshape(&state, &[bh, n, n])?;
        let qt_bh = ops::reshape(&qt, &[bh, 1, n])?;
        let mut out_t = ops::matmul(&qt_bh, &state_bh)?;
        out_t = ops::mul(&out_t, &scale_arr)?;
        out_t = ops::reshape(&out_t, &[batch, 1, heads, n])?;
        ys.push(out_t);
    }

    let refs: Vec<&Array> = ys.iter().collect();
    let out = ops::concat(&refs, 1)?;
    Ok((out, state_in.map(|_| state)))
}

pub(crate) fn node_input_shape(graph: &Graph, id: NodeId) -> Vec<i32> {
    graph
        .node(id)
        .shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static() as i32)
        .collect()
}

/// Same ceiling as `rlx-cpu` `IM2COL_MAX_COL_ELEMS` — refuse MLX im2col when the
/// col matrix would exceed ~2 GB of f32 scratch (ISTFT-as-conv after CT decompose).
pub(crate) const MLX_IM2COL_MAX_COL_ELEMS: usize = 512 * 1024 * 1024;

pub(crate) fn mlx_conv_im2col_too_large(
    graph: &Graph,
    node: &rlx_ir::Node,
    kernel_size: &[usize],
    groups: usize,
) -> bool {
    let in_shape = node_input_shape(graph, node.inputs[0]);
    if in_shape.len() < 3 {
        return false;
    }
    let c_in = in_shape[1].max(0) as usize;
    let g = groups.max(1);
    let c_in_per_g = c_in / g;
    let k: usize = kernel_size.iter().copied().product();
    let out_spatial: usize = node
        .shape
        .dims()
        .iter()
        .skip(2)
        .map(|d| d.unwrap_static())
        .product();
    let col_elems = c_in_per_g.saturating_mul(k).saturating_mul(out_spatial);
    col_elems > MLX_IM2COL_MAX_COL_ELEMS
}

/// Sum broadcast axes of `grad` (shape `full_dims`) down to `target_dims`.
pub(crate) fn mlx_unbroadcast_grad(
    grad: &Array,
    full_dims: &[i32],
    target_dims: &[i32],
) -> Result<Array, MlxError> {
    if full_dims == target_dims {
        return grad.clone_handle();
    }
    let g_rank = full_dims.len();
    let t_rank = target_dims.len();
    let extra = g_rank.saturating_sub(t_rank);
    let mut axes: Vec<i32> = (0..extra).map(|i| i as i32).collect();
    for i in 0..t_rank {
        if target_dims[i] == 1 && full_dims[extra + i] > 1 {
            axes.push((extra + i) as i32);
        }
    }
    let mut current = grad.clone_handle()?;
    for &ax in &axes {
        current = ops::reduce(&current, MlxReduce::Sum, &[ax], true)?;
    }
    if current
        .shape()?
        .iter()
        .map(|&d| d as i32)
        .collect::<Vec<_>>()
        != target_dims
    {
        ops::reshape(&current, target_dims)
    } else {
        Ok(current)
    }
}

/// Flatten each grad to 1-D and concat on axis 0 (packed DiT reverse layout).
pub(crate) fn mlx_pack_flat_grads(grads: &[Array]) -> Result<Array, MlxError> {
    let mut flats = Vec::with_capacity(grads.len());
    for g in grads {
        let n = g.num_elements()? as i32;
        flats.push(ops::reshape(g, &[n])?);
    }
    let refs: Vec<&Array> = flats.iter().collect();
    ops::concat(&refs, 0)
}

/// ONNX `Expand`: broadcast input to the **output node's** shape (not the
/// op's `target_shape` hint, which can be a lower-rank broadcast template).
/// Mirrors CPU/Metal leading-1 padding + stride-0 broadcast semantics.
pub(crate) fn mlx_expand(
    graph: &Graph,
    input_id: NodeId,
    out_node: &rlx_ir::Node,
    x: &Array,
) -> Result<Array, MlxError> {
    let x_rt: Vec<i32> = x.shape()?.iter().map(|&d| d as i32).collect();
    let mut out_graph = node_input_shape(graph, out_node.id);
    let in_graph = node_input_shape(graph, input_id);
    let in_dims = if x_rt.len() == in_graph.len() || !x_rt.is_empty() {
        x_rt.clone()
    } else {
        in_graph.clone()
    };

    // Align padded compile seq (512) with active runtime seq on output shape.
    if in_dims.len() == out_graph.len() {
        for i in 0..in_dims.len() {
            if in_dims[i] > 1 && out_graph[i] > 1 && in_dims[i] != out_graph[i] {
                out_graph[i] = mlx_pick_seq_dim(in_dims[i] as usize, out_graph[i] as usize) as i32;
            }
        }
    }

    let pad = out_graph.len().saturating_sub(in_dims.len());
    let mut padded: Vec<i32> = vec![1; pad];
    padded.extend_from_slice(&in_dims);

    let mut x_adj = x.clone_handle()?;
    for i in 0..padded.len().min(out_graph.len()) {
        if padded[i] > out_graph[i] && out_graph[i] > 1 {
            let rt = x_adj.shape()?;
            if rt.len() == padded.len() {
                let start = vec![0i32; rt.len()];
                let mut stop: Vec<i32> = rt.iter().map(|&d| d as i32).collect();
                stop[i] = out_graph[i];
                x_adj = ops::slice(&x_adj, &start, &stop)?;
                padded[i] = out_graph[i];
            }
        } else if padded[i] != out_graph[i] && padded[i] != 1 && out_graph[i] != 1 {
            return Err(MlxError(format!(
                "Expand: incompatible dim {i} (in={in_d}, out={out_d})",
                in_d = padded[i],
                out_d = out_graph[i]
            )));
        }
    }
    let x = if pad > 0 {
        ops::reshape(&x_adj, &padded)?
    } else {
        x_adj
    };
    ops::broadcast_to(&x, &out_graph)
}

/// Element-wise add with seq/layout alignment for rank-3 Kitten tensors.
pub(crate) fn mlx_add_aligned(a: &Array, b: &Array) -> Result<Array, MlxError> {
    if a.shape()?.len() == 3 && b.shape()?.len() == 3 {
        let (a, b) = mlx_align_rank3_seq_pair(a, b)?;
        return ops::add(&a, &b);
    }
    ops::add(a, b)
}

/// Pick a common seq axis for broadcast: padded compile tables (e.g. 512)
/// narrow to the active row; two compile-scale mismatches (12 vs 14) widen
/// to the larger compile slot count.
pub(crate) fn mlx_pick_seq_dim(a: usize, b: usize) -> usize {
    let (big, small) = if a > b { (a, b) } else { (b, a) };
    if big > 128 && small <= 128 {
        return small;
    }
    if a.max(b) > 128 { a.min(b) } else { a.max(b) }
}

/// When compile-time reshape targets embed padded mel/time axes (514) but
/// runtime tensors are shorter, infer the matching dim from element count.
///
/// Only accepts *small* corrections (absolute ≤16 or relative ≤5%). Large
/// mismatches (e.g. StyleTTS2 `[1,256,1040]` vs a `[1,256,256]` producer) must
/// surface as reshape errors instead of silently lying about the runtime shape
/// and blowing up later in an `ElementwiseRegion` broadcast.
pub(crate) fn mlx_fix_reshape_shape(in_shape: &[usize], target: &[i64]) -> Vec<i32> {
    let target: Vec<i32> = target.iter().map(|&d| d as i32).collect();
    let in_n: i64 = in_shape.iter().map(|&d| d as i64).product::<i64>();
    let out_n: i64 = target.iter().map(|&d| i64::from(d.max(1))).product::<i64>();
    if in_n == out_n {
        return target;
    }
    let small_fix = |declared: i32, candidate: i32| -> bool {
        if declared <= 0 || candidate <= 0 {
            return false;
        }
        let abs = (declared - candidate).unsigned_abs();
        if abs <= 16 {
            return true;
        }
        let ratio = candidate as f32 / declared as f32;
        (ratio - 1.0).abs() <= 0.05
    };
    let mut out = target.clone();
    for i in 0..out.len() {
        if out[i] <= 128 {
            continue;
        }
        let rest: i64 = out
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, &d)| i64::from(d.max(1)))
            .product::<i64>()
            .max(1);
        if rest > 0 && in_n % rest == 0 {
            let candidate = (in_n / rest) as i32;
            if small_fix(out[i], candidate) {
                out[i] = candidate;
                let check: i64 = out.iter().map(|&d| i64::from(d.max(1))).product::<i64>();
                if check == in_n {
                    return out;
                }
                out[i] = target[i];
            }
        }
    }
    for i in 0..out.len() {
        let rest: i64 = out
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, &d)| i64::from(d.max(1)))
            .product::<i64>()
            .max(1);
        if rest > 0 && in_n % rest == 0 {
            let mut trial = out.clone();
            let candidate = (in_n / rest) as i32;
            if !small_fix(trial[i], candidate) {
                continue;
            }
            trial[i] = candidate;
            if trial.iter().map(|&d| i64::from(d.max(1))).product::<i64>() == in_n {
                return trial;
            }
        }
    }
    target
}

/// Narrow `[1,S,C]` (or transpose of `[S,1,C]`) to `len` on axis 1.
pub(crate) fn mlx_narrow_axis1(arr: &Array, len: usize) -> Result<Array, MlxError> {
    let rt = arr.shape()?;
    if rt.len() != 3 || rt[1] <= len {
        return arr.clone_handle();
    }
    let start = vec![0i32; 3];
    let mut stop: Vec<i32> = rt.iter().map(|&d| d as i32).collect();
    stop[1] = len as i32;
    ops::slice(arr, &start, &stop)
}

/// Normalize rank-3 tensors to batch-major; seq-first `[S,1,C]` → `[1,S,C]`.
/// Feature-first `[H,1,L]` (H > 128) → `[1,L,H]`.
/// True if two equal-rank shapes broadcast under standard NumPy rules
/// (each aligned axis pair is equal, or one side is 1).
pub(crate) fn rank3_broadcasts_cleanly(a: &[usize], b: &[usize]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(&x, &y)| x == y || x == 1 || y == 1)
}

pub(crate) fn mlx_batch_major_rank3(arr: &Array) -> Result<Array, MlxError> {
    let s = arr.shape()?;
    if s.len() != 3 {
        return arr.clone_handle();
    }
    if s[0] == 1 {
        return arr.clone_handle();
    }
    if s[1] == 1 {
        if s[0] > 128 && s[2] <= 128 {
            let t = ops::transpose(arr, &[2, 1, 0])?;
            return ops::transpose(&t, &[1, 0, 2]);
        }
        return ops::transpose(arr, &[1, 0, 2]);
    }
    arr.clone_handle()
}

/// Known channel widths in Kitten — do not treat as sequence when paired
/// with a small leading count (e.g. LSTM `num_directions=2`).
pub(crate) const MLX_KITTEN_CHANNEL_DIMS: &[usize] = &[128, 256, 512, 768, 1024];

/// Which axis carries token/time in `[1, ?, ?]` (if any).
pub(crate) fn mlx_rank3_seq_axis(s: &[usize]) -> Option<usize> {
    if s.len() != 3 || s[0] != 1 {
        return None;
    }
    // BERT-style `[1, S, 128|256|512]` (padded seq on axis 1).
    if MLX_KITTEN_CHANNEL_DIMS.contains(&s[2]) && s[1] != s[2] {
        return Some(1);
    }
    if s[1] > 128 && s[2] == 1 {
        return Some(1);
    }
    if s[1] <= 128 && s[2] > 128 {
        return Some(1);
    }
    if s[2] <= 128 && s[1] > 128 {
        return Some(2);
    }
    if s[1] <= 128 && s[2] <= 128 {
        return Some(1);
    }
    None
}

pub(crate) fn mlx_narrow_rank3_axis(
    arr: &Array,
    axis: usize,
    len: usize,
) -> Result<Array, MlxError> {
    let rt = arr.shape()?;
    if rt.len() != 3 || rt[axis] <= len {
        return arr.clone_handle();
    }
    let start = vec![0i32; 3];
    let mut stop: Vec<i32> = rt.iter().map(|&d| d as i32).collect();
    stop[axis] = len as i32;
    ops::slice(arr, &start, &stop)
}

pub(crate) fn mlx_looks_like_channel_vs_small(a1: usize, b1: usize) -> bool {
    let (big, small) = if a1 > b1 { (a1, b1) } else { (b1, a1) };
    // LSTM `num_directions`, harmonic counts — not token rows.
    const STRUCTURAL_SMALL: &[usize] = &[1, 2, 4, 9];
    MLX_KITTEN_CHANNEL_DIMS.contains(&big) && STRUCTURAL_SMALL.contains(&small)
}

/// Align seq-first `[S,1,C]` with batch-major `[1,S,C]`, and narrow padded
/// compile-time seq (e.g. 512) to the active runtime seq before broadcast.
pub(crate) fn mlx_align_rank3_seq_pair(a: &Array, b: &Array) -> Result<(Array, Array), MlxError> {
    let as_ = a.shape()?;
    let bs = b.shape()?;
    if as_.len() != 3 || bs.len() != 3 {
        return Ok((a.clone_handle()?, b.clone_handle()?));
    }
    // If the operands already broadcast under standard NumPy rules, no
    // seq-alignment is needed — and applying the heuristic would destructively
    // reorder a valid pair. The alignment exists only for padded-decode seq
    // buckets (a stale compile seq vs the active runtime seq), which never
    // broadcast cleanly. Example it used to break: the OCR recognition head,
    // logits [S,1,C] + per-class bias [1,1,C] with S>128 and C<=128 —
    // `mlx_batch_major_rank3` flips [163,1,97] to [1,97,163], so the [1,1,97]
    // bias can no longer broadcast.
    if rank3_broadcasts_cleanly(&as_, &bs) {
        return Ok((a.clone_handle()?, b.clone_handle()?));
    }
    let mut a = mlx_batch_major_rank3(a)?;
    let mut b = mlx_batch_major_rank3(b)?;
    let as_ = a.shape()?;
    let bs = b.shape()?;
    if as_[0] != 1 || bs[0] != 1 {
        return Ok((a, b));
    }
    let Some(a_axis) = mlx_rank3_seq_axis(&as_) else {
        return Ok((a, b));
    };
    let Some(b_axis) = mlx_rank3_seq_axis(&bs) else {
        return Ok((a, b));
    };
    if a_axis != b_axis {
        return Ok((a, b));
    }
    let a_len = as_[a_axis];
    let b_len = bs[b_axis];
    if a_len == b_len {
        return Ok((a, b));
    }
    // A size-1 axis is a NumPy broadcast (e.g. a per-feature bias `[1,1,C]`
    // added to `[1,S,C]`), never a padded compile-time seq table (those are
    // >1, like 512). Narrowing the larger operand to 1 here would collapse a
    // real sequence — so let MLX broadcast it natively instead.
    if a_len == 1 || b_len == 1 {
        return Ok((a, b));
    }
    if mlx_looks_like_channel_vs_small(a_len, b_len) {
        return Ok((a, b));
    }
    let seq = mlx_pick_seq_dim(a_len, b_len);
    a = mlx_narrow_rank3_axis(&a, a_axis, seq)?;
    b = mlx_narrow_rank3_axis(&b, b_axis, seq)?;
    Ok((a, b))
}

/// Align concat inputs that mix padded compile seq with active runtime seq.
///
/// This equalizes the *seq* axis of rank-3 inputs to the active runtime
/// length (padded-decode buckets leave stale, longer compile-time seq dims).
/// It must NEVER touch the axis actually being concatenated — sizes along the
/// concat axis are intentionally different (e.g. pooling `[1,1,C]` ++ tokens
/// `[1,N,C]` along axis 1 produces `[1,1+N,C]`), and narrowing it silently
/// drops the data being concatenated.
pub(crate) fn mlx_align_concat_inputs(
    inputs: &[&Array],
    axis: usize,
) -> Result<Vec<Array>, MlxError> {
    let mut out: Vec<Array> = inputs
        .iter()
        .map(|a| a.clone_handle())
        .collect::<Result<_, _>>()?;
    let mut min_seq: Option<usize> = None;
    for a in &out {
        let s = a.shape()?;
        if s.len() == 3 {
            // The "seq" axis is 1 when batch-leading (`[1,S,C]`) or 0 when
            // `[S,1,C]`. Skip it when it is the concat axis.
            let (seq_axis, seq) = if s[0] == 1 {
                (1usize, s[1])
            } else if s[1] == 1 {
                (0usize, s[0])
            } else {
                continue;
            };
            if seq_axis == axis {
                continue;
            }
            min_seq = Some(min_seq.map_or(seq, |m| mlx_pick_seq_dim(m, seq)));
        }
    }
    let Some(seq) = min_seq else {
        return Ok(out);
    };
    for a in &mut out {
        let s = a.shape()?;
        if s.len() != 3 {
            continue;
        }
        if s[0] == 1 && s[1] > seq && axis != 1 {
            *a = mlx_narrow_axis1(a, seq)?;
        } else if s[1] == 1 && s[0] > seq && axis != 0 {
            let t = ops::transpose(a, &[1, 0, 2])?;
            *a = mlx_narrow_axis1(&t, seq)?;
        }
    }
    Ok(out)
}

/// Host-eval ScatterNd / Gather* via the native CPU indexing thunk.
///
/// Unlike [`host_eval_op_f32`], this keeps `I64`/`I32` index buffers as integer
/// bytes. The old path ran `to_f32()` on every input then
/// `run_host_op_node_f32`, so ScatterNd compiled with `indices_i64=1` and
/// re-read float bits as `i64` — corrupting F5's 88 rope scatters per DiT step
/// and causing multi-step drift vs Metal (which uses native CpuIndexing).
pub(crate) fn host_eval_indexing_op(
    graph: &Graph,
    node: &rlx_ir::Node,
    env: &HashMap<NodeId, Array>,
) -> Result<Array, MlxError> {
    let mut arena: Vec<u8> = Vec::new();
    let mut offs: HashMap<NodeId, usize> = HashMap::new();

    let pad8 = |a: &mut Vec<u8>| {
        while !a.len().is_multiple_of(8) {
            a.push(0);
        }
    };

    for &in_id in &node.inputs {
        let arr = ops::contiguous(lookup(env, in_id)?)?;
        let sh = &graph.node(in_id).shape;
        let n = sh.num_elements().unwrap_or(0);
        pad8(&mut arena);
        let off = arena.len();
        offs.insert(in_id, off);
        match sh.dtype() {
            DType::I64 | DType::I32 => {
                arena.extend(materialize_index_i64_bytes(&arr, n)?);
            }
            _ => {
                let f = arr.to_f32()?;
                arena.extend(f.iter().flat_map(|v| v.to_le_bytes()));
            }
        }
    }

    let out_n = node.shape.num_elements().unwrap_or(0);
    pad8(&mut arena);
    let out_off = arena.len();
    arena.resize(out_off + out_n * 4, 0);

    let indices_int = node
        .inputs
        .get(1)
        .map(|&id| matches!(graph.node(id).shape.dtype(), DType::I64 | DType::I32))
        .unwrap_or(false);

    let thunk = rlx_cpu::thunk::indexing_thunk_from_node(graph, node, |id| {
        if id == node.id {
            out_off
        } else {
            *offs.get(&id).expect("indexing host: missing input offset")
        }
    });
    // We staged I64/I32 index inputs as i64 bytes — force the i64 reader even
    // if prepare_f32 left a float-looking leaf somewhere upstream.
    let thunk = if indices_int {
        force_indexing_indices_i64(thunk)
    } else {
        thunk
    };
    unsafe {
        rlx_cpu::thunk::execute_indexing_thunk_on_bytes(arena.as_mut_ptr(), &thunk);
    }
    let out: Vec<f32> = arena[out_off..out_off + out_n * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let out_shape: Vec<usize> = node
        .shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect();
    Array::from_f32_slice(&out, &out_shape, DType::F32)
}

/// Read index tensors as little-endian i64 bytes (one i64 per element).
/// Handles MLX arrays that are truly I64 or already float-encoded integers.
pub(crate) fn materialize_index_i64_bytes(arr: &Array, n: usize) -> Result<Vec<u8>, MlxError> {
    if let Ok(f) = arr.to_f32() {
        if f.len() == n {
            return Ok(f
                .iter()
                .flat_map(|&x| (x.round() as i64).to_le_bytes())
                .collect());
        }
    }
    let b = arr.to_bytes()?;
    if b.len() == n * 8 {
        Ok(b)
    } else if b.len() == n * 4 {
        Ok(b.chunks_exact(4)
            .flat_map(|c| {
                let x = f32::from_le_bytes(c.try_into().unwrap());
                (x.round() as i64).to_le_bytes()
            })
            .collect())
    } else {
        Err(MlxError(format!(
            "indexing host: index buffer len {} for {n} elems (expected {} or {})",
            b.len(),
            n * 8,
            n * 4
        )))
    }
}

pub(crate) fn force_indexing_indices_i64(thunk: rlx_cpu::thunk::Thunk) -> rlx_cpu::thunk::Thunk {
    match thunk {
        rlx_cpu::thunk::Thunk::ScatterNd {
            data,
            indices,
            updates,
            dst,
            data_shape,
            indices_shape,
            data_len,
            updates_len,
            indices_len,
            reduction,
            ..
        } => rlx_cpu::thunk::Thunk::ScatterNd {
            data,
            indices,
            updates,
            dst,
            data_shape,
            indices_shape,
            data_len,
            updates_len,
            indices_len,
            indices_i64: 1,
            reduction,
        },
        rlx_cpu::thunk::Thunk::ScatterElements {
            data,
            indices,
            updates,
            dst,
            data_shape,
            data_len,
            updates_len,
            indices_len,
            axis,
            reduction,
            ..
        } => rlx_cpu::thunk::Thunk::ScatterElements {
            data,
            indices,
            updates,
            dst,
            data_shape,
            data_len,
            updates_len,
            indices_len,
            indices_i64: 1,
            axis,
            reduction,
        },
        rlx_cpu::thunk::Thunk::GatherNd {
            data,
            indices,
            dst,
            data_shape,
            indices_shape,
            data_len,
            indices_len,
            out_len,
            batch_dims,
            ..
        } => rlx_cpu::thunk::Thunk::GatherNd {
            data,
            indices,
            dst,
            data_shape,
            indices_shape,
            data_len,
            indices_len,
            out_len,
            indices_i64: 1,
            batch_dims,
        },
        rlx_cpu::thunk::Thunk::GatherElements {
            data,
            indices,
            dst,
            data_shape,
            indices_shape,
            data_len,
            indices_len,
            out_len,
            data_elem_bytes,
            axis,
            ..
        } => rlx_cpu::thunk::Thunk::GatherElements {
            data,
            indices,
            dst,
            data_shape,
            indices_shape,
            data_len,
            indices_len,
            out_len,
            indices_i64: 1,
            data_elem_bytes,
            axis,
        },
        other => other,
    }
}

/// Host-eval a single nested-body / indexing op via the shared CPU path.
pub(crate) fn host_eval_op_f32(
    graph: &Graph,
    node: &rlx_ir::Node,
    env: &HashMap<NodeId, Array>,
) -> Result<Array, MlxError> {
    let mut vals = std::collections::HashMap::new();
    for &in_id in &node.inputs {
        vals.insert(in_id, ops::contiguous(lookup(env, in_id)?)?.to_f32()?);
    }
    let out = rlx_cpu::thunk::run_host_op_node_f32(graph, node, |id| {
        vals.get(&id).cloned().unwrap_or_default()
    });
    let out_shape: Vec<usize> = node
        .shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect();
    Array::from_f32_slice(&out, &out_shape, DType::F32)
}

/// Materialize gather/scatter indices as I64 (bundle params and TopK
/// outputs are often F32 at the MLX lazy boundary).
pub(crate) fn mlx_indices_i64(idx: &Array) -> Result<Array, MlxError> {
    // Lazy index conversion — must NOT host-eval the index array. A
    // `to_bytes()` round-trip here forces evaluation, which is forbidden
    // inside `mlx::compile` ("Attempting to eval an array during function
    // transformations…") and made any graph with a dynamic-index
    // `Op::Gather` (e.g. token-embedding lookups) crash in Compiled mode.
    // Gather indices are integer-valued, so the truncating f32→i64 cast
    // matches the previous host-side round while staying fully lazy.
    ops::contiguous(&ops::cast(idx, DType::I64)?)
}

/// MLX `layer_norm` / `rms_norm` expect 1-D scale vectors; graph params may
/// be broadcast-expanded to rank 3 (`[1,1,C]` or `[1,S,C]`).
pub(crate) fn mlx_norm_scale_1d(w: &Array) -> Result<Array, MlxError> {
    let s = w.shape()?;
    if s.len() == 1 {
        return w.clone_handle();
    }
    if let Some(&last) = s.last() {
        return ops::reshape(w, &[last as i32]);
    }
    w.clone_handle()
}

/// Prefer runtime shape when rank matches the graph (dynamic seq
/// specialization can leave stale static dims in the IR).
pub(crate) fn runtime_shape_or_graph(
    arr: &Array,
    graph_shape: &[i32],
) -> Result<Vec<i32>, MlxError> {
    let rt = arr.shape()?;
    if rt.len() == graph_shape.len() {
        Ok(rt.iter().map(|&d| d as i32).collect())
    } else {
        Ok(graph_shape.to_vec())
    }
}

/// Batch/seq from runtime hidden `[B,S,H]` when available (graph dims can
/// lag dynamic specialization); fall back to graph shape otherwise.
pub(crate) fn runtime_bsh_dims(hidden: &Array, graph_h: &[i32]) -> Result<(i32, i32), MlxError> {
    let rt = hidden.shape()?;
    if rt.len() == 3 {
        Ok((rt[0] as i32, rt[1] as i32))
    } else if graph_h.len() == 3 {
        Ok((graph_h[0], graph_h[1]))
    } else {
        Err(MlxError(format!(
            "runtime_bsh_dims: expected rank-3 hidden, got runtime {rt:?} graph {graph_h:?}"
        )))
    }
}

/// When the graph flattened leading dims (e.g. `[batch*seq, K]`) but MLX
/// still carries them as `[1, batch*seq, K]`, squeeze the unit batch
/// dim before matmul. Only applied when the trailing dims match exactly.
pub(crate) fn flatten_matmul_lhs_if_needed(
    a: &Array,
    graph_a: &[i32],
    graph_out: &[i32],
) -> Result<Array, MlxError> {
    if graph_a.len() < 2 || graph_out.len() != graph_a.len() {
        return a.clone_handle();
    }
    let a_rt = a.shape()?;
    if a_rt.len() != graph_a.len() + 1 || a_rt[0] != 1 {
        return a.clone_handle();
    }
    let matches = graph_a
        .iter()
        .enumerate()
        .all(|(i, &d)| a_rt[i + 1] == d as usize);
    if matches {
        ops::reshape(a, graph_a)
    } else {
        a.clone_handle()
    }
}

/// Map a graph axis index onto the runtime rank when leading dims were
/// preserved by MLX (graph rank < runtime rank).
pub(crate) fn map_graph_axis_to_runtime(
    axis: usize,
    graph_rank: usize,
    runtime_rank: usize,
) -> usize {
    if runtime_rank <= graph_rank {
        axis
    } else {
        axis + (runtime_rank - graph_rank)
    }
}

/// Expand a fused-region input using the same flat `gid % modulus` addressing
/// used by the device elementwise kernels.
pub(crate) fn tile_region_modulus_input(
    input: &Array,
    modulus: usize,
    out_shape: &[i32],
) -> Result<Array, MlxError> {
    let input_shape = input.shape()?;
    let input_n: usize = input_shape.iter().product();
    if input_n != modulus {
        return Err(MlxError(format!(
            "ElementwiseRegion modulus {modulus} does not match input size {input_n}"
        )));
    }
    let out_n: usize = out_shape.iter().map(|&d| d as usize).product();
    let flat = ops::reshape(input, &[modulus as i32])?;
    let copies = out_n.div_ceil(modulus);
    let mut parts = Vec::with_capacity(copies);
    for _ in 0..copies {
        parts.push(flat.clone_handle()?);
    }
    let refs: Vec<&Array> = parts.iter().collect();
    let repeated = ops::concat(&refs, 0)?;
    let trimmed = if out_n == copies * modulus {
        repeated
    } else {
        ops::slice(&repeated, &[0], &[out_n as i32])?
    };
    ops::reshape(&trimmed, out_shape)
}

/// Evaluate an [`Op::ElementwiseRegion`] (or one batch slice) on MLX arrays.
pub(crate) fn eval_elementwise_region_on_inputs(
    env: &HashMap<NodeId, Array>,
    node_inputs: &[NodeId],
    chain: &[ChainStep],
    prologue: RegionPrologue,
) -> Result<Array, MlxError> {
    let mut input0_up: Option<Array> = None;
    if prologue == RegionPrologue::ResizeNearest2x {
        let x = lookup(env, node_inputs[0])?;
        input0_up = Some(ops::resize_nearest_2x_nchw(x)?);
    }
    let mut steps: Vec<Array> = Vec::with_capacity(chain.len());
    for step in chain {
        let arr = match step {
            ChainStep::Activation(act, x_op) => {
                let x =
                    resolve_region_operand(*x_op, node_inputs, input0_up.as_ref(), env, &steps)?;
                match act {
                    Activation::Gelu => ops::gelu(x)?,
                    Activation::GeluApprox => ops::gelu_approx(x)?,
                    Activation::Silu => ops::silu(x)?,
                    Activation::Relu => ops::unary(x, MlxUnary::Relu)?,
                    Activation::Sigmoid => ops::unary(x, MlxUnary::Sigmoid)?,
                    Activation::Tanh => ops::unary(x, MlxUnary::Tanh)?,
                    Activation::Exp => ops::unary(x, MlxUnary::Exp)?,
                    Activation::Log => ops::unary(x, MlxUnary::Log)?,
                    Activation::Sqrt => ops::unary(x, MlxUnary::Sqrt)?,
                    Activation::Rsqrt => ops::unary(x, MlxUnary::Rsqrt)?,
                    Activation::Neg => ops::unary(x, MlxUnary::Neg)?,
                    Activation::Abs => ops::unary(x, MlxUnary::Abs)?,
                    Activation::Round => ops::unary(x, MlxUnary::Round)?,
                    Activation::Sin => ops::unary(x, MlxUnary::Sin)?,
                    Activation::Cos => ops::unary(x, MlxUnary::Cos)?,
                    Activation::Tan => ops::unary(x, MlxUnary::Tan)?,
                    Activation::Atan => ops::unary(x, MlxUnary::Atan)?,
                }
            }
            ChainStep::Cast(to, x_op) => {
                let x =
                    resolve_region_operand(*x_op, node_inputs, input0_up.as_ref(), env, &steps)?;
                ops::cast(x, *to)?
            }
            ChainStep::Binary(bop, l_op, r_op) => {
                let a =
                    resolve_region_operand(*l_op, node_inputs, input0_up.as_ref(), env, &steps)?;
                let b =
                    resolve_region_operand(*r_op, node_inputs, input0_up.as_ref(), env, &steps)?;
                match bop {
                    BinaryOp::Add => ops::add(a, b)?,
                    BinaryOp::Mul => ops::mul(a, b)?,
                    BinaryOp::Sub => ops::sub(a, b)?,
                    BinaryOp::Div => ops::div(a, b)?,
                    BinaryOp::Max => ops::max(a, b)?,
                    BinaryOp::Min => ops::min(a, b)?,
                    BinaryOp::Pow => ops::pow(a, b)?,
                }
            }
            ChainStep::Compare(cop, l_op, r_op) => {
                let a =
                    resolve_region_operand(*l_op, node_inputs, input0_up.as_ref(), env, &steps)?;
                let b =
                    resolve_region_operand(*r_op, node_inputs, input0_up.as_ref(), env, &steps)?;
                match cop {
                    CmpOp::Eq => ops::eq(a, b)?,
                    CmpOp::Ne => ops::ne(a, b)?,
                    CmpOp::Lt => ops::lt(a, b)?,
                    CmpOp::Le => ops::le(a, b)?,
                    CmpOp::Gt => ops::gt(a, b)?,
                    CmpOp::Ge => ops::ge(a, b)?,
                }
            }
            ChainStep::Where(c_op, t_op, f_op) => {
                let c =
                    resolve_region_operand(*c_op, node_inputs, input0_up.as_ref(), env, &steps)?;
                let t =
                    resolve_region_operand(*t_op, node_inputs, input0_up.as_ref(), env, &steps)?;
                let f =
                    resolve_region_operand(*f_op, node_inputs, input0_up.as_ref(), env, &steps)?;
                ops::select(c, t, f)?
            }
        };
        steps.push(arr);
    }
    steps
        .pop()
        .ok_or_else(|| MlxError("ElementwiseRegion: empty chain has no output".into()))
}

pub(crate) fn resolve_region_operand<'a>(
    op: ChainOperand,
    node_inputs: &[NodeId],
    input0_up: Option<&'a Array>,
    env: &'a HashMap<NodeId, Array>,
    steps: &'a [Array],
) -> Result<&'a Array, MlxError> {
    match op {
        ChainOperand::Input(i) => {
            let i = i as usize;
            if i == 0 {
                if let Some(up) = input0_up {
                    return Ok(up);
                }
            }
            let id = *node_inputs.get(i).ok_or_else(|| {
                MlxError(format!(
                    "ElementwiseRegion: ChainOperand::Input({i}) out of range"
                ))
            })?;
            env.get(&id).ok_or_else(|| {
                MlxError(format!(
                    "ElementwiseRegion: missing input node for Input({i})"
                ))
            })
        }
        ChainOperand::Step(i) => {
            let i = i as usize;
            steps.get(i).ok_or_else(|| {
                MlxError(format!(
                    "ElementwiseRegion: ChainOperand::Step({i}) \
                     references step not yet produced (have {} steps)",
                    steps.len()
                ))
            })
        }
    }
}

pub(crate) fn lookup(env: &HashMap<NodeId, Array>, id: NodeId) -> Result<&Array, MlxError> {
    env.get(&id)
        .ok_or_else(|| MlxError(format!("node {id:?} referenced before being lowered")))
}

pub(crate) fn unsupported<T>(what: String) -> Result<T, MlxError> {
    Err(MlxError(format!("MLX backend: unsupported op {what}")))
}

/// Native (on-device) `Op::Lstm` for `carry = false` — unrolls the time
/// loop into MLX ops so the recurrence stays in the lazy graph and is
/// `mlx::compile`-able (no host `to_f32` roundtrip, unlike the CPU
/// host-eval fallback). Covers uni/bi-directional and multi-layer.
///
/// Packing mirrors `execute_lstm_f32` (the shared CPU kernel):
///   inputs `[x, w_ih, w_hh, bias]`, `x` `[B, S, in]`, flat `w_ih`
///   laid out per-layer as `dirs` blocks of `[4h, in_l]` (row-major
///   gate rows `i, f, g, o`), `w_hh` `[L*D][4h, h]`, `bias` `[L*D][4h]`.
///   Output `y` `[B, S, D*h]` — per direction `dir` owns feature slice
///   `dir*h .. dir*h+h`; the reverse pass writes hidden states at their
///   natural time index.
/// Compute precision for the native recurrent unrolls on MLX. The runtime
/// promotes built-in ops (Lstm/Gru/Rnn) to f32 before lowering (shared with the
/// f32-only CPU kernels), so the graph reaches here in f32. Setting
/// `RLX_MLX_RNN_F16` opts the recurrence into fp16 *compute* on the Metal GPU
/// (matmuls accumulate in f32) — faster/less bandwidth for wide hidden sizes —
/// while inputs/outputs stay f32.
pub(crate) fn rnn_compute_dtype() -> DType {
    if std::env::var_os("RLX_MLX_RNN_F16").is_some() {
        DType::F16
    } else {
        DType::F32
    }
}

/// Cast to `dt` only when it differs from f32 (avoids emitting no-op casts on
/// the default f32 path).
pub(crate) fn to_compute(a: Array, dt: DType) -> Result<Array, MlxError> {
    if dt == DType::F32 {
        Ok(a)
    } else {
        ops::cast(&a, dt)
    }
}

pub(crate) fn native_lstm(
    graph: &Graph,
    env: &HashMap<NodeId, Array>,
    node: &rlx_ir::Node,
    hidden: usize,
    num_layers: usize,
    bidirectional: bool,
) -> Result<Array, MlxError> {
    let x_shape = node_input_shape(graph, node.inputs[0]); // [B, S, in]
    if x_shape.len() != 3 {
        return Err(MlxError(format!(
            "Lstm: x must be rank-3 [B, S, in], got rank {}",
            x_shape.len()
        )));
    }
    let batch = x_shape[0];
    let seq = x_shape[1];
    let input_size = x_shape[2];
    let h = hidden as i32;
    let four_h = 4 * h;
    let dirs: i32 = if bidirectional { 2 } else { 1 };

    let x0 = lookup(env, node.inputs[0])?;
    // `Op::Lstm` packs each parameter stream into per-layer/direction
    // blocks. Synthetic graphs traditionally supply those streams as rank-1,
    // but ONNX imports preserve the source matrices (`w_ih` is commonly
    // `[directions * 4h, input]`). Flatten once so the packed element offsets
    // below address both representations identically.
    let cdt = rnn_compute_dtype();
    let flatten = |input: NodeId| -> Result<Array, MlxError> {
        let a = lookup(env, input)?;
        let n: i32 = node_input_shape(graph, input).iter().product();
        to_compute(ops::reshape(a, &[n])?, cdt)
    };
    let w_ih = flatten(node.inputs[1])?;
    let w_hh = flatten(node.inputs[2])?;
    let bias = flatten(node.inputs[3])?;

    // Running per-layer input `[B, S, in_l]`; layer 0 reads `x`, later
    // layers read the previous layer's `[B, S, D*h]` output.
    let mut layer_in = to_compute(x0.clone_handle()?, cdt)?;
    let mut in_l = input_size;
    // Element cursor into flat `w_ih` (block width `4h*in_l` varies per
    // layer because `in_l` grows to `D*h` after layer 0).
    let mut wih_cursor: i32 = 0;

    for l in 0..num_layers as i32 {
        let wih_block = four_h * in_l;
        let mut dir_outputs: Vec<Array> = Vec::with_capacity(dirs as usize);

        for dir in 0..dirs {
            let ld = l * dirs + dir;

            // Slice + reshape this (layer, dir)'s weights, then transpose
            // to the `x @ Wᵀ` layout MLX matmul wants.
            let wih_off = wih_cursor + dir * wih_block;
            let wih_mat = ops::reshape(
                &ops::slice(&w_ih, &[wih_off], &[wih_off + wih_block])?,
                &[four_h, in_l],
            )?;
            let wih_t = ops::transpose(&wih_mat, &[1, 0])?; // [in_l, 4h]

            let whh_off = ld * four_h * h;
            let whh_mat = ops::reshape(
                &ops::slice(&w_hh, &[whh_off], &[whh_off + four_h * h])?,
                &[four_h, h],
            )?;
            let whh_t = ops::transpose(&whh_mat, &[1, 0])?; // [h, 4h]

            let b_off = ld * four_h;
            let b_row = ops::reshape(
                &ops::slice(&bias, &[b_off], &[b_off + four_h])?,
                &[1, four_h],
            )?; // [1, 4h]

            // Input contribution to the gates for the whole sequence in a
            // single GEMM: gates_x = x @ w_ihᵀ + b  → [B, S, 4h]. Flatten
            // the batch/time axes so it is a plain 2-D matmul.
            let flat_in = ops::reshape(&layer_in, &[batch * seq, in_l])?;
            let gx = ops::add(&ops::matmul(&flat_in, &wih_t)?, &b_row)?;
            let gates_x = ops::reshape(&gx, &[batch, seq, four_h])?;

            // Recurrence. h, c start at zero (carry = false).
            let bh = (batch * h) as usize;
            let zero_shape = [batch as usize, h as usize];
            let mut h_state = to_compute(
                Array::from_f32_slice(&vec![0f32; bh], &zero_shape, DType::F32)?,
                cdt,
            )?;
            let mut c_state = to_compute(
                Array::from_f32_slice(&vec![0f32; bh], &zero_shape, DType::F32)?,
                cdt,
            )?;

            // Hidden states collected in natural time order (index = t).
            let mut h_steps: Vec<Option<Array>> = (0..seq).map(|_| None).collect();

            for step in 0..seq {
                let t = if dir == 0 { step } else { seq - 1 - step };

                // z = gates_x[:, t, :] + h · w_hhᵀ  → [B, 4h].
                let gx_t = ops::reshape(
                    &ops::slice(&gates_x, &[0, t, 0], &[batch, t + 1, four_h])?,
                    &[batch, four_h],
                )?;
                let z = ops::add(&gx_t, &ops::matmul(&h_state, &whh_t)?)?;

                // Split gates (order i, f, g, o) and apply nonlinearities.
                let i_g = ops::unary(&ops::slice(&z, &[0, 0], &[batch, h])?, MlxUnary::Sigmoid)?;
                let f_g = ops::unary(
                    &ops::slice(&z, &[0, h], &[batch, 2 * h])?,
                    MlxUnary::Sigmoid,
                )?;
                let g_g = ops::unary(
                    &ops::slice(&z, &[0, 2 * h], &[batch, 3 * h])?,
                    MlxUnary::Tanh,
                )?;
                let o_g = ops::unary(
                    &ops::slice(&z, &[0, 3 * h], &[batch, 4 * h])?,
                    MlxUnary::Sigmoid,
                )?;

                // c = f⊙c + i⊙g ; h = o ⊙ tanh(c).
                c_state = ops::add(&ops::mul(&f_g, &c_state)?, &ops::mul(&i_g, &g_g)?)?;
                h_state = ops::mul(&o_g, &ops::unary(&c_state, MlxUnary::Tanh)?)?;

                h_steps[t as usize] = Some(ops::reshape(&h_state, &[batch, 1, h])?);
            }

            let refs: Vec<&Array> = h_steps
                .iter()
                .map(|o| o.as_ref().expect("every timestep filled"))
                .collect();
            dir_outputs.push(ops::concat(&refs, 1)?); // [B, S, h]
        }

        // Concatenate directions on the feature axis → [B, S, D*h].
        layer_in = if dirs == 1 {
            dir_outputs.pop().expect("one direction")
        } else {
            let refs: Vec<&Array> = dir_outputs.iter().collect();
            ops::concat(&refs, 2)?
        };
        in_l = dirs * h;
        wih_cursor += dirs * wih_block;
    }

    // Cast the fp16 recurrence result back to the node's (f32) output dtype.
    if cdt != node.shape.dtype() {
        ops::cast(&layer_in, node.shape.dtype())
    } else {
        Ok(layer_in)
    }
}

/// Shared weight-stream flattener for the recurrent unrolls: reshape each
/// packed parameter to rank-1 so per-(layer,dir) element offsets address the
/// rank-1 packed and rank-2 ONNX-matrix forms identically.
pub(crate) fn rnn_flatten(
    graph: &Graph,
    env: &HashMap<NodeId, Array>,
    input: NodeId,
) -> Result<Array, MlxError> {
    let a = lookup(env, input)?;
    let n: i32 = node_input_shape(graph, input).iter().product();
    ops::reshape(a, &[n])
}

/// Native (on-device) `Op::Gru` for `carry = false`. Gate order r, z, n with
/// separate `b_ih`/`b_hh` (the reset gate multiplies the hidden term *after*
/// its bias): `r=σ(xᵣ+hᵣ)`, `z=σ(x_z+h_z)`, `n=tanh(xₙ+r⊙hₙ)`,
/// `h'=(1−z)⊙n+z⊙h`. Packing mirrors `execute_gru_f32`: per-(layer,dir)
/// `w_ih [3h,in_l]`, `w_hh [3h,h]`, `b_ih [3h]`, `b_hh [3h]`. Output `[B,S,D*h]`.
pub(crate) fn native_gru(
    graph: &Graph,
    env: &HashMap<NodeId, Array>,
    node: &rlx_ir::Node,
    hidden: usize,
    num_layers: usize,
    bidirectional: bool,
) -> Result<Array, MlxError> {
    let x_shape = node_input_shape(graph, node.inputs[0]);
    if x_shape.len() != 3 {
        return Err(MlxError(format!(
            "Gru: x must be rank-3 [B, S, in], got rank {}",
            x_shape.len()
        )));
    }
    let batch = x_shape[0];
    let seq = x_shape[1];
    let input_size = x_shape[2];
    let h = hidden as i32;
    let g3 = 3 * h;
    let dirs: i32 = if bidirectional { 2 } else { 1 };

    let cdt = rnn_compute_dtype();
    let x0 = lookup(env, node.inputs[0])?;
    let w_ih = to_compute(rnn_flatten(graph, env, node.inputs[1])?, cdt)?;
    let w_hh = to_compute(rnn_flatten(graph, env, node.inputs[2])?, cdt)?;
    let b_ih = to_compute(rnn_flatten(graph, env, node.inputs[3])?, cdt)?;
    let b_hh = to_compute(rnn_flatten(graph, env, node.inputs[4])?, cdt)?;

    let mut layer_in = to_compute(x0.clone_handle()?, cdt)?;
    let mut in_l = input_size;
    let mut wih_cursor: i32 = 0;

    for l in 0..num_layers as i32 {
        let wih_block = g3 * in_l;
        let mut dir_outputs: Vec<Array> = Vec::with_capacity(dirs as usize);

        for dir in 0..dirs {
            let ld = l * dirs + dir;
            let wih_off = wih_cursor + dir * wih_block;
            let wih_t = ops::transpose(
                &ops::reshape(
                    &ops::slice(&w_ih, &[wih_off], &[wih_off + wih_block])?,
                    &[g3, in_l],
                )?,
                &[1, 0],
            )?; // [in_l, 3h]
            let whh_off = ld * g3 * h;
            let whh_t = ops::transpose(
                &ops::reshape(
                    &ops::slice(&w_hh, &[whh_off], &[whh_off + g3 * h])?,
                    &[g3, h],
                )?,
                &[1, 0],
            )?; // [h, 3h]
            let b_off = ld * g3;
            let bih_row = ops::reshape(&ops::slice(&b_ih, &[b_off], &[b_off + g3])?, &[1, g3])?;
            let bhh_row = ops::reshape(&ops::slice(&b_hh, &[b_off], &[b_off + g3])?, &[1, g3])?;

            // xi = x @ w_ihᵀ + b_ih  → [B, S, 3h] (one GEMM for the sequence).
            let flat_in = ops::reshape(&layer_in, &[batch * seq, in_l])?;
            let xi = ops::reshape(
                &ops::add(&ops::matmul(&flat_in, &wih_t)?, &bih_row)?,
                &[batch, seq, g3],
            )?;

            let bh = (batch * h) as usize;
            let zero_shape = [batch as usize, h as usize];
            let mut h_state = to_compute(
                Array::from_f32_slice(&vec![0f32; bh], &zero_shape, DType::F32)?,
                cdt,
            )?;

            let mut h_steps: Vec<Option<Array>> = (0..seq).map(|_| None).collect();
            for step in 0..seq {
                let t = if dir == 0 { step } else { seq - 1 - step };
                let xi_t = ops::reshape(
                    &ops::slice(&xi, &[0, t, 0], &[batch, t + 1, g3])?,
                    &[batch, g3],
                )?;
                // hi = h · w_hhᵀ + b_hh  → [B, 3h].
                let hi = ops::add(&ops::matmul(&h_state, &whh_t)?, &bhh_row)?;

                let xr = ops::slice(&xi_t, &[0, 0], &[batch, h])?;
                let xz = ops::slice(&xi_t, &[0, h], &[batch, 2 * h])?;
                let xn = ops::slice(&xi_t, &[0, 2 * h], &[batch, 3 * h])?;
                let hr = ops::slice(&hi, &[0, 0], &[batch, h])?;
                let hz = ops::slice(&hi, &[0, h], &[batch, 2 * h])?;
                let hn = ops::slice(&hi, &[0, 2 * h], &[batch, 3 * h])?;

                let r_g = ops::unary(&ops::add(&xr, &hr)?, MlxUnary::Sigmoid)?;
                let z_g = ops::unary(&ops::add(&xz, &hz)?, MlxUnary::Sigmoid)?;
                let n_g = ops::unary(&ops::add(&xn, &ops::mul(&r_g, &hn)?)?, MlxUnary::Tanh)?;
                // h' = (1−z)⊙n + z⊙h = n + z⊙(h − n).
                h_state = ops::add(&n_g, &ops::mul(&z_g, &ops::sub(&h_state, &n_g)?)?)?;
                h_steps[t as usize] = Some(ops::reshape(&h_state, &[batch, 1, h])?);
            }
            let refs: Vec<&Array> = h_steps
                .iter()
                .map(|o| o.as_ref().expect("every timestep filled"))
                .collect();
            dir_outputs.push(ops::concat(&refs, 1)?);
        }

        layer_in = if dirs == 1 {
            dir_outputs.pop().expect("one direction")
        } else {
            let refs: Vec<&Array> = dir_outputs.iter().collect();
            ops::concat(&refs, 2)?
        };
        in_l = dirs * h;
        wih_cursor += dirs * wih_block;
    }
    // Cast the fp16 recurrence result back to the node's (f32) output dtype.
    if cdt != node.shape.dtype() {
        ops::cast(&layer_in, node.shape.dtype())
    } else {
        Ok(layer_in)
    }
}

/// Native (on-device) `Op::Rnn` (Elman) for `carry = false`:
/// `h' = act(x·w_ihᵀ + h·w_hhᵀ + b)`, `act = ReLU` when `relu` else `tanh`.
/// Single merged bias; per-(layer,dir) `w_ih [h,in_l]`, `w_hh [h,h]`, `bias [h]`.
/// Output `[B, S, D*h]`.
pub(crate) fn native_rnn(
    graph: &Graph,
    env: &HashMap<NodeId, Array>,
    node: &rlx_ir::Node,
    hidden: usize,
    num_layers: usize,
    bidirectional: bool,
    relu: bool,
) -> Result<Array, MlxError> {
    let x_shape = node_input_shape(graph, node.inputs[0]);
    if x_shape.len() != 3 {
        return Err(MlxError(format!(
            "Rnn: x must be rank-3 [B, S, in], got rank {}",
            x_shape.len()
        )));
    }
    let batch = x_shape[0];
    let seq = x_shape[1];
    let input_size = x_shape[2];
    let h = hidden as i32;
    let dirs: i32 = if bidirectional { 2 } else { 1 };
    let act = if relu { MlxUnary::Relu } else { MlxUnary::Tanh };

    let cdt = rnn_compute_dtype();
    let x0 = lookup(env, node.inputs[0])?;
    let w_ih = to_compute(rnn_flatten(graph, env, node.inputs[1])?, cdt)?;
    let w_hh = to_compute(rnn_flatten(graph, env, node.inputs[2])?, cdt)?;
    let bias = to_compute(rnn_flatten(graph, env, node.inputs[3])?, cdt)?;

    let mut layer_in = to_compute(x0.clone_handle()?, cdt)?;
    let mut in_l = input_size;
    let mut wih_cursor: i32 = 0;

    for l in 0..num_layers as i32 {
        let wih_block = h * in_l;
        let mut dir_outputs: Vec<Array> = Vec::with_capacity(dirs as usize);

        for dir in 0..dirs {
            let ld = l * dirs + dir;
            let wih_off = wih_cursor + dir * wih_block;
            let wih_t = ops::transpose(
                &ops::reshape(
                    &ops::slice(&w_ih, &[wih_off], &[wih_off + wih_block])?,
                    &[h, in_l],
                )?,
                &[1, 0],
            )?; // [in_l, h]
            let whh_off = ld * h * h;
            let whh_t = ops::transpose(
                &ops::reshape(&ops::slice(&w_hh, &[whh_off], &[whh_off + h * h])?, &[h, h])?,
                &[1, 0],
            )?; // [h, h]
            let b_off = ld * h;
            let b_row = ops::reshape(&ops::slice(&bias, &[b_off], &[b_off + h])?, &[1, h])?;

            // xi = x @ w_ihᵀ + b  → [B, S, h] (bias folded in once).
            let flat_in = ops::reshape(&layer_in, &[batch * seq, in_l])?;
            let xi = ops::reshape(
                &ops::add(&ops::matmul(&flat_in, &wih_t)?, &b_row)?,
                &[batch, seq, h],
            )?;

            let bh = (batch * h) as usize;
            let zero_shape = [batch as usize, h as usize];
            let mut h_state = to_compute(
                Array::from_f32_slice(&vec![0f32; bh], &zero_shape, DType::F32)?,
                cdt,
            )?;

            let mut h_steps: Vec<Option<Array>> = (0..seq).map(|_| None).collect();
            for step in 0..seq {
                let t = if dir == 0 { step } else { seq - 1 - step };
                let xi_t = ops::reshape(
                    &ops::slice(&xi, &[0, t, 0], &[batch, t + 1, h])?,
                    &[batch, h],
                )?;
                // acc = xi_t + h · w_hhᵀ ; h = act(acc).
                let acc = ops::add(&xi_t, &ops::matmul(&h_state, &whh_t)?)?;
                h_state = ops::unary(&acc, act)?;
                h_steps[t as usize] = Some(ops::reshape(&h_state, &[batch, 1, h])?);
            }
            let refs: Vec<&Array> = h_steps
                .iter()
                .map(|o| o.as_ref().expect("every timestep filled"))
                .collect();
            dir_outputs.push(ops::concat(&refs, 1)?);
        }

        layer_in = if dirs == 1 {
            dir_outputs.pop().expect("one direction")
        } else {
            let refs: Vec<&Array> = dir_outputs.iter().collect();
            ops::concat(&refs, 2)?
        };
        in_l = dirs * h;
        wih_cursor += dirs * wih_block;
    }
    // Cast the fp16 recurrence result back to the node's (f32) output dtype.
    if cdt != node.shape.dtype() {
        ops::cast(&layer_in, node.shape.dtype())
    } else {
        Ok(layer_in)
    }
}

/// Zero-inflate a 4-D NHWC array along the two spatial axes by factors
/// (`sh`, `sw`). Produces a new array of shape
/// `[N, (H − 1)·sh + 1, (W − 1)·sw + 1, C]`, with original values at
/// strided positions and zeros between them.
///
/// Workaround for an MLX `conv_general` limitation: when `groups > 1`
/// AND `input_dilation > 1`, the kernel produces incorrect output. We
/// materialize the input dilation explicitly (reshape → pad → reshape
/// per spatial axis) so the downstream `conv_general` can run with
/// `input_dilation=[1,1]`.
pub(crate) fn inflate_spatial_2d(a: &Array, sh: usize, sw: usize) -> Result<Array, MlxError> {
    if sh == 1 && sw == 1 {
        return a.clone_handle();
    }
    let shape = a.shape()?;
    if shape.len() != 4 {
        return Err(MlxError(format!(
            "inflate_spatial_2d: expected rank-4 NHWC, got rank {}",
            shape.len()
        )));
    }
    let n = shape[0] as i32;
    let h = shape[1] as i32;
    let w = shape[2] as i32;
    let c = shape[3] as i32;

    let mut cur = a.clone_handle()?;
    if sh > 1 {
        let sh_i = sh as i32;
        // [N, H, W, C] → [N, H, 1, W, C] → pad axis 2 by (0, sh-1) →
        // [N, H, sh, W, C] → reshape [N, H*sh, W, C] → slice trailing
        // (sh-1) frames so dim becomes (H-1)*sh + 1.
        let r1 = ops::reshape(&cur, &[n, h, 1, w, c])?;
        let padded = ops::pad(
            &r1,
            /*low =*/ &[0, 0, 0, 0, 0],
            /*high=*/ &[0, 0, sh_i - 1, 0, 0],
            /*pad_value=*/ 0.0,
        )?;
        let merged = ops::reshape(&padded, &[n, h * sh_i, w, c])?;
        let new_h = (h - 1) * sh_i + 1;
        cur = ops::slice(&merged, &[0, 0, 0, 0], &[n, new_h, w, c])?;
    }
    if sw > 1 {
        let sw_i = sw as i32;
        let cur_shape = cur.shape()?;
        let cur_h = cur_shape[1] as i32;
        let r1 = ops::reshape(&cur, &[n, cur_h, w, 1, c])?;
        let padded = ops::pad(
            &r1,
            /*low =*/ &[0, 0, 0, 0, 0],
            /*high=*/ &[0, 0, 0, sw_i - 1, 0],
            /*pad_value=*/ 0.0,
        )?;
        let merged = ops::reshape(&padded, &[n, cur_h, w * sw_i, c])?;
        let new_w = (w - 1) * sw_i + 1;
        cur = ops::slice(&merged, &[0, 0, 0, 0], &[n, cur_h, new_w, c])?;
    }
    Ok(cur)
}

/// Zero-inflate a rank-3 NLC array along its single spatial axis by factor `s`
/// (insert `s-1` zeros between elements), giving length `(L-1)*s + 1`.
/// Kept for a future native MLX CT path; Vocos ISTFT currently host-evals CT.
#[allow(dead_code)]
pub(crate) fn inflate_spatial_1d(a: &Array, s: usize) -> Result<Array, MlxError> {
    if s == 1 {
        return a.clone_handle();
    }
    let shape = a.shape()?;
    if shape.len() != 3 {
        return Err(MlxError(format!(
            "inflate_spatial_1d: expected rank-3 NLC, got rank {}",
            shape.len()
        )));
    }
    let n = shape[0] as i32;
    let l = shape[1] as i32;
    let c = shape[2] as i32;
    let si = s as i32;
    let r1 = ops::reshape(a, &[n, l, 1, c])?;
    let padded = ops::pad(&r1, &[0, 0, 0, 0], &[0, 0, si - 1, 0], 0.0)?;
    let merged = ops::reshape(&padded, &[n, l * si, c])?;
    let new_l = (l - 1) * si + 1;
    ops::slice(&merged, &[0, 0, 0], &[n, new_l, c])
}

/// Map `bits` ∈ {8, 4, 2} to its quantization range `q_max`.
pub(crate) fn fq_q_max(bits: u8) -> Result<f32, MlxError> {
    match bits {
        8 => Ok(127.0),
        4 => Ok(7.0),
        2 => Ok(1.0),
        n => Err(MlxError(format!("FakeQuantize: unsupported bits {n}"))),
    }
}

/// PerBatch-style scale: per-channel `max(|x|) / q_max`, floored at
/// `1e-12` so dividing by it never blows up. Returned shape is
/// broadcast-compatible against `x` (via `keep_dim=true` on the reduce).
pub(crate) fn fq_scale_perbatch(
    x: &Array,
    x_shape: &[i32],
    axis: Option<usize>,
    q_max: f32,
    dtype: DType,
) -> Result<Array, MlxError> {
    let abs_x = ops::unary(x, MlxUnary::Abs)?;
    let reduce_axes: Vec<i32> = match axis {
        None => (0..x_shape.len() as i32).collect(),
        Some(c) => (0..x_shape.len() as i32)
            .filter(|&i| i != c as i32)
            .collect(),
    };
    let max_abs = ops::reduce(
        &abs_x,
        MlxReduce::Max,
        &reduce_axes,
        /*keep_dim=*/ true,
    )?;
    let q_max_arr = Array::from_f32_slice(&[q_max], &[1], dtype)?;
    let scale_unclamped = ops::div(&max_abs, &q_max_arr)?;
    let eps = Array::from_f32_slice(&[1e-12], &[1], dtype)?;
    ops::max(&scale_unclamped, &eps)
}

/// Build a broadcast-shaped scale tensor from a 1-D `state` (shape `[C]`
/// for per-channel; `[1]` for per-tensor) so it broadcasts against `x`.
pub(crate) fn fq_scale_from_state(
    state: &Array,
    x_shape: &[i32],
    axis: Option<usize>,
    dtype: DType,
) -> Result<Array, MlxError> {
    let eps = Array::from_f32_slice(&[1e-12], &[1], dtype)?;
    let clamped = ops::max(state, &eps)?;
    match axis {
        None => Ok(clamped),
        Some(c) => {
            let state_dim = state.shape()?;
            let dim_c = state_dim.first().copied().unwrap_or(1) as i32;
            let mut bcast: Vec<i32> = vec![1; x_shape.len()];
            bcast[c] = dim_c;
            ops::reshape(&clamped, &bcast)
        }
    }
}

/// Shared quant + dequant tail of `Op::FakeQuantize`. Same formula
/// regardless of which `scale_mode` produced `scale`.
pub(crate) fn fq_quantize_dequantize(
    x: &Array,
    scale: &Array,
    q_max: f32,
    dtype: DType,
) -> Result<Array, MlxError> {
    let scaled = ops::div(x, scale)?;
    let rounded = ops::unary(&scaled, MlxUnary::Round)?;
    let neg_qmax = Array::from_f32_slice(&[-q_max], &[1], dtype)?;
    let pos_qmax = Array::from_f32_slice(&[q_max], &[1], dtype)?;
    let clamped = ops::max(&rounded, &neg_qmax)?;
    let clamped = ops::min(&clamped, &pos_qmax)?;
    ops::mul(&clamped, scale)
}

/// `[N, C]` one-hot encoding of f32-valued integer labels.
/// `oh[n, c] = 1.0` if `labels[n] == c` else `0.0`.
pub(crate) fn one_hot_2d(
    labels: &Array,
    n: usize,
    c: usize,
    dtype: DType,
) -> Result<Array, MlxError> {
    let arange_data: Vec<f32> = (0..c).map(|i| i as f32).collect();
    let arange = Array::from_f32_slice(&arange_data, &[c], dtype)?;
    let arange_2d = ops::reshape(&arange, &[1, c as i32])?;
    let labels_2d = ops::reshape(labels, &[n as i32, 1])?;
    let mask_bool = ops::eq(&labels_2d, &arange_2d)?;
    ops::cast(&mask_bool, dtype)
}

/// Closed-form derivative of every `Activation` kind. Mirrors
/// `rlx-cpu/src/thunk.rs::activation_backward_kernel`.
pub(crate) fn activation_backward_compose(
    x: &Array,
    dy: &Array,
    kind: Activation,
    dtype: DType,
) -> Result<Array, MlxError> {
    use Activation::*;
    match kind {
        Relu => {
            let zero = Array::from_f32_slice(&[0.0], &[1], dtype)?;
            let mask = ops::gt(x, &zero)?;
            ops::select(&mask, dy, &zero)
        }
        Sigmoid => {
            // dy · σ(x) · (1 − σ(x))
            let s = ops::unary(x, MlxUnary::Sigmoid)?;
            let one = Array::from_f32_slice(&[1.0], &[1], dtype)?;
            let one_minus_s = ops::sub(&one, &s)?;
            let s_compl = ops::mul(&s, &one_minus_s)?;
            ops::mul(dy, &s_compl)
        }
        Tanh => {
            // dy · (1 − tanh²(x))
            let t = ops::unary(x, MlxUnary::Tanh)?;
            let t_sq = ops::mul(&t, &t)?;
            let one = Array::from_f32_slice(&[1.0], &[1], dtype)?;
            let factor = ops::sub(&one, &t_sq)?;
            ops::mul(dy, &factor)
        }
        Silu => {
            // dy · σ(x) · (1 + x · (1 − σ(x)))
            let s = ops::unary(x, MlxUnary::Sigmoid)?;
            let one = Array::from_f32_slice(&[1.0], &[1], dtype)?;
            let one_minus_s = ops::sub(&one, &s)?;
            let x_times = ops::mul(x, &one_minus_s)?;
            let inner = ops::add(&one, &x_times)?;
            let factor = ops::mul(&s, &inner)?;
            ops::mul(dy, &factor)
        }
        Gelu => {
            // dy · (½(1 + erf(x/√2)) + x · φ(x)),  φ(x) = exp(−x²/2)/√(2π)
            const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;
            const INV_SQRT_2PI: f32 = 0.398_942_3;
            let inv_sqrt2 = Array::from_f32_slice(&[INV_SQRT2], &[1], dtype)?;
            let inv_sqrt_2pi = Array::from_f32_slice(&[INV_SQRT_2PI], &[1], dtype)?;
            let half = Array::from_f32_slice(&[0.5], &[1], dtype)?;
            let one = Array::from_f32_slice(&[1.0], &[1], dtype)?;
            let neg_half = Array::from_f32_slice(&[-0.5], &[1], dtype)?;

            let x_sc = ops::mul(x, &inv_sqrt2)?;
            let erf_v = ops::unary(&x_sc, MlxUnary::Erf)?;
            let phi_inner = ops::add(&one, &erf_v)?;
            let phi = ops::mul(&half, &phi_inner)?;
            let x_sq = ops::mul(x, x)?;
            let arg = ops::mul(&x_sq, &neg_half)?;
            let pdf_e = ops::unary(&arg, MlxUnary::Exp)?;
            let pdf = ops::mul(&pdf_e, &inv_sqrt_2pi)?;
            let x_pdf = ops::mul(x, &pdf)?;
            let deriv = ops::add(&phi, &x_pdf)?;
            ops::mul(dy, &deriv)
        }
        GeluApprox => {
            // y = ½ x (1 + tanh(c (x + a x³))), c = √(2/π), a = 0.044715
            // dy/dx = ½(1+t) + ½ x (1−t²) · c (1 + 3 a x²)
            const C: f32 = 0.797_884_6;
            const A: f32 = 0.044_715;
            let half = Array::from_f32_slice(&[0.5], &[1], dtype)?;
            let one = Array::from_f32_slice(&[1.0], &[1], dtype)?;
            let c_arr = Array::from_f32_slice(&[C], &[1], dtype)?;
            let a_arr = Array::from_f32_slice(&[A], &[1], dtype)?;
            let three_a = Array::from_f32_slice(&[3.0 * A], &[1], dtype)?;

            let x_sq = ops::mul(x, x)?;
            let x_cu = ops::mul(&x_sq, x)?;
            let a_x_cu = ops::mul(&a_arr, &x_cu)?;
            let inner_sum = ops::add(x, &a_x_cu)?;
            let inner = ops::mul(&c_arr, &inner_sum)?;
            let t = ops::unary(&inner, MlxUnary::Tanh)?;
            let one_plus_t = ops::add(&one, &t)?;
            let term1 = ops::mul(&half, &one_plus_t)?;
            let t_sq = ops::mul(&t, &t)?;
            let one_minus_t_sq = ops::sub(&one, &t_sq)?;
            let three_a_x_sq = ops::mul(&three_a, &x_sq)?;
            let one_plus_3ax2 = ops::add(&one, &three_a_x_sq)?;
            let dinner = ops::mul(&c_arr, &one_plus_3ax2)?;
            let half_x = ops::mul(&half, x)?;
            let part2_a = ops::mul(&half_x, &one_minus_t_sq)?;
            let term2 = ops::mul(&part2_a, &dinner)?;
            let deriv = ops::add(&term1, &term2)?;
            ops::mul(dy, &deriv)
        }
        Exp => {
            let ex = ops::unary(x, MlxUnary::Exp)?;
            ops::mul(dy, &ex)
        }
        Log => ops::div(dy, x),
        Sqrt => {
            // 0.5 · dy / √x; zero where √x ≤ 0.
            let s = ops::unary(x, MlxUnary::Sqrt)?;
            let zero = Array::from_f32_slice(&[0.0], &[1], dtype)?;
            let half = Array::from_f32_slice(&[0.5], &[1], dtype)?;
            let mask = ops::gt(&s, &zero)?;
            let half_dy = ops::mul(&half, dy)?;
            let raw = ops::div(&half_dy, &s)?;
            ops::select(&mask, &raw, &zero)
        }
        Rsqrt => {
            // −0.5 · dy / (x · √x); zero where √x ≤ 0.
            let s = ops::unary(x, MlxUnary::Sqrt)?;
            let zero = Array::from_f32_slice(&[0.0], &[1], dtype)?;
            let neg_half = Array::from_f32_slice(&[-0.5], &[1], dtype)?;
            let mask = ops::gt(&s, &zero)?;
            let denom = ops::mul(x, &s)?;
            let neg_half_dy = ops::mul(&neg_half, dy)?;
            let raw = ops::div(&neg_half_dy, &denom)?;
            ops::select(&mask, &raw, &zero)
        }
        Neg => ops::unary(dy, MlxUnary::Neg),
        Abs => {
            // sign(x) · dy. CPU reference uses 0 at x=0 (not ±0).
            let zero = Array::from_f32_slice(&[0.0], &[1], dtype)?;
            let one = Array::from_f32_slice(&[1.0], &[1], dtype)?;
            let neg_one = Array::from_f32_slice(&[-1.0], &[1], dtype)?;
            let pos = ops::gt(x, &zero)?;
            let neg = ops::lt(x, &zero)?;
            let inner = ops::select(&neg, &neg_one, &zero)?;
            let sign = ops::select(&pos, &one, &inner)?;
            ops::mul(&sign, dy)
        }
        Round => {
            // STE: pretend Round was identity (zero-grad almost everywhere
            // means the optimizer can't learn through it without this).
            dy.clone_handle()
        }
        Sin => {
            // d/dx sin(x) = cos(x) · upstream.
            let c = ops::unary(x, MlxUnary::Cos)?;
            ops::mul(&c, dy)
        }
        Cos => {
            // d/dx cos(x) = −sin(x) · upstream.
            let s = ops::unary(x, MlxUnary::Sin)?;
            let neg_s = ops::unary(&s, MlxUnary::Neg)?;
            ops::mul(&neg_s, dy)
        }
        Tan => {
            // dy · (1 + tan²(x))
            let t = ops::unary(x, MlxUnary::Tan)?;
            let t2 = ops::mul(&t, &t)?;
            let one = Array::from_f32_slice(&[1.0], &[1], dtype)?;
            let sec2 = ops::add(&one, &t2)?;
            ops::mul(dy, &sec2)
        }
        Atan => {
            // dy · (1 / (1 + x²))
            let x2 = ops::mul(x, x)?;
            let one = Array::from_f32_slice(&[1.0], &[1], dtype)?;
            let denom = ops::add(&one, &x2)?;
            ops::div(dy, &denom)
        }
    }
}
