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

//! `generators` — extracted from the `ops` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

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

use crate::lower::options::{ImportOptions, ImportReport};

use super::*;

pub(super) fn lower_constant_of_shape(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let shape_in = ctx.tensor(&node.inputs[0])?;
    // ONNX `ConstantOfShape` output SHAPE = the *values* of its shape input and
    // its DTYPE = the `value` attribute tensor's dtype (default float32). Derive
    // BOTH from those authoritative sources rather than the ONNX-declared
    // `output_meta`, which for a dynamic-sequence export is symbolic and
    // mis-resolves — the same stale-meta class as the gather / elementwise fixes.
    // Kokoro's ALBERT attention mask builds `ConstantOfShape(value=1.0f,
    // Concat([1, seq]))`: `output_meta` is symbolic `[unk, unk]` AND typed i64
    // (it feeds a Cast), so trusting meta gave a wrong-shape i64 all-ones tensor
    // that downstream ops read as f32 denormals → the whole attention mask
    // (`(1-mask)*finfo.min`) inverted → garbage attention. The shape-input values
    // `[1, 15]` and the float `value` are authoritative.
    let meta_s = output_shape(ctx, node, m, shape_in);
    let value_dtype = node
        .attrs
        .get("value")
        .and_then(|v| v.get("tensor"))
        .and_then(|t| t.get("dtype"))
        .and_then(|d| d.as_str())
        .map(|s| match s {
            "i64" => DType::I64,
            "i32" => DType::I32,
            "bool" => DType::Bool,
            "u8" | "uint8" => DType::U8,
            "i8" | "int8" => DType::I8,
            _ => DType::F32,
        })
        .unwrap_or_else(|| meta_s.dtype());
    let mut out_s = match eval_static_shape_vector(ctx, m, &node.inputs[0], 0) {
        // Non-empty shape vector → tensor with those dims filled by `value`.
        Some(dims) if !dims.is_empty() => {
            let d: Vec<usize> = dims.iter().map(|&x| x.max(0) as usize).collect();
            Shape::new(&d, value_dtype)
        }
        // Empty 1-D shape tensor (`dims=[]` / length-0) → ONNX scalar fill.
        // Soprano's causal-mask graph uses `ConstantOfShape(Constant([]))` as a
        // scalar bool; falling through to stale `output_meta` ([head_dim]=128)
        // inflated the ones mask and poisoned every attention layer.
        // RLX represents scalars as `[1]` (see `resolve_shape` for empty meta).
        Some(dims) if dims.is_empty() => Shape::new(&[1], value_dtype),
        _ => Shape::from_dims(meta_s.dims(), value_dtype),
    };
    // `num_elements()==0` means a Static(0) dim (not an empty rank-0 shape —
    // those become `[1]` above). Never emit a 1-element param binding for a
    // 0-element shape: `specialize_params` asserts binding len == shape elems
    // and panics (MOSS prefill on CUDA). Promote empty stubs to a scalar `[1]`
    // so fill length and shape stay consistent.
    let n = match out_s.num_elements() {
        Some(0) => {
            out_s = Shape::new(&[1], value_dtype);
            1
        }
        Some(n) => n.clamp(1, MAX_STUB_ELEMENTS),
        None => 1,
    };
    // ONNX `ConstantOfShape` fills the output with the scalar in its `value`
    // attribute (default 0). It is NOT always zero — e.g. supertonic's duration
    // predictor / vector_estimator build an all-ones mask by prepending a
    // `ConstantOfShape(value=1)` before the real mask; defaulting to 0 zeroed the
    // first mask position and corrupted the whole downstream product.
    // ONNX `value` attribute is a scalar tensor; the parser records its value at
    // `value.tensor.scalar` (see `onnx_file`). Fall back to a bare number / array
    // form and finally 0 (the ONNX default).
    let val: f64 = node
        .attrs
        .get("value")
        .and_then(|v| {
            v.get("tensor")
                .and_then(|t| t.get("scalar"))
                .and_then(|s| s.as_f64())
                .or_else(|| {
                    v.as_array()
                        .and_then(|a| a.first())
                        .and_then(|x| x.as_f64())
                })
                .or_else(|| v.as_f64())
        })
        .unwrap_or(0.0);
    // Non-f32 fills must be written at their NATIVE width. The f32 param path below
    // stores `val as f32` bytes regardless of dtype; a Bool `ConstantOfShape(value=1)`
    // then read its 1-byte elements out of the 4-byte `1.0f32` pattern `[00,00,80,3F]`
    // → `[0,0,1,1]` repeating (MOSS codec's all-ones attention pre-mask → wrong mask
    // → softmax(all -inf) NaN). Emit an exact-width Constant for integer/bool dtypes.
    match out_s.dtype() {
        DType::I64 => {
            let bytes: Vec<u8> = std::iter::repeat_n(val as i64, n)
                .flat_map(i64::to_le_bytes)
                .collect();
            let id = m.add_node(Op::Constant { data: bytes }, vec![], out_s);
            ctx.env.insert(node.outputs[0].clone(), id);
            return Ok(true);
        }
        DType::I32 => {
            let bytes: Vec<u8> = std::iter::repeat_n(val as i32, n)
                .flat_map(i32::to_le_bytes)
                .collect();
            let id = m.add_node(Op::Constant { data: bytes }, vec![], out_s);
            ctx.env.insert(node.outputs[0].clone(), id);
            return Ok(true);
        }
        DType::Bool | DType::U8 | DType::I8 => {
            let bv: u8 = if val != 0.0 { 1 } else { 0 };
            let bytes = vec![bv; n];
            let id = m.add_node(Op::Constant { data: bytes }, vec![], out_s);
            ctx.env.insert(node.outputs[0].clone(), id);
            return Ok(true);
        }
        _ => {}
    }
    let key = format!("__const_of_shape__/{}", node.outputs[0]);
    let id = m.param(&key, out_s);
    ctx.params.insert(key, vec![val as f32; n]);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

pub(super) fn lower_range(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    if node.inputs.len() < 3 {
        ctx.passthrough_stub(m, node)?;
        return Ok(true);
    }
    // Resolve start/limit/delta from initializers, else fold the (possibly
    // dynamic) shape expression — e.g. the sequence mask's `arange(x.size(2))`
    // where the limit is `Gather(Shape(x), 2)`. Without this the limit defaults
    // to 0 and `arange` collapses to length 1, producing a degenerate `[1,1,1]`
    // mask that broadcasts on CPU/Metal but reshape-fails on MLX.
    let scalar = |ctx: &LowerCtx<'_>, m: &HirMut<'_>, n: &str| -> Option<i64> {
        i64_tensor(&ctx.i64_params, &ctx.params, n)
            .and_then(|v| v.first().copied())
            .or_else(|| eval_static_shape_vector(ctx, m, n, 0).and_then(|v| v.first().copied()))
    };
    let start = scalar(ctx, m, &node.inputs[0]).unwrap_or(0);
    let limit = scalar(ctx, m, &node.inputs[1])
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
    // Keep the sign of `delta`: a DESCENDING range (`arange(T-1, -1, -1)`, used by
    // the Zipformer relative-position rel-shift) has `delta < 0`. Clamping it to
    // `max(1)` silently reversed the step and — with the ascending-only length
    // formula below — collapsed the range to length 1, poisoning the rel-shift
    // indices. `delta == 0` is invalid per ONNX; treat as 1 to avoid div-by-zero.
    let delta = scalar(ctx, m, &node.inputs[2]).unwrap_or(1);
    let delta = if delta == 0 { 1 } else { delta };
    let mut len = if delta > 0 {
        if limit > start {
            ((limit - start) as usize).div_ceil(delta as usize)
        } else {
            0
        }
    } else if start > limit {
        ((start - limit) as usize).div_ceil((-delta) as usize)
    } else {
        0
    };
    // Vocos / Soprano decoder: `arange` limit is `4*(s53-1)+1` from Shape math that
    // often fails to fold at import; shape_propagate may also scribble the value_info
    // as head_dim (128). Prefer an explicit `named_lengths` pin, then meta if sane.
    if let Some(&pinned) = ctx
        .opts
        .named_lengths
        .get("4*s53 - 3")
        .or_else(|| ctx.opts.named_lengths.get("4*s53-3"))
    {
        if pinned > 1
            && (node.name.contains("arange")
                || node
                    .outputs
                    .iter()
                    .any(|o| o == "arange" || o.ends_with("/arange")))
        {
            len = pinned;
        }
    }
    if len <= 1 {
        if let Some(meta) = node.output_meta.first() {
            if let Ok(sh) = resolve_shape(meta, ctx.opts) {
                if sh.rank() == 1 {
                    let meta_len = dim_usize(sh.dim(0), ctx.opts);
                    // Reject head_dim-sized stale meta when a Vocos pin exists.
                    let looks_like_headdim = meta_len == 128 || meta_len == 64;
                    if meta_len > 1 && !looks_like_headdim {
                        len = meta_len;
                    }
                }
            }
        }
    }
    // `Range`'s output dtype follows its operands (ONNX). A FLOAT range must stay
    // f32: RoPE builds `inv_freq = exp(arange(0,dim,2).float() · -log(base)/dim)`,
    // and emitting the arange as i64 makes the downstream `Range·factor` a mixed
    // i64×f32 binary (read as garbage) → `Exp` overflows to +inf → `Cos`/`Sin` NaN →
    // the whole attention NaNs (MOSS codec decoder). Decide from the operands' HIR
    // dtype (authoritative); fall back to i64 for integer ranges (indices / masks).
    let operand_is_float = |inp: &str| -> bool {
        if let Some(&id) = ctx.env.get(inp) {
            matches!(m.shape(id).dtype(), DType::F32 | DType::F16 | DType::F64)
        } else {
            ctx.params.contains_key(inp) && !ctx.i64_params.contains_key(inp)
        }
    };
    let is_float = node.inputs.iter().take(3).any(|inp| operand_is_float(inp));
    let n = len.max(1);
    let (bytes, dtype): (Vec<u8>, DType) = if is_float {
        (
            (0..n)
                .flat_map(|i| ((start + i as i64 * delta) as f32).to_le_bytes())
                .collect(),
            DType::F32,
        )
    } else {
        (
            (0..n)
                .flat_map(|i| (start + i as i64 * delta).to_le_bytes())
                .collect(),
            DType::I64,
        )
    };
    let id = m.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(&[n], dtype),
    );
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

/// ONNX `STFT` (opset 17) → a static constant-DFT-matmul subgraph of existing
/// ops (no new backend op; runs on every backend for free). Requires a STATIC
/// signal length (true once a graph-split binds the decoder's frame count).
///
/// `signal [B,L]` (or `[B,L,1]`), `frame_step`, `window [N]`, `frame_length N`,
/// attr `onesided` → `[B, n_frames, n_bins, 2]` (last axis = [real, imag]).
/// Decomposition (verified bit-close vs onnxruntime): frame the signal via a
/// baked sliding-window index `Gather` (`idx[m,n]=m·step+n`), then two matmuls
/// with the window FOLDED into the DFT matrices —
///   `COSw[n,k]=win[n]·cos(2πkn/N)`, `NSINw[n,k]=−win[n]·sin(2πkn/N)` —
/// so `re=frames·COSw`, `im=frames·NSINw`, interleaved on a new last axis.
pub(super) fn lower_stft(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    use std::f64::consts::PI;
    let signal = ctx.tensor(&node.inputs[0])?;
    let mut sig_s = m.shape(signal).clone();
    // Normalize signal to [B, L] (ONNX allows a trailing channel-1 axis).
    let signal = if sig_s.rank() == 3 && sig_s.dim(2).unwrap_static() == 1 {
        let b = sig_s.dim(0).unwrap_static();
        let l = sig_s.dim(1).unwrap_static();
        let r = m.reshape_(signal, vec![b as i64, l as i64]);
        sig_s = m.shape(r).clone();
        r
    } else {
        signal
    };
    if sig_s.rank() != 2 {
        ctx.unsupported("STFT(signal rank != 2)");
        ctx.passthrough_stub(m, node)?;
        return Ok(true);
    }
    let scalar = |ctx: &LowerCtx<'_>, m: &HirMut<'_>, n: Option<&String>| -> Option<i64> {
        let n = n.filter(|s| !s.is_empty())?;
        i64_tensor(&ctx.i64_params, &ctx.params, n)
            .and_then(|v| v.first().copied())
            .or_else(|| eval_static_shape_vector(ctx, m, n, 0).and_then(|v| v.first().copied()))
    };
    let batch = sig_s.dim(0).unwrap_static();
    let sig_len = sig_s.dim(1).unwrap_static();
    let frame_step = scalar(ctx, m, node.inputs.get(1)).unwrap_or(0) as usize;
    // window values (f32); default rectangular. `frame_length` defaults to the
    // window length when the (optional) 4th input is absent.
    let window: Option<Vec<f32>> = node
        .inputs
        .get(2)
        .filter(|s| !s.is_empty())
        .and_then(|n| ctx.params.get(n).cloned());
    let frame_length = scalar(ctx, m, node.inputs.get(3))
        .map(|v| v as usize)
        .or_else(|| window.as_ref().map(|w| w.len()))
        .unwrap_or(0);
    if frame_length == 0 || frame_step == 0 || sig_len < frame_length {
        ctx.unsupported("STFT(non-static / degenerate frame params)");
        ctx.passthrough_stub(m, node)?;
        return Ok(true);
    }
    let n = frame_length;
    let onesided = node
        .attrs
        .get("onesided")
        .and_then(|v| v.as_i64())
        .unwrap_or(1)
        != 0;
    let n_bins = if onesided { n / 2 + 1 } else { n };
    let n_frames = (sig_len - n) / frame_step + 1;
    let win: Vec<f32> = window.unwrap_or_else(|| vec![1.0; n]);
    if win.len() != n {
        ctx.unsupported("STFT(window length != frame_length)");
        ctx.passthrough_stub(m, node)?;
        return Ok(true);
    }
    // Framing gather: idx[m*n + j] = m*frame_step + j  (sliding, hop < N overlaps).
    let idx: Vec<i64> = (0..n_frames)
        .flat_map(|mf| (0..n).map(move |j| (mf * frame_step + j) as i64))
        .collect();
    let idx_bytes: Vec<u8> = idx.iter().flat_map(|v| v.to_le_bytes()).collect();
    let idx_id = m.add_node(
        Op::Constant { data: idx_bytes },
        vec![],
        Shape::new(&[n_frames * n], DType::I64),
    );
    // Gather frames along the per-signal length axis (axis 1) — one framing
    // pattern applied to EVERY batch row, so this handles batch > 1 (the
    // ChatterBox speech_encoder's mel STFT frames `[198, 512]` — 198 pre-framed
    // rows, each a single 512-sample window → `[198, 1, 257, 2]`).
    let gathered = m.add_node(
        Op::Gather { axis: 1 },
        vec![signal, idx_id],
        Shape::new(&[batch, n_frames * n], DType::F32),
    );
    // DFT via a plain 2-D matmul over `[batch·n_frames, n]` (avoids batched-matmul
    // ambiguity); reshaped back to per-batch frames afterwards.
    let bf = batch * n_frames;
    let frames = m.reshape_(gathered, vec![bf as i64, n as i64]);
    // Window-folded DFT matrices, row-major [N, n_bins].
    let mut cos_w = vec![0f32; n * n_bins];
    let mut nsin_w = vec![0f32; n * n_bins];
    for nn in 0..n {
        for k in 0..n_bins {
            let ang = 2.0 * PI * (k as f64) * (nn as f64) / (n as f64);
            cos_w[nn * n_bins + k] = (win[nn] as f64 * ang.cos()) as f32;
            nsin_w[nn * n_bins + k] = (-(win[nn] as f64) * ang.sin()) as f32;
        }
    }
    let f32c = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let cos_id = m.add_node(
        Op::Constant { data: f32c(&cos_w) },
        vec![],
        Shape::new(&[n, n_bins], DType::F32),
    );
    let nsin_id = m.add_node(
        Op::Constant {
            data: f32c(&nsin_w),
        },
        vec![],
        Shape::new(&[n, n_bins], DType::F32),
    );
    let re = m.add_node(
        Op::MatMul,
        vec![frames, cos_id],
        Shape::new(&[bf, n_bins], DType::F32),
    );
    let im = m.add_node(
        Op::MatMul,
        vec![frames, nsin_id],
        Shape::new(&[bf, n_bins], DType::F32),
    );
    // Interleave [re, im] on a new trailing axis → [batch, n_frames, n_bins, 2].
    let re4 = m.reshape_(re, vec![batch as i64, n_frames as i64, n_bins as i64, 1]);
    let im4 = m.reshape_(im, vec![batch as i64, n_frames as i64, n_bins as i64, 1]);
    let out = m.concat_(vec![re4, im4], 3);
    ctx.env.insert(node.outputs[0].clone(), out);
    Ok(true)
}

/// ONNX `DFT` (opset 17/20) — inverse onesided / `irfft` path used by Soprano's
/// vocoder: complex `[…, n_bins=N/2+1, …, 2]` → complex `[…, N, …, 2]` with
/// imag≈0 (downstream Slice/Squeeze takes the real part as `c2r`). Forward DFT
/// and full non-onesided spectra are stubbed.
pub(super) fn lower_dft(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    use std::f64::consts::PI;
    let inverse = node
        .attrs
        .get("inverse")
        .and_then(|v| v.as_i64())
        .unwrap_or(0)
        != 0;
    let onesided_attr = node
        .attrs
        .get("onesided")
        .and_then(|v| v.as_i64())
        .unwrap_or(0)
        != 0;
    let axis_attr = node.attrs.get("axis").and_then(|v| v.as_i64()).unwrap_or(1);
    let x = ctx.tensor(&node.inputs[0])?;
    let x_s = m.shape(x).clone();
    if x_s.rank() < 2 || x_s.dim(x_s.rank() - 1).unwrap_static() != 2 {
        ctx.unsupported("DFT(expected trailing complex axis of size 2)");
        ctx.passthrough_stub(m, node)?;
        return Ok(true);
    }
    let rank = x_s.rank();
    let axis = if axis_attr < 0 {
        (rank as i64 - 1 + axis_attr) as usize
    } else {
        axis_attr as usize
    };
    if axis >= rank - 1 {
        ctx.unsupported("DFT(axis on complex dim)");
        ctx.passthrough_stub(m, node)?;
        return Ok(true);
    }
    let n_bins = x_s.dim(axis).unwrap_static();
    let dft_length = {
        let scalar = |ctx: &LowerCtx<'_>, m: &HirMut<'_>, n: Option<&String>| -> Option<i64> {
            let n = n.filter(|s| !s.is_empty())?;
            i64_tensor(&ctx.i64_params, &ctx.params, n)
                .and_then(|v| v.first().copied())
                .or_else(|| eval_static_shape_vector(ctx, m, n, 0).and_then(|v| v.first().copied()))
        };
        scalar(ctx, m, node.inputs.get(1))
            .map(|v| v as usize)
            .unwrap_or(n_bins)
    };
    let onesided = onesided_attr || (inverse && n_bins == dft_length / 2 + 1);
    if !(inverse && onesided && n_bins == dft_length / 2 + 1) {
        ctx.unsupported("DFT(only inverse onesided / irfft is lowered)");
        ctx.passthrough_stub(m, node)?;
        return Ok(true);
    }
    let n = dft_length;
    let dims: Vec<usize> = (0..rank).map(|i| x_s.dim(i).unwrap_static()).collect();
    // Move DFT axis to rank-2: […, n_bins, 2] for a BF×bins matmul.
    let mut perm: Vec<usize> = (0..rank).collect();
    perm.remove(axis);
    perm.insert(rank - 2, axis);
    let need_transpose = perm.iter().enumerate().any(|(i, &p)| i != p);
    let x_t = if need_transpose {
        m.transpose_(x, perm.clone())
    } else {
        x
    };
    let t_dims: Vec<usize> = {
        let mut out_dims = vec![0usize; rank];
        for (i, &p) in perm.iter().enumerate() {
            out_dims[i] = dims[p];
        }
        out_dims
    };
    let bf: usize = t_dims[..rank - 2].iter().product::<usize>().max(1);
    let flat = m.reshape_(x_t, vec![bf as i64, n_bins as i64, 2]);
    let re_n = m.narrow_(flat, 2, 0, 1);
    let im_n = m.narrow_(flat, 2, 1, 1);
    let re = m.reshape_(re_n, vec![bf as i64, n_bins as i64]);
    let im = m.reshape_(im_n, vec![bf as i64, n_bins as i64]);
    // ONNX inverse DFT (incl. onesided / IRFFT as implemented by ORT): uniform
    // `1/N` over the provided bins — NOT numpy/pytorch `irfft`'s Hermitian
    // convention (`2/N` on interior bins). Using the Hermitian 2× makes Vocos
    // frames ~2× too hot and destroys correlation with ORT reference PCM.
    let mut cos_w = vec![0f32; n_bins * n];
    let mut nsin_w = vec![0f32; n_bins * n];
    let scale = 1.0 / (n as f64);
    for k in 0..n_bins {
        for nn in 0..n {
            let ang = 2.0 * PI * (k as f64) * (nn as f64) / (n as f64);
            cos_w[k * n + nn] = (scale * ang.cos()) as f32;
            nsin_w[k * n + nn] = (-scale * ang.sin()) as f32;
        }
    }
    let f32c = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let cos_id = m.add_node(
        Op::Constant { data: f32c(&cos_w) },
        vec![],
        Shape::new(&[n_bins, n], DType::F32),
    );
    let nsin_id = m.add_node(
        Op::Constant {
            data: f32c(&nsin_w),
        },
        vec![],
        Shape::new(&[n_bins, n], DType::F32),
    );
    let y_re = m.add_node(
        Op::MatMul,
        vec![re, cos_id],
        Shape::new(&[bf, n], DType::F32),
    );
    let y_im = m.add_node(
        Op::MatMul,
        vec![im, nsin_id],
        Shape::new(&[bf, n], DType::F32),
    );
    let y = m.add_node(
        Op::Binary(BinaryOp::Add),
        vec![y_re, y_im],
        Shape::new(&[bf, n], DType::F32),
    );
    // Pack as complex with zero imag (matches ONNX DFT output before c2r Slice).
    let zero_s = m.add_node(
        Op::Constant {
            data: f32c(&[0f32]),
        },
        vec![],
        Shape::new(&[], DType::F32),
    );
    let y_i = m.add_node(
        Op::Binary(BinaryOp::Mul),
        vec![y, zero_s],
        Shape::new(&[bf, n], DType::F32),
    );
    let y_r = m.reshape_(y, vec![bf as i64, n as i64, 1]);
    let y_i = m.reshape_(y_i, vec![bf as i64, n as i64, 1]);
    let y_c = m.concat_(vec![y_r, y_i], 2);
    let mut t_out = t_dims.clone();
    t_out[rank - 2] = n;
    let y_shaped = m.reshape_(y_c, t_out.iter().map(|&d| d as i64).collect::<Vec<_>>());
    let out = if need_transpose {
        let mut inv = vec![0usize; rank];
        for (i, &p) in perm.iter().enumerate() {
            inv[p] = i;
        }
        m.transpose_(y_shaped, inv)
    } else {
        y_shaped
    };
    ctx.env.insert(node.outputs[0].clone(), out);
    Ok(true)
}

/// ONNX `Einsum` (opset 12+). The `equation` attribute is forwarded verbatim
/// (UTF-8) in the op attrs; the reference kernel parses it at runtime.
pub(super) fn lower_einsum(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let equation = node
        .attrs
        .get("equation")
        .and_then(|v| v.as_str())
        .context("Einsum missing `equation` attribute")?
        .to_string();
    let inputs: Vec<HirNodeId> = node
        .inputs
        .iter()
        .filter(|s| !s.is_empty())
        .map(|n| ctx.tensor(n))
        .collect::<Result<_>>()?;
    let fallback = *inputs.first().context("Einsum needs at least one input")?;
    // Compute the output shape from the equation + the CONCRETE input shapes (all
    // available here in the lowerer). shape_propagate often can't — its `infer_output`
    // bails if an input isn't yet in its env (e.g. MOSS RoPE's `Einsum('bs,d->bsd',
    // positions[1,seq], inv_freq[32])` whose positions come off an unpropagated
    // Slice/Cast chain), leaving the meta empty so the old `output_shape` fallback
    // used input0's shape and dropped the contracted/expanded axes (`[1,seq]` not
    // `[1,seq,32]`), collapsing the rotary embedding.
    let in_dims: Vec<Vec<i64>> = inputs
        .iter()
        .map(|&id| {
            m.shape(id)
                .dims()
                .iter()
                .map(|d| d.unwrap_static() as i64)
                .collect()
        })
        .collect();
    let out_s = einsum_out_shape(&equation, &in_dims, m.shape(fallback).dtype())
        .unwrap_or_else(|| output_shape(ctx, node, m, fallback));
    let id = m.add_node(
        Op::Custom {
            name: "onnx.Einsum".to_string(),
            num_inputs: inputs.len() as u32,
            attrs: equation.into_bytes(),
        },
        inputs,
        out_s,
    );
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

/// Static Einsum output shape from the equation + concrete input dims. Mirrors the
/// label bookkeeping of shape_propagate's `einsum_output_dims` but on `i64` dims →
/// `Shape`. Returns `None` for ellipsis equations, term/rank mismatches, or an
/// output label absent from the inputs (caller falls back to the meta).
fn einsum_out_shape(equation: &str, in_dims: &[Vec<i64>], dtype: DType) -> Option<Shape> {
    let eq: String = equation.chars().filter(|c| !c.is_whitespace()).collect();
    if eq.contains("...") {
        return None;
    }
    let (lhs, rhs) = match eq.split_once("->") {
        Some((l, r)) => (l.to_string(), Some(r.to_string())),
        None => (eq.clone(), None),
    };
    let terms: Vec<&str> = lhs.split(',').collect();
    if terms.len() != in_dims.len() {
        return None;
    }
    let mut size: HashMap<char, i64> = HashMap::new();
    let mut counts: HashMap<char, usize> = HashMap::new();
    for (term, dims) in terms.iter().zip(in_dims) {
        let chars: Vec<char> = term.chars().collect();
        if chars.len() != dims.len() {
            return None;
        }
        for (c, &d) in chars.iter().zip(dims) {
            size.entry(*c).or_insert(d);
            *counts.entry(*c).or_insert(0) += 1;
        }
    }
    let out_chars: Vec<char> = match rhs {
        Some(r) => r.chars().collect(),
        None => {
            // Implicit output: labels appearing exactly once, in sorted order.
            let mut once: Vec<char> = counts
                .iter()
                .filter(|&(_, &n)| n == 1)
                .map(|(&c, _)| c)
                .collect();
            once.sort_unstable();
            once
        }
    };
    let dims: Option<Vec<usize>> = out_chars
        .iter()
        .map(|c| size.get(c).map(|&d| d.max(0) as usize))
        .collect();
    Some(Shape::new(&dims?, dtype))
}

pub(super) fn lower_resize(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let x0 = ctx.tensor(&node.inputs[0])?;
    // Prefer the SCALES-derived output shape (input × scales) — scales are
    // authoritative for Resize, whereas the declared `output_meta` /
    // propagate_shapes result frequently pins the resize to its INPUT shape
    // (symbolic `num_samples` after a graph-split; a scales-less propagate).
    // `resize_output_shape` now Errs when scales aren't readable, so the
    // `output_meta` fallback still covers sizes-based / dynamic resizes.
    let out_s_final = resize_output_shape(m, ctx, node, x0)
        .or_else(|_| resolve_shape(&node.output_meta[0], ctx.opts))
        .unwrap_or_else(|_| m.shape(x0).clone());
    let x = ensure_nchw_4d(m, x0);
    let in_s = m.shape(x).clone();
    let out_s = ncl_to_nchw_shape(&out_s_final);
    let mode = node
        .attrs
        .get("mode")
        .and_then(|v| v.as_str())
        .unwrap_or("nearest");
    // General 1-D LINEAR resize (`[N,C,1,W]→[N,C,1,W']`, e.g. the ISTFTNet NSF
    // m_source `l_sin_gen` phase up/downsample): two baked-index gathers +
    // per-position blend weights. `src=(o+½)·W/W'−½` for half_pixel (else `o·W/W'`),
    // `out[o]=x[lo]·(1−f)+x[hi]·f`. The generic path only emitted a ZERO stub.
    if (mode == "linear" || mode == "bilinear")
        && in_s.rank() == 4
        && out_s.rank() == 4
        && in_s.dim(2).unwrap_static() == 1
        && out_s.dim(2).unwrap_static() == 1
        && in_s.dim(3).unwrap_static() > 1
    {
        let w_in = in_s.dim(3).unwrap_static();
        let w_out = out_s.dim(3).unwrap_static();
        let nc = (in_s.dim(0).unwrap_static() * in_s.dim(1).unwrap_static()) as i64;
        let half = node
            .attrs
            .get("coordinate_transformation_mode")
            .and_then(|v| v.as_str())
            .map(|s| s == "half_pixel")
            .unwrap_or(false);
        let (mut lo, mut hi, mut wlo, mut whi) = (
            Vec::with_capacity(w_out),
            Vec::with_capacity(w_out),
            Vec::with_capacity(w_out),
            Vec::with_capacity(w_out),
        );
        for o in 0..w_out {
            let src = if half {
                (o as f64 + 0.5) * (w_in as f64) / (w_out as f64) - 0.5
            } else {
                o as f64 * (w_in as f64) / (w_out as f64)
            }
            .max(0.0);
            let l = (src.floor() as i64).clamp(0, w_in as i64 - 1);
            let h = (l + 1).clamp(0, w_in as i64 - 1);
            let f = (src - l as f64).clamp(0.0, 1.0) as f32;
            lo.push(l);
            hi.push(h);
            wlo.push(1.0 - f);
            whi.push(f);
        }
        let flat = m.reshape_(x, vec![nc, w_in as i64]);
        let i64c = |v: &[i64]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let f32c = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let out2 = Shape::new(&[nc as usize, w_out], DType::F32);
        let lo_id = m.add_node(
            Op::Constant { data: i64c(&lo) },
            vec![],
            Shape::new(&[w_out], DType::I64),
        );
        let hi_id = m.add_node(
            Op::Constant { data: i64c(&hi) },
            vec![],
            Shape::new(&[w_out], DType::I64),
        );
        let wlo_id = m.add_node(
            Op::Constant { data: f32c(&wlo) },
            vec![],
            Shape::new(&[1, w_out], DType::F32),
        );
        let whi_id = m.add_node(
            Op::Constant { data: f32c(&whi) },
            vec![],
            Shape::new(&[1, w_out], DType::F32),
        );
        let glo = m.add_node(Op::Gather { axis: 1 }, vec![flat, lo_id], out2.clone());
        let ghi = m.add_node(Op::Gather { axis: 1 }, vec![flat, hi_id], out2.clone());
        let plo = m.add_node(Op::Binary(BinaryOp::Mul), vec![glo, wlo_id], out2.clone());
        let phi = m.add_node(Op::Binary(BinaryOp::Mul), vec![ghi, whi_id], out2.clone());
        let sum = m.add_node(Op::Binary(BinaryOp::Add), vec![plo, phi], out2);
        let id = m.reshape_(
            sum,
            vec![
                out_s_final.dim(0).unwrap_static() as i64,
                out_s_final.dim(1).unwrap_static() as i64,
                w_out as i64,
            ],
        );
        ctx.env.insert(node.outputs[0].clone(), id);
        return Ok(true);
    }
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
        // General 1-D nearest length upsample (`[N,C,1,W]→[N,C,1,W']`, StyleTTS2
        // F0/N upsample + ISTFTNet m_source): flatten to `[N*C, W]`, gather each
        // row with the nearest index `idx[o]=floor(o·W/W')` (ONNX asymmetric +
        // floor), reshape back. The generic path below only emitted a ZERO stub.
        if h_in == 1 && h_out == 1 && w_out != w_in && w_in > 0 {
            let nc = (in_s.dim(0).unwrap_static() * in_s.dim(1).unwrap_static()) as i64;
            let flat = m.reshape_(x, vec![nc, w_in as i64]);
            let idx: Vec<i64> = (0..w_out).map(|o| ((o * w_in) / w_out) as i64).collect();
            let idx_id = m.add_node(
                Op::Constant {
                    data: idx.iter().flat_map(|v| v.to_le_bytes()).collect(),
                },
                vec![],
                Shape::new(&[w_out], DType::I64),
            );
            let gathered = m.add_node(
                Op::Gather { axis: 1 },
                vec![flat, idx_id],
                Shape::new(&[nc as usize, w_out], DType::F32),
            );
            let id = m.reshape_(
                gathered,
                vec![
                    out_s_final.dim(0).unwrap_static() as i64,
                    out_s_final.dim(1).unwrap_static() as i64,
                    w_out as i64,
                ],
            );
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

pub(super) fn lower_random_like(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let shape_in = ctx.tensor(
        node.inputs
            .first()
            .context("Random*Like missing shape input")?,
    )?;
    // Random*Like produces a tensor of the SAME shape as its input; the input is
    // authoritative when static (the ONNX `output_meta` can carry a symbolic dim
    // that shape-propagation defaulted — the S3Gen decoder's `RandomNormalLike`
    // over the length-64 mel came back 128).
    let in_s = m.shape(shape_in).clone();
    let out_s = if in_s.is_static() {
        in_s
    } else {
        let mut s = output_shape(ctx, node, m, shape_in);
        if s.rank() == 0 || s.num_elements().unwrap_or(0) == 0 {
            s = m.shape(shape_in).clone();
        }
        s
    };
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

pub(super) fn lower_random(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
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

pub(super) fn lower_pad_as_concat(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    // Constant (zero) or edge (replicate) padding. Pad amounts come from the
    // `pads` attribute (opset < 11) or, since opset 11, the second input tensor —
    // which may be dynamically computed (VITS relative attention pads embeddings
    // to `2*len-1`). Evaluate that tensor and realize the padding as concats:
    // zero tensors for `constant` mode, replicated edge slices for `edge`.
    // (ConvNeXt depthwise convs — supertonic/luxtts/kokoro CFM decoders — use
    // `mode='edge'`; lowering it as zero-pad corrupts the conv near both borders.)
    let x = ctx.tensor(&node.inputs[0])?;
    let in_s = m.shape(x).clone();
    let rank = in_s.rank();
    let mode = node
        .attrs
        .get("mode")
        .and_then(|v| v.as_str())
        .unwrap_or("constant");
    let edge = mode == "edge" || mode == "replicate";
    let reflect = mode == "reflect";
    let pads: Vec<i64> = node
        .attrs
        .get("pads")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|d| d.as_i64()).collect::<Vec<_>>())
        .filter(|v: &Vec<i64>| !v.is_empty())
        .or_else(|| {
            node.inputs.get(1).filter(|s| !s.is_empty()).and_then(|n| {
                eval_i64_shaped(ctx, m, n, 0)
                    .map(|(d, _)| d)
                    .or_else(|| eval_static_shape_vector(ctx, m, n, 0))
            })
        })
        .unwrap_or_default();
    if pads.is_empty() || pads.iter().all(|&p| p == 0) {
        return lower_reshape(m, ctx, node);
    }
    if pads.len() < 2 * rank || pads.iter().any(|&p| p < 0) {
        if ctx.opts.strict {
            anyhow::bail!("Pad at {} has unsupported pads {pads:?}", node.name);
        }
        ctx.passthrough_stub(m, node)?;
        return Ok(true);
    }
    let dt = in_s.dtype();
    let esize = dt.size_bytes().max(1);
    let mut cur = x;
    let mut dims: Vec<usize> = in_s.dims().iter().map(|d| d.unwrap_static()).collect();
    for a in 0..rank {
        // ONNX pads layout: [begin_0..begin_{r-1}, end_0..end_{r-1}].
        // `reflect` mirrors the signal WITHOUT repeating the edge sample and
        // reflects both sides against the ORIGINAL slice, so it must be built in
        // one shot per axis (numpy `pad(mode='reflect')`: `[1,2,3,4]` pad (2,2)
        // → `[3,2,1,2,3,4,3,2]`). Zero-padding it (the old fallthrough) corrupts
        // every reflect-padded frontend — e.g. Kokoro's ISTFTNet source STFT,
        // whose centering pad feeds the phase and blows the whole vocoder up.
        if reflect {
            let before_amt = pads[a].max(0) as usize;
            let after_amt = pads[rank + a].max(0) as usize;
            if before_amt == 0 && after_amt == 0 {
                continue;
            }
            let len_a = dims[a];
            if before_amt >= len_a || after_amt >= len_a {
                // Reflect can mirror at most `len-1` samples; anything larger is
                // an invalid/degenerate pad — bail rather than emit garbage.
                if ctx.opts.strict {
                    anyhow::bail!("Pad(reflect) at {} exceeds axis length", node.name);
                }
                ctx.passthrough_stub(m, node)?;
                return Ok(true);
            }
            let mut parts: Vec<HirNodeId> = Vec::with_capacity(before_amt + 1 + after_amt);
            // before block, output order x[before_amt], x[before_amt-1], …, x[1].
            for idx in (1..=before_amt).rev() {
                parts.push(m.narrow_(cur, a, idx, 1));
            }
            parts.push(cur);
            // after block, output order x[len-2], x[len-3], …, x[len-1-after_amt].
            for k in 0..after_amt {
                parts.push(m.narrow_(cur, a, len_a - 2 - k, 1));
            }
            cur = m.concat_(parts, a);
            dims[a] += before_amt + after_amt;
            continue;
        }
        for (amt, before) in [(pads[a], true), (pads[rank + a], false)] {
            if amt <= 0 {
                continue;
            }
            let pad_block = if edge {
                // Replicate the current edge slice `amt` times along axis `a`.
                // `before` copies element 0; `after` copies the last element
                // (indices already include any before-pad added this iteration).
                let start = if before { 0 } else { dims[a] - 1 };
                let edge_slice = m.narrow_(cur, a, start, 1);
                if amt == 1 {
                    edge_slice
                } else {
                    let copies = vec![edge_slice; amt as usize];
                    m.concat_(copies, a)
                }
            } else {
                let mut zshape = dims.clone();
                zshape[a] = amt as usize;
                let numel: usize = zshape.iter().product();
                m.add_node(
                    Op::Constant {
                        data: vec![0u8; numel * esize],
                    },
                    vec![],
                    Shape::new(&zshape, dt),
                )
            };
            let inputs = if before {
                vec![pad_block, cur]
            } else {
                vec![cur, pad_block]
            };
            cur = m.concat_(inputs, a);
            dims[a] += amt as usize;
        }
    }
    ctx.env.insert(node.outputs[0].clone(), cur);
    Ok(true)
}
