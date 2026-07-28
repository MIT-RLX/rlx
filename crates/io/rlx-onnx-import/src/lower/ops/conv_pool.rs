// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `conv_pool` — extracted from the `ops` module for navigability (see `mod.rs`).

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

/// Decompose a 1D `ConvTranspose` into zero-insertion + a regular forward `Conv`
/// (with the kernel reversed and Cin/Cout transposed at the data level). This is
/// for backends without a native transposed-conv kernel (wgpu / CoreML); it emits
/// only ops they already support (reshape / concat / slice / conv2d) and reuses the
/// proven forward-conv lowering for the heavy lifting. Returns `false` (no rewrite)
/// when the weight is not a static f32 1D initializer, so the caller falls back to
/// the native path.
pub(super) fn lower_conv_transpose_decomposed(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let w_name = node.inputs[1].clone();
    let Some(w_shape) = ctx.init_shapes.get(&w_name).cloned() else {
        return Ok(false);
    };
    let Some(w_data) = ctx.params.get(&w_name).cloned() else {
        return Ok(false);
    };
    if w_shape.len() < 3 {
        return Ok(false);
    }
    // ONNX ConvTranspose weight: `[Cin, Cout/g, kH, kW]`; 1D collapses one spatial dim.
    let cin = w_shape[0];
    let cout = w_shape[1];
    let k: usize = w_shape[2..].iter().product();
    if k == 0 || cin == 0 || cout == 0 || cin * cout * k != w_data.len() {
        return Ok(false);
    }

    let (_kk, st, pad, dil) = onnx_pads(node);
    let stride = st[0].max(st[1]).max(1);
    let dilation = dil[0].max(dil[1]).max(1);
    let (pad_b, pad_e) = (pad[0], pad[1]);
    let out_pad = node
        .attrs
        .get("output_padding")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .and_then(|d| d.as_u64())
        .unwrap_or(0) as usize;

    // Reversed + transposed weight: W'[Cout, Cin, k] = W[Cin, Cout, k-1-j].
    let mut wp = vec![0f32; w_data.len()];
    for co in 0..cout {
        for ci in 0..cin {
            for j in 0..k {
                wp[(co * cin + ci) * k + j] = w_data[(ci * cout + co) * k + (k - 1 - j)];
            }
        }
    }
    let w_key = format!("{w_name}__ctdec_w");
    ctx.params.insert(w_key.clone(), wp);
    let w_node = m.param(&w_key, Shape::new(&[cout, cin, k], DType::F32));
    ctx.env.insert(w_key.clone(), w_node);

    // Normalise input to NCL `[n, c, L]`.
    let x0 = ctx.tensor(&node.inputs[0])?;
    let xs = m.shape(x0).clone();
    let xdims: Vec<usize> = xs.dims().iter().map(|d| d.unwrap_static()).collect();
    if xdims.len() < 2 {
        return Ok(false);
    }
    let n = xdims[0];
    let c = xdims[1];
    if n * c == 0 {
        return Ok(false);
    }
    let total: usize = xdims.iter().product();
    let l = total / (n * c);
    let dt = xs.dtype();
    let x_ncl = if xdims.len() == 3 && xdims[2] == l {
        x0
    } else {
        m.reshape_(x0, vec![n as i64, c as i64, l as i64])
    };

    // Zero-insert along length by `stride` → `[n, c, (l-1)*stride + 1]`.
    // Expand a scalar 0 — do NOT materialize `n·c·l·(stride-1)` zero bytes as a
    // Constant (F5 Vocos ISTFT inflate is hundreds of MB; the old path also
    // forced MLX's subsequent forward-conv im2col into the hundreds of GB).
    let z_ncl = if stride > 1 {
        let z4 = m.reshape_(x_ncl, vec![n as i64, c as i64, l as i64, 1]);
        let gap = stride - 1;
        let zshape = [n, c, l, gap];
        let zero_scalar = m.add_node(
            Op::Constant {
                data: vec![0u8; dt.size_bytes().max(1)],
            },
            vec![],
            Shape::new(&[1], dt),
        );
        let zeros = m.add_node(
            Op::Expand {
                target_shape: zshape.iter().map(|&d| d as i64).collect(),
            },
            vec![zero_scalar],
            Shape::new(&zshape, dt),
        );
        let cat = m.concat_(vec![z4, zeros], 3); // [n, c, l, stride]
        let flat = m.reshape_(cat, vec![n as i64, c as i64, (l * stride) as i64]);
        let keep = (l - 1) * stride + 1;
        m.narrow_(flat, 2, 0, keep)
    } else {
        x_ncl
    };
    let z_key = format!("{}__ctdec_zins", node.name);
    ctx.env.insert(z_key.clone(), z_ncl);

    // Forward conv: pad = dilation*(k-1) - pad_orig, with output_padding on the end.
    let new_pad_b = (dilation * (k - 1)).saturating_sub(pad_b);
    let new_pad_e = (dilation * (k - 1)).saturating_sub(pad_e) + out_pad;
    let mut attrs = node.attrs.clone();
    attrs.insert("kernel_shape".into(), serde_json::json!([k]));
    attrs.insert("strides".into(), serde_json::json!([1]));
    attrs.insert("pads".into(), serde_json::json!([new_pad_b, new_pad_e]));
    attrs.insert("dilations".into(), serde_json::json!([dilation]));
    attrs.insert("group".into(), serde_json::json!(1));
    attrs.remove("output_padding");

    let mut inputs = vec![z_key, w_key];
    if node.inputs.len() >= 3 && !node.inputs[2].is_empty() {
        inputs.push(node.inputs[2].clone()); // bias
    }
    let synth = BundleNode {
        name: format!("{}__ctdec_conv", node.name),
        op: "Conv".to_string(),
        inputs,
        outputs: node.outputs.clone(),
        attrs,
        output_meta: node.output_meta.clone(),
    };
    lower_conv(m, ctx, &synth, false)
}

pub(super) fn lower_conv(
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
    // Decompose a 1D ConvTranspose into zero-insertion + a regular Conv (kernel
    // reversed at the data level) for backends without a native transposed-conv
    // kernel (wgpu / CoreML). Reuses the proven forward-conv lowering.
    if transpose && ctx.opts.decompose_conv_transpose && groups == 1 {
        if lower_conv_transpose_decomposed(m, ctx, node)? {
            return Ok(true);
        }
    }
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
    // Disambiguate BLC vs NCL for a rank-3 forward-conv input using the weight's
    // declared in_channels (concrete, not heuristic). The `is_vocoder_blc` guard
    // in `ensure_nchw_4d` misses non-vocoder BLC tensors — e.g. the VITS FFN /
    // `enc_q` WaveNet carry `[1, L, C]` (channel-last), which would otherwise be
    // read as NCL (channels=L), leaking in_channels into the length dim (a conv
    // over `[1,64,320]` wrongly yields `[1,80,320]` instead of `[1,80,64]`). When
    // the last axis matches in_channels and the middle axis does not, it is
    // unambiguously channel-last — transpose to NCL.
    if !transpose {
        let s = m.shape(x0).clone();
        let w_s = m.shape(w).clone();
        if s.rank() == 3 && w_s.rank() >= 2 {
            let in_ch = w_s.dim(1).unwrap_static() * groups;
            let mid = s.dim(1).unwrap_static();
            let last = s.dim(2).unwrap_static();
            if last == in_ch && mid != in_ch {
                x0 = m.transpose_(x0, vec![0, 2, 1]);
            }
        }
    }
    // Canonicalize rank-4 1-D activations that carry length on H (`[N,C,L,1]`)
    // — STFT/`Transpose` in Kokoro/StyleTTS2 `noise_convs` — to the `[N,C,1,L]`
    // layout that `ensure_nchw_4d` produces for rank-3 NCL. A rank-3 weight
    // always places its kernel on W; without this remap the kernel hits W=1
    // and the length collapses to 1, after which bias `[C]` right-aligns to
    // invent a phantom `[1,C,C]` (then fails reshaping to the real upsample
    // length, e.g. 1040).
    let mut canonicalized_rank4_1d = false;
    if !transpose {
        let s = m.shape(x0).clone();
        let w_s = m.shape(w).clone();
        if s.rank() == 4 && w_s.rank() == 3 {
            let (n, c, h, wd) = (
                s.dim(0).unwrap_static(),
                s.dim(1).unwrap_static(),
                s.dim(2).unwrap_static(),
                s.dim(3).unwrap_static(),
            );
            if h > 1 && wd == 1 {
                x0 = m.reshape_(x0, vec![n as i64, c as i64, 1, h as i64]);
                canonicalized_rank4_1d = true;
            }
        }
    }
    let (mut kernel, stride, pad, dilation) = onnx_pads(node);
    // PyTorch / ONNX Runtime often omit `kernel_shape` and infer it from the
    // weight. `onnx_pads` defaults missing attrs to `[1]`, which turns a 1-D
    // depthwise Conv (Soprano Vocos ConvNeXt, weight `[C,1,3]`, pads=`[1,1]`)
    // into a length-expanding pointwise op (L=161 → 163) and desyncs every
    // residual Add. Prefer the weight's trailing spatial dims when the attr
    // is absent.
    if !node.attrs.contains_key("kernel_shape") {
        let w_s = m.shape(w).clone();
        if w_s.rank() >= 3 {
            let k0 = w_s.dim(2).unwrap_static();
            let k1 = if w_s.rank() >= 4 {
                w_s.dim(3).unwrap_static()
            } else {
                1
            };
            kernel = [k0, k1];
        }
    }
    // ONNX ConvTranspose `output_padding` (extra size added to the OUTPUT length,
    // needed for a stride-2 upsample to double exactly: 74→148, not 74→147). The
    // shape computation below hardcoded 0, so the depthwise `pool/ConvTranspose`
    // in Kokoro's ISTFTNet decoder came out one short and truncated the whole
    // vocoder via a length-mismatched residual Add.
    let out_pad_len: usize = node
        .attrs
        .get("output_padding")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .and_then(|d| d.as_u64())
        .unwrap_or(0) as usize;
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
    // `propagate_shapes` sometimes records a conv's `output_meta` in BLC layout
    // (channels last), while lowering feeds the conv NCL data — so the meta label
    // is a transpose of the tensor the conv actually produces (`[1,64,320]` meta
    // vs `[1,320,64]` data for a 320-out-channel FFN conv). A downstream `Pad`
    // then pads the wrong axis. Detect the mismatch via the weight's true
    // out_channels and recompute from operands so the label matches the data.
    let expected_cout = {
        let w_s = m.shape(w).clone();
        if transpose {
            w_s.dim(1).unwrap_static() * groups
        } else {
            w_s.dim(0).unwrap_static()
        }
    };
    let meta_layout_transposed = !meta_empty
        && rank0 == 3
        && out_shape.rank() == 3
        && out_shape.dim(1).unwrap_static() != expected_cout
        && out_shape.dim(2).unwrap_static() == expected_cout;
    // The recorded meta can carry a STALE output length when the conv's input
    // length is dynamic (MOSS codec's `code_length`): `propagate_shapes` resolved
    // the symbolic length to 1, so a 1×1 `out_proj` conv over `[1,8,4]` was
    // labelled `[1,512,1]` and every codec conv collapsed. Recompute from the
    // concrete HIR input length and re-derive when they disagree. (Forward 1-D
    // convs only; the transpose path already recomputes via its own branches.)
    let meta_len_stale = !meta_empty
        && !transpose
        && rank0 == 3
        && out_shape.rank() == 3
        && out_shape.dim(1).unwrap_static() == expected_cout
        && {
            let li = in_s0.dim(2).unwrap_static();
            let s = stride[0].max(1);
            let eff = dilation[0] * kernel[0].saturating_sub(1);
            let lo = (li + 2 * pad[0]).saturating_sub(eff).saturating_sub(1) / s + 1;
            out_shape.dim(2).unwrap_static() != lo
        };
    // Same STALE-length problem for TRANSPOSE convs. `meta_len_stale` above is
    // forward-only, and the transpose recompute branches below are GATED by this
    // very `if`, so a stale-but-non-empty transpose meta was trusted verbatim —
    // the ChatterBox S3Gen vocoder `ups.*` ConvTransposes (input 24, meta 128,
    // true 192 = 8·24 upsample) kept 128, breaking the whole vocoder length.
    let meta_len_stale_transpose = !meta_empty
        && transpose
        && rank0 == 3
        && out_shape.rank() == 3
        && out_shape.dim(1).unwrap_static() == expected_cout
        && {
            let li = in_s0.dim(2).unwrap_static();
            let lo = rlx_ir::shape::conv_transpose2d_spatial_output(
                li,
                kernel[0],
                stride[0],
                pad[0],
                dilation[0],
                out_pad_len,
            );
            out_shape.dim(2).unwrap_static() != lo
        };
    if meta_empty
        || out_shape.rank() < 2
        || meta_layout_transposed
        || meta_len_stale
        || meta_len_stale_transpose
        || canonicalized_rank4_1d
    {
        let w_s = m.shape(w).clone();
        let wi = w_s.dim(1).unwrap_static();
        let wc = w_s.dim(0).unwrap_static();
        let n = if rank0 > 0 {
            in_s0.dim(0).unwrap_static()
        } else {
            1
        };
        let c_out = if transpose { wi * groups } else { wc };
        let onnx_1d = rank0 == 3
            || canonicalized_rank4_1d
            || (rank0 == 4 && in_s0.dim(2).unwrap_static() == 1);
        if transpose && rank0 == 4 && !onnx_1d {
            let h = in_s0.dim(2).unwrap_static();
            let w = in_s0.dim(3).unwrap_static();
            let h_out = rlx_ir::shape::conv_transpose2d_spatial_output(
                h,
                kernel[0],
                stride[0],
                pad[0],
                dilation[0],
                out_pad_len,
            );
            let w_out = rlx_ir::shape::conv_transpose2d_spatial_output(w, 1, 1, 0, 1, 0);
            out_shape = Shape::new(&[n, c_out, h_out, w_out], in_s0.dtype());
        } else if !transpose
            && rank0 == 4
            && !onnx_1d
            && m.shape(w).rank() == 4
            && in_s0.dim(2).unwrap_static() > 1
            && in_s0.dim(3).unwrap_static() > 1
        {
            // Genuine 2D forward conv: a rank-4 weight `[out,in,kh,kw]` with both
            // input spatial dims > 1 — e.g. the StyleTTS/OpenVoice ReferenceEncoder
            // convolving a spectrogram as an image. The 1D path below computes only
            // the last (width) spatial output and collapses the height dim to 1;
            // compute BOTH here. (A rank-3 weight = 1D conv even on a 4D-shaped
            // input, so it must NOT take this branch.)
            let conv_out = |sz: usize, k: usize, s: usize, p: usize, d: usize| {
                let s = s.max(1);
                let eff = d * k.saturating_sub(1);
                (sz + 2 * p).saturating_sub(eff).saturating_sub(1) / s + 1
            };
            let h = in_s0.dim(2).unwrap_static();
            let w = in_s0.dim(3).unwrap_static();
            let h_out = conv_out(h, kernel[0], stride[0], pad[0], dilation[0]);
            let w_out = conv_out(w, kernel[1], stride[1], pad[1], dilation[1]);
            out_shape = Shape::new(&[n, c_out, h_out, w_out], in_s0.dtype());
        } else {
            let l = if onnx_1d {
                if rank0 == 3 {
                    in_s0.dim(2).unwrap_static()
                } else {
                    // Rank-4 1-D (including remapped `[N,C,L,1]` → `[N,C,1,L]`):
                    // length is on W. Using H here collapses L_out to 1.
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
                    out_pad_len,
                )
            } else if !transpose {
                // Standard conv: (l + pad_begin + pad_end − dilation·(k−1) − 1)/stride
                // + 1. Reduces to `l` for same-padding (decoder, attention 1×1) and
                // correctly shrinks "valid" convs (pad=0, k>1) fed by an explicit Pad
                // (VITS FFN). Uses pad[0]+pad[1] (total, ASYMMETRIC-safe) not 2·pad[0]:
                // the ChatterBox S3Gen pre-lookahead conv has pads=[0,3], so the old
                // symmetric assumption gave 32→29 instead of 32, cascading the whole
                // encoder length. For symmetric pads pad[0]+pad[1] == 2·pad[0].
                let s = stride[0].max(1);
                let eff = dilation[0] * kernel[0].saturating_sub(1);
                (l + pad[0] + pad[1]).saturating_sub(eff).saturating_sub(1) / s + 1
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
        let w_rank = w_s.rank();
        // A 1-D ConvTranspose keeps its length in the W axis (`ensure_nchw_4d`/
        // `ncl_to_nchw_shape` map NCL → `[N,C,1,L]`), so the kernel/stride/pad —
        // AND the weight's kernel axis — must sit in W, not H. The old layout put
        // the kernel in H (size 1): only the middle tap survived and W passed
        // through with kw=1, so a depthwise `pool/ConvTranspose` (StyleTTS2 F0/N
        // predictor, Kokoro) produced garbage (cos 0.08).
        //
        // `Op::ConvTranspose2d` weight layout matches ONNX/PyTorch:
        // `[C_in, C_out/groups, kH, kW]`. Do **not** Cin↔Cout-transpose: that
        // left depthwise CTs unchanged (`C_out/g == 1`) but destroyed dense
        // upsamples (Kokoro ISTFTNet `ups.0`, cos ≈ 0.02 vs ORT).
        let is_1d = rank0 == 3
            || canonicalized_rank4_1d
            || (rank0 == 4 && in_s0.dim(2).unwrap_static() == 1);
        let w_rlx = if w_rank >= 4 {
            w
        } else if is_1d {
            m.reshape_(w, vec![wc as i64, wi as i64, 1, wk as i64])
        } else {
            m.reshape_(w, vec![wc as i64, wi as i64, wk as i64, 1])
        };
        let (k2, s2, p2, d2) = if is_1d {
            (
                [1, kernel[0]],
                [1, stride[0]],
                [0, pad[0]],
                [1, dilation[0]],
            )
        } else {
            (kernel, stride, pad, dilation)
        };
        // 1-D output_padding also belongs on the W axis (see the kernel/weight
        // remap above). CPU folds it into `out_shape`, but keep it consistent for
        // backends that apply it directly.
        let out_pad = if is_1d { [0, out_pad[0]] } else { out_pad };
        m.conv_transpose2d(x, w_rlx, k2, s2, p2, d2, out_pad, groups, out_shape.clone())
    } else if !transpose && rank >= 4 {
        let w_s = m.shape(w).clone();
        let w_rank = w_s.rank();
        // A 1-D forward conv keeps its length in the W axis (`ensure_nchw_4d` maps
        // NCL `[N,C,L]` → `[N,C,1,L]`), so the kernel/stride/pad/dilation — AND the
        // weight's kernel axis — must sit in W, not H. This mirrors the 1-D
        // ConvTranspose fix above: putting the kernel in H (size 1) collapses a real
        // K-tap conv to a single center-tap pointwise op (only the middle weight
        // survives). For a strided STFT-as-conv front-end (F5-TTS mel: kernel 1024 /
        // stride 256) it both mis-strided the length (12000 frames vs 47) AND
        // destroyed the values. Genuine 2-D convs (rank-4 weight) keep both axes.
        let w_1d = w_rank < 4;
        let w_in = if w_rank >= 4 {
            w
        } else {
            let wc = w_s.dim(0).unwrap_static();
            let wi = w_s.dim(1).unwrap_static();
            let wk = w_s.dim(2).unwrap_static();
            m.reshape_(w, vec![wc as i64, wi as i64, 1, wk as i64])
        };
        let k2 = [
            if w_1d { 1 } else { kernel[0] },
            if w_1d { kernel[0] } else { kernel[1] },
        ];
        let s2 = [
            if w_1d { 1 } else { stride[0] },
            if w_1d { stride[0] } else { stride[1] },
        ];
        let p2 = [
            if w_1d { 0 } else { pad[0] },
            if w_1d { pad[0] } else { pad[1] },
        ];
        // Emit `Op::Conv` directly so the real dilation is preserved — the
        // `conv2d` helper hard-codes `dilation=[1,1]`, which silently turns the
        // dilated resblock convs (HiFi-GAN MRF, dilations 1/3/5) into stride-1
        // convs and corrupts the waveform.
        let d2 = [
            if w_1d { 1 } else { dilation[0] },
            if w_1d { dilation[0] } else { dilation[1] },
        ];
        m.add_node(
            Op::Conv {
                kernel_size: k2.to_vec(),
                stride: s2.to_vec(),
                padding: p2.to_vec(),
                dilation: d2.to_vec(),
                groups,
            },
            vec![x, w_in],
            out_shape,
        )
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
    // The conv output is deterministically NCHW `[n, c_out, h, w]` — we just
    // constructed it. Only run the ambiguity-resolving collapse when the channel
    // axis is NOT already the true out_channels, so it cannot misread a large
    // filter dim (e.g. the VITS FFN's 320, absent from `is_typical_channel`) as a
    // length and transpose a correct result into BLC (`[1,320,1,64]`→`[1,64,320]`).
    let id_s = m.shape(id).clone();
    if id_s.rank() != 4 || id_s.dim(1).unwrap_static() != expected_cout {
        id = collapse_duplicate_channel_4d(m, id);
    }
    // Collapse a genuine-1D conv result (NCHW with a singleton spatial axis) back
    // to NCL *before* adding the bias, so `binary_infer` sees a rank-3 operand and
    // never runs its 4D channel-disambiguation on a correct `[1,C,1,L]` (which,
    // for a non-"typical" out-channel like the VITS FFN's 320, would transpose it
    // to BLC). Also collapse when the ONNX input was already rank-4 1-D
    // (`[N,C,1,L]` or remapped `[N,C,L,1]`) — otherwise Kokoro `noise_convs` stay
    // 4D, AdaIN `ReduceMean` hits the singleton H, and Add with ups `[N,C,L]`
    // invents `[N,C,C,L]`.
    let rank4_1d = rank0 == 4
        && !transpose
        && m.shape(w).rank() < 4
        && (in_s0.dim(2).unwrap_static() == 1 || in_s0.dim(3).unwrap_static() == 1);
    if rank0 == 3 || canonicalized_rank4_1d || rank4_1d {
        let cur = m.shape(id).clone();
        if cur.rank() == 4 && cur.dim(1).unwrap_static() == expected_cout {
            let (n, c) = (cur.dim(0).unwrap_static(), cur.dim(1).unwrap_static());
            let (h, w) = (cur.dim(2).unwrap_static(), cur.dim(3).unwrap_static());
            let l = if w == 1 {
                Some(h)
            } else if h == 1 {
                Some(w)
            } else {
                None
            };
            if let Some(l) = l {
                id = m.reshape_(id, vec![n as i64, c as i64, l as i64]);
            }
        }
    }
    if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
        let bias = ctx.tensor(&node.inputs[2])?;
        let act = m.shape(id).clone();
        // The conv bias is per-output-channel `[C]`, so it broadcasts over the
        // batch (and spatial) axes. Its reshaped leading dim MUST be 1 — using
        // the activation's actual batch (`act.dim(0)`) reshapes `[C]` into
        // `[N,C,1]`, which for N>1 asks for N·C elements from a C-element buffer,
        // so batch elements ≥1 read out-of-bounds garbage bias. Silent for the
        // batch-1 inference path (N=1 is a no-op) but corrupts every batched
        // conv (e.g. CFG's batch-2 vector estimator). Broadcast from 1 instead.
        let bias_in = if m.shape(bias).rank() == 1 {
            let bc = m.shape(bias).dim(0).unwrap_static();
            if act.rank() == 4 && act.dim(1).unwrap_static() == bc {
                m.reshape_(bias, vec![1, bc as i64, 1, 1])
            } else if act.rank() == 3 && is_blc_rank3(&act) && act.dim(2).unwrap_static() == bc {
                m.reshape_(bias, vec![1, 1, bc as i64])
            } else if act.rank() == 3
                && (is_ncl_rank3(&act) || is_vocoder_ncl(&act) || is_nc1_rank3(&act))
                && act.dim(1).unwrap_static() == bc
            {
                // Include `[N,C,1]` (`is_nc1_rank3`): leaving bias as `[C]`
                // right-aligns under NumPy rules to invent `[1,C,C]`.
                m.reshape_(bias, vec![1, bc as i64, 1])
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
                vec![1, m.shape(bias).dim(1).unwrap_static() as i64, 1, 1],
            )
        } else {
            bias
        };
        id = binary_infer_add(m, id, bias_in, &node.name);
    }
    // Collapse a 1D-conv result (lowered through NCHW with a singleton spatial axis)
    // back to NCL when the ONNX input was genuinely 1-D (rank-3 NCL or rank-4 with
    // a singleton spatial axis), so it lines up with the rest of a 3D graph for
    // elementwise ops (attention/residual adds) and lets downstream
    // `Shape`/`Gather(axis=2)` read the real length.
    if rank0 == 3 || canonicalized_rank4_1d || rank4_1d {
        let cur = m.shape(id).clone();
        if cur.rank() == 4 {
            let (n, c) = (cur.dim(0).unwrap_static(), cur.dim(1).unwrap_static());
            let (h, w) = (cur.dim(2).unwrap_static(), cur.dim(3).unwrap_static());
            let l = if w == 1 {
                Some(h)
            } else if h == 1 {
                Some(w)
            } else {
                None
            };
            if let Some(l) = l {
                id = m.reshape_(id, vec![n as i64, c as i64, l as i64]);
            }
        }
    }
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

pub(super) fn lower_pool(
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
    // A 1-D ONNX pool over NCL `[N,C,L]` arrives rank-3. The ChatterBox
    // speech_encoder's ECAPA CAM layers pool the ENTIRE time axis with
    // `AveragePool [T,1]` (kernel == time, stride == kernel) → a single global
    // output frame. Lower that global-context case as `Narrow(window) + Reduce`
    // over the time axis: `Reduce` has correct shape inference on every backend
    // (cpu/metal/mlx/wgpu/coreml), whereas promoting to a rank-4 `Op::Pool`
    // tripped a downstream shape-sync pass that reset the pooled length back to
    // the full input length → CPU-kernel OOB. Genuine multi-window 1-D pooling
    // (ol > 1) still promotes to rank-4 `Op::Pool` below.
    let in_s = m.shape(x).clone();
    if in_s.rank() == 3 && op != "GlobalAveragePool" {
        let n = in_s.dim(0).unwrap_static();
        let c = in_s.dim(1).unwrap_static();
        let l = in_s.dim(2).unwrap_static();
        let kh = kernel_size.first().copied().unwrap_or(1).max(1);
        let sh = stride.first().copied().unwrap_or(1).max(1);
        let ph: usize = padding.iter().take(2).sum();
        let ol = ((l + ph).saturating_sub(kh) / sh + 1).max(1);
        if ol == 1 && ph == 0 {
            // Single window starting at 0, covering the first `min(kh, l)` frames
            // (exact ONNX AveragePool/MaxPool window when there is one output and
            // no padding). Reduce over the time axis, keeping it as length 1.
            let win = kh.min(l);
            let windowed = if win < l { m.narrow_(x, 2, 0, win) } else { x };
            let out_s = Shape::new(&[n, c, 1], in_s.dtype());
            let id = match kind {
                ReduceOp::Mean => m.mean(windowed, vec![2], true),
                _ => m.add_node(
                    Op::Reduce {
                        op: kind,
                        axes: vec![2],
                        keep_dim: true,
                    },
                    vec![windowed],
                    out_s,
                ),
            };
            ctx.env.insert(node.outputs[0].clone(), id);
            return Ok(true);
        }
        // ol > 1: genuine multi-window 1-D pooling — promote NCL → NCHW `[N,C,L,1]`
        // (L on the H axis) so a `[kh,1]` kernel pools it, then reshape back. The
        // rank-4 `Op::Pool` needs 2-D kernel/stride/padding (a 1-D kernel left `kw`
        // defaulted, and the CPU kernel then read `kw` columns of a width-1 plane).
        let (on, oc) = (n, c);
        let x4 = m.reshape_(x, vec![n as i64, c as i64, l as i64, 1]);
        let s4 = Shape::new(&[on, oc, ol, 1], in_s.dtype());
        let pooled = m.add_node(
            Op::Pool {
                kind,
                kernel_size: vec![kh, 1],
                stride: vec![sh, 1],
                padding: vec![ph, 0, 0, 0],
            },
            vec![x4],
            s4,
        );
        let id = m.reshape_(pooled, vec![on as i64, oc as i64, ol as i64]);
        ctx.env.insert(node.outputs[0].clone(), id);
        return Ok(true);
    }
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
