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
    // A genuine 3-D convolution: `[N,C,D,H,W]` against a `[Cout,Cin/g,kD,kH,kW]`
    // weight. Everything past this point assumes one or two spatial axes — the
    // `kernel` below is a `[usize; 2]` — so a 3×3×3 weight reaches `conv2d` as
    // `[3,3]` carrying rank-5 tensors. Shape inference accepts that, the
    // conv-bias-activation fuser then matches it (its `cudnn_friendly_conv`
    // guard only inspects the kernel, which looks 2-D), and the failure finally
    // surfaces in the CPU expansion, far from the cause. Handle it here and
    // return rather than teaching the 1-D/2-D path a third axis it would have
    // to carry through every branch below.
    if !transpose && m.shape(x0).rank() == 5 && m.shape(w).rank() == 5 {
        return lower_conv3d(m, ctx, node, x0, w, groups);
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
    // `ensure_nchw_4d` re-guesses NCL-vs-BLC via `is_vocoder_blc`, which decides on
    // `is_typical_channel(last) && mid > last`. That misfires when a genuine NCL
    // input `[1, C, L]` has a length `L` that is itself a power-of-2 "typical
    // channel" value — e.g. a ConvNeXt depthwise input padded to `[1, 256, 64]`
    // (supertonic/luxtts text-encoder, the dilation-2 pad grows 56→64). It then
    // transposes to `[1, 64, 256]`, so at the conv `c_in/groups = 64/256 = 0`, the
    // im2col is empty, and the op degenerates to a bias-only (constant) output —
    // which the flow ODE amplifies into babble. The weight gives the CONCRETE
    // in_channels, so when the middle axis already equals it the tensor is
    // unambiguously NCL (a genuine BLC block has the *sequence length* there, which
    // never equals in_channels): insert the unit H axis directly and skip the
    // heuristic. Forward convs only — a ConvTranspose weight is `[Cin, Cout/g, k]`,
    // so `dim(1)` is not in_channels there.
    // Covers BOTH conv directions: a forward Conv weight is `[Cout, Cin/g, k]`
    // (in_channels = `dim(1)*groups`), a ConvTranspose weight is `[Cin, Cout/g, k]`
    // (in_channels = `dim(0)`). E.g. the MioTTS `wave_conv_upsample` ConvTranspose
    // input `[1, 512, 100]` was flipped to `[1, 100, 512]` because length 100 is a
    // "typical channel" value — corrupting the whole vocoder (output cos 0.038).
    let x = {
        let s = m.shape(x0).clone();
        let w_s = m.shape(w).clone();
        let in_ch = if transpose {
            w_s.dim(0).unwrap_static()
        } else {
            w_s.dim(1).unwrap_static() * groups
        };
        let is_known_ncl = s.rank() == 3
            && w_s.rank() >= 2
            && s.dim(1).unwrap_static() == in_ch
            && s.dim(1).unwrap_static() != s.dim(2).unwrap_static();
        if is_known_ncl {
            let (n, c, l) = (
                s.dim(0).unwrap_static() as i64,
                s.dim(1).unwrap_static() as i64,
                s.dim(2).unwrap_static() as i64,
            );
            m.reshape_(x0, vec![n, c, 1, l])
        } else {
            ensure_nchw_4d(m, x0)
        }
    };
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
    // Same STALE-meta problem for a genuine 2-D conv. `meta_len_stale` above is
    // rank-3 only, so a rank-4 conv whose meta carries a length that
    // `propagate_shapes` guessed from a symbolic dim was trusted verbatim:
    // ChatterBox's speech_encoder fed a correct 742-frame mel `[1, 1, 80, 742]`
    // into a stride-1 3x3 conv and kept the meta's `[1, 32, 80, 128]`, collapsing
    // time for the entire ResNet below it and leaving declared and operand shapes
    // irreconcilable at lowering. Recompute from the concrete HIR input instead —
    // the "genuine 2D forward conv" branch below already knows how, it was just
    // never reached.
    let meta_len_stale_2d = !meta_empty
        && !transpose
        && rank0 == 4
        && out_shape.rank() == 4
        && m.shape(w).rank() == 4
        && in_s0.dim(2).unwrap_static() > 1
        && in_s0.dim(3).unwrap_static() > 1
        && out_shape.dim(1).unwrap_static() == expected_cout
        && {
            let conv_out = |sz: usize, k: usize, st: usize, p: usize, d: usize| {
                let st = st.max(1);
                let eff = d * k.saturating_sub(1);
                (sz + 2 * p).saturating_sub(eff).saturating_sub(1) / st + 1
            };
            let h = conv_out(
                in_s0.dim(2).unwrap_static(),
                kernel[0],
                stride[0],
                pad[0],
                dilation[0],
            );
            let wd = conv_out(
                in_s0.dim(3).unwrap_static(),
                kernel[1],
                stride[1],
                pad[1],
                dilation[1],
            );
            out_shape.dim(2).unwrap_static() != h || out_shape.dim(3).unwrap_static() != wd
        };
    if meta_empty
        || out_shape.rank() < 2
        || meta_layout_transposed
        || meta_len_stale
        || meta_len_stale_transpose
        || meta_len_stale_2d
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

/// Emit ONNX `MaxPool`'s optional second output: the index of each maximum.
///
/// The op declares two outputs and the pooled values are only the first. When
/// the second is left unbound, every consumer of it fails to resolve — which is
/// how a network that upsamples by max-unpooling (FastSurfer's VINN, and any
/// SegNet-style decoder) stops importing, with an error naming the *consumer*
/// rather than the pool.
///
/// # Convention
///
/// ONNX indices are flat over the **whole** `[N,C,H,W]` tensor, not per
/// channel plane — `[0, N·C·H·W)`. PyTorch's are per-plane, so a graph that
/// mixes the two is out by `(n·C + c)·H·W` on every channel but the first.
/// This emits the ONNX convention, which is what the rest of an ONNX graph
/// expects.
///
/// # How
///
/// For a non-overlapping pool the window can be exposed by reshaping, so the
/// index is an `ArgMax` over the window plus arithmetic. Both the arithmetic
/// terms are constants: a window only takes `kh·kw` distinct offsets, so a
/// `Gather` from a small table covers it, and everything else — the plane
/// offset and the window's origin — is fixed by the shapes. That leaves one
/// data-dependent op where a direct implementation would need a new kernel on
/// every backend.
///
/// Overlapping or padded pools cannot be reshaped this way and are refused
/// rather than approximated.
fn emit_pool_indices(
    m: &mut HirMut<'_>,
    x: HirNodeId,
    kernel: [usize; 2],
    stride: [usize; 2],
    pad: &[usize],
    name: &str,
) -> Result<HirNodeId> {
    let in_s = m.shape(x).clone();
    if in_s.rank() != 4 {
        bail!("{name}: pooling indices are only implemented for rank-4 NCHW");
    }
    let dims: Vec<usize> = (0..4).map(|i| in_s.dim(i).unwrap_static()).collect();
    let (n, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    let (kh, kw) = (kernel[0], kernel[1]);

    if pad.iter().any(|&p| p != 0) {
        bail!("{name}: pooling indices with padding are not supported");
    }
    if stride != kernel {
        bail!(
            "{name}: pooling indices need non-overlapping windows (kernel {kernel:?}, \
             stride {stride:?})"
        );
    }
    if kh == 0 || kw == 0 || h % kh != 0 || w % kw != 0 {
        bail!("{name}: {h}x{w} does not tile exactly by {kh}x{kw}");
    }
    let (oh, ow) = (h / kh, w / kw);

    // [N,C,H,W] -> [N,C,OH,kh,OW,kw] -> [N,C,OH,OW,kh,kw] -> [N,C,OH,OW,kh*kw]
    let r1 = m.reshape_(
        x,
        vec![
            n as i64, c as i64, oh as i64, kh as i64, ow as i64, kw as i64,
        ],
    );
    let t = m.transpose_(r1, vec![0, 1, 2, 4, 3, 5]);
    let flat = m.reshape_(
        t,
        vec![n as i64, c as i64, oh as i64, ow as i64, (kh * kw) as i64],
    );

    let picked = m.add_node(
        Op::ArgMax {
            axis: 4,
            keep_dim: false,
        },
        vec![flat],
        Shape::new(&[n, c, oh, ow], DType::I64),
    );

    // Offset within the window, indexed by the argmax: row-major (a, b) sits
    // `a·W + b` from the window's first element.
    let table: Vec<i64> = (0..kh)
        .flat_map(|a| (0..kw).map(move |b| (a * w + b) as i64))
        .collect();
    let table_bytes: Vec<u8> = table.iter().flat_map(|v| v.to_le_bytes()).collect();
    let table_id = m.add_node(
        Op::Constant { data: table_bytes },
        vec![],
        Shape::new(&[kh * kw], DType::I64),
    );
    let offsets = m.add_node(
        Op::Gather { axis: 0 },
        vec![table_id, picked],
        Shape::new(&[n, c, oh, ow], DType::I64),
    );

    // Where each window starts, flattened over the whole tensor.
    let mut base = Vec::with_capacity(n * c * oh * ow);
    for ni in 0..n {
        for ci in 0..c {
            let plane = ((ni * c + ci) * h * w) as i64;
            for y in 0..oh {
                for xw in 0..ow {
                    base.push(plane + ((y * kh) * w + xw * kw) as i64);
                }
            }
        }
    }
    let base_bytes: Vec<u8> = base.iter().flat_map(|v| v.to_le_bytes()).collect();
    let base_id = m.add_node(
        Op::Constant { data: base_bytes },
        vec![],
        Shape::new(&[n, c, oh, ow], DType::I64),
    );

    Ok(binary_infer_add(m, offsets, base_id, name))
}

/// Lower a rank-5 `MaxPool` / `AveragePool` to a 3-spatial-axis [`Op::Pool`].
///
/// `Op::Pool` already carries `Vec` extents and the CPU backend already has a
/// 3-D kernel, so all this has to do is read the attributes at their real
/// length and compute the output shape, instead of going through the
/// two-element `onnx_pads`.
/// ONNX pooling output extent. `ceil_mode=1` rounds the division UP instead of
/// down (`ceil((in + pads - kernel) / stride) + 1`), which the importer ignored
/// entirely — `ceil_mode` appeared nowhere in it. ChatterBox's speaker encoder
/// pools 259 frames with kernel/stride 100 and `ceil_mode=1`: the truth is 3
/// windows, floor gives 2, and the CAM layer then expands the pooled segments
/// back by 100 to a 200-frame tensor instead of 259 — silently wrong shapes and
/// a speaker embedding that no longer identifies the speaker.
fn pool_out_len(size: usize, pad: usize, kernel: usize, stride: usize, ceil_mode: bool) -> usize {
    let stride = stride.max(1);
    let num = (size + pad).saturating_sub(kernel);
    let steps = if ceil_mode {
        num.div_ceil(stride)
    } else {
        num / stride
    };
    steps + 1
}

/// `ceil_mode` attribute of a pooling node (ONNX default 0).
fn pool_ceil_mode(node: &BundleNode) -> bool {
    node.attrs
        .get("ceil_mode")
        .and_then(|v| v.as_i64())
        .unwrap_or(0)
        != 0
}

fn lower_pool3d(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
    x: HirNodeId,
    kind: ReduceOp,
) -> Result<bool> {
    let spatial = |name: &str, default: usize| -> [usize; 3] {
        let v: Vec<usize> = node
            .attrs
            .get(name)
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|d| d.as_u64().map(|x| x as usize))
                    .collect()
            })
            .unwrap_or_default();
        [0, 1, 2].map(|i| v.get(i).copied().unwrap_or(default))
    };
    let kernel = spatial("kernel_shape", 1);
    let stride = spatial("strides", 1);

    // ONNX orders `pads` as all the begins then all the ends. `Op::Pool` takes
    // one number per axis, so an asymmetric pad has nowhere to go.
    let pads: Vec<usize> = node
        .attrs
        .get("pads")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|d| d.as_u64().map(|x| x as usize))
                .collect()
        })
        .unwrap_or_default();
    let pad = if pads.is_empty() {
        [0, 0, 0]
    } else if pads.len() == 6 {
        for i in 0..3 {
            if pads[i] != pads[i + 3] {
                bail!(
                    "{}: asymmetric 3-D pool padding {pads:?} is not supported",
                    node.name
                );
            }
        }
        [pads[0], pads[1], pads[2]]
    } else {
        bail!(
            "{}: expected 6 pad values for a 3-D pool, got {pads:?}",
            node.name
        );
    };

    let in_s = m.shape(x).clone();
    let dims: Vec<usize> = (0..5).map(|i| in_s.dim(i).unwrap_static()).collect();
    let mut out = dims.clone();
    let ceil_mode = pool_ceil_mode(node);
    for a in 0..3 {
        out[2 + a] = pool_out_len(dims[2 + a], 2 * pad[a], kernel[a], stride[a], ceil_mode);
    }
    let id = m.add_node(
        Op::Pool {
            kind,
            kernel_size: kernel.to_vec(),
            stride: stride.to_vec(),
            padding: pad.to_vec(),
        },
        vec![x],
        Shape::new(&out, in_s.dtype()),
    );
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

/// Lower a rank-5 forward `Conv` to [`Op::Conv3d`].
///
/// Deliberately separate from [`lower_conv`]: that function's shape logic is a
/// long sequence of 1-D-versus-2-D disambiguations (BLC vs NCL, `[N,C,L,1]`
/// canonicalisation, vocoder-specific layout fixes) and none of it applies to
/// volumetric data, where `[N,C,D,H,W]` is unambiguous. Kernel extents come
/// from the weight rather than `kernel_shape` for the same reason the 2-D path
/// prefers them: the weight is concrete, the attribute is optional.
fn lower_conv3d(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
    x: HirNodeId,
    w: HirNodeId,
    groups: usize,
) -> Result<bool> {
    let w_s = m.shape(w).clone();
    let kernel = [
        w_s.dim(2).unwrap_static(),
        w_s.dim(3).unwrap_static(),
        w_s.dim(4).unwrap_static(),
    ];
    let spatial = |name: &str, default: usize| -> [usize; 3] {
        let v: Vec<usize> = node
            .attrs
            .get(name)
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|d| d.as_u64().map(|x| x as usize))
                    .collect()
            })
            .unwrap_or_default();
        [0, 1, 2].map(|i| v.get(i).copied().unwrap_or(default))
    };
    let stride = spatial("strides", 1);
    let dilation = spatial("dilations", 1);

    // `Op::Conv3d` takes one padding per axis, so an asymmetric ONNX `pads` has
    // no faithful representation. Say so instead of dropping the end pad, which
    // would shift every downstream voxel by half a kernel.
    let pads: Vec<usize> = node
        .attrs
        .get("pads")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|d| d.as_u64().map(|x| x as usize))
                .collect()
        })
        .unwrap_or_default();
    let auto_pad = node
        .attrs
        .get("auto_pad")
        .and_then(|v| v.as_str())
        .unwrap_or("NOTSET");
    let pad = match auto_pad {
        "VALID" => [0, 0, 0],
        "SAME_UPPER" | "SAME_LOWER" => {
            // Symmetric only when every kernel extent is odd and unstrided;
            // otherwise ONNX puts the extra row on one side and we cannot.
            let same = [0, 1, 2].map(|i| dilation[i] * (kernel[i] - 1) / 2);
            for i in 0..3 {
                if kernel[i].is_multiple_of(2) || stride[i] != 1 {
                    bail!(
                        "{}: {auto_pad} with kernel {kernel:?} stride {stride:?} needs \
                         asymmetric padding, which Conv3d cannot express",
                        node.name
                    );
                }
            }
            same
        }
        _ if pads.is_empty() => [0, 0, 0],
        _ if pads.len() == 6 => {
            for i in 0..3 {
                if pads[i] != pads[i + 3] {
                    bail!(
                        "{}: asymmetric 3-D padding {pads:?} is not supported; \
                         precede the Conv with an explicit Pad",
                        node.name
                    );
                }
            }
            [pads[0], pads[1], pads[2]]
        }
        _ => bail!(
            "{}: expected 6 pad values for a 3-D Conv, got {pads:?}",
            node.name
        ),
    };

    let in_s = m.shape(x).clone();
    let out_shape =
        rlx_ir::shape::conv3d_output_shape(&in_s, &w_s, kernel, stride, pad, dilation, groups)
            .map_err(|e| anyhow!("{}: {e}", node.name))?;
    let mut id = m.add_node(
        Op::Conv3d {
            stride,
            padding: pad,
            dilation,
            groups,
        },
        vec![x, w],
        out_shape,
    );

    // The conv bias is per-output-channel `[C]`. Reshaping it to `[1,C,1,1,1]`
    // rather than leaving it rank-1 matters: a bare `[C]` broadcasts against the
    // trailing axis (W), which for an isotropic volume has the same extent as C
    // in exactly the cases where the mistake is invisible.
    if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
        let bias = ctx.tensor(&node.inputs[2])?;
        let bias_in = if m.shape(bias).rank() == 1 {
            let bc = m.shape(bias).dim(0).unwrap_static();
            m.reshape_(bias, vec![1, bc as i64, 1, 1, 1])
        } else {
            bias
        };
        id = binary_infer_add(m, id, bias_in, &node.name);
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
    // A 3-D pool over `[N,C,D,H,W]`. `onnx_pads` reports two spatial extents,
    // so without this a 2×2×2 MaxPool arrives as 2×2 — and because the CPU
    // backend pairs a 2-element kernel with a rank-5 input by emitting
    // `Thunk::Nop`, the pool silently does nothing at all. The tensor keeps its
    // full depth, every downstream shape is wrong, and a U-Net's skip
    // connections stop lining up, with no error anywhere.
    if m.shape(x).rank() == 5 && op != "GlobalAveragePool" {
        return lower_pool3d(m, ctx, node, x, kind);
    }
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
        let ol = pool_out_len(l, ph, kh, sh, pool_ceil_mode(node)).max(1);
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
            kernel_size: kernel_size.clone(),
            stride: stride.clone(),
            padding: padding.clone(),
        },
        vec![x],
        s,
    );
    ctx.env.insert(node.outputs[0].clone(), id);

    // `MaxPool` may declare a second output, the index of each maximum. Left
    // unbound it is not a missing optimisation — every consumer fails to
    // resolve, and the error names the consumer rather than this.
    if node.outputs.len() > 1 && !node.outputs[1].is_empty() {
        let two = |v: &[usize], d: usize| {
            [
                v.first().copied().unwrap_or(d),
                v.get(1).copied().unwrap_or(d),
            ]
        };
        let indices = emit_pool_indices(
            m,
            x,
            two(&kernel_size, 1),
            two(&stride, 1),
            &padding,
            &node.name,
        )?;
        ctx.env.insert(node.outputs[1].clone(), indices);
    }
    Ok(true)
}

#[cfg(test)]
mod pool_indices_tests {
    /// The index arithmetic `emit_pool_indices` builds, in plain Rust.
    ///
    /// The emitted graph computes `base + table[argmax]`; this is the same
    /// expression evaluated directly, so the tests below check the *arithmetic*
    /// — which is the part that is silently wrong when it is wrong. A wrong
    /// index does not fail; it unpools each maximum to a different place, and
    /// the decoder still produces a smooth, plausible image.
    fn expected_index(
        n: usize,
        c: usize,
        h: usize,
        w: usize,
        kh: usize,
        kw: usize,
        ni: usize,
        ci: usize,
        oy: usize,
        ox: usize,
        argmax: usize,
    ) -> i64 {
        let _ = n;
        let plane = ((ni * c + ci) * h * w) as i64;
        let origin = ((oy * kh) * w + ox * kw) as i64;
        let (a, b) = (argmax / kw, argmax % kw);
        let offset = (a * w + b) as i64;
        plane + origin + offset
    }

    #[test]
    fn indices_are_flat_over_the_whole_tensor_not_per_plane() {
        // ONNX counts over all of `[N,C,H,W]`; PyTorch counts per plane. Mixing
        // them is out by `(n·C + c)·H·W`, so channel 0 looks fine and every
        // other channel unpools into the wrong plane.
        let (n, c, h, w, kh, kw) = (2, 3, 4, 4, 2, 2);
        let first = expected_index(n, c, h, w, kh, kw, 0, 0, 0, 0, 0);
        let second_channel = expected_index(n, c, h, w, kh, kw, 0, 1, 0, 0, 0);
        assert_eq!(first, 0);
        assert_eq!(second_channel, (h * w) as i64);
        let second_batch = expected_index(n, c, h, w, kh, kw, 1, 0, 0, 0, 0);
        assert_eq!(second_batch, (c * h * w) as i64);
    }

    #[test]
    fn each_position_in_a_window_maps_to_its_own_voxel() {
        // The four corners of a 2x2 window, at a window that is not the origin.
        let (h, w, kh, kw) = (4, 4, 2, 2);
        let at = |am| expected_index(1, 1, h, w, kh, kw, 0, 0, 1, 1, am);
        // Window (1,1) starts at row 2, column 2 -> flat 2*4 + 2 = 10.
        assert_eq!(at(0), 10); // (0,0)
        assert_eq!(at(1), 11); // (0,1)
        assert_eq!(at(2), 10 + w as i64); // (1,0)
        assert_eq!(at(3), 11 + w as i64); // (1,1)
    }

    #[test]
    fn every_index_is_distinct_and_in_range() {
        // A non-overlapping pool partitions the tensor, so across all windows
        // and all argmax choices the indices must cover distinct voxels within
        // bounds. A collision would mean two maxima unpool to one place.
        let (n, c, h, w, kh, kw) = (2, 2, 6, 4, 2, 2);
        let mut seen = std::collections::HashSet::new();
        for ni in 0..n {
            for ci in 0..c {
                for oy in 0..h / kh {
                    for ox in 0..w / kw {
                        for am in 0..kh * kw {
                            let i = expected_index(n, c, h, w, kh, kw, ni, ci, oy, ox, am);
                            assert!(i >= 0 && (i as usize) < n * c * h * w, "{i} out of range");
                            // Distinct only within a window's own choice set;
                            // across windows the sets are disjoint.
                            if am == 0 {
                                assert!(seen.insert(i), "window origin {i} repeated");
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(seen.len(), n * c * (h / kh) * (w / kw));
    }

    #[test]
    fn a_one_by_one_pool_is_the_identity_index_map() {
        // This is the case torch's max-unpool decomposition uses to obtain the
        // flat index of every element, so it has to come out as exactly
        // `arange(N·C·H·W)`.
        let (n, c, h, w) = (2, 2, 3, 3);
        let mut got = Vec::new();
        for ni in 0..n {
            for ci in 0..c {
                for y in 0..h {
                    for x in 0..w {
                        got.push(expected_index(n, c, h, w, 1, 1, ni, ci, y, x, 0));
                    }
                }
            }
        }
        let want: Vec<i64> = (0..(n * c * h * w) as i64).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn a_non_square_window_offsets_by_the_row_stride() {
        // 1x2 and 2x1 windows exercise the `a·W + b` term separately.
        assert_eq!(expected_index(1, 1, 2, 4, 1, 2, 0, 0, 0, 1, 1), 3);
        assert_eq!(expected_index(1, 1, 4, 2, 2, 1, 0, 0, 1, 0, 1), 2 * 2 + 2);
    }
}
