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
    let z_ncl = if stride > 1 {
        let z4 = m.reshape_(x_ncl, vec![n as i64, c as i64, l as i64, 1]);
        let gap = stride - 1;
        let zshape = [n, c, l, gap];
        let numel: usize = zshape.iter().product();
        let zeros = m.add_node(
            Op::Constant {
                data: vec![0u8; numel * dt.size_bytes().max(1)],
            },
            vec![],
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
            } else if !transpose {
                // Standard conv: (l + 2·pad − dilation·(k−1) − 1)/stride + 1. Reduces
                // to `l` for same-padding (decoder, attention 1×1) and correctly
                // shrinks "valid" convs (pad=0, k>1) fed by an explicit Pad (VITS FFN).
                let s = stride[0].max(1);
                let eff = dilation[0] * kernel[0].saturating_sub(1);
                (l + 2 * pad[0]).saturating_sub(eff).saturating_sub(1) / s + 1
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
        // Emit `Op::Conv` directly so the real dilation is preserved — the
        // `conv2d` helper hard-codes `dilation=[1,1]`, which silently turns the
        // dilated resblock convs (HiFi-GAN MRF, dilations 1/3/5) into stride-1
        // convs and corrupts the waveform.
        let d2 = [dilation[0], if w_rank >= 4 { dilation[1] } else { 1 }];
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
    // Collapse a 1D-conv result (lowered through NCHW with a singleton spatial axis)
    // back to NCL when the ONNX input was genuinely 3D, so it lines up with the rest
    // of a 3D graph for elementwise ops (attention/residual adds) and lets downstream
    // `Shape`/`Gather(axis=2)` read the real length.
    if rank0 == 3 {
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

