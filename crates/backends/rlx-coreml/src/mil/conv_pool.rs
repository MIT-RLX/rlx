// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// IR → CoreML ML Program (MIL) lowering. Pure data transformation: takes
// an RLX `Graph` plus baked parameter/constant data and produces a
// `proto::Model` ready to serialise into a `.mlpackage`. No FFI, so this
// builds and unit-tests on any host.

//! `conv_pool` — extracted from the `mil` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use super::helpers::simple_op_flex;
use super::helpers::*;
use crate::proto;
use crate::{CoremlError, Result};
use rlx_ir::op::{Activation, CmpOp, MaskKind, ReduceOp};
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Dim, Graph, NodeId, Op, Shape};
use std::collections::HashMap;

use super::*;

impl<'a> LowerCtx<'a> {
    /// Native MaxPool2d backward (training). Routes each window's upstream
    /// gradient to its max position(s) via reshape + reduce_max + select —
    /// O(input size), no dense scatter. On ties EVERY maximum receives the
    /// gradient, matching the shared autodiff decomposition
    /// (`compose_max_pool2d_backward`) — important for the common relu→maxpool
    /// all-zero window (every element equals the max), and keeps ANE gradients
    /// consistent with the CPU/GPU training path.
    ///
    /// Supports the non-overlapping, unpadded case (stride == kernel, pad == 0,
    /// dims divisible by the kernel) — what CNN training uses (e.g. MNIST 2×2/2).
    /// Other configs return `Unsupported` rather than a silently wrong result.
    #[cfg(feature = "training")]
    pub(crate) fn lower_max_pool2d_backward(
        &mut self,
        id: NodeId,
        kernel: &[usize],
        stride: &[usize],
        padding: &[usize],
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let x = self.val(node.inputs[0]);
        let dy = self.val(node.inputs[1]);
        let out_shape = node.shape.clone();
        let dt = out_shape.dtype();
        if out_shape.rank() != 4 {
            return Err(CoremlError::Unsupported(
                "max_pool2d_backward: expected NCHW (rank 4)".into(),
            ));
        }
        let dim = |i: usize| out_shape.dim(i).unwrap_static();
        let (n, c, h, w) = (dim(0), dim(1), dim(2), dim(3));
        let (kh, kw) = (kernel[0], kernel[1]);
        if stride.first() != Some(&kh)
            || stride.get(1) != Some(&kw)
            || padding.iter().any(|&p| p != 0)
            || h % kh != 0
            || w % kw != 0
        {
            return Err(CoremlError::Unsupported(format!(
                "max_pool2d_backward native kernel handles only non-overlapping, \
                 unpadded pooling with divisible dims (stride==kernel, pad==0); got \
                 kernel={kernel:?} stride={stride:?} pad={padding:?} on {h}x{w}"
            )));
        }
        let (ho, wo) = (h / kh, w / kw);
        // Keep every tensor rank ≤ 4 (the ANE reshape limit): fold N·C·Ho into one
        // batch dim. The window view is [B, kh, Wo, kw] with B = N·C·Ho, and each
        // pooling window is the (kh, kw) pair at axes [1, 3].
        let b = n * c * ho;
        let win = Shape::new(&[b, kh, wo, kw], dt);
        let red = Shape::new(&[b, 1, wo, 1], dt);
        let win_b = win.clone().with_dtype(DType::Bool);
        // Reshape [N,C,H,W] → [B,kh,Wo,kw] is a pure reinterpret (row-major) since
        // H=Ho·kh, W=Wo·kw; the final reshape inverts it back to [N,C,H,W].
        let win_dims =
            |kdim: usize, wdim: usize| vec_i32(&[b as i32, kdim as i32, wo as i32, wdim as i32]);

        let xr = format!("{out_name}_xr");
        self.emit(
            "reshape",
            &xr,
            &win,
            vec![
                ("x", bind_name(&x)),
                ("shape", bind_value(win_dims(kh, kw))),
            ],
        )?;
        let ymax = format!("{out_name}_ymax");
        self.emit(
            "reduce_max",
            &ymax,
            &red,
            vec![
                ("x", bind_name(&xr)),
                ("axes", bind_value(vec_i32(&[1, 3]))),
                ("keep_dims", bind_value(scalar_bool(true))),
            ],
        )?;
        // mask = (xr >= ymax) ⟺ (xr == max). On ties, every maximum is marked —
        // matching the shared autodiff decomposition (`compose_max_pool2d_backward`
        // routes dy to all maxima), so ANE gradients stay consistent with the
        // CPU/GPU training path.
        let mask = format!("{out_name}_mask");
        self.emit(
            "greater_equal",
            &mask,
            &win_b,
            vec![("x", bind_name(&xr)), ("y", bind_name(&ymax))],
        )?;
        let zero = format!("{out_name}_zero");
        let zero_op = make_const(&mut self.blob, &zero, &Shape::new(&[1], dt), &[0.0])?;
        self.operations.push(zero_op);
        // route dy (broadcast over the window) to every max position, 0 elsewhere.
        let dyr = format!("{out_name}_dyr");
        self.emit(
            "reshape",
            &dyr,
            &red,
            vec![("x", bind_name(&dy)), ("shape", bind_value(win_dims(1, 1)))],
        )?;
        let dxr = format!("{out_name}_dxr");
        self.emit(
            "select",
            &dxr,
            &win,
            vec![
                ("cond", bind_name(&mask)),
                ("a", bind_name(&dyr)),
                ("b", bind_name(&zero)),
            ],
        )?;
        let op = self.simple_op(
            "reshape",
            out_name,
            &out_shape,
            vec![
                ("x", bind_name(&dxr)),
                (
                    "shape",
                    bind_value(vec_i32(&[n as i32, c as i32, h as i32, w as i32])),
                ),
            ],
        )?;
        self.push_named(id, out_name.to_string(), op);
        Ok(())
    }

    /// 2D conv / conv_transpose, NCHW. Inputs `[x, weight]` (no bias in IR).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn lower_conv(
        &mut self,
        id: NodeId,
        transpose: bool,
        stride: &[usize],
        padding: &[usize],
        dilation: &[usize],
        _output_padding: &[usize],
        groups: usize,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let shape = node.shape.clone();
        let in_shape = self.graph.shape(node.inputs[0]).clone();
        let w_shape = self.graph.shape(node.inputs[1]).clone();
        if std::env::var("RLX_DBG_CONV").is_ok() {
            eprintln!(
                "[mil-conv] transpose={transpose} groups={groups} in={:?} w={:?} out={:?}",
                in_shape.dims(),
                w_shape.dims(),
                shape.dims()
            );
        }
        // rlx lowers ONNX 1D convs as 2D NCHW with a unit H axis and the length in W
        // (`[N,C,1,L]`, weight `[Co,Ci,k,1]`). CoreML's 2D conv would run the k-tap
        // kernel over the singleton H axis, so collapse to a real rank-3 1D conv over
        // the length (matching CPU/MLX/wgpu and onnxruntime).
        // 1D conv packed as 2D NCHW with ONE singleton spatial axis and the
        // length in the other — either `[N,C,1,L]` (length-in-W) or
        // `[N,C,L,1]` (length-in-H, e.g. Whisper's conv frontend) — with
        // weight `[Co,Ci,k,1]`. Collapse to a real rank-3 1D conv over the
        // length (matches CPU/MLX/wgpu/onnxruntime); CoreML's 2D conv over a
        // singleton/degenerate spatial axis is not reliable here.
        // Rank-5 NCDHW (`Conv3d`) skips this collapse and uses MIL `"conv"`
        // with 3-length strides/pad/dilations.
        let one_d = !transpose
            && in_shape.rank() == 4
            && w_shape.rank() == 4
            && w_shape.dim(3).unwrap_static() == 1
            && w_shape.dim(2).unwrap_static() > 1
            && {
                let in_h = in_shape.dim(2).unwrap_static();
                let in_w = in_shape.dim(3).unwrap_static();
                in_h == 1 || in_w == 1
            };
        if one_d {
            let in_h = in_shape.dim(2).unwrap_static();
            let in_w = in_shape.dim(3).unwrap_static();
            let n = in_shape.dim(0).unwrap_static() as i32;
            let c = in_shape.dim(1).unwrap_static() as i32;
            // length is the non-singleton input spatial axis
            let l = if in_h == 1 { in_w } else { in_h } as i32;
            let co = w_shape.dim(0).unwrap_static() as i32;
            let ci = w_shape.dim(1).unwrap_static() as i32;
            let k = w_shape.dim(2).unwrap_static() as i32;
            let out_h = shape.dim(2).unwrap_static();
            let out_w = shape.dim(3).unwrap_static();
            let lo = if out_h == 1 { out_w } else { out_h } as i32;
            let xr = format!("{out_name}_x1d");
            self.emit(
                "reshape",
                &xr,
                &Shape::new(&[n as usize, c as usize, l as usize], DType::F32),
                vec![
                    ("x", bind_name(&self.val(node.inputs[0]))),
                    ("shape", bind_value(vec_i32(&[n, c, l]))),
                ],
            )?;
            let wr = format!("{out_name}_w1d");
            self.emit(
                "reshape",
                &wr,
                &Shape::new(&[co as usize, ci as usize, k as usize], DType::F32),
                vec![
                    ("x", bind_name(&self.val(node.inputs[1]))),
                    ("shape", bind_value(vec_i32(&[co, ci, k]))),
                ],
            )?;
            let cout = format!("{out_name}_c1d");
            self.emit(
                "conv",
                &cout,
                &Shape::new(&[n as usize, co as usize, lo as usize], DType::F32),
                vec![
                    ("x", bind_name(&xr)),
                    ("weight", bind_name(&wr)),
                    ("strides", bind_value(vec_i32(&[stride[0] as i32]))),
                    ("pad_type", bind_value(scalar_str("custom"))),
                    ("pad", bind_value(vec_i32(&pad_begin_end(&[padding[0]])))),
                    ("dilations", bind_value(vec_i32(&[dilation[0] as i32]))),
                    ("groups", bind_value(scalar_i32(groups as i32))),
                ],
            )?;
            let out_dims: Vec<i32> = static_dims(&shape)?.iter().map(|&v| v as i32).collect();
            let op = self.simple_op(
                "reshape",
                out_name,
                &shape,
                vec![
                    ("x", bind_name(&cout)),
                    ("shape", bind_value(vec_i32(&out_dims))),
                ],
            )?;
            self.push_named(id, out_name.to_string(), op);
            return Ok(());
        }
        let x = self.val(node.inputs[0]);
        let mut w = self.val(node.inputs[1]);
        // MIL's `conv_transpose` requires the weight as `[C_in, C_out/groups, *k]`.
        // The importer boxes a grouped/depthwise transposed-conv weight oddly (a
        // depthwise ConvTranspose1d arrives as `[1, C, 1, k]` — C_in in dim 1),
        // which MIL rejects (`KernelChannels(1) != InputChannels(512)`). The flat
        // data is already C_in-major, so reshape it into MIL's layout. (For
        // groups=1 this is a no-op reshape.)
        if transpose {
            let cin = in_shape.dim(1).unwrap_static() as i32;
            let cout = shape.dim(1).unwrap_static() as i32;
            let g = groups.max(1) as i32;
            let spatial: Vec<i32> = (2..w_shape.rank())
                .map(|i| w_shape.dim(i).unwrap_static() as i32)
                .collect();
            let mut tgt = vec![cin, cout / g];
            tgt.extend_from_slice(&spatial);
            let cur: Vec<i32> = w_shape
                .dims()
                .iter()
                .map(|d| d.unwrap_static() as i32)
                .collect();
            if cur != tgt {
                let wr = format!("{out_name}_wt");
                let dims: Vec<usize> = tgt.iter().map(|&d| d as usize).collect();
                self.emit(
                    "reshape",
                    &wr,
                    &Shape::new(&dims, DType::F32),
                    vec![("x", bind_name(&w)), ("shape", bind_value(vec_i32(&tgt)))],
                )?;
                w = wr;
            }
        }
        let strides = vec_usize_i32(stride);
        let dilations = vec_usize_i32(dilation);
        let pad = pad_begin_end(padding);
        let ty = if transpose { "conv_transpose" } else { "conv" };
        let mut binds = vec![
            ("x", bind_name(&x)),
            ("weight", bind_name(&w)),
            ("strides", bind_value(vec_i32(&strides))),
            ("pad_type", bind_value(scalar_str("custom"))),
            ("pad", bind_value(vec_i32(&pad))),
            ("dilations", bind_value(vec_i32(&dilations))),
            ("groups", bind_value(scalar_i32(groups as i32))),
        ];
        if transpose {
            // conv_transpose needs the explicit output shape to resolve the
            // fractionally-strided ambiguity.
            let out_dims: Vec<i32> = static_dims(&shape)?.iter().map(|&v| v as i32).collect();
            binds.push(("output_shape", bind_value(vec_i32(&out_dims))));
        }
        let op = self.simple_op(ty, out_name, &shape, binds)?;
        self.push_named(id, out_name.to_string(), op);
        Ok(())
    }

    /// Conv2d backward w.r.t. weight (NCHW, groups == 1). The weight gradient is a
    /// convolution of the input by the upstream gradient: with N folded as the
    /// contraction channel and the gradient as the (stride-dilated) kernel,
    ///   dWᵀ = conv(xᵀ[Cin,N,H,W], dyᵀ[Cout,N,Hout,Wout], dilation = forward stride)
    /// gives [Cin,Cout,kh,kw]; transpose back to [Cout,Cin,kh,kw]. Inputs [x, dy].
    #[cfg(feature = "training")]
    pub(crate) fn lower_conv2d_backward_weight(
        &mut self,
        id: NodeId,
        stride: &[usize],
        padding: &[usize],
        groups: usize,
        out_name: &str,
    ) -> Result<()> {
        if groups != 1 {
            return Err(CoremlError::Unsupported(
                "conv2d backward weight: only groups == 1".into(),
            ));
        }
        let node = self.graph.node(id);
        let x = self.val(node.inputs[0]);
        let dy = self.val(node.inputs[1]);
        let x_shape = self.graph.shape(node.inputs[0]).clone();
        let dy_shape = self.graph.shape(node.inputs[1]).clone();
        let out_shape = node.shape.clone();
        let dt = out_shape.dtype();
        let xd = |i: usize| x_shape.dim(i).unwrap_static();
        let dd = |i: usize| dy_shape.dim(i).unwrap_static();
        let od = |i: usize| out_shape.dim(i).unwrap_static();
        let (n, cin, h, w) = (xd(0), xd(1), xd(2), xd(3));
        let (cout, hout, wout) = (dd(1), dd(2), dd(3));
        let (kh, kw) = (od(2), od(3));
        let perm = || bind_value(vec_i32(&[1, 0, 2, 3]));

        // xᵀ = [Cin, N, H, W], dyᵀ = [Cout, N, Hout, Wout]
        let xt = format!("{out_name}_xt");
        self.emit(
            "transpose",
            &xt,
            &Shape::new(&[cin, n, h, w], dt),
            vec![("x", bind_name(&x)), ("perm", perm())],
        )?;
        let dyt = format!("{out_name}_dyt");
        self.emit(
            "transpose",
            &dyt,
            &Shape::new(&[cout, n, hout, wout], dt),
            vec![("x", bind_name(&dy)), ("perm", perm())],
        )?;
        // conv(xᵀ, dyᵀ): kernel = dyᵀ over the N contraction channel, dilated by the
        // forward stride so it samples the strided receptive field → [Cin,Cout,kh,kw].
        let dwt = format!("{out_name}_dwt");
        self.emit(
            "conv",
            &dwt,
            &Shape::new(&[cin, cout, kh, kw], dt),
            vec![
                ("x", bind_name(&xt)),
                ("weight", bind_name(&dyt)),
                ("strides", bind_value(vec_i32(&[1, 1]))),
                ("pad_type", bind_value(scalar_str("custom"))),
                ("pad", bind_value(vec_i32(&pad_begin_end(padding)))),
                (
                    "dilations",
                    bind_value(vec_i32(&[stride[0] as i32, stride[1] as i32])),
                ),
                ("groups", bind_value(scalar_i32(1))),
            ],
        )?;
        let op = self.simple_op(
            "transpose",
            out_name,
            &out_shape,
            vec![("x", bind_name(&dwt)), ("perm", perm())],
        )?;
        self.push_named(id, out_name.to_string(), op);
        Ok(())
    }

    /// 2D max/avg pool, NCHW. Avg divides by the full window (pad counts).
    pub(crate) fn lower_pool(
        &mut self,
        id: NodeId,
        kind: ReduceOp,
        kernel: &[usize],
        stride: &[usize],
        padding: &[usize],
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let shape = node.shape.clone();
        let x = self.val(node.inputs[0]);
        let ty = match kind {
            ReduceOp::Max => "max_pool",
            ReduceOp::Mean => "avg_pool",
            other => return Err(CoremlError::Unsupported(format!("pool {other:?}"))),
        };
        let pad = match padding.len() {
            // Symmetric `[h, w]` → MIL `[h_b, h_e, w_b, w_e]`.
            0 => pad_begin_end(&[0, 0]),
            1 | 2 => pad_begin_end(padding),
            // Already begin/end layout (e.g. GlobalAveragePool emits `[0;4]`).
            4 => padding.iter().map(|&p| p as i32).collect(),
            // Unexpected rank — take first two as symmetric H/W.
            _ => pad_begin_end(&padding[..2]),
        };
        let mut binds = vec![
            ("x", bind_name(&x)),
            ("kernel_sizes", bind_value(vec_i32(&vec_usize_i32(kernel)))),
            ("strides", bind_value(vec_i32(&vec_usize_i32(stride)))),
            ("pad_type", bind_value(scalar_str("custom"))),
            ("pad", bind_value(vec_i32(&pad))),
            ("ceil_mode", bind_value(scalar_bool(false))),
        ];
        if matches!(kind, ReduceOp::Mean) {
            binds.push((
                "exclude_padding_from_average",
                bind_value(scalar_bool(false)),
            ));
        }
        let op = self.simple_op(ty, out_name, &shape, binds)?;
        self.push_named(id, out_name.to_string(), op);
        Ok(())
    }
}
