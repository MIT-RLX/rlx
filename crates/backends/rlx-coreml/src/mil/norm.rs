// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// IR → CoreML ML Program (MIL) lowering. Pure data transformation: takes
// an RLX `Graph` plus baked parameter/constant data and produces a
// `proto::Model` ready to serialise into a `.mlpackage`. No FFI, so this
// builds and unit-tests on any host.

//! `norm` — extracted from the `mil` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use super::helpers::simple_op_flex;
use super::helpers::*;
use crate::proto;
use crate::{CoremlError, Result};
use rlx_ir::op::{Activation, AdaNormKind, CmpOp, MaskKind, ReduceOp};
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Dim, Graph, NodeId, Op, Shape};
use std::collections::HashMap;

use super::*;

impl<'a> LowerCtx<'a> {
    /// LayerNorm over the last `axis` dims, with optional affine. The IR
    /// node carries inputs `[x, gamma?, beta?]`.
    pub(crate) fn lower_layer_norm(
        &mut self,
        id: NodeId,
        axis: i32,
        eps: f32,
        out_name: &str,
    ) -> Result<()> {
        // Decompose to primitives rather than emit MIL's native `layer_norm`
        // op: that op requires `gamma`/`beta` to be compile-time CONST, but the
        // ConvNeXt/CFM/VITS family feeds affine params that are runtime values
        // (cast weights, computed tensors) → CoreML load fails "Param 'beta'
        // must be const". `mul`/`add` accept any operand, so the decomposition
        // (mathematically identical) works universally. Mirrors `lower_rms_norm`
        // but centers first.
        let node = self.graph.node(id);
        let x = self.val(node.inputs[0]);
        let rank = node.shape.rank() as i32;
        let norm_axis = (if axis < 0 { axis + rank } else { axis }) as usize;
        let axes: Vec<i32> = (norm_axis as i32..rank).collect();
        let red_shape = reduced_shape(&node.shape, norm_axis);

        // mean = reduce_mean(x, axes); xc = x - mean
        let mean = format!("{out_name}_mean");
        self.operations.push(self.simple_op(
            "reduce_mean",
            &mean,
            &red_shape,
            vec![
                ("x", bind_name(&x)),
                ("axes", bind_value(vec_i32(&axes))),
                ("keep_dims", bind_value(scalar_bool(true))),
            ],
        )?);
        let xc = format!("{out_name}_xc");
        self.operations.push(self.simple_op(
            "sub",
            &xc,
            &node.shape,
            vec![("x", bind_name(&x)), ("y", bind_name(&mean))],
        )?);
        // var = reduce_mean(xc²); inv = rsqrt(var + eps)
        let sq = format!("{out_name}_sq");
        self.operations.push(self.simple_op(
            "mul",
            &sq,
            &node.shape,
            vec![("x", bind_name(&xc)), ("y", bind_name(&xc))],
        )?);
        let var = format!("{out_name}_var");
        self.operations.push(self.simple_op(
            "reduce_mean",
            &var,
            &red_shape,
            vec![
                ("x", bind_name(&sq)),
                ("axes", bind_value(vec_i32(&axes))),
                ("keep_dims", bind_value(scalar_bool(true))),
            ],
        )?);
        let vare = format!("{out_name}_vare");
        self.operations.push(self.simple_op(
            "add",
            &vare,
            &red_shape,
            vec![("x", bind_name(&var)), ("y", bind_value(scalar_f32(eps)))],
        )?);
        let inv = format!("{out_name}_inv");
        self.operations.push(self.simple_op(
            "rsqrt",
            &inv,
            &red_shape,
            vec![
                ("x", bind_name(&vare)),
                ("epsilon", bind_value(scalar_f32(0.0))),
            ],
        )?);

        let has_gamma = node.inputs.len() > 1;
        let has_beta = node.inputs.len() > 2;
        // xn = xc * inv
        let xn_name = if has_gamma || has_beta {
            format!("{out_name}_xn")
        } else {
            out_name.to_string()
        };
        self.operations.push(self.simple_op(
            "mul",
            &xn_name,
            &node.shape,
            vec![("x", bind_name(&xc)), ("y", bind_name(&inv))],
        )?);
        let mut last = xn_name;
        if has_gamma {
            let g = self.val(node.inputs[1]);
            let name = if has_beta {
                format!("{out_name}_xg")
            } else {
                out_name.to_string()
            };
            self.operations.push(self.simple_op(
                "mul",
                &name,
                &node.shape,
                vec![("x", bind_name(&last)), ("y", bind_name(&g))],
            )?);
            last = name;
        }
        if has_beta {
            let b = self.val(node.inputs[2]);
            self.operations.push(self.simple_op(
                "add",
                out_name,
                &node.shape,
                vec![("x", bind_name(&last)), ("y", bind_name(&b))],
            )?);
        }
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// RMSNorm over the trailing dims from `axis`: composed from
    /// primitive MIL ops since the base opset has no `rms_norm`.
    /// `y = x · rsqrt(mean(x², axes) + eps) · gamma`. Inputs `[x, gamma?]`.
    pub(crate) fn lower_rms_norm(
        &mut self,
        id: NodeId,
        axis: i32,
        eps: f32,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let x = self.val(node.inputs[0]);
        let rank = node.shape.rank();
        let norm_axis = if axis < 0 { axis + rank as i32 } else { axis } as usize;
        let axes: Vec<i32> = (norm_axis..rank).map(|a| a as i32).collect();
        let red_shape = reduced_shape(&node.shape, norm_axis);

        // sq = x * x
        let sq = format!("{out_name}_sq");
        self.operations.push(self.simple_op(
            "mul",
            &sq,
            &node.shape,
            vec![("x", bind_name(&x)), ("y", bind_name(&x))],
        )?);
        // ms = reduce_mean(sq, axes, keep_dims=true)
        let ms = format!("{out_name}_ms");
        self.operations.push(self.simple_op(
            "reduce_mean",
            &ms,
            &red_shape,
            vec![
                ("x", bind_name(&sq)),
                ("axes", bind_value(vec_i32(&axes))),
                ("keep_dims", bind_value(scalar_bool(true))),
            ],
        )?);
        // ms_eps = ms + eps
        let mse = format!("{out_name}_mse");
        self.operations.push(self.simple_op(
            "add",
            &mse,
            &red_shape,
            vec![("x", bind_name(&ms)), ("y", bind_value(scalar_f32(eps)))],
        )?);
        // inv = rsqrt(ms_eps)  (eps already folded into ms_eps above)
        let inv = format!("{out_name}_inv");
        self.operations.push(self.simple_op(
            "rsqrt",
            &inv,
            &red_shape,
            vec![
                ("x", bind_name(&mse)),
                ("epsilon", bind_value(scalar_f32(0.0))),
            ],
        )?);

        // RmsNorm carries inputs [x, gamma, beta]; gamma scales and beta
        // shifts (matching the CPU kernel `x·inv·gamma + beta`). Both are
        // optional defensively, though the IR verifier requires all three.
        let has_gamma = node.inputs.len() > 1;
        let has_beta = node.inputs.len() > 2;

        // xn = x * inv  (broadcast)
        let xn_name = if has_gamma || has_beta {
            format!("{out_name}_xn")
        } else {
            out_name.to_string()
        };
        self.operations.push(self.simple_op(
            "mul",
            &xn_name,
            &node.shape,
            vec![("x", bind_name(&x)), ("y", bind_name(&inv))],
        )?);

        let mut last = xn_name;
        if has_gamma {
            let g = self.val(node.inputs[1]);
            let name = if has_beta {
                format!("{out_name}_xg")
            } else {
                out_name.to_string()
            };
            self.operations.push(self.simple_op(
                "mul",
                &name,
                &node.shape,
                vec![("x", bind_name(&last)), ("y", bind_name(&g))],
            )?);
            last = name;
        }
        if has_beta {
            let b = self.val(node.inputs[2]);
            self.operations.push(self.simple_op(
                "add",
                out_name,
                &node.shape,
                vec![("x", bind_name(&last)), ("y", bind_name(&b))],
            )?);
        }
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// `LayerNorm(x + residual [+ bias])` — compose Add then layer-norm.
    pub(crate) fn lower_fused_residual_ln(
        &mut self,
        id: NodeId,
        has_bias: bool,
        eps: f32,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let shape = node.shape.clone();
        let x = self.val(node.inputs[0]);
        let r = self.val(node.inputs[1]);
        let summed = format!("{out_name}_sum");
        self.emit(
            "add",
            &summed,
            &shape,
            vec![("x", bind_name(&x)), ("y", bind_name(&r))],
        )?;
        let pre = if has_bias {
            let bias = self.val(node.inputs[2]);
            let name = format!("{out_name}_bias");
            self.emit(
                "add",
                &name,
                &shape,
                vec![("x", bind_name(&summed)), ("y", bind_name(&bias))],
            )?;
            name
        } else {
            summed
        };
        let (g_idx, b_idx) = if has_bias { (3, 4) } else { (2, 3) };
        let g = self.val(node.inputs[g_idx]);
        let b = self.val(node.inputs[b_idx]);
        // Reuse LayerNorm composition by temporarily wiring through a
        // synthetic path: emit mean/var/affine on `pre`.
        let rank = shape.rank() as i32;
        let axis = -1i32;
        let norm_axis = (if axis < 0 { axis + rank } else { axis }) as usize;
        let axes: Vec<i32> = (norm_axis as i32..rank).collect();
        let red_shape = reduced_shape(&shape, norm_axis);
        let mean = format!("{out_name}_mean");
        self.operations.push(self.simple_op(
            "reduce_mean",
            &mean,
            &red_shape,
            vec![
                ("x", bind_name(&pre)),
                ("axes", bind_value(vec_i32(&axes))),
                ("keep_dims", bind_value(scalar_bool(true))),
            ],
        )?);
        let xc = format!("{out_name}_xc");
        self.operations.push(self.simple_op(
            "sub",
            &xc,
            &shape,
            vec![("x", bind_name(&pre)), ("y", bind_name(&mean))],
        )?);
        let sq = format!("{out_name}_sq");
        self.operations.push(self.simple_op(
            "mul",
            &sq,
            &shape,
            vec![("x", bind_name(&xc)), ("y", bind_name(&xc))],
        )?);
        let var = format!("{out_name}_var");
        self.operations.push(self.simple_op(
            "reduce_mean",
            &var,
            &red_shape,
            vec![
                ("x", bind_name(&sq)),
                ("axes", bind_value(vec_i32(&axes))),
                ("keep_dims", bind_value(scalar_bool(true))),
            ],
        )?);
        let var_eps = format!("{out_name}_ve");
        self.operations.push(self.simple_op(
            "add",
            &var_eps,
            &red_shape,
            vec![("x", bind_name(&var)), ("y", bind_value(scalar_f32(eps)))],
        )?);
        let inv = format!("{out_name}_inv");
        self.operations.push(self.simple_op(
            "rsqrt",
            &inv,
            &red_shape,
            vec![
                ("x", bind_name(&var_eps)),
                ("epsilon", bind_value(scalar_f32(0.0))),
            ],
        )?);
        let normed = format!("{out_name}_n");
        self.operations.push(self.simple_op(
            "mul",
            &normed,
            &shape,
            vec![("x", bind_name(&xc)), ("y", bind_name(&inv))],
        )?);
        let scaled = format!("{out_name}_g");
        self.operations.push(self.simple_op(
            "mul",
            &scaled,
            &shape,
            vec![("x", bind_name(&normed)), ("y", bind_name(&g))],
        )?);
        self.emit(
            "add",
            out_name,
            &shape,
            vec![("x", bind_name(&scaled)), ("y", bind_name(&b))],
        )?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// `RmsNorm(x + residual [+ bias])`.
    pub(crate) fn lower_fused_residual_rms_norm(
        &mut self,
        id: NodeId,
        has_bias: bool,
        eps: f32,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let shape = node.shape.clone();
        let x = self.val(node.inputs[0]);
        let r = self.val(node.inputs[1]);
        let summed = format!("{out_name}_sum");
        self.emit(
            "add",
            &summed,
            &shape,
            vec![("x", bind_name(&x)), ("y", bind_name(&r))],
        )?;
        let pre = if has_bias {
            let bias = self.val(node.inputs[2]);
            let name = format!("{out_name}_bias");
            self.emit(
                "add",
                &name,
                &shape,
                vec![("x", bind_name(&summed)), ("y", bind_name(&bias))],
            )?;
            name
        } else {
            summed
        };
        let g_idx = if has_bias { 3 } else { 2 };
        let g = self.val(node.inputs[g_idx]);
        let rank = shape.rank();
        let norm_axis = rank - 1;
        let axes: Vec<i32> = (norm_axis..rank).map(|a| a as i32).collect();
        let red_shape = reduced_shape(&shape, norm_axis);
        let sq = format!("{out_name}_sq");
        self.operations.push(self.simple_op(
            "mul",
            &sq,
            &shape,
            vec![("x", bind_name(&pre)), ("y", bind_name(&pre))],
        )?);
        let mean = format!("{out_name}_ms");
        self.operations.push(self.simple_op(
            "reduce_mean",
            &mean,
            &red_shape,
            vec![
                ("x", bind_name(&sq)),
                ("axes", bind_value(vec_i32(&axes))),
                ("keep_dims", bind_value(scalar_bool(true))),
            ],
        )?);
        let ve = format!("{out_name}_ve");
        self.operations.push(self.simple_op(
            "add",
            &ve,
            &red_shape,
            vec![("x", bind_name(&mean)), ("y", bind_value(scalar_f32(eps)))],
        )?);
        let inv = format!("{out_name}_inv");
        self.operations.push(self.simple_op(
            "rsqrt",
            &inv,
            &red_shape,
            vec![
                ("x", bind_name(&ve)),
                ("epsilon", bind_value(scalar_f32(0.0))),
            ],
        )?);
        let normed = format!("{out_name}_n");
        self.operations.push(self.simple_op(
            "mul",
            &normed,
            &shape,
            vec![("x", bind_name(&pre)), ("y", bind_name(&inv))],
        )?);
        self.emit(
            "mul",
            out_name,
            &shape,
            vec![("x", bind_name(&normed)), ("y", bind_name(&g))],
        )?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// RMSNorm backward w.r.t. input. Inputs `[x, gamma, beta, dy]`, output = `x`.
    /// Mirrors `compose_rms_norm_backward_input`:
    ///   inv = rsqrt(mean(x², ax) + eps);  dy_g = dy·gamma
    ///   dot = mean(x·dy_g, ax);  dx = inv·(dy_g − x·dot·inv³)
    /// All reductions keep dims so `[...,1]` factors broadcast over `[...,H]`.
    #[cfg(feature = "training")]
    pub(crate) fn lower_rms_norm_backward_input(
        &mut self,
        id: NodeId,
        axis: i32,
        eps: f32,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let x = self.val(node.inputs[0]);
        let gamma = self.val(node.inputs[1]);
        // node.inputs[2] = beta — additive in the forward, so absent from dx.
        let dy = self.val(node.inputs[3]);
        let full = node.shape.clone();
        let rank = full.rank();
        let norm_axis = if axis < 0 { axis + rank as i32 } else { axis } as usize;
        let axes: Vec<i32> = (norm_axis..rank).map(|a| a as i32).collect();
        let red = reduced_shape(&full, norm_axis);
        let red_axes = || bind_value(vec_i32(&axes));
        let keep = || bind_value(scalar_bool(true));

        // inv = rsqrt(mean(x*x, axes) + eps)
        let x2 = format!("{out_name}_x2");
        self.emit(
            "mul",
            &x2,
            &full,
            vec![("x", bind_name(&x)), ("y", bind_name(&x))],
        )?;
        let mx2 = format!("{out_name}_mx2");
        self.emit(
            "reduce_mean",
            &mx2,
            &red,
            vec![
                ("x", bind_name(&x2)),
                ("axes", red_axes()),
                ("keep_dims", keep()),
            ],
        )?;
        let ve = format!("{out_name}_ve");
        self.emit(
            "add",
            &ve,
            &red,
            vec![("x", bind_name(&mx2)), ("y", bind_value(scalar_f32(eps)))],
        )?;
        let inv = format!("{out_name}_inv");
        self.emit(
            "rsqrt",
            &inv,
            &red,
            vec![
                ("x", bind_name(&ve)),
                ("epsilon", bind_value(scalar_f32(0.0))),
            ],
        )?;
        let inv2 = format!("{out_name}_inv2");
        self.emit(
            "mul",
            &inv2,
            &red,
            vec![("x", bind_name(&inv)), ("y", bind_name(&inv))],
        )?;

        // dy_g = dy * gamma  (gamma [H] broadcasts over [...,H])
        let dyg = format!("{out_name}_dyg");
        self.emit(
            "mul",
            &dyg,
            &full,
            vec![("x", bind_name(&dy)), ("y", bind_name(&gamma))],
        )?;
        // dot = mean(x * dy_g, axes)
        let xdyg = format!("{out_name}_xdyg");
        self.emit(
            "mul",
            &xdyg,
            &full,
            vec![("x", bind_name(&x)), ("y", bind_name(&dyg))],
        )?;
        let dot = format!("{out_name}_dot");
        self.emit(
            "reduce_mean",
            &dot,
            &red,
            vec![
                ("x", bind_name(&xdyg)),
                ("axes", red_axes()),
                ("keep_dims", keep()),
            ],
        )?;
        // term2 = x * dot * inv²  (the outer `* inv` below makes the cross term inv³, not inv⁴)
        let xdot = format!("{out_name}_xdot");
        self.emit(
            "mul",
            &xdot,
            &full,
            vec![("x", bind_name(&x)), ("y", bind_name(&dot))],
        )?;
        let term2 = format!("{out_name}_t2");
        self.emit(
            "mul",
            &term2,
            &full,
            vec![("x", bind_name(&xdot)), ("y", bind_name(&inv2))],
        )?;
        // diff = dy_g - term2;  dx = diff * inv
        let diff = format!("{out_name}_diff");
        self.emit(
            "sub",
            &diff,
            &full,
            vec![("x", bind_name(&dyg)), ("y", bind_name(&term2))],
        )?;
        let op = self.simple_op(
            "mul",
            out_name,
            &full,
            vec![("x", bind_name(&diff)), ("y", bind_name(&inv))],
        )?;
        self.push_named(id, out_name.to_string(), op);
        Ok(())
    }

    /// RMSNorm backward w.r.t. gamma. Inputs `[x, gamma, beta, dy]`, output =
    /// `gamma` (`[H]`). Mirrors `compose_rms_norm_backward_gamma`:
    ///   `dgamma = sum_batch(dy · x · rsqrt(mean(x², ax) + eps))`.
    #[cfg(feature = "training")]
    pub(crate) fn lower_rms_norm_backward_gamma(
        &mut self,
        id: NodeId,
        axis: i32,
        eps: f32,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let x = self.val(node.inputs[0]);
        let dy = self.val(node.inputs[3]);
        let gamma_shape = node.shape.clone();
        let x_shape = self.graph.shape(node.inputs[0]).clone();
        let rank = x_shape.rank();
        let norm_axis = if axis < 0 { axis + rank as i32 } else { axis } as usize;
        let axes: Vec<i32> = (norm_axis..rank).map(|a| a as i32).collect();
        let red = reduced_shape(&x_shape, norm_axis);
        let batch_axes: Vec<i32> = (0..rank as i32)
            .filter(|&i| i as usize != norm_axis)
            .collect();

        let x2 = format!("{out_name}_x2");
        self.emit(
            "mul",
            &x2,
            &x_shape,
            vec![("x", bind_name(&x)), ("y", bind_name(&x))],
        )?;
        let mx2 = format!("{out_name}_mx2");
        self.emit(
            "reduce_mean",
            &mx2,
            &red,
            vec![
                ("x", bind_name(&x2)),
                ("axes", bind_value(vec_i32(&axes))),
                ("keep_dims", bind_value(scalar_bool(true))),
            ],
        )?;
        let ve = format!("{out_name}_ve");
        self.emit(
            "add",
            &ve,
            &red,
            vec![("x", bind_name(&mx2)), ("y", bind_value(scalar_f32(eps)))],
        )?;
        let inv = format!("{out_name}_inv");
        self.emit(
            "rsqrt",
            &inv,
            &red,
            vec![
                ("x", bind_name(&ve)),
                ("epsilon", bind_value(scalar_f32(0.0))),
            ],
        )?;
        let xinv = format!("{out_name}_xinv");
        self.emit(
            "mul",
            &xinv,
            &x_shape,
            vec![("x", bind_name(&x)), ("y", bind_name(&inv))],
        )?;
        let prod = format!("{out_name}_prod");
        self.emit(
            "mul",
            &prod,
            &x_shape,
            vec![("x", bind_name(&dy)), ("y", bind_name(&xinv))],
        )?;
        let op = self.simple_op(
            "reduce_sum",
            out_name,
            &gamma_shape,
            vec![
                ("x", bind_name(&prod)),
                ("axes", bind_value(vec_i32(&batch_axes))),
                ("keep_dims", bind_value(scalar_bool(false))),
            ],
        )?;
        self.push_named(id, out_name.to_string(), op);
        Ok(())
    }

    /// RMSNorm backward w.r.t. beta. Inputs `[x, gamma, beta, dy]`, output =
    /// `beta` (`[H]`). `dbeta = sum_batch(dy)`.
    #[cfg(feature = "training")]
    pub(crate) fn lower_rms_norm_backward_beta(
        &mut self,
        id: NodeId,
        _axis: i32,
        _eps: f32,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let dy = self.val(node.inputs[3]);
        let beta_shape = node.shape.clone();
        let rank = self.graph.shape(node.inputs[3]).rank();
        // beta is over the last (feature) axis; reduce every batch axis.
        let batch_axes: Vec<i32> = (0..rank as i32 - 1).collect();
        let op = self.simple_op(
            "reduce_sum",
            out_name,
            &beta_shape,
            vec![
                ("x", bind_name(&dy)),
                ("axes", bind_value(vec_i32(&batch_axes))),
                ("keep_dims", bind_value(scalar_bool(false))),
            ],
        )?;
        self.push_named(id, out_name.to_string(), op);
        Ok(())
    }

    /// Native LayerNorm backward w.r.t. input (axis = -1). Inputs `[x, gamma, dy]`,
    /// output matches `x`. Mirrors `compose_layer_norm_backward_input`:
    ///   `dx = inv_std·(sy − mean(sy) − x_hat·mean(sy·x_hat))`, `sy = dy·γ`,
    ///   `x_hat = (x − mean)·inv_std`, `inv_std = rsqrt(var + eps)`. Composed MIL
    /// with implicit broadcasting (reduced `[..,1]` tensors broadcast over the norm
    /// axis), no decomposition `expand`s.
    #[cfg(feature = "training")]
    pub(crate) fn lower_layer_norm_backward_input(
        &mut self,
        id: NodeId,
        axis: i32,
        eps: f32,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let x = self.val(node.inputs[0]);
        let gamma = self.val(node.inputs[1]);
        let dy = self.val(node.inputs[2]);
        let full = node.shape.clone();
        let rank = full.rank();
        let norm_axis = if axis < 0 { axis + rank as i32 } else { axis } as usize;
        let axes: Vec<i32> = (norm_axis..rank).map(|a| a as i32).collect();
        let red = reduced_shape(&full, norm_axis);
        let red_axes = || bind_value(vec_i32(&axes));
        let keep = || bind_value(scalar_bool(true));

        // mean, centered x
        let mean = format!("{out_name}_mean");
        self.emit(
            "reduce_mean",
            &mean,
            &red,
            vec![
                ("x", bind_name(&x)),
                ("axes", red_axes()),
                ("keep_dims", keep()),
            ],
        )?;
        let xc = format!("{out_name}_xc");
        self.emit(
            "sub",
            &xc,
            &full,
            vec![("x", bind_name(&x)), ("y", bind_name(&mean))],
        )?;
        // var, inv_std = rsqrt(var + eps)
        let xc2 = format!("{out_name}_xc2");
        self.emit(
            "mul",
            &xc2,
            &full,
            vec![("x", bind_name(&xc)), ("y", bind_name(&xc))],
        )?;
        let var = format!("{out_name}_var");
        self.emit(
            "reduce_mean",
            &var,
            &red,
            vec![
                ("x", bind_name(&xc2)),
                ("axes", red_axes()),
                ("keep_dims", keep()),
            ],
        )?;
        let ve = format!("{out_name}_ve");
        self.emit(
            "add",
            &ve,
            &red,
            vec![("x", bind_name(&var)), ("y", bind_value(scalar_f32(eps)))],
        )?;
        let inv_std = format!("{out_name}_invs");
        self.emit(
            "rsqrt",
            &inv_std,
            &red,
            vec![
                ("x", bind_name(&ve)),
                ("epsilon", bind_value(scalar_f32(0.0))),
            ],
        )?;
        let x_hat = format!("{out_name}_xhat");
        self.emit(
            "mul",
            &x_hat,
            &full,
            vec![("x", bind_name(&xc)), ("y", bind_name(&inv_std))],
        )?;

        // sy = dy·γ; its mean; and mean(sy·x_hat)
        let sy = format!("{out_name}_sy");
        self.emit(
            "mul",
            &sy,
            &full,
            vec![("x", bind_name(&dy)), ("y", bind_name(&gamma))],
        )?;
        let m_sy = format!("{out_name}_msy");
        self.emit(
            "reduce_mean",
            &m_sy,
            &red,
            vec![
                ("x", bind_name(&sy)),
                ("axes", red_axes()),
                ("keep_dims", keep()),
            ],
        )?;
        let sy_xh = format!("{out_name}_syxh");
        self.emit(
            "mul",
            &sy_xh,
            &full,
            vec![("x", bind_name(&sy)), ("y", bind_name(&x_hat))],
        )?;
        let m_sxh = format!("{out_name}_msxh");
        self.emit(
            "reduce_mean",
            &m_sxh,
            &red,
            vec![
                ("x", bind_name(&sy_xh)),
                ("axes", red_axes()),
                ("keep_dims", keep()),
            ],
        )?;

        // dx = inv_std·(sy − mean(sy) − x_hat·mean(sy·x_hat))
        let t1 = format!("{out_name}_t1");
        self.emit(
            "sub",
            &t1,
            &full,
            vec![("x", bind_name(&sy)), ("y", bind_name(&m_sy))],
        )?;
        let t2 = format!("{out_name}_t2");
        self.emit(
            "mul",
            &t2,
            &full,
            vec![("x", bind_name(&x_hat)), ("y", bind_name(&m_sxh))],
        )?;
        let t3 = format!("{out_name}_t3");
        self.emit(
            "sub",
            &t3,
            &full,
            vec![("x", bind_name(&t1)), ("y", bind_name(&t2))],
        )?;
        let op = self.simple_op(
            "mul",
            out_name,
            &full,
            vec![("x", bind_name(&inv_std)), ("y", bind_name(&t3))],
        )?;
        self.push_named(id, out_name.to_string(), op);
        Ok(())
    }

    /// Native LayerNorm backward w.r.t. gamma. Inputs `[x, dy]`, output = gamma
    /// shape. Mirrors `compose_layer_norm_backward_gamma`:
    ///   `dgamma = Σ_batch(dy · x_hat)`, `x_hat = (x − mean)·rsqrt(var + eps)`.
    #[cfg(feature = "training")]
    pub(crate) fn lower_layer_norm_backward_gamma(
        &mut self,
        id: NodeId,
        axis: i32,
        eps: f32,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let x = self.val(node.inputs[0]);
        let dy = self.val(node.inputs[1]);
        let gamma_shape = node.shape.clone();
        let x_shape = self.graph.shape(node.inputs[0]).clone();
        let rank = x_shape.rank();
        let norm_axis = if axis < 0 { axis + rank as i32 } else { axis } as usize;
        let axes: Vec<i32> = (norm_axis..rank).map(|a| a as i32).collect();
        let red = reduced_shape(&x_shape, norm_axis);
        let batch_axes: Vec<i32> = (0..rank as i32)
            .filter(|&i| i as usize != norm_axis)
            .collect();
        let red_axes = || bind_value(vec_i32(&axes));
        let keep = || bind_value(scalar_bool(true));

        let mean = format!("{out_name}_mean");
        self.emit(
            "reduce_mean",
            &mean,
            &red,
            vec![
                ("x", bind_name(&x)),
                ("axes", red_axes()),
                ("keep_dims", keep()),
            ],
        )?;
        let xc = format!("{out_name}_xc");
        self.emit(
            "sub",
            &xc,
            &x_shape,
            vec![("x", bind_name(&x)), ("y", bind_name(&mean))],
        )?;
        let xc2 = format!("{out_name}_xc2");
        self.emit(
            "mul",
            &xc2,
            &x_shape,
            vec![("x", bind_name(&xc)), ("y", bind_name(&xc))],
        )?;
        let var = format!("{out_name}_var");
        self.emit(
            "reduce_mean",
            &var,
            &red,
            vec![
                ("x", bind_name(&xc2)),
                ("axes", red_axes()),
                ("keep_dims", keep()),
            ],
        )?;
        let ve = format!("{out_name}_ve");
        self.emit(
            "add",
            &ve,
            &red,
            vec![("x", bind_name(&var)), ("y", bind_value(scalar_f32(eps)))],
        )?;
        let inv_std = format!("{out_name}_invs");
        self.emit(
            "rsqrt",
            &inv_std,
            &red,
            vec![
                ("x", bind_name(&ve)),
                ("epsilon", bind_value(scalar_f32(0.0))),
            ],
        )?;
        let x_hat = format!("{out_name}_xhat");
        self.emit(
            "mul",
            &x_hat,
            &x_shape,
            vec![("x", bind_name(&xc)), ("y", bind_name(&inv_std))],
        )?;
        let prod = format!("{out_name}_prod");
        self.emit(
            "mul",
            &prod,
            &x_shape,
            vec![("x", bind_name(&dy)), ("y", bind_name(&x_hat))],
        )?;
        let op = self.simple_op(
            "reduce_sum",
            out_name,
            &gamma_shape,
            vec![
                ("x", bind_name(&prod)),
                ("axes", bind_value(vec_i32(&batch_axes))),
                ("keep_dims", bind_value(scalar_bool(false))),
            ],
        )?;
        self.push_named(id, out_name.to_string(), op);
        Ok(())
    }

    /// Native GroupNorm backward w.r.t. input (NCHW). Inputs `[x, gamma, beta, dy]`,
    /// output matches `x`. Reshapes `[N,C,H,W] → [N,G,M]` (M = C/G·H·W) so the group
    /// stats are a single last-axis reduce; the affine `sy = dy·γ` is done in NCHW
    /// (γ broadcasts over H,W) before the reshape. Same math as
    /// `compose_group_norm_backward_input`, without the per-group narrow/concat loop.
    #[cfg(feature = "training")]
    pub(crate) fn lower_group_norm_backward_input(
        &mut self,
        id: NodeId,
        num_groups: usize,
        eps: f32,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let x = self.val(node.inputs[0]);
        let gamma = self.val(node.inputs[1]);
        // inputs[2] = beta (additive in the forward, absent from dx)
        let dy = self.val(node.inputs[3]);
        let full = node.shape.clone(); // [N,C,H,W]
        let (n, c, h, w) = (
            full.dim(0).unwrap_static(),
            full.dim(1).unwrap_static(),
            full.dim(2).unwrap_static(),
            full.dim(3).unwrap_static(),
        );
        let dt = full.dtype();
        let m = (c / num_groups) * h * w;
        let grouped = Shape::new(&[n, num_groups, m], dt);
        let red = Shape::new(&[n, num_groups, 1], dt);
        let g3 = || bind_value(vec_i32(&[n as i32, num_groups as i32, m as i32]));
        let red_axis = || bind_value(vec_i32(&[2]));
        let keep = || bind_value(scalar_bool(true));

        // sy = dy·γ in NCHW (γ [C] → [1,C,1,1] broadcasts over H,W)
        let gr = format!("{out_name}_gr");
        self.emit(
            "reshape",
            &gr,
            &Shape::new(&[1, c, 1, 1], dt),
            vec![
                ("x", bind_name(&gamma)),
                ("shape", bind_value(vec_i32(&[1, c as i32, 1, 1]))),
            ],
        )?;
        let sy_nchw = format!("{out_name}_synchw");
        self.emit(
            "mul",
            &sy_nchw,
            &full,
            vec![("x", bind_name(&dy)), ("y", bind_name(&gr))],
        )?;

        // group the channels: x, sy → [N,G,M]
        let xf = format!("{out_name}_xf");
        self.emit(
            "reshape",
            &xf,
            &grouped,
            vec![("x", bind_name(&x)), ("shape", g3())],
        )?;
        let syf = format!("{out_name}_syf");
        self.emit(
            "reshape",
            &syf,
            &grouped,
            vec![("x", bind_name(&sy_nchw)), ("shape", g3())],
        )?;

        // mean, var, inv_std over the group axis
        let mean = format!("{out_name}_mean");
        self.emit(
            "reduce_mean",
            &mean,
            &red,
            vec![
                ("x", bind_name(&xf)),
                ("axes", red_axis()),
                ("keep_dims", keep()),
            ],
        )?;
        let xc = format!("{out_name}_xc");
        self.emit(
            "sub",
            &xc,
            &grouped,
            vec![("x", bind_name(&xf)), ("y", bind_name(&mean))],
        )?;
        let xc2 = format!("{out_name}_xc2");
        self.emit(
            "mul",
            &xc2,
            &grouped,
            vec![("x", bind_name(&xc)), ("y", bind_name(&xc))],
        )?;
        let var = format!("{out_name}_var");
        self.emit(
            "reduce_mean",
            &var,
            &red,
            vec![
                ("x", bind_name(&xc2)),
                ("axes", red_axis()),
                ("keep_dims", keep()),
            ],
        )?;
        let ve = format!("{out_name}_ve");
        self.emit(
            "add",
            &ve,
            &red,
            vec![("x", bind_name(&var)), ("y", bind_value(scalar_f32(eps)))],
        )?;
        let inv_std = format!("{out_name}_invs");
        self.emit(
            "rsqrt",
            &inv_std,
            &red,
            vec![
                ("x", bind_name(&ve)),
                ("epsilon", bind_value(scalar_f32(0.0))),
            ],
        )?;
        let x_hat = format!("{out_name}_xhat");
        self.emit(
            "mul",
            &x_hat,
            &grouped,
            vec![("x", bind_name(&xc)), ("y", bind_name(&inv_std))],
        )?;

        // mean(sy), mean(sy·x_hat) over the group axis
        let m_sy = format!("{out_name}_msy");
        self.emit(
            "reduce_mean",
            &m_sy,
            &red,
            vec![
                ("x", bind_name(&syf)),
                ("axes", red_axis()),
                ("keep_dims", keep()),
            ],
        )?;
        let sy_xh = format!("{out_name}_syxh");
        self.emit(
            "mul",
            &sy_xh,
            &grouped,
            vec![("x", bind_name(&syf)), ("y", bind_name(&x_hat))],
        )?;
        let m_sxh = format!("{out_name}_msxh");
        self.emit(
            "reduce_mean",
            &m_sxh,
            &red,
            vec![
                ("x", bind_name(&sy_xh)),
                ("axes", red_axis()),
                ("keep_dims", keep()),
            ],
        )?;

        // flat_dx = inv_std·(sy − mean(sy) − x_hat·mean(sy·x_hat)); reshape to NCHW
        let t1 = format!("{out_name}_t1");
        self.emit(
            "sub",
            &t1,
            &grouped,
            vec![("x", bind_name(&syf)), ("y", bind_name(&m_sy))],
        )?;
        let t2 = format!("{out_name}_t2");
        self.emit(
            "mul",
            &t2,
            &grouped,
            vec![("x", bind_name(&x_hat)), ("y", bind_name(&m_sxh))],
        )?;
        let t3 = format!("{out_name}_t3");
        self.emit(
            "sub",
            &t3,
            &grouped,
            vec![("x", bind_name(&t1)), ("y", bind_name(&t2))],
        )?;
        let flat_dx = format!("{out_name}_fdx");
        self.emit(
            "mul",
            &flat_dx,
            &grouped,
            vec![("x", bind_name(&t3)), ("y", bind_name(&inv_std))],
        )?;
        let op = self.simple_op(
            "reshape",
            out_name,
            &full,
            vec![
                ("x", bind_name(&flat_dx)),
                (
                    "shape",
                    bind_value(vec_i32(&[n as i32, c as i32, h as i32, w as i32])),
                ),
            ],
        )?;
        self.push_named(id, out_name.to_string(), op);
        Ok(())
    }

    /// Native GroupNorm backward w.r.t. gamma (NCHW). Inputs `[x, dy]`, output =
    /// gamma `[C]`. `dgamma[c] = Σ_{n,h,w} dy·x_hat`, with `x_hat` the group-
    /// normalized `x` (computed in the `[N,G,M]` layout, then reshaped back to NCHW
    /// so the channel reduction over axes {N,H,W} is exact for any batch size).
    #[cfg(feature = "training")]
    pub(crate) fn lower_group_norm_backward_gamma(
        &mut self,
        id: NodeId,
        num_groups: usize,
        eps: f32,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let x = self.val(node.inputs[0]);
        let dy = self.val(node.inputs[1]);
        let gamma_shape = node.shape.clone(); // [C]
        let xs = self.graph.shape(node.inputs[0]).clone(); // [N,C,H,W]
        let (n, c, h, w) = (
            xs.dim(0).unwrap_static(),
            xs.dim(1).unwrap_static(),
            xs.dim(2).unwrap_static(),
            xs.dim(3).unwrap_static(),
        );
        let dt = xs.dtype();
        let m = (c / num_groups) * h * w;
        let grouped = Shape::new(&[n, num_groups, m], dt);
        let red = Shape::new(&[n, num_groups, 1], dt);
        let g3 = || bind_value(vec_i32(&[n as i32, num_groups as i32, m as i32]));
        let red_axis = || bind_value(vec_i32(&[2]));
        let keep = || bind_value(scalar_bool(true));

        let xf = format!("{out_name}_xf");
        self.emit(
            "reshape",
            &xf,
            &grouped,
            vec![("x", bind_name(&x)), ("shape", g3())],
        )?;
        let mean = format!("{out_name}_mean");
        self.emit(
            "reduce_mean",
            &mean,
            &red,
            vec![
                ("x", bind_name(&xf)),
                ("axes", red_axis()),
                ("keep_dims", keep()),
            ],
        )?;
        let xc = format!("{out_name}_xc");
        self.emit(
            "sub",
            &xc,
            &grouped,
            vec![("x", bind_name(&xf)), ("y", bind_name(&mean))],
        )?;
        let xc2 = format!("{out_name}_xc2");
        self.emit(
            "mul",
            &xc2,
            &grouped,
            vec![("x", bind_name(&xc)), ("y", bind_name(&xc))],
        )?;
        let var = format!("{out_name}_var");
        self.emit(
            "reduce_mean",
            &var,
            &red,
            vec![
                ("x", bind_name(&xc2)),
                ("axes", red_axis()),
                ("keep_dims", keep()),
            ],
        )?;
        let ve = format!("{out_name}_ve");
        self.emit(
            "add",
            &ve,
            &red,
            vec![("x", bind_name(&var)), ("y", bind_value(scalar_f32(eps)))],
        )?;
        let inv_std = format!("{out_name}_invs");
        self.emit(
            "rsqrt",
            &inv_std,
            &red,
            vec![
                ("x", bind_name(&ve)),
                ("epsilon", bind_value(scalar_f32(0.0))),
            ],
        )?;
        let x_hat_g = format!("{out_name}_xhatg");
        self.emit(
            "mul",
            &x_hat_g,
            &grouped,
            vec![("x", bind_name(&xc)), ("y", bind_name(&inv_std))],
        )?;
        // back to NCHW so the channel-aligned reduction is unambiguous
        let x_hat = format!("{out_name}_xhat");
        self.emit(
            "reshape",
            &x_hat,
            &xs,
            vec![
                ("x", bind_name(&x_hat_g)),
                (
                    "shape",
                    bind_value(vec_i32(&[n as i32, c as i32, h as i32, w as i32])),
                ),
            ],
        )?;
        let prod = format!("{out_name}_prod");
        self.emit(
            "mul",
            &prod,
            &xs,
            vec![("x", bind_name(&dy)), ("y", bind_name(&x_hat))],
        )?;
        let op = self.simple_op(
            "reduce_sum",
            out_name,
            &gamma_shape,
            vec![
                ("x", bind_name(&prod)),
                ("axes", bind_value(vec_i32(&[0, 2, 3]))),
                ("keep_dims", bind_value(scalar_bool(false))),
            ],
        )?;
        self.push_named(id, out_name.to_string(), op);
        Ok(())
    }

    /// Native GroupNorm backward w.r.t. beta (NCHW). Inputs `[x, dy]` (x unused),
    /// output = beta `[C] = Σ_{n,h,w} dy` — a single channel-aligned reduce_sum.
    #[cfg(feature = "training")]
    pub(crate) fn lower_group_norm_backward_beta(
        &mut self,
        id: NodeId,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let dy = self.val(node.inputs[1]);
        let beta_shape = node.shape.clone();
        let op = self.simple_op(
            "reduce_sum",
            out_name,
            &beta_shape,
            vec![
                ("x", bind_name(&dy)),
                ("axes", bind_value(vec_i32(&[0, 2, 3]))),
                ("keep_dims", bind_value(scalar_bool(false))),
            ],
        )?;
        self.push_named(id, out_name.to_string(), op);
        Ok(())
    }

    /// Inference batch norm with frozen stats. Inputs `[x, gamma, beta,
    /// mean, var]`, channel-last: `(x - mean)·rsqrt(var+eps)·gamma + beta`,
    /// all per-channel `[C]` broadcasting over the trailing axis.
    pub(crate) fn lower_batch_norm(&mut self, id: NodeId, eps: f32, out_name: &str) -> Result<()> {
        let node = self.graph.node(id);
        let shape = node.shape.clone();
        let c = dim_static(&shape, shape.rank() - 1)?;
        let cs = Shape::new(&[c], DType::F32);
        let x = self.val(node.inputs[0]);
        let gamma = self.val(node.inputs[1]);
        let beta = self.val(node.inputs[2]);
        let mean = self.val(node.inputs[3]);
        let var = self.val(node.inputs[4]);

        let veps = format!("{out_name}_veps");
        self.emit(
            "add",
            &veps,
            &cs,
            vec![("x", bind_name(&var)), ("y", bind_value(scalar_f32(eps)))],
        )?;
        let inv = format!("{out_name}_inv");
        self.emit(
            "rsqrt",
            &inv,
            &cs,
            vec![
                ("x", bind_name(&veps)),
                ("epsilon", bind_value(scalar_f32(0.0))),
            ],
        )?;
        let xc = format!("{out_name}_xc");
        self.emit(
            "sub",
            &xc,
            &shape,
            vec![("x", bind_name(&x)), ("y", bind_name(&mean))],
        )?;
        let t = format!("{out_name}_t");
        self.emit(
            "mul",
            &t,
            &shape,
            vec![("x", bind_name(&xc)), ("y", bind_name(&inv))],
        )?;
        let t2 = format!("{out_name}_t2");
        self.emit(
            "mul",
            &t2,
            &shape,
            vec![("x", bind_name(&t)), ("y", bind_name(&gamma))],
        )?;
        self.emit(
            "add",
            out_name,
            &shape,
            vec![("x", bind_name(&t2)), ("y", bind_name(&beta))],
        )?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// GroupNorm over NCHW. Inputs `[x, gamma, beta]`; normalises over
    /// `(C/G)·H·W` within each of `G` groups, then per-channel affine.
    pub(crate) fn lower_group_norm(
        &mut self,
        id: NodeId,
        groups: usize,
        eps: f32,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let shape = node.shape.clone();
        let d = static_dims(&shape)?;
        if d.len() != 4 {
            return Err(CoremlError::Unsupported("group_norm: only NCHW".into()));
        }
        let (n, c, h, w) = (d[0], d[1], d[2], d[3]);
        let inner = (c / groups as i64) * h * w;
        let x = self.val(node.inputs[0]);
        let gamma = self.val(node.inputs[1]);
        let beta = self.val(node.inputs[2]);

        let grp = Shape::new(&[n as usize, groups, inner as usize], DType::F32);
        let red = Shape::new(&[n as usize, groups, 1], DType::F32);
        let xr = format!("{out_name}_xr");
        self.reshape_to(&x, &[n, groups as i64, inner], &grp, &xr)?;
        let normb = self.normalize_chain(out_name, &xr, &grp, &red, &[2], eps)?;
        // back to NCHW
        let nb = format!("{out_name}_nb");
        self.reshape_to(&normb, &[n, c, h, w], &shape, &nb)?;
        self.affine_nchw(out_name, &nb, &shape, &gamma, &beta, c)
    }

    /// LayerNorm over the channel axis of NCHW (per spatial position).
    /// Inputs `[x, gamma, beta]`.
    pub(crate) fn lower_layer_norm2d(
        &mut self,
        id: NodeId,
        eps: f32,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let shape = node.shape.clone();
        let d = static_dims(&shape)?;
        if d.len() != 4 {
            return Err(CoremlError::Unsupported("layer_norm2d: only NCHW".into()));
        }
        let (n, c, h, w) = (d[0], d[1], d[2], d[3]);
        let red = Shape::new(&[n as usize, 1, h as usize, w as usize], DType::F32);
        let x = self.val(node.inputs[0]);
        let gamma = self.val(node.inputs[1]);
        let beta = self.val(node.inputs[2]);
        let norm = self.normalize_chain(out_name, &x, &shape, &red, &[1], eps)?;
        self.affine_nchw(out_name, &norm, &shape, &gamma, &beta, c)
    }

    /// `(in - mean)·rsqrt(var+eps)` reducing over `axes` (keep dims).
    /// Returns the normalised value name.
    pub(crate) fn normalize_chain(
        &mut self,
        out: &str,
        input: &str,
        full: &Shape,
        red: &Shape,
        axes: &[i32],
        eps: f32,
    ) -> Result<String> {
        let mean = format!("{out}_mean");
        self.emit(
            "reduce_mean",
            &mean,
            red,
            vec![
                ("x", bind_name(input)),
                ("axes", bind_value(vec_i32(axes))),
                ("keep_dims", bind_value(scalar_bool(true))),
            ],
        )?;
        let xc = format!("{out}_nc");
        self.emit(
            "sub",
            &xc,
            full,
            vec![("x", bind_name(input)), ("y", bind_name(&mean))],
        )?;
        let sq = format!("{out}_sq");
        self.emit(
            "mul",
            &sq,
            full,
            vec![("x", bind_name(&xc)), ("y", bind_name(&xc))],
        )?;
        let var = format!("{out}_var");
        self.emit(
            "reduce_mean",
            &var,
            red,
            vec![
                ("x", bind_name(&sq)),
                ("axes", bind_value(vec_i32(axes))),
                ("keep_dims", bind_value(scalar_bool(true))),
            ],
        )?;
        let veps = format!("{out}_veps");
        self.emit(
            "add",
            &veps,
            red,
            vec![("x", bind_name(&var)), ("y", bind_value(scalar_f32(eps)))],
        )?;
        let inv = format!("{out}_ninv");
        self.emit(
            "rsqrt",
            &inv,
            red,
            vec![
                ("x", bind_name(&veps)),
                ("epsilon", bind_value(scalar_f32(0.0))),
            ],
        )?;
        let norm = format!("{out}_norm");
        self.emit(
            "mul",
            &norm,
            full,
            vec![("x", bind_name(&xc)), ("y", bind_name(&inv))],
        )?;
        Ok(norm)
    }

    /// Per-channel affine for NCHW: `out = norm·γ[1,C,1,1] + β[1,C,1,1]`.
    pub(crate) fn affine_nchw(
        &mut self,
        out_name: &str,
        norm: &str,
        shape: &Shape,
        gamma: &str,
        beta: &str,
        c: i64,
    ) -> Result<()> {
        let g4 = format!("{out_name}_g4");
        let b4 = format!("{out_name}_b4");
        let c4 = Shape::new(&[1, c as usize, 1, 1], DType::F32);
        self.reshape_to(gamma, &[1, c, 1, 1], &c4, &g4)?;
        self.reshape_to(beta, &[1, c, 1, 1], &c4, &b4)?;
        let scaled = format!("{out_name}_sc");
        self.emit(
            "mul",
            &scaled,
            shape,
            vec![("x", bind_name(norm)), ("y", bind_name(&g4))],
        )?;
        self.emit(
            "add",
            out_name,
            shape,
            vec![("x", bind_name(&scaled)), ("y", bind_name(&b4))],
        )?;
        // Caller registers the node mapping.
        Ok(())
    }

    /// DiT adaLN-Zero forward: `norm(x)·(1+scale)+shift` with broadcast scale/shift.
    /// Affine-free norm via [`normalize_chain`] (LN) or RMS `x·rsqrt(mean(x²)+eps)`.
    pub(crate) fn lower_ada_layer_norm(
        &mut self,
        id: NodeId,
        norm: AdaNormKind,
        eps: f32,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let x = self.val(node.inputs[0]);
        let scale = self.val(node.inputs[1]);
        let shift = self.val(node.inputs[2]);
        let x_shape = node.shape.clone();
        let rank = x_shape.rank();
        let norm_axis = rank - 1;
        let axes: Vec<i32> = (norm_axis as i32..rank as i32).collect();
        let red = reduced_shape(&x_shape, norm_axis);

        let n = match norm {
            AdaNormKind::LayerNorm => {
                self.normalize_chain(out_name, &x, &x_shape, &red, &axes, eps)?
            }
            AdaNormKind::RmsNorm => {
                let sq = format!("{out_name}_nsq");
                self.emit(
                    "mul",
                    &sq,
                    &x_shape,
                    vec![("x", bind_name(&x)), ("y", bind_name(&x))],
                )?;
                let ms = format!("{out_name}_nms");
                self.emit(
                    "reduce_mean",
                    &ms,
                    &red,
                    vec![
                        ("x", bind_name(&sq)),
                        ("axes", bind_value(vec_i32(&axes))),
                        ("keep_dims", bind_value(scalar_bool(true))),
                    ],
                )?;
                let mse = format!("{out_name}_nmse");
                self.emit(
                    "add",
                    &mse,
                    &red,
                    vec![("x", bind_name(&ms)), ("y", bind_value(scalar_f32(eps)))],
                )?;
                let inv = format!("{out_name}_ninv");
                self.emit(
                    "rsqrt",
                    &inv,
                    &red,
                    vec![
                        ("x", bind_name(&mse)),
                        ("epsilon", bind_value(scalar_f32(0.0))),
                    ],
                )?;
                let nn = format!("{out_name}_n");
                self.emit(
                    "mul",
                    &nn,
                    &x_shape,
                    vec![("x", bind_name(&x)), ("y", bind_name(&inv))],
                )?;
                nn
            }
        };

        let n_el = x_shape
            .num_elements()
            .ok_or_else(|| CoremlError::Unsupported("ada forward: dynamic x numel".into()))?;
        let ones = format!("{out_name}_ones");
        self.operations.push(make_const(
            &mut self.blob,
            &ones,
            &x_shape,
            &vec![1.0f32; n_el],
        )?);
        let one_plus = format!("{out_name}_1p");
        self.emit(
            "add",
            &one_plus,
            &x_shape,
            vec![("x", bind_name(&ones)), ("y", bind_name(&scale))],
        )?;
        let scaled = format!("{out_name}_sc");
        self.emit(
            "mul",
            &scaled,
            &x_shape,
            vec![("x", bind_name(&n)), ("y", bind_name(&one_plus))],
        )?;
        self.emit(
            "add",
            out_name,
            &x_shape,
            vec![("x", bind_name(&scaled)), ("y", bind_name(&shift))],
        )?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// DiT gated residual forward: `x + gate·y` with broadcast gate.
    pub(crate) fn lower_gated_residual(&mut self, id: NodeId, out_name: &str) -> Result<()> {
        let node = self.graph.node(id);
        let x = self.val(node.inputs[0]);
        let y = self.val(node.inputs[1]);
        let gate = self.val(node.inputs[2]);
        let shape = node.shape.clone();
        let gated = format!("{out_name}_gy");
        self.emit(
            "mul",
            &gated,
            &shape,
            vec![("x", bind_name(&gate)), ("y", bind_name(&y))],
        )?;
        self.emit(
            "add",
            out_name,
            &shape,
            vec![("x", bind_name(&x)), ("y", bind_name(&gated))],
        )?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// Packed DiT adaLN reverse → 1-D `[dx ∥ dscale ∥ dshift]`.
    /// Mirrors `compose_ada_layer_norm_backward` with implicit MIL broadcast
    /// (no Expand-with-ones) and the native LN/RMS input-gradient math.
    #[cfg(feature = "training")]
    pub(crate) fn lower_ada_layer_norm_backward(
        &mut self,
        id: NodeId,
        norm: AdaNormKind,
        eps: f32,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let x = self.val(node.inputs[0]);
        let scale = self.val(node.inputs[1]);
        // inputs[2] = shift — additive in the forward, absent from dx/dscale.
        let dy = self.val(node.inputs[3]);
        let x_shape = self.graph.shape(node.inputs[0]).clone();
        let scale_shape = self.graph.shape(node.inputs[1]).clone();
        let out_shape = node.shape.clone();
        let rank = x_shape.rank();
        let norm_axis = rank - 1;
        let axes: Vec<i32> = (norm_axis as i32..rank as i32).collect();
        let red = reduced_shape(&x_shape, norm_axis);

        // Affine-free forward norm `n` (γ=1, β=0).
        let n = match norm {
            AdaNormKind::LayerNorm => {
                self.normalize_chain(out_name, &x, &x_shape, &red, &axes, eps)?
            }
            AdaNormKind::RmsNorm => {
                let sq = format!("{out_name}_nsq");
                self.emit(
                    "mul",
                    &sq,
                    &x_shape,
                    vec![("x", bind_name(&x)), ("y", bind_name(&x))],
                )?;
                let ms = format!("{out_name}_nms");
                self.emit(
                    "reduce_mean",
                    &ms,
                    &red,
                    vec![
                        ("x", bind_name(&sq)),
                        ("axes", bind_value(vec_i32(&axes))),
                        ("keep_dims", bind_value(scalar_bool(true))),
                    ],
                )?;
                let mse = format!("{out_name}_nmse");
                self.emit(
                    "add",
                    &mse,
                    &red,
                    vec![("x", bind_name(&ms)), ("y", bind_value(scalar_f32(eps)))],
                )?;
                let inv = format!("{out_name}_ninv");
                self.emit(
                    "rsqrt",
                    &inv,
                    &red,
                    vec![
                        ("x", bind_name(&mse)),
                        ("epsilon", bind_value(scalar_f32(0.0))),
                    ],
                )?;
                let nn = format!("{out_name}_n");
                self.emit(
                    "mul",
                    &nn,
                    &x_shape,
                    vec![("x", bind_name(&x)), ("y", bind_name(&inv))],
                )?;
                nn
            }
        };

        // one_plus = ones(x) + scale (broadcast); dn = dy · one_plus
        let n_el = x_shape
            .num_elements()
            .ok_or_else(|| CoremlError::Unsupported("ada reverse: dynamic x numel".into()))?;
        let ones = format!("{out_name}_ones");
        self.operations.push(make_const(
            &mut self.blob,
            &ones,
            &x_shape,
            &vec![1.0f32; n_el],
        )?);
        let one_plus = format!("{out_name}_1p");
        self.emit(
            "add",
            &one_plus,
            &x_shape,
            vec![("x", bind_name(&ones)), ("y", bind_name(&scale))],
        )?;
        let dn = format!("{out_name}_dn");
        self.emit(
            "mul",
            &dn,
            &x_shape,
            vec![("x", bind_name(&dy)), ("y", bind_name(&one_plus))],
        )?;

        // dx via LN/RMS input reverse with γ=1 (dn plays the role of dy·γ).
        let dx = match norm {
            AdaNormKind::LayerNorm => {
                self.mil_layer_norm_dx(out_name, &x, &dn, &x_shape, &red, &axes, eps)?
            }
            AdaNormKind::RmsNorm => {
                self.mil_rms_norm_dx(out_name, &x, &dn, &x_shape, &red, &axes, eps)?
            }
        };

        // dscale = unbroadcast(dy · n); dshift = unbroadcast(dy)
        let dsf = format!("{out_name}_dsf");
        self.emit(
            "mul",
            &dsf,
            &x_shape,
            vec![("x", bind_name(&dy)), ("y", bind_name(&n))],
        )?;
        let dscale = format!("{out_name}_ds");
        self.mil_unbroadcast(&dsf, &x_shape, &scale_shape, &dscale)?;
        let dshift = format!("{out_name}_dt");
        self.mil_unbroadcast(&dy, &x_shape, &scale_shape, &dshift)?;

        self.mil_pack_flat_grads(
            id,
            out_name,
            &out_shape,
            &[
                (&dx, &x_shape),
                (&dscale, &scale_shape),
                (&dshift, &scale_shape),
            ],
        )
    }

    /// Packed DiT gated residual reverse → 1-D `[dx ∥ dy ∥ dgate]`.
    /// Mirrors `compose_gated_residual_backward`.
    #[cfg(feature = "training")]
    pub(crate) fn lower_gated_residual_backward(
        &mut self,
        id: NodeId,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        // inputs[0] = x — dx = dy (identity).
        let y = self.val(node.inputs[1]);
        let gate = self.val(node.inputs[2]);
        let dy = self.val(node.inputs[3]);
        let x_shape = self.graph.shape(node.inputs[0]).clone();
        let gate_shape = self.graph.shape(node.inputs[2]).clone();
        let out_shape = node.shape.clone();

        let dx = dy.clone();
        let dy_out = format!("{out_name}_dy");
        self.emit(
            "mul",
            &dy_out,
            &x_shape,
            vec![("x", bind_name(&dy)), ("y", bind_name(&gate))],
        )?;
        let dgf = format!("{out_name}_dgf");
        self.emit(
            "mul",
            &dgf,
            &x_shape,
            vec![("x", bind_name(&dy)), ("y", bind_name(&y))],
        )?;
        let dgate = format!("{out_name}_dg");
        self.mil_unbroadcast(&dgf, &x_shape, &gate_shape, &dgate)?;

        self.mil_pack_flat_grads(
            id,
            out_name,
            &out_shape,
            &[(&dx, &x_shape), (&dy_out, &x_shape), (&dgate, &gate_shape)],
        )
    }

    /// LayerNorm input reverse with `sy = dy` already (γ=1). Returns `dx` name.
    #[cfg(feature = "training")]
    fn mil_layer_norm_dx(
        &mut self,
        prefix: &str,
        x: &str,
        sy: &str,
        full: &Shape,
        red: &Shape,
        axes: &[i32],
        eps: f32,
    ) -> Result<String> {
        let red_axes = || bind_value(vec_i32(axes));
        let keep = || bind_value(scalar_bool(true));
        let mean = format!("{prefix}_ln_mean");
        self.emit(
            "reduce_mean",
            &mean,
            red,
            vec![
                ("x", bind_name(x)),
                ("axes", red_axes()),
                ("keep_dims", keep()),
            ],
        )?;
        let xc = format!("{prefix}_ln_xc");
        self.emit(
            "sub",
            &xc,
            full,
            vec![("x", bind_name(x)), ("y", bind_name(&mean))],
        )?;
        let xc2 = format!("{prefix}_ln_xc2");
        self.emit(
            "mul",
            &xc2,
            full,
            vec![("x", bind_name(&xc)), ("y", bind_name(&xc))],
        )?;
        let var = format!("{prefix}_ln_var");
        self.emit(
            "reduce_mean",
            &var,
            red,
            vec![
                ("x", bind_name(&xc2)),
                ("axes", red_axes()),
                ("keep_dims", keep()),
            ],
        )?;
        let ve = format!("{prefix}_ln_ve");
        self.emit(
            "add",
            &ve,
            red,
            vec![("x", bind_name(&var)), ("y", bind_value(scalar_f32(eps)))],
        )?;
        let inv = format!("{prefix}_ln_inv");
        self.emit(
            "rsqrt",
            &inv,
            red,
            vec![
                ("x", bind_name(&ve)),
                ("epsilon", bind_value(scalar_f32(0.0))),
            ],
        )?;
        let x_hat = format!("{prefix}_ln_xh");
        self.emit(
            "mul",
            &x_hat,
            full,
            vec![("x", bind_name(&xc)), ("y", bind_name(&inv))],
        )?;
        let m_sy = format!("{prefix}_ln_msy");
        self.emit(
            "reduce_mean",
            &m_sy,
            red,
            vec![
                ("x", bind_name(sy)),
                ("axes", red_axes()),
                ("keep_dims", keep()),
            ],
        )?;
        let sy_xh = format!("{prefix}_ln_syxh");
        self.emit(
            "mul",
            &sy_xh,
            full,
            vec![("x", bind_name(sy)), ("y", bind_name(&x_hat))],
        )?;
        let m_sxh = format!("{prefix}_ln_msxh");
        self.emit(
            "reduce_mean",
            &m_sxh,
            red,
            vec![
                ("x", bind_name(&sy_xh)),
                ("axes", red_axes()),
                ("keep_dims", keep()),
            ],
        )?;
        let t1 = format!("{prefix}_ln_t1");
        self.emit(
            "sub",
            &t1,
            full,
            vec![("x", bind_name(sy)), ("y", bind_name(&m_sy))],
        )?;
        let t2 = format!("{prefix}_ln_t2");
        self.emit(
            "mul",
            &t2,
            full,
            vec![("x", bind_name(&x_hat)), ("y", bind_name(&m_sxh))],
        )?;
        let t3 = format!("{prefix}_ln_t3");
        self.emit(
            "sub",
            &t3,
            full,
            vec![("x", bind_name(&t1)), ("y", bind_name(&t2))],
        )?;
        let dx = format!("{prefix}_dx");
        self.emit(
            "mul",
            &dx,
            full,
            vec![("x", bind_name(&inv)), ("y", bind_name(&t3))],
        )?;
        Ok(dx)
    }

    /// RMSNorm input reverse with `dy_g = dy` already (γ=1). Returns `dx` name.
    #[cfg(feature = "training")]
    fn mil_rms_norm_dx(
        &mut self,
        prefix: &str,
        x: &str,
        dyg: &str,
        full: &Shape,
        red: &Shape,
        axes: &[i32],
        eps: f32,
    ) -> Result<String> {
        let red_axes = || bind_value(vec_i32(axes));
        let keep = || bind_value(scalar_bool(true));
        let x2 = format!("{prefix}_rms_x2");
        self.emit(
            "mul",
            &x2,
            full,
            vec![("x", bind_name(x)), ("y", bind_name(x))],
        )?;
        let mx2 = format!("{prefix}_rms_mx2");
        self.emit(
            "reduce_mean",
            &mx2,
            red,
            vec![
                ("x", bind_name(&x2)),
                ("axes", red_axes()),
                ("keep_dims", keep()),
            ],
        )?;
        let ve = format!("{prefix}_rms_ve");
        self.emit(
            "add",
            &ve,
            red,
            vec![("x", bind_name(&mx2)), ("y", bind_value(scalar_f32(eps)))],
        )?;
        let inv = format!("{prefix}_rms_inv");
        self.emit(
            "rsqrt",
            &inv,
            red,
            vec![
                ("x", bind_name(&ve)),
                ("epsilon", bind_value(scalar_f32(0.0))),
            ],
        )?;
        let inv2 = format!("{prefix}_rms_inv2");
        self.emit(
            "mul",
            &inv2,
            red,
            vec![("x", bind_name(&inv)), ("y", bind_name(&inv))],
        )?;
        let xdyg = format!("{prefix}_rms_xdyg");
        self.emit(
            "mul",
            &xdyg,
            full,
            vec![("x", bind_name(x)), ("y", bind_name(dyg))],
        )?;
        let dot = format!("{prefix}_rms_dot");
        self.emit(
            "reduce_mean",
            &dot,
            red,
            vec![
                ("x", bind_name(&xdyg)),
                ("axes", red_axes()),
                ("keep_dims", keep()),
            ],
        )?;
        let xdot = format!("{prefix}_rms_xdot");
        self.emit(
            "mul",
            &xdot,
            full,
            vec![("x", bind_name(x)), ("y", bind_name(&dot))],
        )?;
        let term2 = format!("{prefix}_rms_t2");
        self.emit(
            "mul",
            &term2,
            full,
            vec![("x", bind_name(&xdot)), ("y", bind_name(&inv2))],
        )?;
        let diff = format!("{prefix}_rms_diff");
        self.emit(
            "sub",
            &diff,
            full,
            vec![("x", bind_name(dyg)), ("y", bind_name(&term2))],
        )?;
        let dx = format!("{prefix}_dx");
        self.emit(
            "mul",
            &dx,
            full,
            vec![("x", bind_name(&diff)), ("y", bind_name(&inv))],
        )?;
        Ok(dx)
    }

    /// Sum broadcast axes of `src` down to `tgt` (numpy unbroadcast).
    #[cfg(feature = "training")]
    fn mil_unbroadcast(
        &mut self,
        src: &str,
        src_shape: &Shape,
        tgt: &Shape,
        out: &str,
    ) -> Result<()> {
        if src_shape == tgt {
            let dims: Vec<i64> = (0..tgt.rank())
                .map(|i| tgt.dim(i).unwrap_static() as i64)
                .collect();
            return self.reshape_to(src, &dims, tgt, out);
        }
        let g_rank = src_shape.rank();
        let t_rank = tgt.rank();
        let extra = g_rank.saturating_sub(t_rank);
        let mut axes: Vec<usize> = (0..extra).collect();
        for i in 0..t_rank {
            let g_dim = src_shape.dim(extra + i);
            let t_dim = tgt.dim(i);
            if matches!(t_dim, Dim::Static(1)) && !matches!(g_dim, Dim::Static(1)) {
                axes.push(extra + i);
            }
        }
        let mut current = src.to_string();
        let mut running_dims: Vec<Dim> = (0..g_rank).map(|i| src_shape.dim(i)).collect();
        for (step, &ax) in axes.iter().enumerate() {
            running_dims[ax] = Dim::Static(1);
            let step_shape = Shape::from_dims(&running_dims, tgt.dtype());
            let name = format!("{out}_ub{step}");
            self.emit(
                "reduce_sum",
                &name,
                &step_shape,
                vec![
                    ("x", bind_name(&current)),
                    ("axes", bind_value(vec_i32(&[ax as i32]))),
                    ("keep_dims", bind_value(scalar_bool(true))),
                ],
            )?;
            current = name;
        }
        let dims: Vec<i64> = (0..t_rank)
            .map(|i| match tgt.dim(i) {
                Dim::Static(n) => n as i64,
                Dim::Dynamic(_) => -1,
            })
            .collect();
        self.reshape_to(&current, &dims, tgt, out)
    }

    /// Flatten each grad and concat on axis 0 into the packed 1-D output.
    #[cfg(feature = "training")]
    fn mil_pack_flat_grads(
        &mut self,
        id: NodeId,
        out_name: &str,
        out_shape: &Shape,
        grads: &[(&str, &Shape)],
    ) -> Result<()> {
        let mut flats = Vec::with_capacity(grads.len());
        for (i, &(name, shape)) in grads.iter().enumerate() {
            let n = shape.num_elements().ok_or_else(|| {
                CoremlError::Unsupported("dit packed reverse: dynamic grad numel".into())
            })?;
            let flat_shape = Shape::new(&[n], shape.dtype());
            let flat = format!("{out_name}_f{i}");
            self.reshape_to(name, &[n as i64], &flat_shape, &flat)?;
            flats.push(flat);
        }
        let op = self.simple_op(
            "concat",
            out_name,
            out_shape,
            vec![
                ("values", bind_names(&flats)),
                ("axis", bind_value(scalar_i32(0))),
                ("interleave", bind_value(scalar_bool(false))),
            ],
        )?;
        self.push_named(id, out_name.to_string(), op);
        Ok(())
    }
}
