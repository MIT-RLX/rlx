// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
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
use rlx_ir::op::{Activation, CmpOp, MaskKind, ReduceOp};
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
        let node = self.graph.node(id);
        let x = self.val(node.inputs[0]);
        let rank = node.shape.rank() as i32;
        let norm_axis = if axis < 0 { axis + rank } else { axis };
        // MIL layer_norm normalises over `axes`; RLX LayerNorm is over the
        // trailing dims from `axis` onward.
        let axes: Vec<i32> = (norm_axis..rank).collect();
        let mut binds = vec![
            ("x", bind_name(&x)),
            ("axes", bind_value(vec_i32(&axes))),
            ("epsilon", bind_value(scalar_f32(eps))),
        ];
        if node.inputs.len() > 1 {
            let g = self.val(node.inputs[1]);
            binds.push(("gamma", bind_name(&g)));
        }
        if node.inputs.len() > 2 {
            let b = self.val(node.inputs[2]);
            binds.push(("beta", bind_name(&b)));
        }
        let op = self.simple_op("layer_norm", out_name, &node.shape, binds)?;
        self.push_named(id, out_name.to_string(), op);
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
}
