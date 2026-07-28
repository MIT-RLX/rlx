// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! HLO composition for training backward / QAT ops (LayerNorm/RmsNorm/
//! GroupNorm bwd, Rope/Cumsum/Gather bwd, FakeQuantize*, Conv2d bwd,
//! MaxPool2dBackward via select-and-scatter).

use crate::hlo::{ConvDimNumbers, ScatterDimNumbers, Shape, Window, WindowDim, prim, prim_of};
use rlx_ir::op::SteKind;
use rlx_ir::{DType, NodeId};

use super::*;

impl<'a> LowerCtx<'a> {
    fn resolve_norm_axis(&self, dims: &[i64], axis: i32) -> i64 {
        let rank = dims.len() as i32;
        if axis < 0 {
            (rank + axis) as i64
        } else {
            axis as i64
        }
    }

    /// LayerNorm input reverse: `dx = inv_std · (sy − mean(sy) − x̂ · mean(sy·x̂))`
    /// with `sy = dy · γ`. Axis = −1 only (IR contract).
    pub(crate) fn lower_layer_norm_backward_input(
        &mut self,
        x_id: NodeId,
        gamma_id: NodeId,
        dy_id: NodeId,
        axis: i32,
        eps: f32,
        out: Shape,
    ) -> i64 {
        let x = self.hlo(x_id);
        let gamma = self.hlo(gamma_id);
        let dy = self.hlo(dy_id);
        let x_dims = self.ir_shape_dims(x_id);
        let x_dt = self.dtype(x_id);
        let prim_ty = prim_of(x_dt);
        let ax = self.resolve_norm_axis(&x_dims, axis);
        assert_eq!(
            ax,
            (x_dims.len() - 1) as i64,
            "rlx-tpu LayerNormBackwardInput: axis must be −1"
        );
        let g_dims = self.ir_shape_dims(gamma_id);
        let g_b = self.broadcast_param_to_axis(gamma, &g_dims, ax, &x_dims, prim_ty);
        let sy = self.entry.binary("multiply", dy, g_b, out.clone());
        // Reuse γ=1 helper with sy as the upstream.
        self.lower_layernorm_dx_gamma1_hlo(x, sy, eps, out, x_dt)
    }

    /// `dγ = Σ_{leading} dy · x̂`.
    pub(crate) fn lower_layer_norm_backward_gamma(
        &mut self,
        x_id: NodeId,
        dy_id: NodeId,
        axis: i32,
        eps: f32,
        out: Shape,
    ) -> i64 {
        let x = self.hlo(x_id);
        let dy = self.hlo(dy_id);
        let x_dims = self.ir_shape_dims(x_id);
        let x_dt = self.dtype(x_id);
        let prim_ty = prim_of(x_dt);
        let ax = self.resolve_norm_axis(&x_dims, axis);
        assert_eq!(
            ax,
            (x_dims.len() - 1) as i64,
            "rlx-tpu LayerNormBackwardGamma: axis must be −1"
        );
        let x_shape = Shape::array(prim_ty, &x_dims);
        let xhat = self.lower_xhat_layernorm(x, eps, x_shape.clone(), x_dt, ax);
        let prod = self.entry.binary("multiply", dy, xhat, x_shape);
        let out_dims = out.dimensions.clone();
        self.sum_unbroadcast_hlo(prod, &x_dims, &out_dims, x_dt)
    }

    fn lower_xhat_layernorm(&mut self, x: i64, eps: f32, out: Shape, x_dt: DType, ax: i64) -> i64 {
        let x_dims = out.dimensions.clone();
        let prim_ty = prim_of(x_dt);
        let mut reduced = x_dims.clone();
        reduced[ax as usize] = 1;
        let kept_shape = Shape::array(prim_ty, &reduced);
        let summed_dims: Vec<i64> = x_dims
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if i == ax as usize { None } else { Some(d) })
            .collect();
        let summed_shape = Shape::array(prim_ty, &summed_dims);
        let n_elems = x_dims[ax as usize] as f32;
        let n_c = self.const_in_dtype(prim_ty, n_elems);
        let n_b = self.entry.broadcast(n_c, &[], summed_shape.clone());

        let pre_sum = self.reduce_one(x, ax, "add", 0.0, x_dt, summed_dims.clone());
        let mean = self
            .entry
            .binary("divide", pre_sum, n_b, summed_shape.clone());
        let mean_kept = self.entry.reshape(mean, kept_shape.clone());
        let mean_b = self.broadcast_align(mean_kept, &reduced, out.clone());
        let centered = self.entry.binary("subtract", x, mean_b, out.clone());
        let sq = self
            .entry
            .binary("multiply", centered, centered, out.clone());
        let sq_sum = self.reduce_one(sq, ax, "add", 0.0, x_dt, summed_dims);
        let var = self.entry.binary("divide", sq_sum, n_b, summed_shape);
        let var_kept = self.entry.reshape(var, kept_shape);
        let eps_c = self.const_in_dtype(prim_ty, eps);
        let eps_b = self
            .entry
            .broadcast(eps_c, &[], Shape::array(prim_ty, &reduced));
        let var_eps = self
            .entry
            .binary("add", var_kept, eps_b, Shape::array(prim_ty, &reduced));
        let inv_std = self
            .entry
            .unary("rsqrt", var_eps, Shape::array(prim_ty, &reduced));
        let inv_std_b = self.broadcast_align(inv_std, &reduced, out.clone());
        self.entry.binary("multiply", centered, inv_std_b, out)
    }

    /// RMSNorm input reverse with γ (β unused). Axis = −1 only.
    pub(crate) fn lower_rms_norm_backward_input(
        &mut self,
        x_id: NodeId,
        gamma_id: NodeId,
        _beta_id: NodeId,
        dy_id: NodeId,
        axis: i32,
        eps: f32,
        out: Shape,
    ) -> i64 {
        let x = self.hlo(x_id);
        let gamma = self.hlo(gamma_id);
        let dy = self.hlo(dy_id);
        let x_dims = self.ir_shape_dims(x_id);
        let x_dt = self.dtype(x_id);
        let prim_ty = prim_of(x_dt);
        let ax = self.resolve_norm_axis(&x_dims, axis);
        assert_eq!(
            ax,
            (x_dims.len() - 1) as i64,
            "rlx-tpu RmsNormBackwardInput: axis must be −1"
        );
        let g_dims = self.ir_shape_dims(gamma_id);
        let g_b = self.broadcast_param_to_axis(gamma, &g_dims, ax, &x_dims, prim_ty);
        let sy = self.entry.binary("multiply", dy, g_b, out.clone());
        self.lower_rms_norm_dx_gamma1_hlo(x, sy, eps, out, x_dt)
    }

    /// `dγ = Σ_{leading} dy · x̂` with `x̂ = x / √(mean(x²) + ε)`. Axis = −1 only.
    pub(crate) fn lower_rms_norm_backward_gamma(
        &mut self,
        x_id: NodeId,
        _gamma_id: NodeId,
        _beta_id: NodeId,
        dy_id: NodeId,
        axis: i32,
        eps: f32,
        out: Shape,
    ) -> i64 {
        let x = self.hlo(x_id);
        let dy = self.hlo(dy_id);
        let x_dims = self.ir_shape_dims(x_id);
        let x_dt = self.dtype(x_id);
        let prim_ty = prim_of(x_dt);
        let ax = self.resolve_norm_axis(&x_dims, axis);
        assert_eq!(
            ax,
            (x_dims.len() - 1) as i64,
            "rlx-tpu RmsNormBackwardGamma: axis must be −1"
        );
        let x_shape = Shape::array(prim_ty, &x_dims);
        let mut reduced = x_dims.clone();
        reduced[ax as usize] = 1;
        let kept_shape = Shape::array(prim_ty, &reduced);
        let summed_dims: Vec<i64> = x_dims
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if i == ax as usize { None } else { Some(d) })
            .collect();
        let summed_shape = Shape::array(prim_ty, &summed_dims);
        let n_elems = x_dims[ax as usize] as f32;
        let n_c = self.const_in_dtype(prim_ty, n_elems);
        let n_b = self.entry.broadcast(n_c, &[], summed_shape.clone());

        let sq = self.entry.binary("multiply", x, x, x_shape.clone());
        let sq_sum = self.reduce_one(sq, ax, "add", 0.0, x_dt, summed_dims);
        let sq_mean = self
            .entry
            .binary("divide", sq_sum, n_b, summed_shape.clone());
        let eps_c = self.const_in_dtype(prim_ty, eps);
        let eps_b = self.entry.broadcast(eps_c, &[], summed_shape.clone());
        let var_eps = self
            .entry
            .binary("add", sq_mean, eps_b, summed_shape.clone());
        let inv = self.entry.unary("rsqrt", var_eps, summed_shape);
        let inv_kept = self.entry.reshape(inv, kept_shape);
        let inv_b = self.broadcast_align(inv_kept, &reduced, x_shape.clone());
        let x_hat = self.entry.binary("multiply", x, inv_b, x_shape.clone());
        let prod = self.entry.binary("multiply", dy, x_hat, x_shape);
        let out_dims = out.dimensions.clone();
        self.sum_unbroadcast_hlo(prod, &x_dims, &out_dims, x_dt)
    }

    /// `dβ = Σ_{leading} dy`. Axis = −1 only.
    pub(crate) fn lower_rms_norm_backward_beta(
        &mut self,
        _x_id: NodeId,
        _gamma_id: NodeId,
        _beta_id: NodeId,
        dy_id: NodeId,
        axis: i32,
        out: Shape,
    ) -> i64 {
        let dy = self.hlo(dy_id);
        let dy_dims = self.ir_shape_dims(dy_id);
        let dy_dt = self.dtype(dy_id);
        let ax = self.resolve_norm_axis(&dy_dims, axis);
        assert_eq!(
            ax,
            (dy_dims.len() - 1) as i64,
            "rlx-tpu RmsNormBackwardBeta: axis must be −1"
        );
        let out_dims = out.dimensions.clone();
        self.sum_unbroadcast_hlo(dy, &dy_dims, &out_dims, dy_dt)
    }

    /// GroupNorm NCHW input reverse via reshape → LN-style → reshape.
    pub(crate) fn lower_group_norm_backward_input(
        &mut self,
        x_id: NodeId,
        gamma_id: NodeId,
        _beta_id: NodeId,
        dy_id: NodeId,
        num_groups: usize,
        eps: f32,
        out: Shape,
    ) -> i64 {
        let x = self.hlo(x_id);
        let gamma = self.hlo(gamma_id);
        let dy = self.hlo(dy_id);
        let x_dims = self.ir_shape_dims(x_id);
        assert_eq!(x_dims.len(), 4, "rlx-tpu GroupNormBackwardInput: NCHW only");
        let (n, c, h, w) = (x_dims[0], x_dims[1], x_dims[2], x_dims[3]);
        let g = num_groups as i64;
        assert!(
            c % g == 0,
            "rlx-tpu GroupNormBackwardInput: C not divisible by groups"
        );
        let cpg = c / g;
        let inner = cpg * h * w;
        let x_dt = self.dtype(x_id);
        let prim_ty = prim_of(x_dt);

        let shape5 = Shape::array(prim_ty, &[n, g, cpg, h, w]);
        let shape3 = Shape::array(prim_ty, &[n, g, inner]);
        let x5 = self.entry.reshape(x, shape5.clone());
        let dy5 = self.entry.reshape(dy, shape5);
        let x3 = self.entry.reshape(x5, shape3.clone());
        let dy3 = self.entry.reshape(dy5, shape3.clone());

        // γ [C] → [1,G,Cpg,1] → broadcast → [N,G,inner]
        let g_dims = self.ir_shape_dims(gamma_id);
        let gamma_1d = if g_dims.as_slice() == [c] {
            gamma
        } else {
            self.entry.reshape(gamma, Shape::array(prim_ty, &[c]))
        };
        let gamma_g = self
            .entry
            .reshape(gamma_1d, Shape::array(prim_ty, &[1, g, cpg, 1]));
        let gamma_b = self.broadcast_align(
            gamma_g,
            &[1, g, cpg, 1],
            Shape::array(prim_ty, &[n, g, cpg, h * w]),
        );
        let gamma_flat = self
            .entry
            .reshape(gamma_b, Shape::array(prim_ty, &[n, g, inner]));
        let sy = self
            .entry
            .binary("multiply", dy3, gamma_flat, shape3.clone());

        let dx3 = self.lower_layernorm_dx_gamma1_hlo(x3, sy, eps, shape3, x_dt);
        let dx5 = self
            .entry
            .reshape(dx3, Shape::array(prim_ty, &[n, g, cpg, h, w]));
        self.entry.reshape(dx5, out)
    }

    /// GroupNorm `dγ`: reshape → LN-style `x̂`, then `Σ_{N,H,W} dy · x̂` → `[C]`.
    pub(crate) fn lower_group_norm_backward_gamma(
        &mut self,
        x_id: NodeId,
        dy_id: NodeId,
        num_groups: usize,
        eps: f32,
        out: Shape,
    ) -> i64 {
        let x = self.hlo(x_id);
        let dy = self.hlo(dy_id);
        let x_dims = self.ir_shape_dims(x_id);
        assert_eq!(x_dims.len(), 4, "rlx-tpu GroupNormBackwardGamma: NCHW only");
        let (n, c, h, w) = (x_dims[0], x_dims[1], x_dims[2], x_dims[3]);
        let g = num_groups as i64;
        let cpg = c / g;
        let inner = cpg * h * w;
        let x_dt = self.dtype(x_id);
        let prim_ty = prim_of(x_dt);

        let shape5 = Shape::array(prim_ty, &[n, g, cpg, h, w]);
        let shape3 = Shape::array(prim_ty, &[n, g, inner]);
        let x5 = self.entry.reshape(x, shape5);
        let x3 = self.entry.reshape(x5, shape3.clone());
        let xhat3 = self.lower_xhat_layernorm(x3, eps, shape3, x_dt, 2);
        let xhat = self
            .entry
            .reshape(xhat3, Shape::array(prim_ty, &[n, c, h, w]));
        let prod = self
            .entry
            .binary("multiply", dy, xhat, Shape::array(prim_ty, &[n, c, h, w]));
        // Sum over N, H, W → [C]
        let out_dims = out.dimensions.clone();
        self.sum_unbroadcast_hlo(prod, &[n, c, h, w], &out_dims, x_dt)
    }

    /// GroupNorm `dβ = Σ_{N,H,W} dy` → `[C]` (NCHW).
    pub(crate) fn lower_group_norm_backward_beta(&mut self, dy_id: NodeId, out: Shape) -> i64 {
        let dy = self.hlo(dy_id);
        let dy_dims = self.ir_shape_dims(dy_id);
        assert_eq!(dy_dims.len(), 4, "rlx-tpu GroupNormBackwardBeta: NCHW only");
        let dy_dt = self.dtype(dy_id);
        let out_dims = out.dimensions.clone();
        self.sum_unbroadcast_hlo(dy, &dy_dims, &out_dims, dy_dt)
    }

    /// Cumsum VJP: inclusive/exclusive suffix sum via `total − cumsum`.
    pub(crate) fn lower_cumsum_backward(
        &mut self,
        dy_id: NodeId,
        axis: i32,
        exclusive: bool,
        out: Shape,
    ) -> i64 {
        let dy_dims = self.ir_shape_dims(dy_id);
        let dy_dt = self.dtype(dy_id);
        let prim_ty = prim_of(dy_dt);
        let rank = dy_dims.len() as i32;
        let ax = if axis < 0 {
            (rank + axis) as i64
        } else {
            axis as i64
        };
        let total = {
            let dy = self.hlo(dy_id);
            let red_dims: Vec<i64> = dy_dims
                .iter()
                .enumerate()
                .filter_map(|(i, &d)| if i == ax as usize { None } else { Some(d) })
                .collect();
            let reduced = self.reduce_one(dy, ax, "add", 0.0, dy_dt, red_dims);
            let mut kept = dy_dims.clone();
            kept[ax as usize] = 1;
            let kept_h = self.entry.reshape(reduced, Shape::array(prim_ty, &kept));
            self.broadcast_align(kept_h, &kept, out.clone())
        };
        // exclusive bwd → inclusive cumsum; inclusive bwd → exclusive cumsum
        let pref = self.lower_cumsum(dy_id, axis, !exclusive, out.clone());
        self.entry.binary("subtract", total, pref, out)
    }

    /// Gather VJP: scatter-add `dy` into a zero table at `indices`.
    pub(crate) fn lower_gather_backward(
        &mut self,
        dy_id: NodeId,
        indices_id: NodeId,
        axis: i32,
        out: Shape,
    ) -> i64 {
        let dy = self.hlo(dy_id);
        let idx = self.hlo(indices_id);
        let idx_dt = self.dtype(indices_id);
        let idx_dims = self.ir_shape_dims(indices_id);
        let idx_s32 = if matches!(idx_dt, DType::I32 | DType::I64 | DType::U32) {
            idx
        } else {
            self.entry.convert(idx, Shape::array(prim::S32, &idx_dims))
        };
        let table_dims = out.dimensions.clone();
        let rank = table_dims.len() as i32;
        let ax = if axis < 0 {
            (rank + axis) as usize
        } else {
            axis as usize
        };
        let prim_ty = out.element_type;
        let zero = self.const_in_dtype(prim_ty, 0.0);
        let dest = self.entry.broadcast(zero, &[], out.clone());
        let combiner = self.reducer("add", prim_ty);

        let idx_rank = idx_dims.len() as i64;
        let n_offset = (table_dims.len() - 1) as i64;
        let update_window_dims: Vec<i64> = (idx_rank..idx_rank + n_offset).collect();
        let dn = ScatterDimNumbers {
            update_window_dims,
            inserted_window_dims: vec![ax as i64],
            scatter_dims_to_operand_dims: vec![ax as i64],
            index_vector_dim: idx_rank,
        };
        self.entry.scatter(dest, idx_s32, dy, &combiner, dn, out)
    }

    /// RoPE backward = forward with negated sin (NeoX rotate-half, `n_rot` support).
    pub(crate) fn lower_rope_backward(
        &mut self,
        dy_id: NodeId,
        cos_id: NodeId,
        sin_id: NodeId,
        head_dim: usize,
        n_rot: usize,
        out: Shape,
    ) -> i64 {
        let dy = self.hlo(dy_id);
        let cos = self.hlo(cos_id);
        let sin = self.hlo(sin_id);
        let dims = self.ir_shape_dims(dy_id);
        let dt = self.dtype(dy_id);
        let prim_ty = prim_of(dt);
        let last = dims.len() - 1;
        let rot = n_rot.min(head_dim);
        let half = rot / 2;
        assert!(
            dims[last] >= rot as i64,
            "rlx-tpu RopeBackward: last dim {} < n_rot {n_rot}",
            dims[last]
        );

        let sin_dims = self.ir_shape_dims(sin_id);
        let neg = self.const_in_dtype(prim_ty, -1.0);
        let neg_s = self
            .entry
            .broadcast(neg, &[], Shape::array(prim_ty, &sin_dims));
        let sin_neg = self
            .entry
            .binary("multiply", sin, neg_s, Shape::array(prim_ty, &sin_dims));

        let strides = vec![1i64; dims.len()];
        let mut starts1 = vec![0i64; dims.len()];
        let mut limits1 = dims.clone();
        let mut starts2 = vec![0i64; dims.len()];
        let mut limits2 = dims.clone();
        starts1[last] = 0;
        limits1[last] = half as i64;
        starts2[last] = half as i64;
        limits2[last] = rot as i64;
        let mut half_dims = dims.clone();
        half_dims[last] = half as i64;
        let half_shape = Shape::array(prim_ty, &half_dims);

        let mut rot_stop = dims.clone();
        rot_stop[last] = rot as i64;
        let rot_part = self.entry.slice(
            dy,
            &vec![0i64; dims.len()],
            &rot_stop,
            &strides,
            Shape::array(prim_ty, &rot_stop),
        );
        let x1 = self
            .entry
            .slice(rot_part, &starts1, &limits1, &strides, half_shape.clone());
        let x2 = self
            .entry
            .slice(rot_part, &starts2, &limits2, &strides, half_shape.clone());

        let cos_dims = self.ir_shape_dims(cos_id);
        let cos_b = self.broadcast_to_target(cos, &cos_dims, half_shape.clone());
        let sin_b = self.broadcast_to_target(sin_neg, &sin_dims, half_shape.clone());

        let x1c = self.entry.binary("multiply", x1, cos_b, half_shape.clone());
        let x2s = self.entry.binary("multiply", x2, sin_b, half_shape.clone());
        let r1 = self.entry.binary("subtract", x1c, x2s, half_shape.clone());
        let x1s = self.entry.binary("multiply", x1, sin_b, half_shape.clone());
        let x2c = self.entry.binary("multiply", x2, cos_b, half_shape.clone());
        let r2 = self.entry.binary("add", x1s, x2c, half_shape);
        let rotated = self
            .entry
            .concat(&[r1, r2], last as i64, Shape::array(prim_ty, &rot_stop));

        if dims[last] == rot as i64 {
            rotated
        } else {
            let mut tail_start = vec![0i64; dims.len()];
            tail_start[last] = rot as i64;
            let tail = self.entry.slice(
                dy,
                &tail_start,
                &dims,
                &strides,
                Shape::array(prim_ty, &{
                    let mut t = dims.clone();
                    t[last] = dims[last] - rot as i64;
                    t
                }),
            );
            self.entry.concat(&[rotated, tail], last as i64, out)
        }
    }

    /// FakeQuantizeLSQ forward = Fixed-scale fake-quant.
    pub(crate) fn lower_fake_quantize_lsq(
        &mut self,
        inputs: &[NodeId],
        bits: u8,
        axis: Option<usize>,
        out: Shape,
    ) -> i64 {
        use rlx_ir::op::ScaleMode;
        self.lower_fake_quantize(inputs, bits, axis, ScaleMode::Fixed, out)
    }

    /// STE / clipped / tanh / hard-tanh fake-quant VJP (PerBatch scale recompute).
    pub(crate) fn lower_fake_quantize_backward(
        &mut self,
        x_id: NodeId,
        dy_id: NodeId,
        bits: u8,
        axis: Option<usize>,
        ste: SteKind,
        out: Shape,
    ) -> i64 {
        let q_max = match bits {
            8 => 127.0f32,
            4 => 7.0,
            2 => 1.0,
            n => panic!("rlx-tpu FakeQuantizeBackward: unsupported bits {n}"),
        };
        let x = self.hlo(x_id);
        let dy = self.hlo(dy_id);
        let dims = out.dimensions.clone();
        let prim_ty = out.element_type;

        let abs_x = self.entry.unary("abs", x, out.clone());
        let (max_abs, kept) = self.reduce_abs_max_for_fake_quant(abs_x, axis, &dims, prim_ty);
        let q = self.const_in_dtype(prim_ty, q_max);
        let q_shape = if kept.is_empty() {
            Shape::scalar(prim_ty)
        } else {
            Shape::array(prim_ty, &kept)
        };
        let q_b = self.entry.broadcast(q, &[], q_shape.clone());
        let s = self.entry.binary("divide", max_abs, q_b, q_shape.clone());
        let eps = self.const_in_dtype(prim_ty, 1e-12);
        let eps_b = self.entry.broadcast(eps, &[], q_shape.clone());
        let s = self.entry.binary("maximum", s, eps_b, q_shape.clone());
        let scale = if kept.is_empty() {
            self.entry.broadcast(s, &[], out.clone())
        } else {
            self.broadcast_align(s, &kept, out.clone())
        };

        let zero = self.const_in_dtype(prim_ty, 0.0);
        let zero_b = self.entry.broadcast(zero, &[], out.clone());
        let one = self.const_in_dtype(prim_ty, 1.0);
        let one_b = self.entry.broadcast(one, &[], out.clone());
        let q_max_c = self.const_in_dtype(prim_ty, q_max);
        let q_max_b = self.entry.broadcast(q_max_c, &[], out.clone());

        match ste {
            SteKind::Identity => dy,
            SteKind::ClippedIdentity => {
                let bound = self.entry.binary("multiply", scale, q_max_b, out.clone());
                let abs_x = self.entry.unary("abs", x, out.clone());
                let pred = self.entry.compare(abs_x, bound, "LE", Shape::pred(&dims));
                self.entry.select(pred, dy, zero_b, out)
            }
            SteKind::Tanh => {
                let scaled = self.entry.binary("divide", x, scale, out.clone());
                let t = self.entry.unary("tanh", scaled, out.clone());
                let t_sq = self.entry.binary("multiply", t, t, out.clone());
                let factor = self.entry.binary("subtract", one_b, t_sq, out.clone());
                self.entry.binary("multiply", dy, factor, out)
            }
            SteKind::HardTanh => {
                let bound = self.entry.binary("multiply", scale, q_max_b, out.clone());
                let scaled = self.entry.binary("divide", x, bound, out.clone());
                let abs_scaled = self.entry.unary("abs", scaled, out.clone());
                let one_minus = self
                    .entry
                    .binary("subtract", one_b, abs_scaled, out.clone());
                let attenuation = self.entry.binary("maximum", one_minus, zero_b, out.clone());
                self.entry.binary("multiply", dy, attenuation, out)
            }
        }
    }

    /// LSQ STE for `x`: `dx = dy` where `|x/s| ≤ q_max`, else 0.
    pub(crate) fn lower_fake_quantize_lsq_backward_x(
        &mut self,
        x_id: NodeId,
        scale_id: NodeId,
        dy_id: NodeId,
        bits: u8,
        axis: Option<usize>,
        out: Shape,
    ) -> i64 {
        let q_max = match bits {
            8 => 127.0f32,
            4 => 7.0,
            2 => 1.0,
            n => panic!("rlx-tpu FakeQuantizeLSQBackwardX: unsupported bits {n}"),
        };
        let x = self.hlo(x_id);
        let scale = self.hlo(scale_id);
        let dy = self.hlo(dy_id);
        let dims = out.dimensions.clone();
        let prim_ty = out.element_type;
        let s_dims = self.ir_shape_dims(scale_id);
        let scale_b = self.broadcast_q_param(scale, &s_dims, axis, &dims, prim_ty);
        let eps = self.const_in_dtype(prim_ty, 1e-12);
        let eps_b = self.entry.broadcast(eps, &[], out.clone());
        let s = self.entry.binary("maximum", scale_b, eps_b, out.clone());
        let z = self.entry.binary("divide", x, s, out.clone());
        let abs_z = self.entry.unary("abs", z, out.clone());
        let q = self.const_in_dtype(prim_ty, q_max);
        let q_b = self.entry.broadcast(q, &[], out.clone());
        let zero = self.const_in_dtype(prim_ty, 0.0);
        let zero_b = self.entry.broadcast(zero, &[], out.clone());
        let pred = self.entry.compare(abs_z, q_b, "LE", Shape::pred(&dims));
        self.entry.select(pred, dy, zero_b, out)
    }

    /// LSQ scale gradient: `Σ ψ(x/s) · dy` with
    /// `ψ(z) = −z + round(z)` inside range, else `sign(z)·q_max`.
    pub(crate) fn lower_fake_quantize_lsq_backward_scale(
        &mut self,
        x_id: NodeId,
        scale_id: NodeId,
        dy_id: NodeId,
        bits: u8,
        axis: Option<usize>,
        out: Shape,
    ) -> i64 {
        let q_max = match bits {
            8 => 127.0f32,
            4 => 7.0,
            2 => 1.0,
            n => panic!("rlx-tpu FakeQuantizeLSQBackwardScale: unsupported bits {n}"),
        };
        let x = self.hlo(x_id);
        let scale = self.hlo(scale_id);
        let dy = self.hlo(dy_id);
        let x_dims = self.ir_shape_dims(x_id);
        let x_dt = self.dtype(x_id);
        let prim_ty = prim_of(x_dt);
        let x_shape = Shape::array(prim_ty, &x_dims);
        let s_dims = self.ir_shape_dims(scale_id);
        let scale_b = self.broadcast_q_param(scale, &s_dims, axis, &x_dims, prim_ty);
        let eps = self.const_in_dtype(prim_ty, 1e-12);
        let eps_b = self.entry.broadcast(eps, &[], x_shape.clone());
        let s = self
            .entry
            .binary("maximum", scale_b, eps_b, x_shape.clone());
        let z = self.entry.binary("divide", x, s, x_shape.clone());
        let rounded = self.entry.unary("round-nearest-even", z, x_shape.clone());
        let neg_z = self.entry.unary("negate", z, x_shape.clone());
        let inner_psi = self.entry.binary("add", neg_z, rounded, x_shape.clone());

        let q = self.const_in_dtype(prim_ty, q_max);
        let q_b = self.entry.broadcast(q, &[], x_shape.clone());
        let neg_q = self.const_in_dtype(prim_ty, -q_max);
        let neg_q_b = self.entry.broadcast(neg_q, &[], x_shape.clone());
        let zero = self.const_in_dtype(prim_ty, 0.0);
        let zero_b = self.entry.broadcast(zero, &[], x_shape.clone());
        let pos = self.entry.compare(z, zero_b, "GT", Shape::pred(&x_dims));
        let outer_psi = self.entry.select(pos, q_b, neg_q_b, x_shape.clone());

        let abs_z = self.entry.unary("abs", z, x_shape.clone());
        let in_range = self.entry.compare(abs_z, q_b, "LE", Shape::pred(&x_dims));
        let psi = self
            .entry
            .select(in_range, inner_psi, outer_psi, x_shape.clone());
        let contrib = self.entry.binary("multiply", psi, dy, x_shape);
        let out_dims = out.dimensions.clone();
        self.sum_unbroadcast_hlo(contrib, &x_dims, &out_dims, x_dt)
    }

    /// `dx = ConvGeneralDilated(dy, flip(w^T))` with lhs dilation = stride
    /// (MLX / JAX ConvGeneralDilated VJP; NCHW).
    pub(crate) fn lower_conv2d_backward_input(
        &mut self,
        dy_id: NodeId,
        w_id: NodeId,
        kernel_size: &[usize],
        stride: &[usize],
        padding: &[usize],
        dilation: &[usize],
        groups: usize,
        out: Shape,
    ) -> i64 {
        assert_eq!(kernel_size.len(), 2, "rlx-tpu Conv2dBackwardInput: 2D only");
        let dy = self.hlo(dy_id);
        let w = self.hlo(w_id);
        let dy_dims = self.ir_shape_dims(dy_id);
        let w_dims = self.ir_shape_dims(w_id);
        let dx_dims = out.dimensions.clone();
        assert_eq!(dy_dims.len(), 4, "rlx-tpu Conv2dBackwardInput: NCHW dy");
        assert_eq!(w_dims.len(), 4, "rlx-tpu Conv2dBackwardInput: NCHW w");
        assert_eq!(dx_dims.len(), 4, "rlx-tpu Conv2dBackwardInput: NCHW dx");

        let g = groups as i64;
        let c_in = dx_dims[1];
        let c_out = dy_dims[1];
        assert!(
            c_in % g == 0 && c_out % g == 0,
            "rlx-tpu Conv2dBackwardInput: groups must divide channels"
        );
        let c_in_per_g = c_in / g;
        let c_out_per_g = c_out / g;
        let (h, w_in) = (dx_dims[2], dx_dims[3]);
        let (h_out, w_out) = (dy_dims[2], dy_dims[3]);
        let (kh, kw) = (w_dims[2], w_dims[3]);
        let (sh, sw) = (stride[0] as i64, stride[1] as i64);
        let (ph, pw) = (padding[0] as i64, padding[1] as i64);
        let (dh, dw) = (dilation[0] as i64, dilation[1] as i64);
        let prim_ty = out.element_type;

        // Weight: [C_out, C_in/g, kH, kW] → [C_in, C_out/g, kH, kW]
        // (group transpose of MLX Convolution::vjp).
        let w_t = if g == 1 {
            self.entry.transpose(
                w,
                &[1, 0, 2, 3],
                Shape::array(prim_ty, &[c_in, c_out, kh, kw]),
            )
        } else {
            let split = self.entry.reshape(
                w,
                Shape::array(prim_ty, &[g, c_out_per_g, c_in_per_g, kh, kw]),
            );
            let perm = self.entry.transpose(
                split,
                &[0, 2, 3, 4, 1],
                Shape::array(prim_ty, &[g, c_in_per_g, kh, kw, c_out_per_g]),
            );
            let flat = self
                .entry
                .reshape(perm, Shape::array(prim_ty, &[c_in, kh, kw, c_out_per_g]));
            self.entry.transpose(
                flat,
                &[0, 3, 1, 2],
                Shape::array(prim_ty, &[c_in, c_out_per_g, kh, kw]),
            )
        };

        // pad_lo = D·(K−1) − P; pad_hi = H − 1 − S·(H_out−1) + P
        let pad_lo_h = dh * (kh - 1) - ph;
        let pad_lo_w = dw * (kw - 1) - pw;
        let pad_hi_h = h - 1 - sh * (h_out - 1) + ph;
        let pad_hi_w = w_in - 1 - sw * (w_out - 1) + pw;

        let window = Window {
            dimensions: vec![
                WindowDim {
                    size: kh,
                    stride: 1,
                    padding_low: pad_lo_h,
                    padding_high: pad_hi_h,
                    window_dilation: dh,
                    base_dilation: sh,
                    window_reversal: true,
                },
                WindowDim {
                    size: kw,
                    stride: 1,
                    padding_low: pad_lo_w,
                    padding_high: pad_hi_w,
                    window_dilation: dw,
                    base_dilation: sw,
                    window_reversal: true,
                },
            ],
        };
        // dy [N,C_out,...] × w_t [C_in, C_out/g, kH, kW] → [N,C_in,...]
        let cdn = ConvDimNumbers {
            input_batch_dim: 0,
            input_feature_dim: 1,
            input_spatial_dims: vec![2, 3],
            kernel_output_feature_dim: 0,
            kernel_input_feature_dim: 1,
            kernel_spatial_dims: vec![2, 3],
            output_batch_dim: 0,
            output_feature_dim: 1,
            output_spatial_dims: vec![2, 3],
        };
        self.entry.convolution(dy, w_t, window, cdn, g, out)
    }

    /// `dw` via ConvGeneralDilated with batch↔feature dim numbers (MLX /
    /// JAX VJP; NCHW). Output layout [C_out, C_in/g, kH, kW].
    pub(crate) fn lower_conv2d_backward_weight(
        &mut self,
        x_id: NodeId,
        dy_id: NodeId,
        kernel_size: &[usize],
        stride: &[usize],
        padding: &[usize],
        dilation: &[usize],
        groups: usize,
        out: Shape,
    ) -> i64 {
        assert_eq!(
            kernel_size.len(),
            2,
            "rlx-tpu Conv2dBackwardWeight: 2D only"
        );
        let x = self.hlo(x_id);
        let dy = self.hlo(dy_id);
        let x_dims = self.ir_shape_dims(x_id);
        let dy_dims = self.ir_shape_dims(dy_id);
        let dw_dims = out.dimensions.clone();
        assert_eq!(x_dims.len(), 4, "rlx-tpu Conv2dBackwardWeight: NCHW x");
        assert_eq!(dy_dims.len(), 4, "rlx-tpu Conv2dBackwardWeight: NCHW dy");
        assert_eq!(dw_dims.len(), 4, "rlx-tpu Conv2dBackwardWeight: NCHW dw");

        let g = groups as i64;
        let n_batch = x_dims[0];
        let c_in = x_dims[1];
        let c_out = dy_dims[1];
        assert!(
            c_in % g == 0 && c_out % g == 0,
            "rlx-tpu Conv2dBackwardWeight: groups must divide channels"
        );
        let c_in_per_g = c_in / g;
        let (h, w_in) = (x_dims[2], x_dims[3]);
        let (h_out, w_out) = (dy_dims[2], dy_dims[3]);
        let (kh, kw) = (dw_dims[2], dw_dims[3]);
        let (sh, sw) = (stride[0] as i64, stride[1] as i64);
        let (ph, pw) = (padding[0] as i64, padding[1] as i64);
        let (dh, dw_d) = (dilation[0] as i64, dilation[1] as i64);
        let prim_ty = out.element_type;

        // pad_lo = P; pad_hi = S·(H_out−1)+1 − H + D·(K−1)+1 − P − 1
        let pad_lo_h = ph;
        let pad_lo_w = pw;
        let pad_hi_h = sh * (h_out - 1) + 1 - h + dh * (kh - 1) + 1 - ph - 1;
        let pad_hi_w = sw * (w_out - 1) + 1 - w_in + dw_d * (kw - 1) + 1 - pw - 1;

        // Physical layout matching MLX:
        //   x  → [C_in/g or C_in, H, W, N' ] with N' = N or g·N
        //   dy → [C_out, H_out, W_out, N]
        // then transpose result [C_in…, kH, kW, C_out…] → [C_out, C_in/g, kH, kW].
        let (lhs, rhs, feature_groups) = if g == 1 {
            let lhs = self.entry.transpose(
                x,
                &[1, 2, 3, 0],
                Shape::array(prim_ty, &[c_in, h, w_in, n_batch]),
            );
            let rhs = self.entry.transpose(
                dy,
                &[1, 2, 3, 0],
                Shape::array(prim_ty, &[c_out, h_out, w_out, n_batch]),
            );
            (lhs, rhs, 1i64)
        } else {
            let split = self
                .entry
                .reshape(x, Shape::array(prim_ty, &[n_batch, g, c_in_per_g, h, w_in]));
            let perm = self.entry.transpose(
                split,
                &[2, 3, 4, 1, 0],
                Shape::array(prim_ty, &[c_in_per_g, h, w_in, g, n_batch]),
            );
            let lhs = self.entry.reshape(
                perm,
                Shape::array(prim_ty, &[c_in_per_g, h, w_in, g * n_batch]),
            );
            let rhs = self.entry.transpose(
                dy,
                &[1, 2, 3, 0],
                Shape::array(prim_ty, &[c_out, h_out, w_out, n_batch]),
            );
            (lhs, rhs, g)
        };

        // lhs [C_in', H, W, N'], rhs [C_out, H_out, W_out, N] with
        // dim numbers: batch=0 (feature of x), feature=3 (batch of x),
        // kernel feature in/out = 3 / 0, output batch=0 feature=3 →
        // [C_in', kH, kW, C_out/g].
        let cdn = ConvDimNumbers {
            input_batch_dim: 0,
            input_feature_dim: 3,
            input_spatial_dims: vec![1, 2],
            kernel_input_feature_dim: 3,
            kernel_output_feature_dim: 0,
            kernel_spatial_dims: vec![1, 2],
            output_batch_dim: 0,
            output_feature_dim: 3,
            output_spatial_dims: vec![1, 2],
        };
        let window = Window {
            dimensions: vec![
                WindowDim {
                    size: h_out,
                    stride: dh,
                    padding_low: pad_lo_h,
                    padding_high: pad_hi_h,
                    window_dilation: sh,
                    base_dilation: 1,
                    window_reversal: false,
                },
                WindowDim {
                    size: w_out,
                    stride: dw_d,
                    padding_low: pad_lo_w,
                    padding_high: pad_hi_w,
                    window_dilation: sw,
                    base_dilation: 1,
                    window_reversal: false,
                },
            ],
        };
        let raw_shape = Shape::array(prim_ty, &[c_in_per_g, kh, kw, c_out]);
        let raw = self
            .entry
            .convolution(lhs, rhs, window, cdn, feature_groups, raw_shape);
        // [C_in/g, kH, kW, C_out] → [C_out, C_in/g, kH, kW]
        self.entry.transpose(raw, &[3, 0, 1, 2], out)
    }

    /// MaxPool2d VJP via HLO `select-and-scatter` (GE select + add scatter).
    /// Window geometry matches forward `Pool` / `reduce-window` (NCHW).
    pub(crate) fn lower_max_pool2d_backward(
        &mut self,
        x_id: NodeId,
        dy_id: NodeId,
        kernel_size: &[usize],
        stride: &[usize],
        padding: &[usize],
        out: Shape,
    ) -> i64 {
        assert_eq!(kernel_size.len(), 2, "rlx-tpu MaxPool2dBackward: 2D only");
        assert_eq!(stride.len(), 2, "rlx-tpu MaxPool2dBackward: 2D only");
        assert_eq!(padding.len(), 2, "rlx-tpu MaxPool2dBackward: 2D only");
        let x = self.hlo(x_id);
        let dy = self.hlo(dy_id);
        let x_dims = self.ir_shape_dims(x_id);
        assert_eq!(x_dims.len(), 4, "rlx-tpu MaxPool2dBackward: NCHW x");
        let dy_dims = self.ir_shape_dims(dy_id);
        assert_eq!(dy_dims.len(), 4, "rlx-tpu MaxPool2dBackward: NCHW dy");
        let x_dt = self.dtype(x_id);
        let prim_ty = prim_of(x_dt);

        let select = self
            .builder
            .make_ge_select(&format!("maxpool_ge_sel_{prim_ty}"), prim_ty);
        let scatter = self.reducer("add", prim_ty);
        let init = self.const_in_dtype(prim_ty, 0.0);

        let mut window_dims = vec![
            WindowDim {
                size: 1,
                stride: 1,
                padding_low: 0,
                padding_high: 0,
                window_dilation: 1,
                base_dilation: 1,
                window_reversal: false,
            };
            4
        ];
        for i in 0..2 {
            window_dims[2 + i] = WindowDim {
                size: kernel_size[i] as i64,
                stride: stride[i] as i64,
                padding_low: padding[i] as i64,
                padding_high: padding[i] as i64,
                window_dilation: 1,
                base_dilation: 1,
                window_reversal: false,
            };
        }
        let window = Window {
            dimensions: window_dims,
        };
        self.entry
            .select_and_scatter(x, dy, init, &select, &scatter, window, out)
    }
}
