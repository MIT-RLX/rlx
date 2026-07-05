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

//! `broadcast` — extracted from the `lower` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use crate::hlo::{
    Computation, ConvDimNumbers, DotDimNumbers, GatherDimNumbers, HloBuilder, Literal, LiteralData,
    ProgramShape, ScatterDimNumbers, Shape, Window, WindowDim, prim, prim_of,
};
use rlx_ir::op::{
    Activation, BinaryOp, ChainOperand, ChainStep, CmpOp, MaskKind, ReduceOp, RegionPrologue,
    TransformStep,
};
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, NodeId, Op};
use std::collections::HashMap;

use super::*;

impl<'a> LowerCtx<'a> {
    /// Broadcast `x` of shape `x_shape` to `target_shape` by aligning
    /// every axis where x has size 1 vs target's size > 1. HLO's
    /// `broadcast` only adds new leading dims; we use `broadcast_in_dim`
    /// semantics by emitting a `reshape` to drop the size-1 dims first
    /// and then a broadcast that places the surviving dims at their
    /// original positions.
    pub(crate) fn broadcast_align(&self, x: i64, x_shape: &[i64], target: Shape) -> i64 {
        let target_dims = target.dimensions.clone();
        debug_assert_eq!(
            x_shape.len(),
            target_dims.len(),
            "broadcast_align expects same rank"
        );
        // Identity broadcast — x already at target.
        if x_shape == target_dims.as_slice() {
            return x;
        }
        // Drop size-1 axes that target wants to expand.
        let surviving_axes: Vec<i64> = (0..x_shape.len() as i64)
            .filter(|&i| {
                let xi = x_shape[i as usize];
                let ti = target_dims[i as usize];
                xi == ti
            })
            .collect();
        let surviving_dims: Vec<i64> = surviving_axes
            .iter()
            .map(|&i| x_shape[i as usize])
            .collect();
        let small = if surviving_dims.len() == x_shape.len() {
            x
        } else {
            let elt = target.element_type;
            self.entry.reshape(x, Shape::array(elt, &surviving_dims))
        };
        self.entry.broadcast(small, &surviving_axes, target)
    }

    /// Build a scale/zero-point broadcast for `Op::Quantize` /
    /// `Op::Dequantize`. `axis = None` → scalar broadcast (per-tensor);
    /// `axis = Some(d)` → 1-D constant of length `out_dims[d]`
    /// broadcast along the channel axis.
    pub(crate) fn broadcast_q_factor(
        &self,
        axis: Option<usize>,
        values: &[f32],
        out_dims: &[i64],
        prim_ty: i32,
    ) -> i64 {
        let out_shape = Shape::array(prim_ty, out_dims);
        match axis {
            None => {
                let v = values.first().copied().unwrap_or(0.0);
                let c = self.const_in_dtype(prim_ty, v);
                self.entry.broadcast(c, &[], out_shape)
            }
            Some(d) => {
                // Materialize a [N] f32 constant where N = out_dims[d].
                // Convert to target dtype if needed, then broadcast
                // along axis d (broadcast_dims = [d]).
                let n = out_dims[d];
                debug_assert_eq!(
                    values.len() as i64,
                    n,
                    "Quantize/Dequantize: per-channel values len ({}) \
                     must match output dim[{}] ({})",
                    values.len(),
                    d,
                    n
                );
                let lit = crate::hlo::Literal {
                    shape: Shape::array(prim::F32, &[n]),
                    data: crate::hlo::LiteralData::F32(values.to_vec()),
                };
                let c = self.entry.constant(lit);
                let c = if prim_ty == prim::F32 {
                    c
                } else {
                    self.entry.convert(c, Shape::array(prim_ty, &[n]))
                };
                self.entry.broadcast(c, &[d as i64], out_shape)
            }
        }
    }

    // ── Binary ─────────────────────────────────────────────────

    /// Bring two operands to a common rank-aligned shape against
    /// `target_dims`. HLO requires both binary operands to have the
    /// same shape; we use `broadcast_align` to lift each one to target.
    pub(crate) fn broadcast_pair_to(
        &self,
        a: i64,
        b: i64,
        a_id: NodeId,
        b_id: NodeId,
        target_dims: &[i64],
    ) -> (i64, i64) {
        let a_dims = self.ir_shape_dims(a_id);
        let b_dims = self.ir_shape_dims(b_id);
        let a_dt = self.dtype(a_id);
        let b_dt = self.dtype(b_id);
        let target_a = Shape::array(prim_of(a_dt), target_dims);
        let target_b = Shape::array(prim_of(b_dt), target_dims);
        let a2 = self.broadcast_to_target(a, &a_dims, target_a);
        let b2 = self.broadcast_to_target(b, &b_dims, target_b);
        (a2, b2)
    }

    /// Broadcast `x` to `target_shape`. Adds leading dims when
    /// `x_dims.len() < target.rank()`, or replicates size-1 axes when
    /// rank matches.
    pub(crate) fn broadcast_to_target(&self, x: i64, x_dims: &[i64], target: Shape) -> i64 {
        let target_dims = target.dimensions.clone();
        if x_dims == target_dims.as_slice() {
            return x;
        }
        if x_dims.len() < target_dims.len() {
            // Pad to right (broadcast adds leading dims).
            let target_rank = target_dims.len();
            let broadcast_dims: Vec<i64> = (target_rank - x_dims.len()..target_rank)
                .map(|i| i as i64)
                .collect();
            // The intermediate shape is x's dims placed at trailing
            // positions of target, with leading dims taken from
            // target. HLO infers the result shape from `target`.
            return self.entry.broadcast(x, &broadcast_dims, target);
        }
        self.broadcast_align(x, x_dims, target)
    }

    // ── ElementwiseRegion ─────────────────────────────────────

    /// Lift a 1-D normalization parameter (shape `[axis_size]`) up to
    /// the layout `x` uses, by reshaping to size-1 in every axis
    /// except `axis` then broadcasting.
    pub(crate) fn broadcast_param_to_axis(
        &self,
        p: i64,
        p_dims: &[i64],
        axis: i64,
        x_dims: &[i64],
        prim_ty: i32,
    ) -> i64 {
        let target = Shape::array(prim_ty, x_dims);
        if p_dims == x_dims {
            return p;
        }
        if p_dims.len() == 1 {
            // [N] → [1,1,...,N,...,1] then broadcast.
            let mut padded = vec![1i64; x_dims.len()];
            padded[axis as usize] = p_dims[0];
            let r = self.entry.reshape(p, Shape::array(prim_ty, &padded));
            return self.broadcast_align(r, &padded, target);
        }
        self.broadcast_to_target(p, p_dims, target)
    }

    // ── RmsNorm ────────────────────────────────────────────────
}
