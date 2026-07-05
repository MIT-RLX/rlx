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

//! `apply` — extracted from the `lower` module for navigability (see `mod.rs`).

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
    /// Synthesize and add a causal mask to QK^T in HLO using
    /// `iota` + `compare` + `select`. Avoids materializing a mask
    /// tensor on the host.
    pub(crate) fn apply_causal_mask(
        &self,
        scaled: i64,
        qk_shape: Shape,
        _s_q: i64,
        _s_k: i64,
        prim_ty: i32,
    ) -> i64 {
        let q_idx = self
            .entry
            .iota(2, Shape::array(prim::S32, &qk_shape.dimensions));
        let k_idx = self
            .entry
            .iota(3, Shape::array(prim::S32, &qk_shape.dimensions));
        let mask = self
            .entry
            .compare(q_idx, k_idx, "GE", Shape::pred(&qk_shape.dimensions));
        let neg_inf = self.const_in_dtype(prim_ty, f32::NEG_INFINITY);
        let neg_inf_b = self.entry.broadcast(neg_inf, &[], qk_shape.clone());
        self.entry.select(mask, scaled, neg_inf_b, qk_shape)
    }

    /// Sliding-window mask: q attends to k in [q-w, q].
    pub(crate) fn apply_sliding_window_mask(
        &self,
        scaled: i64,
        qk_shape: Shape,
        _s_q: i64,
        _s_k: i64,
        w: i64,
        prim_ty: i32,
    ) -> i64 {
        let q_idx = self
            .entry
            .iota(2, Shape::array(prim::S32, &qk_shape.dimensions));
        let k_idx = self
            .entry
            .iota(3, Shape::array(prim::S32, &qk_shape.dimensions));
        let lower = self
            .entry
            .compare(q_idx, k_idx, "GE", Shape::pred(&qk_shape.dimensions));
        // q - k <= w  →  k >= q - w
        let qmw = self.entry.constant(Literal {
            shape: Shape::scalar(prim::S32),
            data: LiteralData::S32(vec![w as i32]),
        });
        let qmw_b = self
            .entry
            .broadcast(qmw, &[], Shape::array(prim::S32, &qk_shape.dimensions));
        let q_minus_w = self.entry.binary(
            "subtract",
            q_idx,
            qmw_b,
            Shape::array(prim::S32, &qk_shape.dimensions),
        );
        let upper = self
            .entry
            .compare(k_idx, q_minus_w, "GE", Shape::pred(&qk_shape.dimensions));
        let mask = self
            .entry
            .binary("and", lower, upper, Shape::pred(&qk_shape.dimensions));
        let neg_inf = self.const_in_dtype(prim_ty, f32::NEG_INFINITY);
        let neg_inf_b = self.entry.broadcast(neg_inf, &[], qk_shape.clone());
        self.entry.select(mask, scaled, neg_inf_b, qk_shape)
    }

    // ── Rope ───────────────────────────────────────────────────
}
