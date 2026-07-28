// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `const_ops` — extracted from the `lower` module for navigability (see `mod.rs`).

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
    /// Constant scalar in an arbitrary primitive type — used for
    /// reduction inits, normalization eps, RoPE constants.
    pub(crate) fn const_scalar_f32(&self, v: f32) -> i64 {
        self.entry.constant_f32_scalar(v)
    }

    /// Scalar constant in the given primitive dtype (F32 or down-cast).
    pub(crate) fn const_in_dtype(&self, prim_ty: i32, v: f32) -> i64 {
        let f = self.entry.constant_f32_scalar(v);
        if prim_ty == prim::F32 {
            f
        } else {
            self.entry.convert(f, Shape::scalar(prim_ty))
        }
    }
}
