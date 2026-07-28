// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `ir` — extracted from the `lower` module for navigability (see `mod.rs`).

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
    pub(crate) fn ir_shape_dims(&self, nid: NodeId) -> Vec<i64> {
        ir_dims(&self.graph.node(nid).shape)
    }

    pub(crate) fn ir_shape(&self, nid: NodeId) -> Shape {
        let n = self.graph.node(nid);
        Shape::array(prim_of(n.shape.dtype()), &ir_dims(&n.shape))
    }
}
