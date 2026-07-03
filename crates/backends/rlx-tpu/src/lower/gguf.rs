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

//! `gguf` — extracted from the `lower` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use std::collections::HashMap;
use rlx_ir::op::{
    Activation, BinaryOp, ChainOperand, ChainStep, CmpOp, MaskKind, ReduceOp, RegionPrologue,
    TransformStep,
};
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, NodeId, Op};
use crate::hlo::{
    Computation, ConvDimNumbers, DotDimNumbers, GatherDimNumbers, HloBuilder, Literal, LiteralData,
    ProgramShape, ScatterDimNumbers, Shape, Window, WindowDim, prim, prim_of,
};

use super::*;

impl<'a> LowerCtx<'a> {
    /// Packed GGUF bytes for a weight node used by `DequantMatMul` lowering.
    /// Packed GGUF bytes for a weight node used by `DequantMatMul` lowering.
    pub(crate) fn gguf_weight_is_deferred(&self, w_id: NodeId) -> bool {
        match &self.graph.node(w_id).op {
            Op::Param { name } => self.param_bytes.and_then(|m| m.get(name)).is_none(),
            _ => false,
        }
    }


    pub(crate) fn gguf_weight_bytes(&self, w_id: NodeId) -> Vec<u8> {
        match &self.graph.node(w_id).op {
            Op::Constant { data } => data.to_vec(),
            Op::Param { name } => {
                if let Some(map) = self.param_bytes {
                    if let Some(bytes) = map.get(name) {
                        return bytes.clone();
                    }
                }
                panic!(
                    "rlx-tpu: GGUF DequantMatMul weight '{name}' is a runtime Param without \
                     compile-time bytes. Pass `LowerParamBytes` via \
                     `lower_graph_with_rng_and_params` or \
                     `TpuExecutable::compile_rng_with_param_bytes`."
                );
            }
            other => panic!("rlx-tpu: GGUF weight node must be Constant/Param, got {other:?}"),
        }
    }

}
