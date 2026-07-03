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

//! `output` — extracted from the `backend` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use std::collections::HashMap;
use std::sync::Arc;
use rlx_ir::op::{Activation, BinaryOp, CmpOp, MaskKind, ReduceOp};
use rlx_ir::{Graph, NodeId, Op};
use std::sync::Mutex;
use crate::arena::{Arena, HalfDtype, plan_f32_uniform};
use crate::device::{RocmContext, rocm_blas, rocm_blas_lt, rocm_context, rocm_dnn};
use crate::hip::{HipBuffer, HipDeviceptr};
use crate::hipblas::{
    HipblasComputeType, HipblasContext, HipblasDatatype, HipblasOperation, hipblas_gemm_default,
};
use crate::hipblaslt::HipblasLtContext;
use crate::host_staging::F32HostSlot;
use crate::miopen::MiopenContext;

use super::*;

impl RocmExecutable {
    pub fn output_slots(&self) -> &[(usize, usize)] {
        &self.output_slots
    }


    /// Declared graph-output dtypes, in `graph.outputs` order. Used by
    /// the runtime wrapper's `run_typed` to narrow f32 outputs back to
    /// the declared dtype on the way out.
    pub fn output_dtypes(&self) -> Vec<rlx_ir::DType> {
        self.graph
            .outputs
            .iter()
            .map(|&id| self.graph.node(id).shape.dtype())
            .collect()
    }

}
