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

//! `fill` — extracted from the `backend` module for navigability (see `mod.rs`).

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
    pub(crate) fn fill_output_staging_indices(&mut self, indices: &[usize]) {
        unsafe {
            let _ = (self.ctx.runtime.hip_stream_sync)(self.ctx.default_stream);
        }
        for &i in indices {
            let id = self.graph.outputs[i];
            let off_f32 = self.arena.offset(id) / 4;
            let elems = self.graph.node(id).shape.num_elements().unwrap_or(0);
            let src = self.arena.buffer.ptr + (off_f32 as u64) * 4;
            debug_assert_eq!(self.output_staging[i].len(), elems);
            self.output_staging[i]
                .dtoh(&self.ctx.runtime, src, elems)
                .expect("rlx-rocm: partial output download failed");
        }
    }


    pub(crate) fn fill_output_staging_all(&mut self) {
        unsafe {
            let _ = (self.ctx.runtime.hip_stream_sync)(self.ctx.default_stream);
        }
        for (i, &id) in self.graph.outputs.iter().enumerate() {
            let off_f32 = self.arena.offset(id) / 4;
            let elems = self.graph.node(id).shape.num_elements().unwrap_or(0);
            let src = self.arena.buffer.ptr + (off_f32 as u64) * 4;
            debug_assert_eq!(self.output_staging[i].len(), elems);
            self.output_staging[i]
                .dtoh(&self.ctx.runtime, src, elems)
                .expect("rlx-rocm: output download failed");
        }
    }

}
