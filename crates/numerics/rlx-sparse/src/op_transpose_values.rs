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

//! `transpose_values` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]

#![allow(unused_imports)]

use std::sync::Arc;

use rlx_ir::{DType, Graph, Node, NodeId, Op, OpExtension, Shape, VjpContext, register_op};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef, register_cpu_kernel};

// ── Op names (stable strings; downstream callers use these to look
//    up the registered op or build `Op::Custom` directly) ─────────

use super::*;

pub(crate) struct SparseTransposeValuesExt;


impl OpExtension for SparseTransposeValuesExt {
    fn name(&self) -> &str {
        SPARSE_TRANSPOSE_VALUES
    }
    fn num_inputs(&self) -> usize {
        5
    } // values, col_idx, row_ptr, col_idx_T, row_ptr_T

    fn infer_shape(&self, inputs: &[&Shape], _attrs: &[u8]) -> Shape {
        // Output shape = same as values (length nnz).
        inputs[0].clone()
    }

    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // VJP of transpose is itself transpose (with patterns swapped).
        // dL/d(values_A) = transpose_values(upstream, col_idx_T, row_ptr_T,
        //                                    col_idx, row_ptr).
        let cidx_a = ctx.fwd_map[&node.inputs[1]];
        let rptr_a = ctx.fwd_map[&node.inputs[2]];
        let cidx_at = ctx.fwd_map[&node.inputs[3]];
        let rptr_at = ctx.fwd_map[&node.inputs[4]];
        let g_vals = ctx.bwd.custom_op(
            SPARSE_TRANSPOSE_VALUES,
            Vec::new(),
            vec![ctx.upstream, cidx_at, rptr_at, cidx_a, rptr_a],
        );
        vec![(0, g_vals)]
    }
}


#[cfg(feature = "cpu")]
pub(crate) struct SparseTransposeValuesCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for SparseTransposeValuesCpu {
    fn name(&self) -> &str {
        SPARSE_TRANSPOSE_VALUES
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let values = inputs[0].expect_f64("transpose_values values")?;
        let col_idx = inputs[1].expect_i32("transpose_values col_idx")?;
        let row_ptr = inputs[2].expect_i32("transpose_values row_ptr")?;
        let col_idx_t = inputs[3].expect_i32("transpose_values col_idx_T")?;
        let row_ptr_t = inputs[4].expect_i32("transpose_values row_ptr_T")?;
        let out = output.expect_f64_mut("transpose_values out")?;
        algos::transpose_values(values, col_idx, row_ptr, col_idx_t, row_ptr_t, out)
    }
}

// ── PCG (Jacobi preconditioner) ───────────────────────────────────

