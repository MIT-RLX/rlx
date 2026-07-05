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

//! `values_grad` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]
#![allow(unused_imports)]

use std::sync::Arc;

use rlx_ir::{DType, Graph, Node, NodeId, Op, OpExtension, Shape, VjpContext, register_op};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef, register_cpu_kernel};

// ── Op names (stable strings; downstream callers use these to look
//    up the registered op or build `Op::Custom` directly) ─────────

use super::*;

pub(crate) struct SparseValuesGradExt;

impl OpExtension for SparseValuesGradExt {
    fn name(&self) -> &str {
        SPARSE_VALUES_GRAD
    }
    fn num_inputs(&self) -> usize {
        4
    } // col_idx, row_ptr, u, v

    fn infer_shape(&self, inputs: &[&Shape], _attrs: &[u8]) -> Shape {
        // Output shape == col_idx shape (length nnz, F64).
        let col_idx = inputs[0];
        assert_eq!(
            col_idx.dtype(),
            DType::I32,
            "values_grad: col_idx must be I32"
        );
        let nnz = col_idx
            .num_elements()
            .expect("values_grad: col_idx must have static shape");
        Shape::new(&[nnz], DType::F64)
    }
    // Non-differentiable (it's itself a gradient kernel; second-order
    // derivatives are out of v1 scope).
}

#[cfg(feature = "cpu")]
pub(crate) struct SparseValuesGradCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for SparseValuesGradCpu {
    fn name(&self) -> &str {
        SPARSE_VALUES_GRAD
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let col_idx = inputs[0].expect_i32("values_grad col_idx")?;
        let row_ptr = inputs[1].expect_i32("values_grad row_ptr")?;
        let u = inputs[2].expect_f64("values_grad u")?;
        let v = inputs[3].expect_f64("values_grad v")?;
        let out = output.expect_f64_mut("values_grad out")?;
        algos::values_grad(col_idx, row_ptr, u, v, out)
    }
}

// ── Sparse LU Solve (general / non-symmetric) ─────────────────────
//
// 7-input variant of `sparse_lu_solve`. Forward identity to the
// symmetric version (kernel only reads inputs 0..=3). The last 3
// inputs are the transpose CSR triplet used by the VJP for the
// adjoint solve `dL/db = solve(Aᵀ, dL/dx)`. Use this for non-
// symmetric matrices where reusing the forward triplet for the
// adjoint would be wrong.
