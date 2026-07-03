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

//! `ilu_pcg` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]

#![allow(unused_imports)]

use std::sync::Arc;

use rlx_ir::{DType, Graph, Node, NodeId, Op, OpExtension, Shape, VjpContext, register_op};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef, register_cpu_kernel};

// ── Op names (stable strings; downstream callers use these to look
//    up the registered op or build `Op::Custom` directly) ─────────

use super::*;

pub(crate) struct SparseIluPcgExt;


impl OpExtension for SparseIluPcgExt {
    fn name(&self) -> &str {
        SPARSE_ILU_PCG_SOLVE
    }
    fn num_inputs(&self) -> usize {
        4
    }
    fn infer_shape(&self, inputs: &[&Shape], _attrs: &[u8]) -> Shape {
        inputs[3].clone()
    }
    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // Same SPD adjoint shape as plain CG/PCG — reuse the kernel.
        let vals = ctx.fwd_map[&node.inputs[0]];
        let cidx = ctx.fwd_map[&node.inputs[1]];
        let rptr = ctx.fwd_map[&node.inputs[2]];
        let attrs = match &node.op {
            Op::Custom { attrs, .. } => attrs.clone(),
            _ => Vec::new(),
        };
        let g_b = ctx.bwd.custom_op(
            SPARSE_ILU_PCG_SOLVE,
            attrs,
            vec![vals, cidx, rptr, ctx.upstream],
        );
        let y_fwd = ctx.fwd_map[&node.id];
        let raw_grad =
            ctx.bwd
                .custom_op(SPARSE_VALUES_GRAD, Vec::new(), vec![cidx, rptr, g_b, y_fwd]);
        let raw_shape = ctx.bwd.node(raw_grad).shape.clone();
        let g_vals = ctx
            .bwd
            .activation(rlx_ir::op::Activation::Neg, raw_grad, raw_shape);
        vec![(0, g_vals), (3, g_b)]
    }
}


#[cfg(feature = "cpu")]
pub(crate) struct SparseIluPcgCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for SparseIluPcgCpu {
    fn name(&self) -> &str {
        SPARSE_ILU_PCG_SOLVE
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let values = inputs[0].expect_f64("ilu_pcg values")?;
        let col_idx = inputs[1].expect_i32("ilu_pcg col_idx")?;
        let row_ptr = inputs[2].expect_i32("ilu_pcg row_ptr")?;
        let b = inputs[3].expect_f64("ilu_pcg b")?;
        let out = output.expect_f64_mut("ilu_pcg x")?;
        let (max_iter, tol) = decode_cg_attrs(attrs)?;
        algos::ilu_pcg_solve(values, col_idx, row_ptr, b, out, max_iter, tol)
    }
}

// ── Sparse Cholesky (direct) ──────────────────────────────────────

