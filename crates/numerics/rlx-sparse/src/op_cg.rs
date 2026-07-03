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

//! `cg` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]

#![allow(unused_imports)]

use std::sync::Arc;

use rlx_ir::{DType, Graph, Node, NodeId, Op, OpExtension, Shape, VjpContext, register_op};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef, register_cpu_kernel};

// ── Op names (stable strings; downstream callers use these to look
//    up the registered op or build `Op::Custom` directly) ─────────

use super::*;

pub(super) fn decode_cg_attrs(attrs: &[u8]) -> Result<(u32, f64), String> {
    if attrs.len() != 12 {
        return Err(format!(
            "cg_solve: attrs must be 12 bytes (u32 max_iter + f64 tol), got {}",
            attrs.len()
        ));
    }
    let max_iter = u32::from_le_bytes(attrs[0..4].try_into().unwrap());
    let tol = f64::from_le_bytes(attrs[4..12].try_into().unwrap());
    Ok((max_iter, tol))
}


pub(crate) struct SparseCgExt;


impl OpExtension for SparseCgExt {
    fn name(&self) -> &str {
        SPARSE_CG_SOLVE
    }
    fn num_inputs(&self) -> usize {
        4
    }

    fn infer_shape(&self, inputs: &[&Shape], _attrs: &[u8]) -> Shape {
        inputs[3].clone()
    }

    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // Same shape as SparseLuExt::vjp — iterative solver, same
        // closed-form adjoint. The adjoint solve recurses into
        // sparse_cg_solve with the forward's attrs (same tolerance
        // and iteration cap).
        let vals_b = ctx.fwd_map[&node.inputs[0]];
        let cidx_b = ctx.fwd_map[&node.inputs[1]];
        let rptr_b = ctx.fwd_map[&node.inputs[2]];
        let attrs = match &node.op {
            Op::Custom { attrs, .. } => attrs.clone(),
            _ => Vec::new(),
        };
        let g_b = ctx.bwd.custom_op(
            SPARSE_CG_SOLVE,
            attrs,
            vec![vals_b, cidx_b, rptr_b, ctx.upstream],
        );
        let y_fwd = ctx.fwd_map[&node.id];
        let raw_grad = ctx.bwd.custom_op(
            SPARSE_VALUES_GRAD,
            Vec::new(),
            vec![cidx_b, rptr_b, g_b, y_fwd],
        );
        let raw_shape = ctx.bwd.node(raw_grad).shape.clone();
        let g_vals = ctx
            .bwd
            .activation(rlx_ir::op::Activation::Neg, raw_grad, raw_shape);

        vec![(0, g_vals), (3, g_b)]
    }
}


#[cfg(feature = "cpu")]
pub(crate) struct SparseCgCpu;


#[cfg(feature = "cpu")]
impl CpuKernel for SparseCgCpu {
    fn name(&self) -> &str {
        SPARSE_CG_SOLVE
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let values = inputs[0].expect_f64("cg_solve values")?;
        let col_idx = inputs[1].expect_i32("cg_solve col_idx")?;
        let row_ptr = inputs[2].expect_i32("cg_solve row_ptr")?;
        let b = inputs[3].expect_f64("cg_solve b")?;
        let out = output.expect_f64_mut("cg_solve x")?;
        let (max_iter, tol) = decode_cg_attrs(attrs)?;
        algos::cg_solve(values, col_idx, row_ptr, b, out, max_iter, tol)
    }
}

// ── Sparse Values Gradient (`dL/dvalues` building block) ─────────

