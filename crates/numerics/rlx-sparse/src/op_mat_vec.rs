// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `mat_vec` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]
#![allow(unused_imports)]

use std::sync::Arc;

use rlx_ir::{DType, Graph, Node, NodeId, Op, OpExtension, Shape, VjpContext, register_op};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef, register_cpu_kernel};

// ── Op names (stable strings; downstream callers use these to look
//    up the registered op or build `Op::Custom` directly) ─────────

use super::*;

pub(crate) struct SparseMatVecExt;

impl OpExtension for SparseMatVecExt {
    fn name(&self) -> &str {
        SPARSE_MAT_VEC
    }
    fn num_inputs(&self) -> usize {
        4
    }

    fn infer_shape(&self, inputs: &[&Shape], _attrs: &[u8]) -> Shape {
        // y has the shape of x (A is n×n by convention).
        inputs[3].clone()
    }

    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // y = A·x. Closed-form gradients:
        //   dL/dx = Aᵀ · upstream                  [v1 symmetric → mat_vec(A, ...)]
        //   dL/dvalues[k] = upstream[row(k)] · x[col(k)]   gathered at nonzero k
        let vals_b = ctx.fwd_map[&node.inputs[0]];
        let cidx_b = ctx.fwd_map[&node.inputs[1]];
        let rptr_b = ctx.fwd_map[&node.inputs[2]];
        let x_bwd = ctx.fwd_map[&node.inputs[3]];

        let g_x = ctx.bwd.custom_op(
            SPARSE_MAT_VEC,
            Vec::new(),
            vec![vals_b, cidx_b, rptr_b, ctx.upstream],
        );
        let g_vals = ctx.bwd.custom_op(
            SPARSE_VALUES_GRAD,
            Vec::new(),
            vec![cidx_b, rptr_b, ctx.upstream, x_bwd],
        );
        vec![(0, g_vals), (3, g_x)]
    }
}

#[cfg(feature = "cpu")]
pub(crate) struct SparseMatVecCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for SparseMatVecCpu {
    fn name(&self) -> &str {
        SPARSE_MAT_VEC
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let values = inputs[0].expect_f64("mat_vec values")?;
        let col_idx = inputs[1].expect_i32("mat_vec col_idx")?;
        let row_ptr = inputs[2].expect_i32("mat_vec row_ptr")?;
        let x = inputs[3].expect_f64("mat_vec x")?;
        let out = output.expect_f64_mut("mat_vec y")?;
        algos::mat_vec(values, col_idx, row_ptr, x, out)
    }
}

// ── Conjugate Gradient ────────────────────────────────────────────
