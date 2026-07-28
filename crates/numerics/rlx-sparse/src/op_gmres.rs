// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `gmres` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]
#![allow(unused_imports)]

use std::sync::Arc;

use rlx_ir::{DType, Graph, Node, NodeId, Op, OpExtension, Shape, VjpContext, register_op};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef, register_cpu_kernel};

// ── Op names (stable strings; downstream callers use these to look
//    up the registered op or build `Op::Custom` directly) ─────────

use super::*;

pub(crate) struct SparseGmresExt;

impl OpExtension for SparseGmresExt {
    fn name(&self) -> &str {
        SPARSE_GMRES_SOLVE
    }
    fn num_inputs(&self) -> usize {
        7
    }
    fn infer_shape(&self, inputs: &[&Shape], _attrs: &[u8]) -> Shape {
        inputs[3].clone()
    }
    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // Same closed-form adjoint as LU general. Recurse into
        // gmres_solve against (Aᵀ, upstream) with the same attrs.
        let vals_a = ctx.fwd_map[&node.inputs[0]];
        let cidx_a = ctx.fwd_map[&node.inputs[1]];
        let rptr_a = ctx.fwd_map[&node.inputs[2]];
        let vals_at = ctx.fwd_map[&node.inputs[4]];
        let cidx_at = ctx.fwd_map[&node.inputs[5]];
        let rptr_at = ctx.fwd_map[&node.inputs[6]];
        let attrs = match &node.op {
            Op::Custom { attrs, .. } => attrs.clone(),
            _ => Vec::new(),
        };
        let g_b = ctx.bwd.custom_op(
            SPARSE_GMRES_SOLVE,
            attrs,
            vec![
                vals_at,
                cidx_at,
                rptr_at,
                ctx.upstream,
                vals_a,
                cidx_a,
                rptr_a,
            ],
        );
        let y_fwd = ctx.fwd_map[&node.id];
        let raw_grad = ctx.bwd.custom_op(
            SPARSE_VALUES_GRAD,
            Vec::new(),
            vec![cidx_a, rptr_a, g_b, y_fwd],
        );
        let raw_shape = ctx.bwd.node(raw_grad).shape.clone();
        let g_vals = ctx
            .bwd
            .activation(rlx_ir::op::Activation::Neg, raw_grad, raw_shape);
        vec![(0, g_vals), (3, g_b)]
    }
}

#[cfg(feature = "cpu")]
pub(crate) struct SparseGmresCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for SparseGmresCpu {
    fn name(&self) -> &str {
        SPARSE_GMRES_SOLVE
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let values = inputs[0].expect_f64("gmres values")?;
        let col_idx = inputs[1].expect_i32("gmres col_idx")?;
        let row_ptr = inputs[2].expect_i32("gmres row_ptr")?;
        let b = inputs[3].expect_f64("gmres b")?;
        let out = output.expect_f64_mut("gmres out")?;
        let (max_iter, tol) = decode_cg_attrs(attrs)?;
        algos::gmres_solve(values, col_idx, row_ptr, b, out, max_iter, tol)
    }
}

// ── Sparse Transpose Values ───────────────────────────────────────
