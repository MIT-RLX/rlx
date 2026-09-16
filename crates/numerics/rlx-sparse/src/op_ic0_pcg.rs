// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `ic0_pcg` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]
#![allow(unused_imports)]

use std::sync::Arc;

use rlx_ir::{DType, Graph, Node, NodeId, Op, OpExtension, Shape, VjpContext, register_op};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef, register_cpu_kernel};

// ── Op names (stable strings; downstream callers use these to look
//    up the registered op or build `Op::Custom` directly) ─────────

use super::*;

pub(crate) struct SparseIc0PcgExt;

impl OpExtension for SparseIc0PcgExt {
    fn name(&self) -> &str {
        SPARSE_IC0_PCG_SOLVE
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
            SPARSE_IC0_PCG_SOLVE,
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
pub(crate) struct SparseIc0PcgCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for SparseIc0PcgCpu {
    fn name(&self) -> &str {
        SPARSE_IC0_PCG_SOLVE
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let values = inputs[0].expect_f64("ic0_pcg values")?;
        let col_idx = inputs[1].expect_i32("ic0_pcg col_idx")?;
        let row_ptr = inputs[2].expect_i32("ic0_pcg row_ptr")?;
        let b = inputs[3].expect_f64("ic0_pcg b")?;
        let out = output.expect_f64_mut("ic0_pcg x")?;
        let (max_iter, tol) = decode_cg_attrs(attrs)?;
        algos::ic0_pcg(values, col_idx, row_ptr, b, out, max_iter, tol)
    }
}

// ── Sparse Cholesky (direct) ──────────────────────────────────────
