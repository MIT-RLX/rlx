// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `lu` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]
#![allow(unused_imports)]

use std::sync::Arc;

use rlx_ir::{DType, Graph, Node, NodeId, Op, OpExtension, Shape, VjpContext, register_op};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef, register_cpu_kernel};

// ── Op names (stable strings; downstream callers use these to look
//    up the registered op or build `Op::Custom` directly) ─────────

use super::*;

pub(crate) struct SparseLuExt;

impl OpExtension for SparseLuExt {
    fn name(&self) -> &str {
        SPARSE_LU_SOLVE
    }
    fn num_inputs(&self) -> usize {
        4
    }

    fn infer_shape(&self, inputs: &[&Shape], _attrs: &[u8]) -> Shape {
        let values = inputs[0];
        let col_idx = inputs[1];
        let row_ptr = inputs[2];
        let b = inputs[3];
        assert_eq!(values.dtype(), DType::F64, "sparse_lu: values must be F64");
        assert_eq!(
            col_idx.dtype(),
            DType::I32,
            "sparse_lu: col_idx must be I32"
        );
        assert_eq!(
            row_ptr.dtype(),
            DType::I32,
            "sparse_lu: row_ptr must be I32"
        );
        assert_eq!(b.dtype(), DType::F64, "sparse_lu: b must be F64");
        b.clone()
    }

    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // y = solve(A, b). Closed-form gradients:
        //   dL/db = solve(Aᵀ, dL/dy)         [v1 symmetric → reuse A]
        //   dL/dvalues[k] = -dL/db[row(k)] · y_fwd[col(k)]   gathered at nonzero k
        let vals_b = ctx.fwd_map[&node.inputs[0]];
        let cidx_b = ctx.fwd_map[&node.inputs[1]];
        let rptr_b = ctx.fwd_map[&node.inputs[2]];

        let g_b = ctx.bwd.custom_op(
            SPARSE_LU_SOLVE,
            Vec::new(),
            vec![vals_b, cidx_b, rptr_b, ctx.upstream],
        );

        // y is the forward solve output, mirrored in the bwd graph
        // by `grad_with_loss`'s up-front fwd→bwd Node copy. Look it up
        // via ctx.fwd_map[&node.id].
        let y_fwd = ctx.fwd_map[&node.id];
        let raw_grad = ctx.bwd.custom_op(
            SPARSE_VALUES_GRAD,
            Vec::new(),
            vec![cidx_b, rptr_b, g_b, y_fwd],
        );
        // The values gradient is `-dL/db ⊗ y`, so negate the gather.
        let raw_shape = ctx.bwd.node(raw_grad).shape.clone();
        let g_vals = ctx
            .bwd
            .activation(rlx_ir::op::Activation::Neg, raw_grad, raw_shape);

        vec![(0, g_vals), (3, g_b)]
    }
}

#[cfg(feature = "cpu")]
pub(crate) struct SparseLuCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for SparseLuCpu {
    fn name(&self) -> &str {
        SPARSE_LU_SOLVE
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let values = inputs[0].expect_f64("sparse_lu values")?;
        let col_idx = inputs[1].expect_i32("sparse_lu col_idx")?;
        let row_ptr = inputs[2].expect_i32("sparse_lu row_ptr")?;
        let b = inputs[3].expect_f64("sparse_lu b")?;
        let out = output.expect_f64_mut("sparse_lu output")?;
        algos::lu_solve(values, col_idx, row_ptr, b, out)
    }
}

// ── Sparse Mat-Vec ────────────────────────────────────────────────
