// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `solve_triangular` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]
#![allow(unused_imports)]

use rlx_ir::infer::GraphExt;
use rlx_ir::{Node, NodeId, OpExtension, Shape, VjpContext};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef};

// ── Op names ─────────────────────────────────────────────────────

use super::*;

pub(crate) struct SolveTriangularExt;

impl OpExtension for SolveTriangularExt {
    fn name(&self) -> &str {
        LINALG_SOLVE_TRIANGULAR
    }
    fn num_inputs(&self) -> usize {
        2
    } // A (n×n), B (n×nrhs)
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        let b = inputs[1];
        b.clone()
    }
    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // y = solve(op(A), B). Closed form:
        //   dL/dB = (op(A))⁻ᵀ · upstream
        // = solve(op(A), upstream, transpose_flag_flipped). Only dL/dB
        // is implemented in v1; dL/dA is a triangular outer-product
        // gather (mirrors sparse-LU's values gradient — mechanical
        // follow-up).
        let attrs = match &node.op {
            rlx_ir::Op::Custom { attrs, .. } => attrs.clone(),
            _ => return vec![],
        };
        let lower = attrs.first().copied().unwrap_or(1) != 0;
        let transpose_a = attrs.get(1).copied().unwrap_or(0) != 0;
        let a_bwd = ctx.fwd_map[&node.inputs[0]];
        let new_attrs = vec![
            if lower { 1u8 } else { 0 },
            if !transpose_a { 1 } else { 0 },
        ];
        let g_b = ctx.bwd.custom_op(
            LINALG_SOLVE_TRIANGULAR,
            new_attrs,
            vec![a_bwd, ctx.upstream],
        );
        vec![(1, g_b)]
    }
    fn jvp(&self, node: &Node, ctx: &mut rlx_ir::JvpContext) -> Option<NodeId> {
        // y = solve(A, B); dy = solve(A, dB - dA·y).
        let attrs = match &node.op {
            rlx_ir::Op::Custom { attrs, .. } => attrs.clone(),
            _ => return None,
        };
        let a = ctx.fwd_map[&node.inputs[0]];
        let y = ctx.fwd_map[&node.id];
        let y_shape = ctx.bwd.shape(y).clone();
        let rhs = match (ctx.tangents[0], ctx.tangents[1]) {
            (Some(t_a), Some(t_b)) => {
                let prod = ctx.bwd.matmul(t_a, y, y_shape.clone());
                ctx.bwd
                    .binary(rlx_ir::op::BinaryOp::Sub, t_b, prod, y_shape.clone())
            }
            (Some(t_a), None) => {
                let prod = ctx.bwd.matmul(t_a, y, y_shape.clone());
                ctx.bwd
                    .activation(rlx_ir::op::Activation::Neg, prod, y_shape.clone())
            }
            (None, Some(t_b)) => t_b,
            (None, None) => return None,
        };
        Some(
            ctx.bwd
                .custom_op(LINALG_SOLVE_TRIANGULAR, attrs, vec![a, rhs]),
        )
    }
}

#[cfg(feature = "cpu")]
pub(crate) struct SolveTriangularCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for SolveTriangularCpu {
    fn name(&self) -> &str {
        LINALG_SOLVE_TRIANGULAR
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("solve_triangular A")?;
        let b = inputs[1].expect_f64("solve_triangular B")?;
        let out = output.expect_f64_mut("solve_triangular out")?;
        let lower = attrs.first().copied().unwrap_or(1) != 0;
        let transpose_a = attrs.get(1).copied().unwrap_or(0) != 0;
        let n_sq = a.len();
        let n = (n_sq as f64).sqrt() as usize;
        if n * n != n_sq {
            return Err(format!("solve_triangular: A length {n_sq} not n²"));
        }
        let nrhs = b.len() / n;
        algos::solve_triangular(a, b, n, nrhs, lower, transpose_a, out)
    }
}

// ── Symmetric Eigendecomposition ─────────────────────────────────
