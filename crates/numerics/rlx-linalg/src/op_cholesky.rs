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

//! `cholesky` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]


#![allow(unused_imports)]

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Node, NodeId, OpExtension, Shape, VjpContext};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef};

// ── Op names ─────────────────────────────────────────────────────

use super::*;

pub(crate) struct CholeskyExt;


impl OpExtension for CholeskyExt {
    fn name(&self) -> &str {
        LINALG_CHOLESKY
    }
    fn num_inputs(&self) -> usize {
        1
    } // A: [n, n]
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        let a = inputs[0];
        assert_eq!(a.dtype(), DType::F64, "cholesky: A must be F64");
        assert_eq!(a.rank(), 2, "cholesky: A must be 2D");
        a.clone()
    }
    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // Closed-form Murray 2016 dL/dA via the cholesky_backward op.
        // Forward L = chol(A); upstream is dL/dL.
        let l_fwd = ctx.fwd_map[&node.id];
        let attrs = match &node.op {
            rlx_ir::Op::Custom { attrs, .. } => attrs.clone(),
            _ => Vec::new(),
        };
        let g_a = ctx
            .bwd
            .custom_op(LINALG_CHOLESKY_BACKWARD, attrs, vec![l_fwd, ctx.upstream]);
        vec![(0, g_a)]
    }
    fn jvp(&self, node: &Node, ctx: &mut rlx_ir::JvpContext) -> Option<NodeId> {
        // t_L = L · phi(L⁻¹·dA·L⁻ᵀ).
        let t_a = ctx.tangents[0]?;
        let l = ctx.fwd_map[&node.id];
        let attrs = match &node.op {
            rlx_ir::Op::Custom { attrs, .. } => attrs.clone(),
            _ => return None,
        };
        Some(ctx.bwd.custom_op(LINALG_CHOLESKY_JVP, attrs, vec![l, t_a]))
    }
}


#[cfg(feature = "cpu")]
pub(crate) struct CholeskyCpu;


#[cfg(feature = "cpu")]
impl CpuKernel for CholeskyCpu {
    fn name(&self) -> &str {
        LINALG_CHOLESKY
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("cholesky A")?;
        let out = output.expect_f64_mut("cholesky out")?;
        let lower = attrs.first().copied().unwrap_or(1) != 0;
        let n_sq = a.len();
        let n = (n_sq as f64).sqrt() as usize;
        if n * n != n_sq {
            return Err(format!("cholesky: A length {n_sq} not n²"));
        }
        algos::cholesky(a, n, lower, out)
    }
}

// ── Solve Triangular ─────────────────────────────────────────────

