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

//! `log_det` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]


#![allow(unused_imports)]

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Node, NodeId, OpExtension, Shape, VjpContext};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef};

// ── Op names ─────────────────────────────────────────────────────

use super::*;

pub(crate) struct LogDetExt;


impl OpExtension for LogDetExt {
    fn name(&self) -> &str {
        LINALG_LOGDET
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        let a = inputs[0];
        assert_eq!(a.dtype(), DType::F64, "logdet: A must be F64");
        assert_eq!(a.rank(), 2, "logdet: A must be 2D");
        // Scalar output.
        Shape::new(&[1], DType::F64)
    }
    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // dL/dA = dL/d(logdet) · A⁻¹  via the logdet_backward kernel.
        let a_bwd = ctx.fwd_map[&node.inputs[0]];
        let g_a = ctx.bwd.custom_op(
            LINALG_LOGDET_BACKWARD,
            Vec::new(),
            vec![a_bwd, ctx.upstream],
        );
        vec![(0, g_a)]
    }
    fn jvp(&self, node: &Node, ctx: &mut rlx_ir::JvpContext) -> Option<NodeId> {
        // d/dt log|det(A(t))| = tr(A⁻¹·dA) = tr(solve(A, dA)).
        let t_a = ctx.tangents[0]?;
        let a = ctx.fwd_map[&node.inputs[0]];
        let n = match ctx.bwd.shape(a).dim(0) {
            rlx_ir::Dim::Static(v) => v,
            _ => return None,
        };
        let x = ctx.bwd.dense_solve(a, t_a, Shape::new(&[n, n], DType::F64));
        let d = ctx.bwd.custom_op(LINALG_DIAG_EXTRACT, Vec::new(), vec![x]);
        // forward output is shape [1] (length-1 tensor), so keep_dim=true.
        Some(ctx.bwd.sum(d, vec![0], true))
    }
}


#[cfg(feature = "cpu")]
pub(crate) struct LogDetCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for LogDetCpu {
    fn name(&self) -> &str {
        LINALG_LOGDET
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("logdet A")?;
        let out = output.expect_f64_mut("logdet out")?;
        let n_sq = a.len();
        let n = (n_sq as f64).sqrt() as usize;
        if n * n != n_sq {
            return Err(format!("logdet: A length {n_sq} not n²"));
        }
        algos::logdet(a, n, out)
    }
}

// ── SlogDet ───────────────────────────────────────────────────────

