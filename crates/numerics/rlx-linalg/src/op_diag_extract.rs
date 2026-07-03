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

//! `diag_extract` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]


#![allow(unused_imports)]

use rlx_ir::{DType, Node, NodeId, OpExtension, Shape, VjpContext};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef};

// ── Op names ─────────────────────────────────────────────────────

use super::*;

pub(crate) struct DiagExtractExt;


impl OpExtension for DiagExtractExt {
    fn name(&self) -> &str {
        LINALG_DIAG_EXTRACT
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        let a = inputs[0];
        assert_eq!(a.dtype(), DType::F64, "diag_extract: A must be F64");
        assert_eq!(a.rank(), 2, "diag_extract: A must be 2D");
        let n = match a.dim(0) {
            rlx_ir::Dim::Static(v) => v,
            _ => panic!("diag_extract: dynamic dim"),
        };
        Shape::new(&[n], DType::F64)
    }
    fn vjp(&self, _node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // dL/dA = diag_set(upstream).
        let g_a = ctx
            .bwd
            .custom_op(LINALG_DIAG_SET, Vec::new(), vec![ctx.upstream]);
        vec![(0, g_a)]
    }
    fn jvp(&self, _node: &Node, ctx: &mut rlx_ir::JvpContext) -> Option<NodeId> {
        // Linear op: dy = diag_extract(dA).
        let t_a = ctx.tangents[0]?;
        Some(
            ctx.bwd
                .custom_op(LINALG_DIAG_EXTRACT, Vec::new(), vec![t_a]),
        )
    }
    fn vmap(&self, node: &Node, ctx: &mut rlx_ir::VmapContext) -> Option<NodeId> {
        // Batched A: [B, n, n] → [B, n]. Unroll over the static batch
        // dim: per batch, Narrow + Reshape + diag_extract + Reshape;
        // then Concat along axis 0. Works for any dtype (no assumptions
        // about Gather's f32-only kernel).
        if !ctx.is_batched[0] {
            return None;
        }
        let n = match node.shape.dim(0) {
            rlx_ir::Dim::Static(n) => n,
            _ => return None,
        };
        let b = ctx.batch_size;
        let a_b = ctx.lifted_inputs[0];
        let mut per_batch: Vec<NodeId> = Vec::with_capacity(b);
        for k in 0..b {
            let slice = ctx.out.add_node(
                rlx_ir::Op::Narrow {
                    axis: 0,
                    start: k,
                    len: 1,
                },
                vec![a_b],
                Shape::new(&[1, n, n], DType::F64),
            );
            let mat = ctx.out.add_node(
                rlx_ir::Op::Reshape {
                    new_shape: vec![n as i64, n as i64],
                },
                vec![slice],
                Shape::new(&[n, n], DType::F64),
            );
            let d = ctx
                .out
                .custom_op(LINALG_DIAG_EXTRACT, Vec::new(), vec![mat]);
            let d_3d = ctx.out.add_node(
                rlx_ir::Op::Reshape {
                    new_shape: vec![1, n as i64],
                },
                vec![d],
                Shape::new(&[1, n], DType::F64),
            );
            per_batch.push(d_3d);
        }
        Some(ctx.out.add_node(
            rlx_ir::Op::Concat { axis: 0 },
            per_batch,
            Shape::new(&[b, n], DType::F64),
        ))
    }
}


#[cfg(feature = "cpu")]
pub(crate) struct DiagExtractCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for DiagExtractCpu {
    fn name(&self) -> &str {
        LINALG_DIAG_EXTRACT
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("diag_extract A")?;
        let out = output.expect_f64_mut("diag_extract out")?;
        let n = out.len();
        if a.len() != n * n {
            return Err(format!("diag_extract: A {} ≠ n²={}·{}", a.len(), n, n));
        }
        algos::diag_extract(a, n, out)
    }
}

