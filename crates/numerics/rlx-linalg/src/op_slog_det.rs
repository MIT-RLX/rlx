// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `slog_det` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]
#![allow(unused_imports)]

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Node, NodeId, OpExtension, Shape, VjpContext};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef};

// ── Op names ─────────────────────────────────────────────────────

use super::*;

pub(crate) struct SlogDetExt;

impl OpExtension for SlogDetExt {
    fn name(&self) -> &str {
        LINALG_SLOGDET
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        let a = inputs[0];
        assert_eq!(a.dtype(), DType::F64, "slogdet: A must be F64");
        assert_eq!(a.rank(), 2, "slogdet: A must be 2D");
        Shape::new(&[2], DType::F64)
    }
    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // upstream is dL/d(packed[2]). Extract index-1 (logabsdet grad)
        // via Narrow; sign component is non-differentiable.
        let a_bwd = ctx.fwd_map[&node.inputs[0]];
        let dl_d_logabs = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: 1,
                len: 1,
            },
            vec![ctx.upstream],
            Shape::new(&[1], DType::F64),
        );
        let g_a = ctx.bwd.custom_op(
            LINALG_SLOGDET_BACKWARD,
            Vec::new(),
            vec![a_bwd, dl_d_logabs],
        );
        vec![(0, g_a)]
    }
    fn jvp(&self, node: &Node, ctx: &mut rlx_ir::JvpContext) -> Option<NodeId> {
        // Output is packed [sign, log|det|]. Sign is non-differentiable
        // (zero tangent); log|det| tangent is tr(A⁻¹·dA) like logdet.
        let t_a = ctx.tangents[0]?;
        let a = ctx.fwd_map[&node.inputs[0]];
        let n = match ctx.bwd.shape(a).dim(0) {
            rlx_ir::Dim::Static(v) => v,
            _ => return None,
        };
        let x = ctx.bwd.dense_solve(a, t_a, Shape::new(&[n, n], DType::F64));
        let d = ctx.bwd.custom_op(LINALG_DIAG_EXTRACT, Vec::new(), vec![x]);
        let t_logabs = ctx.bwd.sum(d, vec![0], true); // [1]
        let zero = ctx.bwd.add_node(
            rlx_ir::Op::Constant {
                data: 0.0_f64.to_le_bytes().to_vec(),
            },
            vec![],
            Shape::new(&[1], DType::F64),
        );
        Some(ctx.bwd.add_node(
            rlx_ir::Op::Concat { axis: 0 },
            vec![zero, t_logabs],
            Shape::new(&[2], DType::F64),
        ))
    }
}

#[cfg(feature = "cpu")]
pub(crate) struct SlogDetCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for SlogDetCpu {
    fn name(&self) -> &str {
        LINALG_SLOGDET
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("slogdet A")?;
        let out = output.expect_f64_mut("slogdet out")?;
        let n_sq = a.len();
        let n = (n_sq as f64).sqrt() as usize;
        if n * n != n_sq {
            return Err(format!("slogdet: A length {n_sq} not n²"));
        }
        algos::slogdet(a, n, out)
    }
}
