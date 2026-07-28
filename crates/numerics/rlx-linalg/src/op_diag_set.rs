// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `diag_set` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]
#![allow(unused_imports)]

use rlx_ir::{DType, Node, NodeId, OpExtension, Shape, VjpContext};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef};

// ── Op names ─────────────────────────────────────────────────────

use super::*;

pub(crate) struct DiagSetExt;

impl OpExtension for DiagSetExt {
    fn name(&self) -> &str {
        LINALG_DIAG_SET
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        let v = inputs[0];
        assert_eq!(v.dtype(), DType::F64, "diag_set: v must be F64");
        assert_eq!(v.rank(), 1, "diag_set: v must be 1D");
        let n = match v.dim(0) {
            rlx_ir::Dim::Static(v) => v,
            _ => panic!("diag_set: dynamic dim"),
        };
        Shape::new(&[n, n], DType::F64)
    }
    fn vjp(&self, _node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // dL/dv = diag_extract(upstream).
        let g_v = ctx
            .bwd
            .custom_op(LINALG_DIAG_EXTRACT, Vec::new(), vec![ctx.upstream]);
        vec![(0, g_v)]
    }
    fn jvp(&self, _node: &Node, ctx: &mut rlx_ir::JvpContext) -> Option<NodeId> {
        // Linear op: dM = diag_set(dv).
        let t_v = ctx.tangents[0]?;
        Some(ctx.bwd.custom_op(LINALG_DIAG_SET, Vec::new(), vec![t_v]))
    }
    fn vmap(&self, node: &Node, ctx: &mut rlx_ir::VmapContext) -> Option<NodeId> {
        // Batched v: [B, n] → [B, n, n]. Per-batch unroll mirroring
        // diag_extract's vmap.
        if !ctx.is_batched[0] {
            return None;
        }
        let n = match node.shape.dim(0) {
            rlx_ir::Dim::Static(n) => n,
            _ => return None,
        };
        let b = ctx.batch_size;
        let v_b = ctx.lifted_inputs[0];
        let mut per_batch: Vec<NodeId> = Vec::with_capacity(b);
        for k in 0..b {
            let slice = ctx.out.add_node(
                rlx_ir::Op::Narrow {
                    axis: 0,
                    start: k,
                    len: 1,
                },
                vec![v_b],
                Shape::new(&[1, n], DType::F64),
            );
            let vec1d = ctx.out.add_node(
                rlx_ir::Op::Reshape {
                    new_shape: vec![n as i64],
                },
                vec![slice],
                Shape::new(&[n], DType::F64),
            );
            let m = ctx.out.custom_op(LINALG_DIAG_SET, Vec::new(), vec![vec1d]);
            let m_3d = ctx.out.add_node(
                rlx_ir::Op::Reshape {
                    new_shape: vec![1, n as i64, n as i64],
                },
                vec![m],
                Shape::new(&[1, n, n], DType::F64),
            );
            per_batch.push(m_3d);
        }
        Some(ctx.out.add_node(
            rlx_ir::Op::Concat { axis: 0 },
            per_batch,
            Shape::new(&[b, n, n], DType::F64),
        ))
    }
}

#[cfg(feature = "cpu")]
pub(crate) struct DiagSetCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for DiagSetCpu {
    fn name(&self) -> &str {
        LINALG_DIAG_SET
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _: &[u8],
    ) -> Result<(), String> {
        let v = inputs[0].expect_f64("diag_set v")?;
        let out = output.expect_f64_mut("diag_set out")?;
        let n = v.len();
        if out.len() != n * n {
            return Err(format!("diag_set: out {} ≠ n²={}·{}", out.len(), n, n));
        }
        algos::diag_set(v, n, out)
    }
}

// ── Expm ──────────────────────────────────────────────────────────
