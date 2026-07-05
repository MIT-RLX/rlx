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

//! `lstsq` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]
#![allow(unused_imports)]

use rlx_ir::{DType, Node, NodeId, OpExtension, Shape, VjpContext};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef};

// ── Op names ─────────────────────────────────────────────────────

use super::*;

pub(crate) struct LstsqExt;

impl OpExtension for LstsqExt {
    fn name(&self) -> &str {
        LINALG_LSTSQ
    }
    fn num_inputs(&self) -> usize {
        2
    } // A (m×n), b (m)
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        let a = inputs[0];
        let b = inputs[1];
        assert_eq!(a.dtype(), DType::F64, "lstsq: A must be F64");
        assert_eq!(a.rank(), 2, "lstsq: A must be 2D");
        assert_eq!(b.rank(), 1, "lstsq: b must be 1D");
        let n = match a.dim(1) {
            rlx_ir::Dim::Static(v) => v,
            _ => panic!("lstsq: dynamic dim"),
        };
        Shape::new(&[n], DType::F64)
    }
    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        let a_bwd = ctx.fwd_map[&node.inputs[0]];
        let b_bwd = ctx.fwd_map[&node.inputs[1]];
        let x_bwd = ctx.fwd_map[&node.id];
        let g_a = ctx.bwd.custom_op(
            LINALG_LSTSQ_BACKWARD_A,
            Vec::new(),
            vec![a_bwd, x_bwd, b_bwd, ctx.upstream],
        );
        let g_b = ctx.bwd.custom_op(
            LINALG_LSTSQ_BACKWARD_B,
            Vec::new(),
            vec![a_bwd, ctx.upstream],
        );
        vec![(0, g_a), (1, g_b)]
    }
}

#[cfg(feature = "cpu")]
pub(crate) struct LstsqCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for LstsqCpu {
    fn name(&self) -> &str {
        LINALG_LSTSQ
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("lstsq A")?;
        let b = inputs[1].expect_f64("lstsq b")?;
        let out = output.expect_f64_mut("lstsq out")?;
        let m = b.len();
        let n = out.len();
        if a.len() != m * n {
            return Err(format!("lstsq: A len {} != m·n = {}·{}", a.len(), m, n));
        }
        algos::lstsq(a, b, m, n, out)
    }
}
