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

//! `expm` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]


#![allow(unused_imports)]

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Node, NodeId, OpExtension, Shape, VjpContext};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef};

// ── Op names ─────────────────────────────────────────────────────

use super::*;

pub(crate) struct ExpmExt;


impl OpExtension for ExpmExt {
    fn name(&self) -> &str {
        LINALG_EXPM
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        let a = inputs[0];
        assert_eq!(a.dtype(), DType::F64, "expm: A must be F64");
        assert_eq!(a.rank(), 2, "expm: A must be 2D");
        a.clone()
    }
    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        let a_bwd = ctx.fwd_map[&node.inputs[0]];
        let g_a = ctx
            .bwd
            .custom_op(LINALG_EXPM_BACKWARD, Vec::new(), vec![a_bwd, ctx.upstream]);
        vec![(0, g_a)]
    }
    fn jvp(&self, _node: &Node, ctx: &mut rlx_ir::JvpContext) -> Option<NodeId> {
        // Frechet derivative via augmented-matrix kernel.
        let t_a = ctx.tangents[0]?;
        let a = ctx.fwd_map[&_node.inputs[0]];
        Some(ctx.bwd.custom_op(LINALG_EXPM_JVP, Vec::new(), vec![a, t_a]))
    }
}


#[cfg(feature = "cpu")]
pub(crate) struct ExpmCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for ExpmCpu {
    fn name(&self) -> &str {
        LINALG_EXPM
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("expm A")?;
        let out = output.expect_f64_mut("expm out")?;
        let n_sq = a.len();
        let n = (n_sq as f64).sqrt() as usize;
        if n * n != n_sq {
            return Err(format!("expm: A length {n_sq} not n²"));
        }
        algos::expm(a, n, out)
    }
}

