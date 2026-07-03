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

//! `lstsq_backward_b` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]


#![allow(unused_imports)]

use rlx_ir::{DType, OpExtension, Shape};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef};

// ── Op names ─────────────────────────────────────────────────────

use super::*;

pub(crate) struct LstsqBackwardBExt;


impl OpExtension for LstsqBackwardBExt {
    fn name(&self) -> &str {
        LINALG_LSTSQ_BACKWARD_B
    }
    fn num_inputs(&self) -> usize {
        2
    } // A, dL/dx
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        let a = inputs[0];
        let m = match a.dim(0) {
            rlx_ir::Dim::Static(v) => v,
            _ => panic!("lstsq_bwd_b: dynamic dim"),
        };
        Shape::new(&[m], DType::F64)
    }
}


#[cfg(feature = "cpu")]
pub(crate) struct LstsqBackwardBCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for LstsqBackwardBCpu {
    fn name(&self) -> &str {
        LINALG_LSTSQ_BACKWARD_B
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("lstsq_bwd_b A")?;
        let dl_dx = inputs[1].expect_f64("lstsq_bwd_b dL/dx")?;
        let out = output.expect_f64_mut("lstsq_bwd_b out")?;
        let m = out.len();
        let n = dl_dx.len();
        if a.len() != m * n {
            return Err(format!("lstsq_bwd_b: A {} ≠ m·n = {}·{}", a.len(), m, n));
        }
        algos::lstsq_backward_b(a, dl_dx, m, n, out)
    }
}

// ── Cholesky JVP ──────────────────────────────────────────────────

