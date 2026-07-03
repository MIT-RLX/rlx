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

//! `cholesky_jvp` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]


#![allow(unused_imports)]

use rlx_ir::infer::GraphExt;
use rlx_ir::{OpExtension, Shape};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef};

// ── Op names ─────────────────────────────────────────────────────

use super::*;

pub(crate) struct CholeskyJvpExt;


impl OpExtension for CholeskyJvpExt {
    fn name(&self) -> &str {
        LINALG_CHOLESKY_JVP
    }
    fn num_inputs(&self) -> usize {
        2
    } // L, dA
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        inputs[0].clone()
    }
}


#[cfg(feature = "cpu")]
pub(crate) struct CholeskyJvpCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for CholeskyJvpCpu {
    fn name(&self) -> &str {
        LINALG_CHOLESKY_JVP
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let l = inputs[0].expect_f64("chol_jvp L")?;
        let da = inputs[1].expect_f64("chol_jvp dA")?;
        let out = output.expect_f64_mut("chol_jvp out")?;
        let lower = attrs.first().copied().unwrap_or(1) != 0;
        let n_sq = l.len();
        let n = (n_sq as f64).sqrt() as usize;
        if n * n != n_sq {
            return Err(format!("chol_jvp: n²={n_sq}"));
        }
        algos::cholesky_jvp(l, da, n, lower, out)
    }
}

// ── Backward ops ──────────────────────────────────────────────────

