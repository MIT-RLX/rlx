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

//! `lstsq_backward_a` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]
#![allow(unused_imports)]

use rlx_ir::{OpExtension, Shape};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef};

// ── Op names ─────────────────────────────────────────────────────

use super::*;

pub(crate) struct LstsqBackwardAExt;

impl OpExtension for LstsqBackwardAExt {
    fn name(&self) -> &str {
        LINALG_LSTSQ_BACKWARD_A
    }
    fn num_inputs(&self) -> usize {
        4
    } // A, x, b, dL/dx
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        inputs[0].clone()
    }
}

#[cfg(feature = "cpu")]
pub(crate) struct LstsqBackwardACpu;

#[cfg(feature = "cpu")]
impl CpuKernel for LstsqBackwardACpu {
    fn name(&self) -> &str {
        LINALG_LSTSQ_BACKWARD_A
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("lstsq_bwd_a A")?;
        let x = inputs[1].expect_f64("lstsq_bwd_a x")?;
        let b = inputs[2].expect_f64("lstsq_bwd_a b")?;
        let dl_dx = inputs[3].expect_f64("lstsq_bwd_a dL/dx")?;
        let out = output.expect_f64_mut("lstsq_bwd_a out")?;
        let m = b.len();
        let n = x.len();
        if a.len() != m * n || dl_dx.len() != n || out.len() != m * n {
            return Err(format!("lstsq_bwd_a: shape mismatch (m={m}, n={n})"));
        }
        algos::lstsq_backward_a(a, x, b, dl_dx, m, n, out)
    }
}
