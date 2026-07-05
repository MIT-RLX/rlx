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

//! `expm_jvp` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]
#![allow(unused_imports)]

use rlx_ir::infer::GraphExt;
use rlx_ir::{OpExtension, Shape};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef};

// ── Op names ─────────────────────────────────────────────────────

use super::*;

pub(crate) struct ExpmJvpExt;

impl OpExtension for ExpmJvpExt {
    fn name(&self) -> &str {
        LINALG_EXPM_JVP
    }
    fn num_inputs(&self) -> usize {
        2
    } // A, dA
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        inputs[0].clone()
    }
}

#[cfg(feature = "cpu")]
pub(crate) struct ExpmJvpCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for ExpmJvpCpu {
    fn name(&self) -> &str {
        LINALG_EXPM_JVP
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("expm_jvp A")?;
        let da = inputs[1].expect_f64("expm_jvp dA")?;
        let out = output.expect_f64_mut("expm_jvp out")?;
        let n_sq = a.len();
        let n = (n_sq as f64).sqrt() as usize;
        if n * n != n_sq {
            return Err(format!("expm_jvp: A length {n_sq} not n²"));
        }
        algos::expm_jvp(a, da, n, out)
    }
}

// ── QR JVP ────────────────────────────────────────────────────────
