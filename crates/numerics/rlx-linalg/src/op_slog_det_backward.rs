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

//! `slog_det_backward` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]


#![allow(unused_imports)]

use rlx_ir::infer::GraphExt;
use rlx_ir::{OpExtension, Shape};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef};

// ── Op names ─────────────────────────────────────────────────────

use super::*;

pub(crate) struct SlogDetBackwardExt;


impl OpExtension for SlogDetBackwardExt {
    fn name(&self) -> &str {
        LINALG_SLOGDET_BACKWARD
    }
    fn num_inputs(&self) -> usize {
        2
    } // A, dL/d(logabsdet)
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        inputs[0].clone()
    }
}


#[cfg(feature = "cpu")]
pub(crate) struct SlogDetBackwardCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for SlogDetBackwardCpu {
    fn name(&self) -> &str {
        LINALG_SLOGDET_BACKWARD
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("slogdet_bwd A")?;
        let dl_d = inputs[1].expect_f64("slogdet_bwd dL/d(logabsdet)")?;
        let out = output.expect_f64_mut("slogdet_bwd out")?;
        if dl_d.len() != 1 {
            return Err(format!(
                "slogdet_bwd: gradient must be scalar, got {}",
                dl_d.len()
            ));
        }
        let n_sq = a.len();
        let n = (n_sq as f64).sqrt() as usize;
        if n * n != n_sq {
            return Err(format!("slogdet_bwd: A length {n_sq} not n²"));
        }
        algos::slogdet_backward(a, dl_d[0], n, out)
    }
}

// ── Diag extract / set ────────────────────────────────────────────

