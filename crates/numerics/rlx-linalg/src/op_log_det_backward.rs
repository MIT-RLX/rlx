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

//! `log_det_backward` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]


#![allow(unused_imports)]

use rlx_ir::infer::GraphExt;
use rlx_ir::{OpExtension, Shape};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef};

// ── Op names ─────────────────────────────────────────────────────

use super::*;

pub(crate) struct LogDetBackwardExt;


impl OpExtension for LogDetBackwardExt {
    fn name(&self) -> &str {
        LINALG_LOGDET_BACKWARD
    }
    fn num_inputs(&self) -> usize {
        2
    } // A, dL/d(logdet) (scalar)
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        inputs[0].clone()
    }
}


#[cfg(feature = "cpu")]
pub(crate) struct LogDetBackwardCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for LogDetBackwardCpu {
    fn name(&self) -> &str {
        LINALG_LOGDET_BACKWARD
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("logdet_bwd A")?;
        let dl_d_lg = inputs[1].expect_f64("logdet_bwd dL/d(logdet)")?;
        let out = output.expect_f64_mut("logdet_bwd out")?;
        if dl_d_lg.len() != 1 {
            return Err(format!(
                "logdet_bwd: dL/d(logdet) must be scalar, got len {}",
                dl_d_lg.len()
            ));
        }
        let n_sq = a.len();
        let n = (n_sq as f64).sqrt() as usize;
        if n * n != n_sq {
            return Err(format!("logdet_bwd: A length {n_sq} not n²"));
        }
        algos::logdet_backward(a, dl_d_lg[0], n, out)
    }
}

// ── Public builder API ───────────────────────────────────────────

