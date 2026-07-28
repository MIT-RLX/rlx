// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cholesky_backward` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]
#![allow(unused_imports)]

use rlx_ir::infer::GraphExt;
use rlx_ir::{OpExtension, Shape};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef};

// ── Op names ─────────────────────────────────────────────────────

use super::*;

pub(crate) struct CholeskyBackwardExt;

impl OpExtension for CholeskyBackwardExt {
    fn name(&self) -> &str {
        LINALG_CHOLESKY_BACKWARD
    }
    fn num_inputs(&self) -> usize {
        2
    } // L, dL/dL
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        inputs[0].clone() // dL/dA has the same shape as L (= A)
    }
    // No second-order VJP (returns empty).
}

#[cfg(feature = "cpu")]
pub(crate) struct CholeskyBackwardCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for CholeskyBackwardCpu {
    fn name(&self) -> &str {
        LINALG_CHOLESKY_BACKWARD
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let l = inputs[0].expect_f64("chol_bwd L")?;
        let dl_dl = inputs[1].expect_f64("chol_bwd dL/dL")?;
        let out = output.expect_f64_mut("chol_bwd out")?;
        let lower = attrs.first().copied().unwrap_or(1) != 0;
        let n_sq = l.len();
        let n = (n_sq as f64).sqrt() as usize;
        if n * n != n_sq {
            return Err(format!("chol_bwd: n²={n_sq}"));
        }
        algos::cholesky_backward(l, dl_dl, n, lower, out)
    }
}
