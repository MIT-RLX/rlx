// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `eigh_backward` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]
#![allow(unused_imports)]

use rlx_ir::{DType, OpExtension, Shape};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef};

// ── Op names ─────────────────────────────────────────────────────

use super::*;

pub(crate) struct EighBackwardExt;

impl OpExtension for EighBackwardExt {
    fn name(&self) -> &str {
        LINALG_EIGH_BACKWARD
    }
    fn num_inputs(&self) -> usize {
        4
    } // λ, V_flat, dL/dλ, dL/dV_flat
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        // dL/dA has shape [n, n] where n = inputs[0].len.
        let n = inputs[0]
            .num_elements()
            .expect("eigh_bwd: λ must have static shape");
        Shape::new(&[n, n], DType::F64)
    }
}

#[cfg(feature = "cpu")]
pub(crate) struct EighBackwardCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for EighBackwardCpu {
    fn name(&self) -> &str {
        LINALG_EIGH_BACKWARD
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let lambda = inputs[0].expect_f64("eigh_bwd λ")?;
        let v_flat = inputs[1].expect_f64("eigh_bwd V")?;
        let dl_dl = inputs[2].expect_f64("eigh_bwd dL/dλ")?;
        let dl_dv = inputs[3].expect_f64("eigh_bwd dL/dV")?;
        let out = output.expect_f64_mut("eigh_bwd out")?;
        let n = lambda.len();
        algos::eigh_backward(lambda, v_flat, dl_dl, dl_dv, n, out)
    }
}
