// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `eigh_jvp` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]
#![allow(unused_imports)]

use rlx_ir::{DType, OpExtension, Shape};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef};

// ── Op names ─────────────────────────────────────────────────────

use super::*;

pub(crate) struct EighJvpExt;

impl OpExtension for EighJvpExt {
    fn name(&self) -> &str {
        LINALG_EIGH_JVP
    }
    fn num_inputs(&self) -> usize {
        3
    } // λ, V_flat, dA_flat
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        let n = inputs[0]
            .num_elements()
            .expect("eigh_jvp: λ must have static shape");
        Shape::new(&[n + n * n], DType::F64)
    }
}

#[cfg(feature = "cpu")]
pub(crate) struct EighJvpCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for EighJvpCpu {
    fn name(&self) -> &str {
        LINALG_EIGH_JVP
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _: &[u8],
    ) -> Result<(), String> {
        let lambda = inputs[0].expect_f64("eigh_jvp λ")?;
        let v_flat = inputs[1].expect_f64("eigh_jvp V")?;
        let da_flat = inputs[2].expect_f64("eigh_jvp dA")?;
        let out = output.expect_f64_mut("eigh_jvp out")?;
        let n = lambda.len();
        algos::eigh_jvp(lambda, v_flat, da_flat, n, out)
    }
}
