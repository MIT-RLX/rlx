// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `pinv_backward` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]
#![allow(unused_imports)]

use rlx_ir::{OpExtension, Shape};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef};

// ── Op names ─────────────────────────────────────────────────────

use super::*;

pub(crate) struct PinvBackwardExt;

impl OpExtension for PinvBackwardExt {
    fn name(&self) -> &str {
        LINALG_PINV_BACKWARD
    }
    fn num_inputs(&self) -> usize {
        3
    } // A, Y, dL/dY
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        inputs[0].clone() // dL/dA shape == A shape
    }
}

#[cfg(feature = "cpu")]
pub(crate) struct PinvBackwardCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for PinvBackwardCpu {
    fn name(&self) -> &str {
        LINALG_PINV_BACKWARD
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("pinv_bwd A")?;
        let y = inputs[1].expect_f64("pinv_bwd Y")?;
        let g = inputs[2].expect_f64("pinv_bwd dL/dY")?;
        let out = output.expect_f64_mut("pinv_bwd out")?;
        // Recover m, n: a.len() = m·n, y.len() = n·m. Need m alone.
        // Y has shape n×m and a has shape m×n; out has shape m×n.
        // Use out.len() = m·n, and recover m via gcd? Better: encode
        // attrs again. v1: take attrs[0..4] as m (u32 LE).
        let attrs = _attrs;
        if attrs.len() < 4 {
            return Err("pinv_bwd: attrs must encode m (u32 LE)".into());
        }
        let m = u32::from_le_bytes(attrs[..4].try_into().unwrap()) as usize;
        if m == 0 || a.len() % m != 0 {
            return Err(format!("pinv_bwd: bad attrs m={m}"));
        }
        let n = a.len() / m;
        algos::pinv_backward(a, y, g, m, n, out)
    }
}

// ── Lstsq ─────────────────────────────────────────────────────────
