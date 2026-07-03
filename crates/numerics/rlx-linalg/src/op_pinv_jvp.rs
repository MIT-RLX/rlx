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

//! `pinv_jvp` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]


#![allow(unused_imports)]

use rlx_ir::{DType, OpExtension, Shape};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef};

// ── Op names ─────────────────────────────────────────────────────

use super::*;

pub(crate) struct PinvJvpExt;


impl OpExtension for PinvJvpExt {
    fn name(&self) -> &str {
        LINALG_PINV_JVP
    }
    fn num_inputs(&self) -> usize {
        2
    } // A, dA
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        // Output = pinv shape [n, m] (transpose of A's [m, n]).
        let a = inputs[0];
        let m = match a.dim(0) {
            rlx_ir::Dim::Static(v) => v,
            _ => panic!("pinv_jvp: dynamic dim"),
        };
        let n = match a.dim(1) {
            rlx_ir::Dim::Static(v) => v,
            _ => panic!("pinv_jvp: dynamic dim"),
        };
        Shape::new(&[n, m], DType::F64)
    }
}


#[cfg(feature = "cpu")]
pub(crate) struct PinvJvpCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for PinvJvpCpu {
    fn name(&self) -> &str {
        LINALG_PINV_JVP
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("pinv_jvp A")?;
        let da = inputs[1].expect_f64("pinv_jvp dA")?;
        let out = output.expect_f64_mut("pinv_jvp out")?;
        // Recover m from attrs (encoded by pinv builder).
        if attrs.len() < 4 {
            return Err("pinv_jvp: attrs must encode m (u32 LE)".into());
        }
        let m = u32::from_le_bytes(attrs[..4].try_into().unwrap()) as usize;
        if m == 0 || a.len() % m != 0 {
            return Err(format!("pinv_jvp: bad attrs m={m}"));
        }
        let n = a.len() / m;
        algos::pinv_jvp(a, da, m, n, out)
    }
}

