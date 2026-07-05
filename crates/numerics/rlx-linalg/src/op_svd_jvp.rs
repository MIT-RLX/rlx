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

//! `svd_jvp` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]
#![allow(unused_imports)]

use rlx_ir::{DType, OpExtension, Shape};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef};

// ── Op names ─────────────────────────────────────────────────────

use super::*;

pub(crate) struct SvdJvpExt;

impl OpExtension for SvdJvpExt {
    fn name(&self) -> &str {
        LINALG_SVD_JVP
    }
    fn num_inputs(&self) -> usize {
        4
    } // U_flat, s, Vt_flat, dA_flat
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        let u_len = inputs[0].num_elements().expect("svd_jvp: dynamic shape");
        let s_len = inputs[1].num_elements().expect("svd_jvp: dynamic shape");
        let vt_len = inputs[2].num_elements().expect("svd_jvp: dynamic shape");
        Shape::new(&[u_len + s_len + vt_len], DType::F64)
    }
}

#[cfg(feature = "cpu")]
pub(crate) struct SvdJvpCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for SvdJvpCpu {
    fn name(&self) -> &str {
        LINALG_SVD_JVP
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _: &[u8],
    ) -> Result<(), String> {
        let u = inputs[0].expect_f64("svd_jvp U")?;
        let s = inputs[1].expect_f64("svd_jvp s")?;
        let vt = inputs[2].expect_f64("svd_jvp Vt")?;
        let da = inputs[3].expect_f64("svd_jvp dA")?;
        let out = output.expect_f64_mut("svd_jvp out")?;
        let k = s.len();
        let m = u.len() / k;
        let n = vt.len() / k;
        algos::svd_jvp(u, s, vt, da, m, n, out)
    }
}

// ── Pinv JVP ──────────────────────────────────────────────────────
