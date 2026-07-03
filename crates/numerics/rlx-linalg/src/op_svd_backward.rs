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

//! `svd_backward` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]


#![allow(unused_imports)]

use rlx_ir::{DType, OpExtension, Shape};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef};

// ── Op names ─────────────────────────────────────────────────────

use super::*;

pub(crate) struct SvdBackwardExt;


impl OpExtension for SvdBackwardExt {
    fn name(&self) -> &str {
        LINALG_SVD_BACKWARD
    }
    fn num_inputs(&self) -> usize {
        6
    } // U, s, Vt, dU, ds, dVt
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        // dL/dA shape is [m, n]. Recover from input lengths:
        //   U: m·k flat, s: k flat, Vt: k·n flat. k = s.len.
        let k = inputs[1].num_elements().expect("svd_bwd: dynamic shape");
        let u_len = inputs[0].num_elements().expect("svd_bwd: dynamic shape");
        let vt_len = inputs[2].num_elements().expect("svd_bwd: dynamic shape");
        let m = u_len / k;
        let n = vt_len / k;
        Shape::new(&[m, n], DType::F64)
    }
}


#[cfg(feature = "cpu")]
pub(crate) struct SvdBackwardCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for SvdBackwardCpu {
    fn name(&self) -> &str {
        LINALG_SVD_BACKWARD
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let u = inputs[0].expect_f64("svd_bwd U")?;
        let s = inputs[1].expect_f64("svd_bwd s")?;
        let vt = inputs[2].expect_f64("svd_bwd Vt")?;
        let dl_du = inputs[3].expect_f64("svd_bwd dL/dU")?;
        let dl_ds = inputs[4].expect_f64("svd_bwd dL/ds")?;
        let dl_dvt = inputs[5].expect_f64("svd_bwd dL/dVt")?;
        let out = output.expect_f64_mut("svd_bwd out")?;
        let k = s.len();
        let m = u.len() / k;
        let n = vt.len() / k;
        algos::svd_backward(u, s, vt, dl_du, dl_ds, dl_dvt, m, n, out)
    }
}

