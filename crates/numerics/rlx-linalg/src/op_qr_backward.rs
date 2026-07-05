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

//! `qr_backward` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]
#![allow(unused_imports)]

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, OpExtension, Shape};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef};

// ── Op names ─────────────────────────────────────────────────────

use super::*;

pub(crate) struct QrBackwardExt;

impl OpExtension for QrBackwardExt {
    fn name(&self) -> &str {
        LINALG_QR_BACKWARD
    }
    fn num_inputs(&self) -> usize {
        4
    } // Q_flat, R_flat, dL/dQ_flat, dL/dR_flat
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        // Q has m·k flat; R has k·n flat. dL/dA has shape [m, n].
        // Recover m, n from input lengths assuming k = min(m, n).
        // For the test cases we know m ≥ n (k = n), so m = q_len / n
        // and n = r_len / k. Build shape [m, n].
        let q_len = inputs[0].num_elements().expect("qr_bwd: dynamic shape");
        let r_len = inputs[1].num_elements().expect("qr_bwd: dynamic shape");
        // For thin QR with k = n: q_len = m·n, r_len = n·n.
        let n = (r_len as f64).sqrt() as usize;
        assert_eq!(n * n, r_len, "qr_bwd: R must be square (m≥n thin QR)");
        let m = q_len / n;
        Shape::new(&[m, n], DType::F64)
    }
}

#[cfg(feature = "cpu")]
pub(crate) struct QrBackwardCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for QrBackwardCpu {
    fn name(&self) -> &str {
        LINALG_QR_BACKWARD
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let q = inputs[0].expect_f64("qr_bwd Q")?;
        let r = inputs[1].expect_f64("qr_bwd R")?;
        let dl_dq = inputs[2].expect_f64("qr_bwd dL/dQ")?;
        let dl_dr = inputs[3].expect_f64("qr_bwd dL/dR")?;
        let out = output.expect_f64_mut("qr_bwd out")?;
        let r_len = r.len();
        let n = (r_len as f64).sqrt() as usize;
        if n * n != r_len {
            return Err(format!("qr_bwd: R must be n²={r_len}"));
        }
        let m = q.len() / n;
        if m * n != q.len() {
            return Err(format!("qr_bwd: Q shape {}/n={n} not int", q.len()));
        }
        algos::qr_backward(q, r, dl_dq, dl_dr, m, n, out)
    }
}
