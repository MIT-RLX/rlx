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

//! `qr_jvp` op registration — split from `lib.rs` (see `register()`).

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]


#![allow(unused_imports)]

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, OpExtension, Shape};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef};

// ── Op names ─────────────────────────────────────────────────────

use super::*;

pub(crate) struct QrJvpExt;


impl OpExtension for QrJvpExt {
    fn name(&self) -> &str {
        LINALG_QR_JVP
    }
    fn num_inputs(&self) -> usize {
        3
    } // Q_flat, R_flat, dA_flat
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        // Output packed [dQ (m·k), dR (k·n)] — same length as Q + R together.
        let q_len = inputs[0].num_elements().expect("qr_jvp: dynamic shape");
        let r_len = inputs[1].num_elements().expect("qr_jvp: dynamic shape");
        Shape::new(&[q_len + r_len], DType::F64)
    }
}


#[cfg(feature = "cpu")]
pub(crate) struct QrJvpCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for QrJvpCpu {
    fn name(&self) -> &str {
        LINALG_QR_JVP
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _: &[u8],
    ) -> Result<(), String> {
        let q = inputs[0].expect_f64("qr_jvp Q")?;
        let r = inputs[1].expect_f64("qr_jvp R")?;
        let da = inputs[2].expect_f64("qr_jvp dA")?;
        let out = output.expect_f64_mut("qr_jvp out")?;
        let r_len = r.len();
        let n = (r_len as f64).sqrt() as usize;
        if n * n != r_len {
            return Err(format!(
                "qr_jvp: R must be square (m≥n thin QR), got len {r_len}"
            ));
        }
        let m = q.len() / n;
        algos::qr_jvp(q, r, da, m, n, out)
    }
}

// ── SVD JVP ───────────────────────────────────────────────────────

