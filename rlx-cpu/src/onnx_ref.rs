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

//! Reference CPU kernels for generic `Op::Custom("onnx.*")` ops.

use std::sync::Arc;

use crate::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef, register_cpu_kernel};

const MOD: &str = "onnx.Mod";
const IS_NAN: &str = "onnx.IsNaN";
const CONCAT_FROM_SEQUENCE: &str = "onnx.ConcatFromSequence";

struct ModKernel;
struct IsNaNKernel;

impl CpuKernel for ModKernel {
    fn name(&self) -> &str {
        MOD
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs
            .first()
            .ok_or("onnx.Mod: missing a")?
            .expect_f32("a")?;
        let b = inputs
            .get(1)
            .ok_or("onnx.Mod: missing b")?
            .expect_f32("b")?;
        let out = output.expect_f32_mut("out")?;
        let fmod = attrs.first().copied().unwrap_or(0) != 0;
        let n = a.len().min(b.len()).min(out.len());
        for i in 0..n {
            out[i] = if fmod {
                a[i] % b[i]
            } else {
                let q = (a[i] / b[i]).trunc();
                a[i] - q * b[i]
            };
        }
        Ok(())
    }
}

impl CpuKernel for IsNaNKernel {
    fn name(&self) -> &str {
        IS_NAN
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let x = inputs
            .first()
            .ok_or("onnx.IsNaN: missing input")?
            .expect_f32("x")?;
        let out = output.expect_bool_mut("out")?;
        let n = x.len().min(out.len());
        for i in 0..n {
            out[i] = u8::from(x[i].is_nan());
        }
        Ok(())
    }
}

/// Register generic ONNX reference kernels (Mod, IsNaN, …).
pub fn register_onnx_reference_kernels() {
    register_cpu_kernel(Arc::new(ModKernel));
    register_cpu_kernel(Arc::new(IsNaNKernel));
}

pub fn onnx_concat_from_sequence_name() -> &'static str {
    CONCAT_FROM_SEQUENCE
}
