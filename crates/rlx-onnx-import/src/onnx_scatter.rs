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

//! Minimal ONNX `ScatterElements` CPU kernel for bundle compile-check.

use std::sync::Arc;

use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef, register_cpu_kernel};

pub const SCATTER_ELEMENTS: &str = "onnx.ScatterElements";

fn shape_usize(sh: &rlx_ir::Shape) -> Vec<usize> {
    sh.dims()
        .iter()
        .map(|d| match d {
            rlx_ir::Dim::Static(n) => *n,
            rlx_ir::Dim::Dynamic(_) => 0,
        })
        .collect()
}

fn shape_for_buffer(buf_len: usize, shape: &[usize]) -> Vec<usize> {
    let want: usize = shape.iter().product::<usize>().max(1);
    if want == buf_len {
        shape.to_vec()
    } else {
        vec![buf_len.max(1)]
    }
}

fn scatter_elements_f32(
    out: &mut [f32],
    data_shape: &[usize],
    indices: &[i64],
    updates: &[f32],
    axis: i32,
) {
    if out.is_empty() {
        return;
    }
    let data_shape = shape_for_buffer(out.len(), data_shape);
    if data_shape.len() == 1 {
        for (i, &idx) in indices.iter().enumerate() {
            let j = idx.max(0) as usize;
            if j < out.len() && i < updates.len() {
                out[j] = updates[i];
            }
        }
        return;
    }
    let rank = data_shape.len();
    let axis = if axis < 0 { rank as i32 + axis } else { axis } as usize;
    let axis = axis.min(rank.saturating_sub(1));
    let outer: usize = data_shape[..axis].iter().product::<usize>();
    let inner: usize = data_shape[axis..].iter().skip(1).product::<usize>().max(1);
    let axis_dim = data_shape.get(axis).copied().unwrap_or(1);
    for o in 0..outer {
        for i in 0..inner {
            let flat_i = o * axis_dim * inner + i;
            if flat_i >= indices.len() {
                continue;
            }
            let row = indices[flat_i].max(0) as usize;
            let dst = o * axis_dim * inner + row.min(axis_dim.saturating_sub(1)) * inner + i;
            if dst < out.len() && flat_i < updates.len() {
                out[dst] = updates[flat_i];
            }
        }
    }
}

struct ScatterElementsKernel;

impl CpuKernel for ScatterElementsKernel {
    fn name(&self) -> &str {
        SCATTER_ELEMENTS
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        if inputs.len() < 3 {
            return Err(format!("expected 3 inputs, got {}", inputs.len()));
        }
        let axis = if attrs.len() >= 4 {
            i32::from_le_bytes(attrs[0..4].try_into().unwrap())
        } else {
            0
        };
        if inputs[0].shape().dtype() == rlx_ir::DType::I64 {
            let out = output.expect_i64_mut("output")?;
            let indices = inputs[1].expect_i64("indices")?;
            let updates = inputs[2].expect_i64("updates")?;
            if let Some(data) = inputs[0].as_i64() {
                if !std::ptr::eq(data.as_ptr(), out.as_ptr()) {
                    let n = data.len().min(out.len());
                    out[..n].copy_from_slice(&data[..n]);
                }
            }
            let n = out.len().min(indices.len()).min(updates.len());
            for i in 0..n {
                let j = indices[i].max(0) as usize;
                if j < out.len() {
                    out[j] = updates[i];
                }
            }
            let _ = axis;
        } else {
            let out = output.expect_f32_mut("output")?;
            let indices = inputs[1].expect_i64("indices")?;
            let updates = inputs[2].expect_f32("updates")?;
            if let Some(data) = inputs[0].as_f32() {
                if !std::ptr::eq(data.as_ptr(), out.as_ptr()) {
                    out.copy_from_slice(data);
                }
            }
            let data_shape = shape_for_buffer(out.len(), &shape_usize(inputs[0].shape()));
            scatter_elements_f32(out, &data_shape, indices, updates, axis);
        }
        Ok(())
    }
}

pub fn register_onnx_scatter_elements_kernel() {
    register_cpu_kernel(Arc::new(ScatterElementsKernel));
}
