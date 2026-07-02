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
//! Fused multi-scale deformable attention on MLX (host delegate via `rlx_cpu`).

use crate::array::{Array, MlxError};
use crate::op_registry::{MlxKernel, register_mlx_kernel};
use rlx_ir::Shape;
use std::sync::Arc;

pub const OP_NAME: &str = "gdino.ms_deform_attn";

struct MsDeformAttnMlx;

impl MlxKernel for MsDeformAttnMlx {
    fn name(&self) -> &str {
        OP_NAME
    }

    fn execute(
        &self,
        inputs: &[&Array],
        output_shape: &Shape,
        attrs: &[u8],
    ) -> Result<Array, MlxError> {
        let ins: Vec<Vec<f32>> = inputs
            .iter()
            .map(|a| {
                let bytes = a.to_bytes()?;
                Ok(bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect())
            })
            .collect::<Result<_, MlxError>>()?;
        let in_refs: Vec<&[f32]> = ins.iter().map(|v| v.as_slice()).collect();
        let out_dims: Vec<usize> = output_shape
            .dims()
            .iter()
            .map(|d| d.unwrap_static())
            .collect();
        let out_elems: usize = out_dims.iter().product();
        let mut out = vec![0f32; out_elems];
        rlx_cpu::ms_deform_attn::execute(&in_refs, attrs, &mut out).map_err(MlxError)?;
        Array::from_f32_slice(&out, &out_dims, output_shape.dtype())
    }
}

pub fn register() {
    register_mlx_kernel(Arc::new(MsDeformAttnMlx));
}
