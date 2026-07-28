// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fused multi-scale deformable attention on Metal (host delegate, unified
//! memory — reads the GPU buffers in place and runs the shared `rlx_cpu` math).

use crate::op_registry::{MetalKernel, register_metal_kernel};
use rlx_ir::Shape;
use std::sync::Arc;

pub const OP_NAME: &str = "gdino.ms_deform_attn";

#[derive(Debug)]
struct MsDeformAttnMetal;

impl MetalKernel for MsDeformAttnMetal {
    fn name(&self) -> &str {
        OP_NAME
    }

    fn execute(
        &self,
        inputs: &[(&[u8], &Shape)],
        output: (&mut [u8], &Shape),
        attrs: &[u8],
    ) -> Result<(), String> {
        let ins: Vec<Vec<f32>> = inputs
            .iter()
            .map(|(bytes, _)| {
                if !bytes.len().is_multiple_of(4) {
                    return Err("ms_deform_attn: non-f32-aligned input".to_string());
                }
                Ok(bytemuck::cast_slice::<u8, f32>(bytes).to_vec())
            })
            .collect::<Result<_, String>>()?;
        let in_refs: Vec<&[f32]> = ins.iter().map(|v| v.as_slice()).collect();
        let out = output.0;
        if !out.len().is_multiple_of(4) {
            return Err("ms_deform_attn: non-f32-aligned output".into());
        }
        let out_f32 = bytemuck::cast_slice_mut::<u8, f32>(out);
        rlx_cpu::ms_deform_attn::execute(&in_refs, attrs, out_f32)
    }
}

pub fn register() {
    register_metal_kernel(Arc::new(MsDeformAttnMetal));
}
