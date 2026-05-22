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
// RLX — LLaDA2 group-limited gate on Metal (host delegate, unified memory).

use crate::op_registry::{MetalKernel, register_metal_kernel};
use rlx_ir::Shape;
use std::sync::Arc;

pub const OP_NAME: &str = "llada2.group_limited_gate";

#[derive(Debug)]
struct Llada2GateMetal;

impl MetalKernel for Llada2GateMetal {
    fn name(&self) -> &str {
        OP_NAME
    }

    fn execute(
        &self,
        inputs: &[(&[u8], &Shape)],
        output: (&mut [u8], &Shape),
        attrs: &[u8],
    ) -> Result<(), String> {
        let sig_bytes = inputs[0].0;
        let route_bytes = inputs[1].0;
        let out_bytes = output.0;
        if sig_bytes.len() % 4 != 0 || route_bytes.len() % 4 != 0 || out_bytes.len() % 4 != 0 {
            return Err("gate: non-f32-aligned buffers".into());
        }
        let sig = bytemuck::cast_slice::<u8, f32>(sig_bytes);
        let route = bytemuck::cast_slice::<u8, f32>(route_bytes);
        let out = bytemuck::cast_slice_mut::<u8, f32>(out_bytes);
        rlx_cpu::llada2_gate::execute_gate_f32(sig, route, out, attrs)
    }
}

pub fn register() {
    register_metal_kernel(Arc::new(Llada2GateMetal));
}
