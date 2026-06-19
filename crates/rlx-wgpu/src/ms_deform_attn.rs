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

//! Host-side `Op::Custom("gdino.ms_deform_attn")` for wgpu arenas. Reads the
//! input buffers back, runs the shared `rlx_cpu` fused deformable kernel, and
//! writes the result. Matches the registry-based Metal/MLX delegates.

use crate::buffer::Arena;

/// `in_offs` is `(byte_off, byte_len)` per input (11 total). Output is `[nq, d]`
/// f32 at `out_byte_off` spanning `out_bytes`.
#[allow(clippy::too_many_arguments)]
pub fn run_ms_deform_attn(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    in_offs: &[(u32, u32)],
    out_byte_off: usize,
    out_bytes: usize,
    attrs: &[u8],
) {
    let ins: Vec<Vec<f32>> = in_offs
        .iter()
        .map(|&(off, len)| {
            let bytes = arena.read_bytes_range(device, queue, off as usize, len as usize);
            bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
        })
        .collect();
    let in_refs: Vec<&[f32]> = ins.iter().map(|v| v.as_slice()).collect();
    let mut out = vec![0f32; out_bytes / 4];
    rlx_cpu::ms_deform_attn::execute(&in_refs, attrs, &mut out)
        .expect("rlx-wgpu: ms_deform_attn host kernel");
    arena.write_bytes_range(queue, out_byte_off, bytemuck::cast_slice(&out));
}
