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
//! Host-side `Op::Custom("gdino.ms_deform_attn")` for ROCm arenas: stage the
//! whole f32 arena to the host, run the shared `rlx_cpu` fused kernel in place,
//! copy back. Mirrors `llada2_gate_host`.

use crate::device::RocmContext;
use crate::hip::HipBuffer;

#[allow(clippy::too_many_arguments)]
pub fn run_ms_deform_attn(
    ctx: &RocmContext,
    buffer: &HipBuffer<f32>,
    arena_size_bytes: usize,
    in_offs: &[(u32, u32)],
    out_off: usize,
    out_len: usize,
    attrs: &[u8],
) {
    let rt = &ctx.runtime;
    let n_f32 = arena_size_bytes / 4;
    let mut host = vec![0f32; n_f32];

    unsafe {
        let _ = (rt.hip_stream_sync)(ctx.default_stream);
        let _ = (rt.hip_memcpy_dtoh)(host.as_mut_ptr() as *mut _, buffer.ptr, n_f32 * 4);
    }

    let offs: Vec<(usize, usize)> = in_offs
        .iter()
        .map(|&(o, l)| (o as usize, l as usize))
        .collect();
    rlx_cpu::ms_deform_attn::execute_in_arena(&mut host, &offs, out_off, out_len, attrs)
        .expect("rlx-rocm: ms_deform_attn execute failed");

    unsafe {
        let _ = (rt.hip_memcpy_htod)(buffer.ptr, host.as_ptr() as *const _, n_f32 * 4);
    }
}
