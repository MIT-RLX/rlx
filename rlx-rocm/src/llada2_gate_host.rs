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
//! Host-side `Op::Custom("llada2.group_limited_gate")` for ROCm arenas.

use crate::device::RocmContext;
use crate::hip::HipBuffer;

pub fn run_llada2_group_limited_gate(
    ctx: &RocmContext,
    buffer: &HipBuffer<f32>,
    arena_size_bytes: usize,
    sig_f32_off: usize,
    route_f32_off: usize,
    out_f32_off: usize,
    n_elems: usize,
    attrs: &[u8],
) {
    let rt = &ctx.runtime;
    let n_f32 = arena_size_bytes / 4;
    let mut host = vec![0f32; n_f32];

    unsafe {
        let _ = (rt.hip_stream_sync)(ctx.default_stream);
        let _ = (rt.hip_memcpy_dtoh)(
            host.as_mut_ptr() as *mut _,
            buffer.ptr,
            n_f32 * 4,
        );
    }

    rlx_cpu::llada2_gate::execute_gate_in_f32_arena(
        &mut host,
        sig_f32_off,
        route_f32_off,
        out_f32_off,
        n_elems,
        attrs,
    )
    .expect("rlx-rocm: llada2 gate execute failed");

    unsafe {
        let _ = (rt.hip_memcpy_htod)(
            buffer.ptr,
            host.as_ptr() as *const _,
            n_f32 * 4,
        );
    }
}
