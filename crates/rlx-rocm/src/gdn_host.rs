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
//! Host-side `Op::GatedDeltaNet` for ROCm device arenas (D2H → CPU → H2D).

use crate::device::RocmContext;
use crate::hip::HipBuffer;

pub fn run_gated_delta_net(
    ctx: &RocmContext,
    buffer: &HipBuffer<f32>,
    arena_size_bytes: usize,
    q_byte_off: usize,
    k_byte_off: usize,
    v_byte_off: usize,
    g_byte_off: usize,
    beta_byte_off: usize,
    state_byte_off: usize,
    dst_byte_off: usize,
    batch: usize,
    seq: usize,
    heads: usize,
    state_size: usize,
    use_carry: bool,
) {
    assert!(
        state_size <= rlx_cpu::gdn::GDN_MAX_STATE,
        "rlx-rocm GatedDeltaNet: state_size {state_size} > {}",
        rlx_cpu::gdn::GDN_MAX_STATE
    );

    let rt = &ctx.runtime;
    let n_f32 = arena_size_bytes / 4;
    let mut host = vec![0f32; n_f32];

    unsafe {
        let _ = (rt.hip_stream_sync)(ctx.default_stream);
        let _ = (rt.hip_memcpy_dtoh)(host.as_mut_ptr() as *mut _, buffer.ptr, n_f32 * 4);
    }

    unsafe {
        rlx_cpu::thunk::execute_gated_delta_net_f32(
            q_byte_off,
            k_byte_off,
            v_byte_off,
            g_byte_off,
            beta_byte_off,
            if use_carry { state_byte_off } else { 0 },
            dst_byte_off,
            batch,
            seq,
            heads,
            state_size,
            host.as_mut_ptr() as *mut u8,
        );
    }

    unsafe {
        let _ = (rt.hip_memcpy_htod)(buffer.ptr, host.as_ptr() as *const _, n_f32 * 4);
    }
}
