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
//! Host-side `Op::Lstm` for ROCm device arenas (D2H → CPU → H2D).

use crate::device::RocmContext;
use crate::hip::HipBuffer;

pub fn run_lstm(
    ctx: &RocmContext,
    buffer: &HipBuffer<f32>,
    arena_size_bytes: usize,
    x_byte_off: usize,
    w_ih_byte_off: usize,
    w_hh_byte_off: usize,
    bias_byte_off: usize,
    h0_byte_off: usize,
    c0_byte_off: usize,
    dst_byte_off: usize,
    batch: usize,
    seq: usize,
    input_size: usize,
    hidden: usize,
    num_layers: usize,
    bidirectional: bool,
    carry: bool,
) {
    let rt = &ctx.runtime;
    let n_f32 = arena_size_bytes / 4;
    let mut host = vec![0f32; n_f32];

    unsafe {
        let _ = (rt.hip_stream_sync)(ctx.default_stream);
        let _ = (rt.hip_memcpy_dtoh)(host.as_mut_ptr() as *mut _, buffer.ptr, n_f32 * 4);
    }

    unsafe {
        rlx_cpu::thunk::execute_lstm_f32(
            x_byte_off,
            w_ih_byte_off,
            w_hh_byte_off,
            bias_byte_off,
            h0_byte_off,
            c0_byte_off,
            dst_byte_off,
            batch,
            seq,
            input_size,
            hidden,
            num_layers,
            bidirectional,
            carry,
            host.as_mut_ptr() as *mut u8,
        );
    }

    unsafe {
        let _ = (rt.hip_memcpy_htod)(buffer.ptr, host.as_ptr() as *const _, n_f32 * 4);
    }
}
