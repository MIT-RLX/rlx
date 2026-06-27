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
//! Host-side `Op::ConvTranspose2d` for wgpu arenas (readback → CPU → writeback).
//! wgpu has no native transposed-conv kernel; the decoder upsample in vision
//! U-Nets is a handful of ops, so the CPU reference round-trip is acceptable.

use crate::buffer::Arena;

#[allow(clippy::too_many_arguments)]
pub fn run_conv_transpose2d(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: usize,
    weight: usize,
    dst: usize,
    n: usize,
    c_in: usize,
    h: usize,
    w_in: usize,
    c_out: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
    groups: usize,
) {
    let mut host = arena.read_bytes_range(device, queue, 0, arena.size);
    unsafe {
        rlx_cpu::thunk::execute_conv_transpose2d_nchw_f32(
            src,
            weight,
            dst,
            n,
            c_in,
            h,
            w_in,
            c_out,
            h_out,
            w_out,
            kh,
            kw,
            sh,
            sw,
            ph,
            pw,
            dh,
            dw,
            groups,
            host.as_mut_ptr(),
        );
    }
    arena.write_bytes_range(queue, 0, &host);
}
