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
//! Host-side `Op::Custom("llada2.group_limited_gate")` for wgpu arenas.

use crate::buffer::Arena;

pub fn run_llada2_group_limited_gate(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    sig_byte_off: usize,
    route_byte_off: usize,
    out_byte_off: usize,
    n_elems: usize,
    attrs: &[u8],
) {
    let mut host = arena.read_bytes_range(device, queue, 0, arena.size);
    let host_f32: &mut [f32] = bytemuck::cast_slice_mut(&mut host);
    let sig_f32_off = sig_byte_off / 4;
    let route_f32_off = route_byte_off / 4;
    let out_f32_off = out_byte_off / 4;
    rlx_cpu::llada2_gate::execute_gate_in_f32_arena(
        host_f32,
        sig_f32_off,
        route_f32_off,
        out_f32_off,
        n_elems,
        attrs,
    )
    .expect("rlx-wgpu: llada2 gate execute failed");
    arena.write_bytes_range(queue, 0, &host);
}
