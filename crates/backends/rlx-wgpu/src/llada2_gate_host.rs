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
use crate::host_stage::WgpuArena;

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
    debug_assert_eq!(sig_byte_off % 4, 0);
    debug_assert_eq!(route_byte_off % 4, 0);
    debug_assert_eq!(out_byte_off % 4, 0);
    let mut a = WgpuArena {
        arena,
        device,
        queue,
        size_bytes: arena.size,
    };
    rlx_gpu_host::run_llada2_group_limited_gate(
        &mut a,
        sig_byte_off / 4,
        route_byte_off / 4,
        out_byte_off / 4,
        n_elems,
        attrs,
    );
}
