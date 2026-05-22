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
//! Host-side `Op::GatedDeltaNet` for wgpu arenas (readback → CPU → writeback).

use crate::buffer::Arena;

pub fn run_gated_delta_net(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
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
        "rlx-wgpu GatedDeltaNet: state_size {state_size} > {}",
        rlx_cpu::gdn::GDN_MAX_STATE
    );

    let mut host = arena.read_bytes_range(device, queue, 0, arena.size);
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
            host.as_mut_ptr(),
        );
    }
    arena.write_bytes_range(queue, 0, &host);
}
