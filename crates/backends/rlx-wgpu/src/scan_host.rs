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

//! Host-side `Op::Scan` for wgpu arenas — the readback analogue of Metal's
//! unified-memory `ScanHost`. wgpu has no host-addressable arena, so we read
//! the span covering the scan's inputs/output to the CPU, run the compiled
//! body loop, and write the span back. Enables recurrences (e.g. IIR
//! `biquad`) on the wgpu backend.

use crate::buffer::Arena;
use rlx_cpu::thunk::ScanBodyPlan;

#[allow(clippy::too_many_arguments)]
pub fn run_scan(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    plan: &ScanBodyPlan,
    outer_init_off: usize,
    outer_final_off: usize,
    length: u32,
    save_trajectory: bool,
    xs_outer: &[(usize, usize)],
    bcast_outer: &[(usize, usize)],
) {
    let cb = plan.carry_bytes;
    let out_len = if save_trajectory {
        length as usize * cb
    } else {
        cb
    };

    // Contiguous arena span covering every referenced region; run the CPU scan
    // against a host copy with offsets rebased to the span start.
    let mut lo = outer_init_off.min(outer_final_off);
    let mut hi = (outer_init_off + cb).max(outer_final_off + out_len);
    for &(o, t) in bcast_outer {
        lo = lo.min(o);
        hi = hi.max(o + t);
    }
    for &(o, ps) in xs_outer {
        lo = lo.min(o);
        hi = hi.max(o + length as usize * ps);
    }
    let span_len = hi - lo;

    let mut host = arena.read_bytes_range(device, queue, lo, span_len);
    let xs_rel: Vec<(usize, usize)> = xs_outer.iter().map(|&(o, ps)| (o - lo, ps)).collect();
    let bcast_rel: Vec<(usize, usize)> = bcast_outer.iter().map(|&(o, t)| (o - lo, t)).collect();
    unsafe {
        rlx_cpu::thunk::execute_scan_host(
            host.as_mut_ptr(),
            plan,
            outer_init_off - lo,
            outer_final_off - lo,
            length,
            save_trajectory,
            &xs_rel,
            &bcast_rel,
        );
    }
    arena.write_bytes_range(queue, lo, &host);
}
