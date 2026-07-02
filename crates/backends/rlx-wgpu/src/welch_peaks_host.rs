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

use crate::buffer::Arena;

pub fn run_welch_peaks(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    spec_byte_off: usize,
    dst_byte_off: usize,
    welch_batch: usize,
    n_fft: usize,
    n_segments: usize,
    k: usize,
) {
    let spec_len = welch_batch * n_segments * n_fft * 2;
    let dst_len = welch_batch * k * 2;
    let span_off = spec_byte_off.min(dst_byte_off);
    let span_end = (spec_byte_off + spec_len * 4).max(dst_byte_off + dst_len * 4);
    let span_len = span_end - span_off;

    let mut host = arena.read_bytes_range(device, queue, span_off, span_len);
    unsafe {
        rlx_cpu::thunk::execute_welch_peaks_f32(
            spec_byte_off - span_off,
            dst_byte_off - span_off,
            welch_batch,
            n_fft,
            n_segments,
            k,
            host.as_mut_ptr(),
        );
    }
    arena.write_bytes_range(queue, span_off, &host);
}
