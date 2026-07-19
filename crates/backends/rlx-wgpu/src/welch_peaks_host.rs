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

//! Host-side `Op::WelchPeaks` for wgpu arenas (partial-span D2H → CPU → H2D).
//!
//! Thin adapter over the shared [`rlx_gpu_host`] implementation.

use crate::buffer::Arena;
use crate::host_stage::WgpuArena;

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
    let mut a = WgpuArena {
        arena,
        device,
        queue,
        size_bytes: 0,
    };
    rlx_gpu_host::run_welch_peaks(
        &mut a,
        spec_byte_off,
        dst_byte_off,
        welch_batch,
        n_fft,
        n_segments,
        k,
        /* pre_sync */ false,
    );
}
