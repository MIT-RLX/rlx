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

//! Host-side `Op::Fft` for wgpu arenas.
//!
//! Thin adapter over [`rlx_gpu_host::run_fft1d`].

use crate::buffer::Arena;
use crate::host_stage::WgpuArena;
use rlx_ir::DType;

pub fn run_fft1d(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src_byte_off: usize,
    dst_byte_off: usize,
    outer: usize,
    n_complex: usize,
    inverse: bool,
    norm_tag: u32,
    dtype: DType,
) {
    let mut a = WgpuArena {
        arena,
        device,
        queue,
        size_bytes: 0,
    };
    rlx_gpu_host::run_fft1d(
        &mut a,
        src_byte_off,
        dst_byte_off,
        outer,
        n_complex,
        inverse,
        norm_tag,
        dtype,
    );
}
