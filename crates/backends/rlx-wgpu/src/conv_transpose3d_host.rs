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

//! Host-side `Op::ConvTranspose3d` for wgpu arenas.
//!
//! Thin adapter over [`rlx_gpu_host::run_conv_transpose3d_ncdhw`].

use crate::buffer::Arena;
use crate::host_stage::WgpuArena;

/// Stage arena slices and run NCDHW transposed conv3d on the host.
#[allow(clippy::too_many_arguments)]
pub fn run_conv_transpose3d(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: usize,
    weight: usize,
    dst: usize,
    n: usize,
    c_in: usize,
    d: usize,
    h: usize,
    w_in: usize,
    c_out: usize,
    d_out: usize,
    h_out: usize,
    w_out: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    sd: usize,
    sh: usize,
    sw: usize,
    pd: usize,
    ph: usize,
    pw: usize,
    dd: usize,
    dh: usize,
    dw: usize,
    groups: usize,
) {
    let mut a = WgpuArena {
        arena,
        device,
        queue,
        size_bytes: arena.size,
    };
    rlx_gpu_host::run_conv_transpose3d_ncdhw(
        &mut a, src, weight, dst, n, c_in, d, h, w_in, c_out, d_out, h_out, w_out, kd, kh, kw, sd,
        sh, sw, pd, ph, pw, dd, dh, dw, groups,
    );
}
