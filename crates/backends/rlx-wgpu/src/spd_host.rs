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

//! CPU host-fallback for Riemannian / SPD-manifold ops on the wgpu backend.
//!
//! Thin adapter over [`rlx_gpu_host::run_spd_spans`]. Eigen-decompositions have
//! no WGSL kernel, so these run on the CPU reference between GPU segments.

use crate::buffer::Arena;
use crate::host_stage::WgpuArena;
use rlx_ir::{Op, Shape};

pub use rlx_gpu_host::{SpdInput, is_spd_host};

/// Run one SPD op on the CPU reference against the wgpu arena. `inputs[i]` is
/// the declared shape + arena byte offset of operand `i`; the f32 result is
/// written to `out_byte_off` (`out_shape`-sized).
pub fn run_spd(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    op: &Op,
    inputs: &[SpdInput],
    out_shape: &Shape,
    out_byte_off: usize,
) {
    let mut a = WgpuArena {
        arena,
        device,
        queue,
        size_bytes: 0,
    };
    rlx_gpu_host::run_spd_spans(&mut a, op, inputs, out_shape, out_byte_off);
}
