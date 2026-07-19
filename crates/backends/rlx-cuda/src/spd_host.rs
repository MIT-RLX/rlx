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

//! Host-side SPD-manifold ops for CUDA device arenas (D2H → CPU → H2D).
//!
//! Thin adapter over [`rlx_gpu_host::run_spd`].

use crate::host_stage::CudaArena;
use cudarc::driver::{CudaSlice, CudaStream};
use rlx_ir::{Op, Shape};
use std::sync::Arc;

/// One SPD op against the device arena. `inputs` / `out_off` are **f32 element**
/// offsets (historical SpdHost layout — not byte offsets).
#[allow(clippy::too_many_arguments)]
pub fn run_spd(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    arena_size_bytes: usize,
    op: &Op,
    out_off: usize,
    out_shape: &Shape,
    inputs: &[(usize, Shape)],
) {
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: arena_size_bytes,
    };
    rlx_gpu_host::run_spd(&mut arena, op, out_off, out_shape, inputs);
}
