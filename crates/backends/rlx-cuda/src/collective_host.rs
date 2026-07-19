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

//! Host-side `Op::Custom("collective.*")` for CUDA arenas.
//!
//! Thin adapter over [`rlx_gpu_host::run_collective_f32`].

use crate::host_stage::CudaArena;
use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

pub use rlx_gpu_host::COLLECTIVE_OPS;

#[allow(clippy::too_many_arguments)]
pub fn run_collective(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    _arena_size_bytes: usize,
    name: &str,
    in_off: usize,
    in_len: usize,
    out_off: usize,
    out_len: usize,
    attrs: &[u8],
) {
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: 0,
    };
    rlx_gpu_host::run_collective_f32(&mut arena, name, in_off, in_len, out_off, out_len, attrs);
}
