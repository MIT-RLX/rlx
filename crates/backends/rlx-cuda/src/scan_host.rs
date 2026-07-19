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

//! Host-side `Op::Scan` / HostOp / indexing for CUDA arenas.
//!
//! Thin adapters over [`rlx_gpu_host`] (`run_scan` / `run_host_op` / `run_indexing`).

use crate::host_stage::CudaArena;
use cudarc::driver::{CudaSlice, CudaStream};
use rlx_cpu::thunk::{HostOpDesc, INDEXING_CONTIGUOUS_SPAN_CAP, IndexingThunk, ScanHostDesc};
use std::sync::Arc;

pub fn run_scan(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    arena_size_bytes: usize,
    desc: &ScanHostDesc,
) {
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: arena_size_bytes,
    };
    rlx_gpu_host::run_scan(&mut arena, desc);
}

pub fn run_host_op(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    arena_size_bytes: usize,
    desc: &HostOpDesc,
) {
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: arena_size_bytes,
    };
    rlx_gpu_host::run_host_op(&mut arena, desc);
}

pub fn run_indexing(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    arena_size_bytes: usize,
    thunk: &IndexingThunk,
) {
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: arena_size_bytes,
    };
    rlx_gpu_host::run_indexing(
        &mut arena,
        arena_size_bytes,
        thunk,
        INDEXING_CONTIGUOUS_SPAN_CAP,
    );
}
