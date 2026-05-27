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
//! Host-side `Op::Custom("llada2.group_limited_gate")` for CUDA arenas.

use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

pub fn run_llada2_group_limited_gate(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    arena_size_bytes: usize,
    sig_f32_off: usize,
    route_f32_off: usize,
    out_f32_off: usize,
    n_elems: usize,
    attrs: &[u8],
) {
    let n_f32 = arena_size_bytes / 4;
    stream
        .synchronize()
        .expect("rlx-cuda: llada2 gate pre-sync failed");

    let mut host = vec![0f32; n_f32];
    stream
        .memcpy_dtoh(&buffer.slice(..), &mut host)
        .expect("rlx-cuda: llada2 gate dtoh failed");

    rlx_cpu::llada2_gate::execute_gate_in_f32_arena(
        &mut host,
        sig_f32_off,
        route_f32_off,
        out_f32_off,
        n_elems,
        attrs,
    )
    .expect("rlx-cuda: llada2 gate execute failed");

    stream
        .memcpy_htod(&host, &mut buffer.slice_mut(..))
        .expect("rlx-cuda: llada2 gate htod failed");
}
