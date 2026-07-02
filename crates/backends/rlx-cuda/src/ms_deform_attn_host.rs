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
//! Host-side `Op::Custom("gdino.ms_deform_attn")` for CUDA arenas: stage the
//! whole f32 arena to the host, run the shared `rlx_cpu` fused kernel in place,
//! copy back. Mirrors `llada2_gate_host`.

use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

#[allow(clippy::too_many_arguments)]
pub fn run_ms_deform_attn(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    arena_size_bytes: usize,
    in_offs: &[(u32, u32)],
    out_off: usize,
    out_len: usize,
    attrs: &[u8],
) {
    let n_f32 = arena_size_bytes / 4;
    stream
        .synchronize()
        .expect("rlx-cuda: ms_deform_attn pre-sync failed");

    let mut host = vec![0f32; n_f32];
    stream
        .memcpy_dtoh(&buffer.slice(..), &mut host)
        .expect("rlx-cuda: ms_deform_attn dtoh failed");

    let offs: Vec<(usize, usize)> = in_offs
        .iter()
        .map(|&(o, l)| (o as usize, l as usize))
        .collect();
    rlx_cpu::ms_deform_attn::execute_in_arena(&mut host, &offs, out_off, out_len, attrs)
        .expect("rlx-cuda: ms_deform_attn execute failed");

    stream
        .memcpy_htod(&host, &mut buffer.slice_mut(..))
        .expect("rlx-cuda: ms_deform_attn htod failed");
}
