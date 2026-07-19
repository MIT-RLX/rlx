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

//! Host-staged fallbacks for irreducible primitives with no CUDA kernel yet:
//! `Reverse`, `ArgMax`/`ArgMin`, `AxialRope2d`. Thin adapters over the shared
//! [`rlx_gpu_host`] implementations — each stages the touched device span to
//! host, runs the verified rlx-cpu reference, and writes back.

use crate::host_stage::CudaArena;
use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

/// Batch-general reverse/flip (dtype-agnostic).
#[allow(clippy::too_many_arguments)]
pub fn run_reverse(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    src: usize,
    dst: usize,
    dims: &[u32],
    rev_mask: &[bool],
    elem_bytes: usize,
) {
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: 0,
    };
    rlx_gpu_host::run_reverse(&mut arena, src, dst, dims, rev_mask, elem_bytes);
}

/// ArgMax/ArgMin (f32-encoded indices) over the middle `reduced` axis.
#[allow(clippy::too_many_arguments)]
pub fn run_argreduce(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    src: usize,
    dst: usize,
    outer: usize,
    reduced: usize,
    inner: usize,
    is_max: bool,
) {
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: 0,
    };
    rlx_gpu_host::run_argreduce(&mut arena, src, dst, outer, reduced, inner, is_max);
}

/// Axial 2-D RoPE on `[batch, seq, hidden]`.
#[allow(clippy::too_many_arguments)]
pub fn run_axial_rope2d(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    src: usize,
    dst: usize,
    batch: usize,
    seq: usize,
    hidden: usize,
    end_x: usize,
    end_y: usize,
    head_dim: usize,
    num_heads: usize,
    theta: f32,
    repeat_factor: usize,
) {
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: 0,
    };
    rlx_gpu_host::run_axial_rope2d(
        &mut arena,
        src,
        dst,
        batch,
        seq,
        hidden,
        end_x,
        end_y,
        head_dim,
        num_heads,
        theta,
        repeat_factor,
    );
}
