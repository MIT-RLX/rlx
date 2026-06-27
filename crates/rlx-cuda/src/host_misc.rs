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
//! `Reverse`, `ArgMax`/`ArgMin`, `AxialRope2d`. Each stages the touched device
//! span to host, runs the verified rlx-cpu reference, and writes back — the
//! same pattern as `im2col_host`. Correct by construction; a native CUDA
//! kernel can replace any of these when a workload makes the round-trip hurt.

use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

/// Sync, copy `[span_start, span_end)` device→host, run `body` against the host
/// base (offsets must be span-relative), copy host→device. Byte offsets are
/// f32-aligned (arena allocations are f32).
fn stage_span(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    span_start: usize,
    span_end: usize,
    body: impl FnOnce(*mut u8),
) {
    if span_end <= span_start {
        return;
    }
    stream
        .synchronize()
        .expect("rlx-cuda host_misc: pre-sync failed");
    let span_start_f32 = span_start / 4;
    let span_end_f32 = span_end.div_ceil(4);
    let mut host = vec![0u8; (span_end_f32 - span_start_f32) * 4];
    stream
        .memcpy_dtoh(
            &buffer.slice(span_start_f32..span_end_f32),
            bytemuck::cast_slice_mut(&mut host),
        )
        .expect("rlx-cuda host_misc: dtoh failed");
    body(host.as_mut_ptr());
    stream
        .memcpy_htod(
            bytemuck::cast_slice(&host),
            &mut buffer.slice_mut(span_start_f32..span_end_f32),
        )
        .expect("rlx-cuda host_misc: htod failed");
}

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
    let total: usize = dims.iter().map(|&d| d as usize).product::<usize>().max(1);
    let bytes = total * elem_bytes;
    let span_start = src.min(dst);
    let span_end = (src + bytes).max(dst + bytes);
    stage_span(stream, buffer, span_start, span_end, |base| unsafe {
        rlx_cpu::thunk::execute_reverse(
            src - span_start,
            dst - span_start,
            dims,
            rev_mask,
            elem_bytes,
            base,
        );
    });
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
    let in_bytes = outer * reduced * inner * 4;
    let out_bytes = outer * inner * 4;
    let span_start = src.min(dst);
    let span_end = (src + in_bytes).max(dst + out_bytes);
    stage_span(stream, buffer, span_start, span_end, |base| unsafe {
        rlx_cpu::thunk::execute_argreduce_f32(
            src - span_start,
            dst - span_start,
            outer,
            reduced,
            inner,
            is_max,
            base,
        );
    });
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
    let bytes = batch * seq * hidden * 4;
    let span_start = src.min(dst);
    let span_end = (src + bytes).max(dst + bytes);
    stage_span(stream, buffer, span_start, span_end, |base| unsafe {
        rlx_cpu::thunk::execute_axial_rope2d_f32(
            src - span_start,
            dst - span_start,
            batch,
            seq,
            hidden,
            end_x,
            end_y,
            head_dim,
            num_heads,
            theta,
            repeat_factor,
            base,
        );
    });
}
