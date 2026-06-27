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

//! Host-staged fallbacks for irreducible primitives with no HIP kernel yet:
//! `Reverse`, `ArgMax`/`ArgMin`, `AxialRope2d`. Each stages the touched device
//! span to host, runs the verified rlx-cpu reference, and writes back — the
//! same pattern as `im2col_host`. Correct by construction.

use crate::device::RocmContext;
use crate::hip::HipBuffer;

/// Sync, copy `[span_start, span_end)` device→host, run `body` against the host
/// base (offsets must be span-relative), copy host→device.
fn stage_span(
    ctx: &RocmContext,
    buffer: &HipBuffer<f32>,
    span_start: usize,
    span_end: usize,
    body: impl FnOnce(*mut u8),
) {
    let span_len = span_end.saturating_sub(span_start);
    if span_len == 0 {
        return;
    }
    let rt = &ctx.runtime;
    unsafe {
        let _ = (rt.hip_stream_sync)(ctx.default_stream);
    }
    let mut host = vec![0u8; span_len];
    unsafe {
        let _ = (rt.hip_memcpy_dtoh)(
            host.as_mut_ptr() as *mut _,
            buffer.ptr + span_start as u64,
            span_len,
        );
    }
    body(host.as_mut_ptr());
    unsafe {
        let _ = (rt.hip_memcpy_htod)(
            buffer.ptr + span_start as u64,
            host.as_ptr() as *const _,
            span_len,
        );
    }
}

/// Batch-general reverse/flip (dtype-agnostic).
#[allow(clippy::too_many_arguments)]
pub fn run_reverse(
    ctx: &RocmContext,
    buffer: &HipBuffer<f32>,
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
    stage_span(ctx, buffer, span_start, span_end, |base| unsafe {
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
    ctx: &RocmContext,
    buffer: &HipBuffer<f32>,
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
    stage_span(ctx, buffer, span_start, span_end, |base| unsafe {
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
    ctx: &RocmContext,
    buffer: &HipBuffer<f32>,
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
    stage_span(ctx, buffer, span_start, span_end, |base| unsafe {
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
