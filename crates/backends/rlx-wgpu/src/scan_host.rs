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

//! Host-side `Op::Scan` / nested-body ops for wgpu arenas.
//!
//! Scan / HostOp / indexing go through [`rlx_gpu_host`] (span-rebased). Concat
//! and Expand stay here — they need per-piece reads on sharded arenas.

use crate::buffer::Arena;
use crate::host_stage::WgpuArena;
use rlx_cpu::thunk::{HostOpDesc, ScanHostDesc};

pub fn run_scan(arena: &Arena, device: &wgpu::Device, queue: &wgpu::Queue, desc: &ScanHostDesc) {
    let mut a = WgpuArena {
        arena,
        device,
        queue,
        size_bytes: 0,
    };
    rlx_gpu_host::run_scan_span(&mut a, desc.clone());
}

pub fn run_host_op(arena: &Arena, device: &wgpu::Device, queue: &wgpu::Queue, desc: &HostOpDesc) {
    let mut a = WgpuArena {
        arena,
        device,
        queue,
        size_bytes: 0,
    };
    rlx_gpu_host::run_host_op_span(&mut a, desc.clone());
}

pub fn run_indexing(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    thunk: &rlx_cpu::thunk::IndexingThunk,
) {
    let max_buf = device.limits().max_buffer_size as usize;
    let mut a = WgpuArena {
        arena,
        device,
        queue,
        size_bytes: 0,
    };
    rlx_gpu_host::run_indexing(&mut a, 0, thunk, max_buf);
}

/// Host Concat for tensors that cannot share one wgpu storage bind window
/// (e.g. a 0.5 GiB Concat input at arena offset 0 and an output near 8 GiB on
/// a sharded F5 DiT arena). Reads each input separately so the span need not
/// be contiguous.
pub fn run_concat_host(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    dst_byte_off: usize,
    outer: u32,
    inner: u32,
    total_axis: u32,
    inputs: &[(usize, u32, u32)],
) {
    let starts: Vec<u32> = {
        let mut s = 0u32;
        inputs
            .iter()
            .map(|&(_, axis_len, _)| {
                let start = s;
                s += axis_len;
                start
            })
            .collect()
    };
    run_concat_host_pieces(
        arena,
        device,
        queue,
        dst_byte_off,
        outer,
        inner,
        total_axis,
        inputs,
        &starts,
        /*clear=*/ true,
    );
}

/// Write selected Concat pieces into `dst` at the given axis starts. When
/// `clear` is false, leaves other columns untouched (GPU already filled them).
pub fn run_concat_host_pieces(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    dst_byte_off: usize,
    outer: u32,
    inner: u32,
    total_axis: u32,
    inputs: &[(usize, u32, u32)],
    starts: &[u32],
    _clear: bool,
) {
    let outer = outer as usize;
    let inner = inner as usize;
    let total_axis = total_axis as usize;
    let row_stride = total_axis.saturating_mul(inner);
    for (i, &(src_off, axis_len, numel)) in inputs.iter().enumerate() {
        let copy_per_row = (axis_len as usize).saturating_mul(inner);
        let dst_col_off = (starts[i] as usize).saturating_mul(inner);
        let bytes =
            arena.read_bytes_range(device, queue, src_off, (numel as usize).saturating_mul(4));
        for o in 0..outer {
            let src_begin = o.saturating_mul(copy_per_row).saturating_mul(4);
            let src_end = src_begin
                .saturating_add(copy_per_row.saturating_mul(4))
                .min(bytes.len());
            if src_begin >= src_end {
                continue;
            }
            let dst = dst_byte_off.saturating_add(
                o.saturating_mul(row_stride)
                    .saturating_add(dst_col_off)
                    .saturating_mul(4),
            );
            write_bytes_range_chunked(arena, queue, dst, &bytes[src_begin..src_end]);
        }
    }
}

/// `queue.write_buffer` can fail or truncate on very large payloads (Metal
/// staging). Write in ≤64 MiB chunks.
fn write_bytes_range_chunked(arena: &Arena, queue: &wgpu::Queue, byte_off: usize, data: &[u8]) {
    const CHUNK: usize = 64 * 1024 * 1024;
    let mut off = 0;
    while off < data.len() {
        let n = (data.len() - off).min(CHUNK);
        let n = if off + n < data.len() { n & !3 } else { n };
        if n == 0 {
            break;
        }
        arena.write_bytes_range(queue, byte_off + off, &data[off..off + n]);
        off += n;
    }
}

/// Host Expand (broadcast) — used when the GPU path cannot cover the full
/// output on a sharded arena.
pub fn run_expand_host(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    in_byte_off: usize,
    out_byte_off: usize,
    in_dims: &[u32],
    out_dims: &[u32],
) {
    let rank = out_dims.len();
    assert_eq!(in_dims.len(), rank);
    let in_elems: usize = in_dims
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1);
    let out_elems: usize = out_dims
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1);
    let inp = arena.read_bytes_range(device, queue, in_byte_off, in_elems * 4);
    let src: Vec<f32> = inp
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let mut in_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        in_strides[i] = in_strides[i + 1] * in_dims[i + 1] as usize;
    }
    for i in 0..rank {
        if in_dims[i] == 1 && out_dims[i] != 1 {
            in_strides[i] = 0;
        }
    }
    let mut out = vec![0f32; out_elems];
    for i in 0..out_elems {
        let mut rem = i;
        let mut src_idx = 0usize;
        for a in (0..rank).rev() {
            let dim = out_dims[a] as usize;
            let c = rem % dim;
            rem /= dim;
            src_idx += c * in_strides[a];
        }
        out[i] = src[src_idx.min(src.len().saturating_sub(1))];
    }
    let out_bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
    write_bytes_range_chunked(arena, queue, out_byte_off, &out_bytes);
}
