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
//!
//! Structural host steps (Expand / Narrow / Transpose / Concat) optionally
//! share a [`HostTensorCache`] with HostOp/Conv so Kitten discrete Vulkan can
//! stay on the host across ~900 PCIe round-trips per infer.

use crate::buffer::Arena;
use crate::host_stage::WgpuArena;
use rlx_cpu::thunk::{HostOpDesc, ScanHostDesc};
use rlx_gpu_host::{DeviceArena, HostTensorCache};
use std::sync::Arc;

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
    run_host_op_with_cache(arena, device, queue, desc, None);
}

pub fn run_host_op_with_cache(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    desc: &HostOpDesc,
    cache: Option<&mut HostTensorCache>,
) {
    let mut a = WgpuArena {
        arena,
        device,
        queue,
        size_bytes: 0,
    };
    // Packed staging: sharded Kitten arenas would otherwise D2H multi-GiB
    // contiguous spans between far-apart operands on every HostOp.
    rlx_gpu_host::run_host_op_packed_cached(&mut a, desc, cache);
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

fn load_f32(
    a: &mut WgpuArena<'_>,
    byte_off: usize,
    n: usize,
    mut cache: Option<&mut HostTensorCache>,
) -> Arc<[f32]> {
    if n == 0 {
        return Arc::from([]);
    }
    if let Some(c) = cache.as_ref() {
        if let Some(hit) = c.get_arc_covering(byte_off, n) {
            return hit;
        }
    }
    if let Some(c) = cache.as_mut() {
        c.flush_offset(a, byte_off);
    }
    let mut v = vec![0f32; n];
    a.dtoh(byte_off, bytemuck::cast_slice_mut(v.as_mut_slice()));
    Arc::from(v)
}

fn store_f32_out(
    arena: &Arena,
    _device: &wgpu::Device,
    queue: &wgpu::Queue,
    byte_off: usize,
    data: Vec<f32>,
    cache: Option<&mut HostTensorCache>,
) {
    if data.is_empty() {
        return;
    }
    // Defer H2D when a mirror is active — Expand→HostOp/BufferCopy(host) stay
    // on the CPU. Device-reading steps (GPU BufferCopy, Gather, MatMul, …)
    // flush first via `run.rs`. Opt out with `RLX_WGPU_HOST_EAGER_H2D=1`.
    let defer = cache.is_some() && !rlx_ir::env::flag("RLX_WGPU_HOST_EAGER_H2D");
    if !defer {
        write_bytes_range_chunked(
            arena,
            queue,
            byte_off,
            bytemuck::cast_slice(data.as_slice()),
        );
    }
    if let Some(c) = cache {
        c.insert(byte_off, data, defer);
    }
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
    run_concat_host_cached(
        arena,
        device,
        queue,
        dst_byte_off,
        outer,
        inner,
        total_axis,
        inputs,
        None,
    );
}

pub fn run_concat_host_cached(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    dst_byte_off: usize,
    outer: u32,
    inner: u32,
    total_axis: u32,
    inputs: &[(usize, u32, u32)],
    cache: Option<&mut HostTensorCache>,
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
    run_concat_host_pieces_cached(
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
        cache,
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
    clear: bool,
) {
    run_concat_host_pieces_cached(
        arena,
        device,
        queue,
        dst_byte_off,
        outer,
        inner,
        total_axis,
        inputs,
        starts,
        clear,
        None,
    );
}

pub fn run_concat_host_pieces_cached(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    dst_byte_off: usize,
    outer: u32,
    inner: u32,
    total_axis: u32,
    inputs: &[(usize, u32, u32)],
    starts: &[u32],
    clear: bool,
    mut cache: Option<&mut HostTensorCache>,
) {
    let outer = outer as usize;
    let inner = inner as usize;
    let total_axis = total_axis as usize;
    let row_stride = total_axis.saturating_mul(inner);
    let out_elems = outer.saturating_mul(row_stride);
    // Partial GPU+host concat must see device columns — no deferred mirror.
    let use_cache = clear;
    let mut a = WgpuArena {
        arena,
        device,
        queue,
        size_bytes: 0,
    };
    let mut out = if use_cache {
        vec![0f32; out_elems]
    } else {
        Vec::new()
    };
    for (i, &(src_off, axis_len, numel)) in inputs.iter().enumerate() {
        let copy_per_row = (axis_len as usize).saturating_mul(inner);
        let dst_col_off = (starts[i] as usize).saturating_mul(inner);
        let src = if use_cache {
            load_f32(&mut a, src_off, numel as usize, cache.as_deref_mut())
        } else {
            let bytes =
                arena.read_bytes_range(device, queue, src_off, (numel as usize).saturating_mul(4));
            Arc::from(
                bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect::<Vec<_>>(),
            )
        };
        for o in 0..outer {
            let src_begin = o.saturating_mul(copy_per_row);
            let src_end = src_begin.saturating_add(copy_per_row).min(src.len());
            if src_begin >= src_end {
                continue;
            }
            if use_cache {
                let dst_begin = o.saturating_mul(row_stride).saturating_add(dst_col_off);
                out[dst_begin..dst_begin + (src_end - src_begin)]
                    .copy_from_slice(&src[src_begin..src_end]);
            } else {
                let dst = dst_byte_off.saturating_add(
                    o.saturating_mul(row_stride)
                        .saturating_add(dst_col_off)
                        .saturating_mul(4),
                );
                let bytes: Vec<u8> = src[src_begin..src_end]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                write_bytes_range_chunked(arena, queue, dst, &bytes);
            }
        }
    }
    if use_cache {
        store_f32_out(arena, device, queue, dst_byte_off, out, cache.take());
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
    run_expand_host_cached(
        arena,
        device,
        queue,
        in_byte_off,
        out_byte_off,
        in_dims,
        out_dims,
        None,
    );
}

pub fn run_expand_host_cached(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    in_byte_off: usize,
    out_byte_off: usize,
    in_dims: &[u32],
    out_dims: &[u32],
    mut cache: Option<&mut HostTensorCache>,
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
    let mut a = WgpuArena {
        arena,
        device,
        queue,
        size_bytes: 0,
    };
    let src = load_f32(&mut a, in_byte_off, in_elems, cache.as_deref_mut());
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
        for ax in (0..rank).rev() {
            let dim = out_dims[ax] as usize;
            let c = rem % dim;
            rem /= dim;
            src_idx += c * in_strides[ax];
        }
        out[i] = src[src_idx.min(src.len().saturating_sub(1))];
    }
    store_f32_out(arena, device, queue, out_byte_off, out, cache.take());
}

/// Host transpose for a virtual arena. Read and write only the operand spans,
/// avoiding a multi-gigabyte contiguous staging allocation between shards.
pub fn run_transpose_host(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    in_byte_off: usize,
    out_byte_off: usize,
    in_dims: &[u32],
    out_dims: &[u32],
    in_strides: &[usize],
) {
    run_transpose_host_cached(
        arena,
        device,
        queue,
        in_byte_off,
        out_byte_off,
        in_dims,
        out_dims,
        in_strides,
        None,
    );
}

pub fn run_transpose_host_cached(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    in_byte_off: usize,
    out_byte_off: usize,
    in_dims: &[u32],
    out_dims: &[u32],
    in_strides: &[usize],
    mut cache: Option<&mut HostTensorCache>,
) {
    let rank = out_dims.len();
    assert_eq!(in_dims.len(), rank);
    assert_eq!(in_strides.len(), rank);
    let in_elems = in_dims
        .iter()
        .fold(1usize, |n, &d| n.saturating_mul(d as usize));
    let out_elems = out_dims
        .iter()
        .fold(1usize, |n, &d| n.saturating_mul(d as usize));
    let mut a = WgpuArena {
        arena,
        device,
        queue,
        size_bytes: 0,
    };
    let src = load_f32(&mut a, in_byte_off, in_elems, cache.as_deref_mut());
    let mut out = vec![0f32; out_elems];
    for (out_idx, dst) in out.iter_mut().enumerate() {
        let mut rem = out_idx;
        let mut src_idx = 0usize;
        for axis in (0..rank).rev() {
            let dim = out_dims[axis] as usize;
            let coord = rem % dim;
            rem /= dim;
            src_idx = src_idx.saturating_add(coord.saturating_mul(in_strides[axis]));
        }
        *dst = src[src_idx];
    }
    store_f32_out(arena, device, queue, out_byte_off, out, cache.take());
}

/// Host Narrow for virtual arenas. It stages the source tensor independently
/// and writes each selected row directly to the destination slot.
#[allow(clippy::too_many_arguments)]
pub fn run_narrow_host(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    in_byte_off: usize,
    out_byte_off: usize,
    outer: u32,
    inner: u32,
    axis_in_size: u32,
    start: u32,
    axis_out_size: u32,
) {
    run_narrow_host_cached(
        arena,
        device,
        queue,
        in_byte_off,
        out_byte_off,
        outer,
        inner,
        axis_in_size,
        start,
        axis_out_size,
        None,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn run_narrow_host_cached(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    in_byte_off: usize,
    out_byte_off: usize,
    outer: u32,
    inner: u32,
    axis_in_size: u32,
    start: u32,
    axis_out_size: u32,
    mut cache: Option<&mut HostTensorCache>,
) {
    let outer = outer as usize;
    let inner = inner as usize;
    let axis_in_size = axis_in_size as usize;
    let start = start as usize;
    let axis_out_size = axis_out_size as usize;
    assert!(start.saturating_add(axis_out_size) <= axis_in_size);
    let in_row = axis_in_size.saturating_mul(inner);
    let out_row = axis_out_size.saturating_mul(inner);
    let mut a = WgpuArena {
        arena,
        device,
        queue,
        size_bytes: 0,
    };
    let src = load_f32(
        &mut a,
        in_byte_off,
        outer.saturating_mul(in_row),
        cache.as_deref_mut(),
    );
    let mut out = vec![0f32; outer.saturating_mul(out_row)];
    for row in 0..outer {
        let src_start = row
            .saturating_mul(in_row)
            .saturating_add(start.saturating_mul(inner));
        let dst_start = row.saturating_mul(out_row);
        out[dst_start..dst_start + out_row].copy_from_slice(&src[src_start..src_start + out_row]);
    }
    store_f32_out(arena, device, queue, out_byte_off, out, cache.take());
}
