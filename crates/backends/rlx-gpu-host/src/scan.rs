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

//! Shared Scan / HostOp / indexing host-fallbacks over [`DeviceArena`].
//!
//! Discrete GPUs (CUDA/ROCm) mirror the full arena for Scan/HostOp via
//! [`run_scan`] / [`run_host_op`]. wgpu prefers the span-rebased variants
//! [`run_scan_span`] / [`run_host_op_span`] so only the touched region crosses
//! the bus. Indexing always stages touched regions (contiguous or packed).

use crate::DeviceArena;
use rlx_cpu::thunk::{
    HostOpDesc, HostOpSpan, INDEXING_CONTIGUOUS_SPAN_CAP, IndexingHostSpan, IndexingThunk,
    ScanHostDesc, ScanHostSpan, execute_indexing_thunk_on_bytes, indexing_thunk_dst_region,
    indexing_thunk_regions, remap_indexing_thunk_offsets,
};

/// Full-arena Scan staging (CUDA/ROCm style).
pub fn run_scan<A: DeviceArena>(a: &mut A, desc: &ScanHostDesc) {
    let arena_size_bytes = a.arena_bytes();
    rlx_cpu::rlx_scan_stage_d2h! {
        arena_size_bytes = arena_size_bytes,
        desc = desc,
        sync = { a.sync(); },
        dtoh = |host| {
            a.dtoh(0, bytemuck::cast_slice_mut(host.as_mut_slice()));
        },
        htod = |host| {
            a.htod(0, bytemuck::cast_slice(host.as_slice()));
        },
    }
}

/// Full-arena HostOp staging (CUDA/ROCm style).
pub fn run_host_op<A: DeviceArena>(a: &mut A, desc: &HostOpDesc) {
    let arena_size_bytes = a.arena_bytes();
    rlx_cpu::rlx_host_op_stage_d2h! {
        arena_size_bytes = arena_size_bytes,
        desc = desc,
        sync = { a.sync(); },
        dtoh = |host| {
            a.dtoh(0, bytemuck::cast_slice_mut(host.as_mut_slice()));
        },
        htod = |host| {
            a.htod(0, bytemuck::cast_slice(host.as_slice()));
        },
    }
}

/// Span-rebased Scan (wgpu / any backend that shouldn't mirror multi-GiB arenas).
pub fn run_scan_span<A: DeviceArena>(a: &mut A, desc: ScanHostDesc) {
    let span = ScanHostSpan::from_desc(desc);
    a.sync();
    let mut host = vec![0u8; span.len()];
    a.dtoh(span.lo, &mut host);
    unsafe {
        rlx_cpu::rlx_execute_scan_on_bytes!(host.as_mut_ptr(), &span.desc);
    }
    a.htod(span.lo, &host);
}

/// Span-rebased HostOp.
pub fn run_host_op_span<A: DeviceArena>(a: &mut A, desc: HostOpDesc) {
    let span = HostOpSpan::from_desc(desc);
    a.sync();
    let mut host = vec![0u8; span.len()];
    a.dtoh(span.lo, &mut host);
    unsafe {
        rlx_cpu::rlx_execute_host_op_on_bytes!(host.as_mut_ptr(), &span.desc);
    }
    a.htod(span.lo, &host);
}

fn packed_region_off(regions: &[(usize, usize)], packed_at: &[usize], old: usize) -> usize {
    for (i, &(off, _)) in regions.iter().enumerate() {
        if off == old {
            return packed_at[i];
        }
    }
    panic!("rlx-gpu-host indexing pack: missing region for off={old}");
}

fn dtoh_bytes<A: DeviceArena>(a: &mut A, off: usize, nbytes: usize) -> Vec<u8> {
    assert!(
        off.is_multiple_of(4) && nbytes.is_multiple_of(4),
        "rlx-gpu-host indexing: dtoh off={off} nbytes={nbytes} must be f32-aligned"
    );
    let mut host = vec![0u8; nbytes];
    a.dtoh(off, &mut host);
    host
}

fn htod_bytes<A: DeviceArena>(a: &mut A, off: usize, host: &[u8]) {
    assert!(
        off.is_multiple_of(4) && host.len().is_multiple_of(4),
        "rlx-gpu-host indexing: htod off={off} len={} must be f32-aligned",
        host.len()
    );
    a.htod(off, host);
}

/// Indexing host-fallback: contiguous span when small, else packed regions.
///
/// Contiguous path writes **destination only** back to the device (CUDA policy)
/// so aliased live arena slots are not stomped. Optional full-arena mirror via
/// `RLX_INDEXING_FULL_ARENA=1` for A/B bisects.
pub fn run_indexing<A: DeviceArena>(
    a: &mut A,
    arena_size_bytes: usize,
    thunk: &IndexingThunk,
    contiguous_span_cap: usize,
) {
    if rlx_ir::env::flag("RLX_INDEXING_FULL_ARENA")
        || rlx_ir::env::flag("RLX_CUDA_INDEXING_FULL_ARENA")
    {
        let nbytes = if arena_size_bytes > 0 {
            arena_size_bytes
        } else {
            a.arena_bytes()
        };
        rlx_cpu::rlx_indexing_stage_d2h! {
            arena_size_bytes = nbytes,
            thunk = thunk.inner(),
            sync = { a.sync(); },
            dtoh = |host| {
                a.dtoh(0, bytemuck::cast_slice_mut(host.as_mut_slice()));
            },
            htod = |host| {
                a.htod(0, bytemuck::cast_slice(host.as_slice()));
            },
        }
        return;
    }

    let inner = thunk.inner().clone();
    let regions = indexing_thunk_regions(&inner);
    let mut lo = regions[0].0;
    let mut hi = regions[0].0 + regions[0].1;
    for &(off, n) in &regions[1..] {
        lo = lo.min(off);
        hi = hi.max(off + n);
    }
    let span_len = hi.saturating_sub(lo);
    let cap = contiguous_span_cap.min(INDEXING_CONTIGUOUS_SPAN_CAP);

    a.sync();

    if span_len > 0 && span_len <= cap && lo.is_multiple_of(4) && span_len.is_multiple_of(4) {
        let span = IndexingHostSpan::from_thunk(inner);
        let mut host = dtoh_bytes(a, span.lo, span.len());
        unsafe {
            execute_indexing_thunk_on_bytes(host.as_mut_ptr(), &span.thunk);
        }
        let (dst_old, dst_nbytes) = indexing_thunk_dst_region(thunk.inner());
        let dst_rel = dst_old - span.lo;
        htod_bytes(a, dst_old, &host[dst_rel..dst_rel + dst_nbytes]);
        return;
    }

    let mut host: Vec<u8> = Vec::new();
    let mut packed_at: Vec<usize> = Vec::with_capacity(regions.len());
    for &(off, n) in &regions {
        let bytes = dtoh_bytes(a, off, n);
        let at = host.len();
        packed_at.push(at);
        host.extend_from_slice(&bytes);
        while !host.len().is_multiple_of(8) {
            host.push(0);
        }
    }
    let packed = remap_indexing_thunk_offsets(inner.clone(), |old| {
        packed_region_off(&regions, &packed_at, old)
    });
    unsafe {
        execute_indexing_thunk_on_bytes(host.as_mut_ptr(), &packed);
    }
    let (dst_old, dst_nbytes) = indexing_thunk_dst_region(&inner);
    let dst_packed = packed_region_off(&regions, &packed_at, dst_old);
    htod_bytes(a, dst_old, &host[dst_packed..dst_packed + dst_nbytes]);
}
