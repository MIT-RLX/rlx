// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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

/// Host-side tensor mirror for consecutive HostOp / ConvHost chains.
///
/// Avoids re-DTOHing activations that a prior host step just produced (Kitten
/// on discrete wgpu: hundreds of Binary/Activation HostOps per chunk). Cleared
/// whenever a GPU compute pass may overwrite the arena.
///
/// Outputs can be **deferred**: kept only in the mirror until
/// [`HostTensorCache::flush_to_device`] before the next GPU pass, so a long
/// host-only chain does one batched H2D instead of one per op.
#[derive(Default)]
pub struct HostTensorCache {
    map: std::collections::HashMap<usize, std::sync::Arc<[f32]>>,
    /// Byte offsets that exist in `map` but may not yet be on the device.
    dirty: std::collections::HashSet<usize>,
}

impl HostTensorCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.dirty.clear();
    }

    pub fn has_deferred_writes(&self) -> bool {
        !self.dirty.is_empty()
    }

    pub fn get(&self, byte_off: usize) -> Option<&[f32]> {
        self.map.get(&byte_off).map(|v| &v[..])
    }

    pub fn get_arc(&self, byte_off: usize) -> Option<std::sync::Arc<[f32]>> {
        self.map.get(&byte_off).cloned()
    }

    /// Cache hit when the mirror covers at least `n` f32s (HostOp may pad to
    /// `size_bytes` while Expand/Narrow ask for `num_elements`).
    pub fn get_arc_covering(&self, byte_off: usize, n: usize) -> Option<std::sync::Arc<[f32]>> {
        let hit = self.map.get(&byte_off)?;
        if hit.len() >= n {
            Some(hit.clone())
        } else {
            None
        }
    }

    pub fn is_dirty(&self, byte_off: usize) -> bool {
        self.dirty.contains(&byte_off)
    }

    pub fn invalidate(&mut self, byte_off: usize) {
        self.map.remove(&byte_off);
        self.dirty.remove(&byte_off);
    }

    pub fn insert(&mut self, byte_off: usize, data: Vec<f32>, defer_htod: bool) {
        if defer_htod {
            self.dirty.insert(byte_off);
        } else {
            self.dirty.remove(&byte_off);
        }
        // Preserve a longer entry's tail. The cache is keyed by byte offset,
        // but *views alias their parent's slot*: a `Narrow(start=0)` of a
        // concat has the parent's exact offset. Replacing outright means the
        // shorter view write destroys the parent's mirror beyond its own
        // length, and a sibling view at an interior offset — `Narrow(start=6)`
        // reading parent+144 — then finds nothing in the cache, nothing dirty
        // to flush, and reads a device region the deferred parent never wrote:
        // zeros, which propagate until the whole graph output is zero.
        //
        // The arena bytes past `data` are unchanged by this write, so the
        // mirror must keep showing them.
        let merged: std::sync::Arc<[f32]> = match self.map.get(&byte_off) {
            Some(prev) if prev.len() > data.len() => {
                let mut v = prev.to_vec();
                v[..data.len()].copy_from_slice(&data);
                std::sync::Arc::from(v)
            }
            _ => std::sync::Arc::<[f32]>::from(data),
        };
        self.map.insert(byte_off, merged);
    }

    /// Write all deferred host outputs to the device (call before a GPU pass).
    pub fn flush_to_device<A: DeviceArena>(&mut self, a: &mut A) {
        if self.dirty.is_empty() {
            return;
        }
        let offs: Vec<usize> = self.dirty.iter().copied().collect();
        for off in offs {
            if let Some(v) = self.map.get(&off) {
                if !v.is_empty() {
                    a.htod(off, bytemuck::cast_slice(v));
                }
            }
        }
        self.dirty.clear();
    }

    /// Flush one deferred offset (before a cache-miss D2H that would otherwise
    /// read zeros for a HostOp that deferred its write).
    pub fn flush_offset<A: DeviceArena>(&mut self, a: &mut A, byte_off: usize) {
        if !self.dirty.remove(&byte_off) {
            return;
        }
        if let Some(v) = self.map.get(&byte_off) {
            if !v.is_empty() {
                a.htod(byte_off, bytemuck::cast_slice(v));
            }
        }
    }
}

/// Per-tensor HostOp staging (sharded / multi-GiB arenas).
///
/// Unlike [`run_host_op_span`], this never mirrors the contiguous gap between
/// operands — each input is read independently, evaluated on the host, and the
/// output is written back. Required when inputs sit on different bind stripes
/// (NVIDIA Kitten: 2 GiB storage bind + ~3.7 GiB arena).
///
/// When `cache` is provided, inputs already produced by a prior host step are
/// reused without a device round-trip; the output is stored for the next host
/// step. Always writes the result back to the device (GPU MatMul may consume it).
pub fn run_host_op_packed<A: DeviceArena>(a: &mut A, desc: &HostOpDesc) {
    run_host_op_packed_cached(a, desc, None);
}

/// Like [`run_host_op_packed`], with an optional [`HostTensorCache`].
pub fn run_host_op_packed_cached<A: DeviceArena>(
    a: &mut A,
    desc: &HostOpDesc,
    mut cache: Option<&mut HostTensorCache>,
) {
    use rlx_cpu::thunk::eval_single_op_f32;
    a.sync();
    if rlx_ir::env::flag("RLX_WGPU_DBG_HOST_OP") {
        eprintln!(
            "[host_op] {:?} out={:?} out_off={:#x} ins={:?}",
            desc.op.kind(),
            desc.out_shape.dims(),
            desc.out_byte_off,
            desc.inputs
                .iter()
                .map(|(o, s)| (*o, s.dims(), s.num_elements()))
                .collect::<Vec<_>>(),
        );
    }
    // Stage inputs: prefer host-mirror hits (no DTOH), else read from device.
    // Keep Arc clones so refs stay valid through eval without re-DTOH copies.
    let staged: Vec<(rlx_ir::Shape, std::sync::Arc<[f32]>)> = desc
        .inputs
        .iter()
        .map(|(off, sh)| {
            let n_elems = sh.num_elements().unwrap_or(0);
            let n = sh.size_bytes().unwrap_or(0).div_ceil(4).max(n_elems);
            if let Some(c) = cache.as_ref() {
                if let Some(hit) = c.get_arc_covering(*off, n_elems) {
                    return (sh.clone(), hit);
                }
            }
            // Deferred producer at this offset but length mismatch — push to
            // device before D2H so we don't read zeros.
            if let Some(c) = cache.as_mut() {
                c.flush_offset(a, *off);
            }
            let mut v = vec![0f32; n];
            if n > 0 {
                a.dtoh(*off, bytemuck::cast_slice_mut(v.as_mut_slice()));
            }
            (sh.clone(), std::sync::Arc::<[f32]>::from(v))
        })
        .collect();
    let refs: Vec<(rlx_ir::Shape, &[f32])> =
        staged.iter().map(|(sh, v)| (sh.clone(), &v[..])).collect();
    let y = eval_single_op_f32(&desc.op, &desc.out_shape, &refs);
    if !y.is_empty() {
        // Defer H2D when a host mirror is active: long HostOp chains (Kitten
        // NSF elementwise) batch one flush before the next GPU / Expand /
        // Concat. Opt out with `RLX_WGPU_HOST_EAGER_H2D=1`.
        let defer = cache.is_some() && !rlx_ir::env::flag("RLX_WGPU_HOST_EAGER_H2D");
        if !defer {
            a.htod(desc.out_byte_off, bytemuck::cast_slice(y.as_slice()));
        }
        if let Some(c) = cache.as_mut() {
            c.insert(desc.out_byte_off, y, defer);
        }
    }
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

    let region_bytes: usize = regions.iter().map(|&(_, n)| n).sum();
    // Prefer packed per-region transfers when the bounding span is mostly holes
    // (Kitten wave: ~88 MiB span for a few small index/update tensors).
    let span_dense = region_bytes > 0 && span_len <= region_bytes.saturating_mul(2);

    if rlx_ir::env::flag("RLX_CUDA_INDEXING_TRACE") {
        eprintln!(
            "[cuda_indexing] regions={} span_lo={lo} span_hi={hi} span_len={span_len} \
             region_bytes={region_bytes} dense={span_dense} cap={}",
            regions.len(),
            contiguous_span_cap.min(INDEXING_CONTIGUOUS_SPAN_CAP)
        );
        for (i, &(off, n)) in regions.iter().enumerate() {
            eprintln!("  region[{i}] off={off} nbytes={n}");
        }
    }

    if span_len > 0
        && span_len <= cap
        && span_dense
        && lo.is_multiple_of(4)
        && span_len.is_multiple_of(4)
    {
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

#[cfg(test)]
mod host_cache_tests {
    use super::HostTensorCache;

    /// A shorter write at the same offset must not truncate a longer entry.
    ///
    /// The cache is keyed by byte offset, but *views alias their parent's slot*:
    /// `Narrow(start=0)` of a concat carries the parent's exact offset. If the
    /// view's write replaced the entry outright, the parent's mirror beyond the
    /// view's length would be lost — and a sibling view at an interior offset
    /// (`Narrow(start=6)`, i.e. parent + 144 B) would then miss the cache, find
    /// nothing dirty to flush, and read a device region the deferred parent
    /// never wrote. That returned zeros, which propagated until the whole graph
    /// output was zero: the `logeig`/`reeig`/`biquad`/`iirfilt` failures on
    /// discrete Vulkan, where these ops all run on the host.
    #[test]
    fn shorter_write_preserves_a_longer_entrys_tail() {
        let mut c = HostTensorCache::new();
        // Parent: 8 elements at offset 4896.
        c.insert(4896, (0..8).map(|i| i as f32).collect(), true);
        // View of its first half, same offset — must not drop elements 4..8.
        c.insert(4896, vec![100.0, 101.0, 102.0, 103.0], true);

        let got = c.get_arc(4896).expect("entry still present");
        assert_eq!(
            &got[..],
            &[100.0, 101.0, 102.0, 103.0, 4.0, 5.0, 6.0, 7.0],
            "shorter write truncated the parent's mirror"
        );
        // The interior read a sibling view performs must still be satisfied.
        assert!(
            c.get_arc_covering(4896, 8).is_some(),
            "parent no longer covers its full length"
        );
    }

    /// A longer write at the same offset legitimately replaces the entry.
    #[test]
    fn longer_write_replaces_wholesale() {
        let mut c = HostTensorCache::new();
        c.insert(64, vec![1.0, 2.0], true);
        c.insert(64, vec![9.0, 8.0, 7.0], true);
        assert_eq!(&c.get_arc(64).unwrap()[..], &[9.0, 8.0, 7.0]);
    }
}
