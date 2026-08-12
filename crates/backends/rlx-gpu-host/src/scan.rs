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
    /// Byte offsets that exist in `map` but may not yet be on the device, in
    /// **host write order** (`dirty_set` mirrors it for O(1) membership).
    ///
    /// Order is load-bearing, not cosmetic: mirror entries routinely *overlap*
    /// in the arena — a view aliasing its parent's slot (see [`Self::insert`]),
    /// or a slot the memory planner reused after its first tenant died. Two
    /// overlapping deferred writes must reach the device in the order the host
    /// produced them, exactly as the eager (non-deferred) path would, otherwise
    /// a stale longer entry lands *after* the live shorter one inside its span
    /// and silently clobbers it. A `HashSet` here made that a per-process coin
    /// flip (randomly-seeded iteration order): `Cholesky`'s dead `[5,5]` L slot
    /// overwrote the live `LogDet` scalar the planner had placed inside it, so
    /// wgpu linalg parity failed ~47% of runs with `logdet == 0.0`.
    dirty: Vec<usize>,
    dirty_set: std::collections::HashSet<usize>,
}

impl HostTensorCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.dirty.clear();
        self.dirty_set.clear();
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
        self.dirty_set.contains(&byte_off)
    }

    /// Queue `byte_off` as the most recent deferred write. Re-writing an offset
    /// moves it to the back so the flush keeps matching host program order.
    fn mark_dirty(&mut self, byte_off: usize) {
        if !self.dirty_set.insert(byte_off) {
            if let Some(pos) = self.dirty.iter().position(|&o| o == byte_off) {
                self.dirty.remove(pos);
            }
        }
        self.dirty.push(byte_off);
    }

    fn unmark_dirty(&mut self, byte_off: usize) -> bool {
        if !self.dirty_set.remove(&byte_off) {
            return false;
        }
        if let Some(pos) = self.dirty.iter().position(|&o| o == byte_off) {
            self.dirty.remove(pos);
        }
        true
    }

    pub fn invalidate(&mut self, byte_off: usize) {
        self.map.remove(&byte_off);
        self.unmark_dirty(byte_off);
    }

    pub fn insert(&mut self, byte_off: usize, data: Vec<f32>, defer_htod: bool) {
        if defer_htod {
            self.mark_dirty(byte_off);
        } else {
            self.unmark_dirty(byte_off);
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

    /// Write all deferred host outputs to the device (call before a GPU pass),
    /// oldest write first — see the `dirty` field docs for why order matters.
    pub fn flush_to_device<A: DeviceArena>(&mut self, a: &mut A) {
        if self.dirty.is_empty() {
            return;
        }
        let offs = std::mem::take(&mut self.dirty);
        for off in offs {
            if let Some(v) = self.map.get(&off) {
                if !v.is_empty() {
                    a.htod(off, bytemuck::cast_slice(v));
                }
            }
        }
        self.dirty_set.clear();
    }

    /// Flush one deferred offset (before a cache-miss D2H that would otherwise
    /// read zeros for a HostOp that deferred its write).
    pub fn flush_offset<A: DeviceArena>(&mut self, a: &mut A, byte_off: usize) {
        if !self.unmark_dirty(byte_off) {
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

    /// Byte-addressed arena double that records the H2D order.
    struct VecArena {
        bytes: Vec<u8>,
        writes: Vec<(usize, usize)>,
    }

    impl VecArena {
        fn new(n: usize) -> Self {
            Self {
                bytes: vec![0u8; n],
                writes: Vec::new(),
            }
        }

        fn f32_at(&self, byte_off: usize) -> f32 {
            f32::from_le_bytes(self.bytes[byte_off..byte_off + 4].try_into().unwrap())
        }
    }

    impl crate::DeviceArena for VecArena {
        fn arena_bytes(&self) -> usize {
            self.bytes.len()
        }
        fn sync(&mut self) {}
        fn dtoh(&mut self, byte_off: usize, dst: &mut [u8]) {
            dst.copy_from_slice(&self.bytes[byte_off..byte_off + dst.len()]);
        }
        fn htod(&mut self, byte_off: usize, src: &[u8]) {
            self.writes.push((byte_off, src.len()));
            self.bytes[byte_off..byte_off + src.len()].copy_from_slice(src);
        }
    }

    /// Overlapping deferred writes must reach the device in host write order.
    ///
    /// The memory planner reuses a dead tensor's slot, so a later, smaller live
    /// tensor can sit *inside* an earlier entry's span. This is the real layout
    /// from `linalg_backend_parity`: `Cholesky`'s `[5,5]` L at byte 240 (100 B)
    /// dies at the `TriangularSolve`, and the planner puts the `LogDet` scalar at
    /// byte 256 — 16 B into L's slot. Flushing L *after* the scalar overwrites it
    /// with `L[4]`, the upper-triangle zero. While `dirty` was a `HashSet` its
    /// randomly-seeded iteration order made that a ~47%-per-process coin flip.
    #[test]
    fn overlapping_deferred_writes_flush_in_write_order() {
        let mut c = HostTensorCache::new();
        // Cholesky L: 25 f32 at 240. Index 4 is an upper-triangle zero.
        let mut l = vec![0f32; 25];
        l[0] = 2.35;
        c.insert(240, l, true);
        // TriangularSolve x: 5 f32 at 512 (disjoint).
        c.insert(512, vec![0.4, 0.25, 0.16, -0.02, -0.31], true);
        // LogDet scalar at 256 — inside L's dead slot, written last.
        c.insert(256, vec![8.534641], true);

        let mut a = VecArena::new(1024);
        c.flush_to_device(&mut a);

        assert_eq!(
            a.writes,
            vec![(240, 100), (512, 20), (256, 4)],
            "deferred writes must flush oldest-first, not in hash order"
        );
        assert_eq!(
            a.f32_at(256),
            8.534641,
            "stale L slot clobbered the live LogDet scalar"
        );
        assert!(!c.has_deferred_writes(), "flush must drain the dirty list");
    }

    /// Re-writing an offset moves it to the back: last host write still wins.
    #[test]
    fn rewriting_an_offset_flushes_it_last() {
        let mut c = HostTensorCache::new();
        c.insert(0, vec![1.0; 8], true); // spans bytes 0..32
        c.insert(16, vec![7.0], true); // interior of the above
        c.insert(0, vec![2.0; 8], true); // re-write of the parent, newest

        let mut a = VecArena::new(64);
        c.flush_to_device(&mut a);

        assert_eq!(a.writes, vec![(16, 4), (0, 32)]);
        assert_eq!(a.f32_at(16), 2.0, "newest write to the span must win");
    }

    /// `flush_offset` drains just that offset and leaves the rest ordered.
    #[test]
    fn flush_offset_drains_one_entry_and_keeps_order() {
        let mut c = HostTensorCache::new();
        c.insert(0, vec![1.0], true);
        c.insert(64, vec![2.0], true);
        c.insert(128, vec![3.0], true);

        let mut a = VecArena::new(256);
        c.flush_offset(&mut a, 64);
        assert_eq!(a.writes, vec![(64, 4)]);
        assert!(!c.is_dirty(64));

        c.flush_to_device(&mut a);
        assert_eq!(a.writes, vec![(64, 4), (0, 4), (128, 4)]);
    }

    /// A non-deferred (eager) write clears any pending deferred entry for it.
    #[test]
    fn eager_write_unmarks_a_pending_deferred_entry() {
        let mut c = HostTensorCache::new();
        c.insert(32, vec![1.0], true);
        assert!(c.is_dirty(32));
        // Caller already wrote this one straight to the device.
        c.insert(32, vec![5.0], false);
        assert!(!c.is_dirty(32));
        assert!(!c.has_deferred_writes());

        let mut a = VecArena::new(64);
        c.flush_to_device(&mut a);
        assert!(a.writes.is_empty(), "eager write must not re-flush");
    }
}
