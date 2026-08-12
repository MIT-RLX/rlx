// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared device-resident parameters via CUDA virtual memory.
//!
//! # Why
//!
//! Every [`CudaExecutable`](crate::backend) owns its own [`Arena`](crate::arena)
//! and stages every `Op::Param` into it at compile time. One generate run
//! compiles several executables (a prefill-cache graph plus decode buckets, the
//! latter compiled lazily as context grows), so the *same* weights are uploaded
//! once per executable. Measured on Qwen3.5-0.8B: `token_embd.weight` is 1.017 GB
//! and was uploaded **3×** — ~2 GB of duplicated VRAM (peak 3586 MiB, ~2.9 GB of
//! it three copies of one tensor), ~6 GB of redundant HtoD, and a ~100 ms stall
//! each time a bucket compiles mid-generation.
//!
//! # How
//!
//! Kernels take a single `float* arena` plus integer offsets, so params and
//! activations must live in ONE address space — that rules out simply putting
//! params in a second buffer without touching every kernel. CUDA's virtual memory
//! API gives us the same effect for free: reserve a virtual range per arena, then
//! map the *same physical pages* for the parameter block into every arena at the
//! same relative offset, followed by that arena's private activation pages.
//!
//! ```text
//!   exec A VA:  [ shared params ][ A's activations ]
//!                  └── same physical allocation ──┐
//!   exec B VA:  [ shared params ][ B's activations ]
//! ```
//!
//! Mapping one physical allocation at several virtual addresses is documented
//! driver behaviour. **No kernel changes.**
//!
//! # Growable, name-addressed
//!
//! Graphs hold *different subsets* of a model's params — on Qwen3.5 the prefill
//! graph has 401 and the decode bucket 397. So the region cannot be a layout
//! hashed from one graph's param set (that was the first design, and it would
//! have silently never shared). Instead the region is a **bump allocator keyed by
//! `(name, bytes)`**: the first graph to mention a param owns its slot, later
//! graphs reuse it, and unseen params extend the region with fresh physical
//! blocks.
//!
//! Keying on `(name, bytes)` rather than name alone keeps two models that happen
//! to reuse a name (`blk.0.attn_qkv.weight`) at different sizes from colliding.
//!
//! An arena maps every block that exists **at its creation**, and places its
//! activations directly after them. A later arena sees a larger region and a
//! larger `act_base`; earlier arenas are untouched and never address the blocks
//! they lack — which is correct, since a graph only reads params it was compiled
//! with.
//!
//! Opt-in via `RLX_CUDA_SHARED_PARAMS=1` while it is validated beyond one
//! model/GPU; the default path is the private per-executable arena.

use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, MutexGuard, OnceLock};

use cudarc::driver::sys as cu;

/// Allocation properties for device-pinned VMM memory on `device`.
fn alloc_prop(device: cu::CUdevice) -> cu::CUmemAllocationProp {
    let mut prop: cu::CUmemAllocationProp = unsafe { std::mem::zeroed() };
    prop.type_ = cu::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop
}

/// Driver-recommended allocation granularity (typically 2 MiB). Every physical
/// block and virtual reservation must be a multiple of it.
pub fn granularity(device: cu::CUdevice) -> Result<usize, cu::CUresult> {
    let prop = alloc_prop(device);
    let mut gran: usize = 0;
    let r = unsafe {
        cu::cuMemGetAllocationGranularity(
            &mut gran,
            &prop,
            cu::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
        )
    };
    if r != cu::CUresult::CUDA_SUCCESS || gran == 0 {
        return Err(r);
    }
    Ok(gran)
}

#[inline]
fn round_up(v: usize, to: usize) -> usize {
    if to == 0 { v } else { v.div_ceil(to) * to }
}

/// A physical VMM allocation. Handles are driver-refcounted: releasing drops our
/// reference, and the pages survive until the last mapping is unmapped.
struct PhysMem {
    handle: cu::CUmemGenericAllocationHandle,
    bytes: usize,
}

impl PhysMem {
    fn new(device: cu::CUdevice, bytes: usize, gran: usize) -> Result<Self, cu::CUresult> {
        let bytes = round_up(bytes.max(1), gran);
        let prop = alloc_prop(device);
        let mut handle: cu::CUmemGenericAllocationHandle = 0;
        let r = unsafe { cu::cuMemCreate(&mut handle, bytes, &prop, 0) };
        if r != cu::CUresult::CUDA_SUCCESS {
            return Err(r);
        }
        Ok(Self { handle, bytes })
    }
}

impl Drop for PhysMem {
    fn drop(&mut self) {
        unsafe { cu::cuMemRelease(self.handle) };
    }
}

/// The set of physical blocks an arena should map, captured at arena creation.
#[derive(Clone)]
pub struct RegionSnapshot {
    /// `(byte offset within the region, handle, byte length)`.
    blocks: Vec<(usize, cu::CUmemGenericAllocationHandle, usize)>,
    /// Total bytes backed — also the arena's `act_base`.
    pub committed: usize,
}

/// Growable, name-addressed parameter region shared by every executable on a
/// device.
pub struct SharedParamRegion {
    device: cu::CUdevice,
    gran: usize,
    blocks: Vec<(usize, PhysMem)>,
    committed: usize,
    /// `(name, bytes)` → byte offset within the region.
    layout: HashMap<(String, usize), usize>,
    cursor: usize,
    uploaded: HashSet<(String, usize)>,
}

impl SharedParamRegion {
    fn new(device: cu::CUdevice) -> Result<Self, cu::CUresult> {
        Ok(Self {
            device,
            gran: granularity(device)?,
            blocks: Vec::new(),
            committed: 0,
            layout: HashMap::new(),
            cursor: 0,
            uploaded: HashSet::new(),
        })
    }

    /// Back at least `need` bytes with physical memory, adding blocks as needed.
    fn ensure_capacity(&mut self, need: usize) -> Result<(), cu::CUresult> {
        while self.committed < need {
            // Fixed 64 MiB chunks: enough that a 400-param model makes tens of
            // blocks rather than hundreds, without the overshoot of geometric
            // growth (at ~3 GB committed, a `committed/2` chunk would reserve a
            // spare 1.5 GB for one small tensor — measured as VRAM going UP).
            const CHUNK: usize = 64 * 1024 * 1024;
            let want = (need - self.committed).max(CHUNK.min(need)).max(self.gran);
            let block = PhysMem::new(self.device, want, self.gran)?;
            let bytes = block.bytes;
            self.blocks.push((self.committed, block));
            self.committed += bytes;
        }
        Ok(())
    }

    /// Byte offset of `(name, bytes)`, allocating a slot on first sight.
    pub fn slot(&mut self, name: &str, bytes: usize) -> Result<usize, cu::CUresult> {
        let key = (name.to_string(), bytes);
        if let Some(off) = self.layout.get(&key) {
            return Ok(*off);
        }
        SLOTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // 256-byte alignment keeps every slot legal for the vector loads and
        // cuBLAS calls that read params directly.
        let off = round_up(self.cursor, 256);
        self.ensure_capacity(off + bytes)?;
        self.layout.insert(key, off);
        self.cursor = off + bytes;
        Ok(off)
    }

    /// True the first time a given param is claimed for upload; false after, so
    /// only one executable pays for it.
    pub fn claim_upload(&mut self, name: &str, bytes: usize) -> bool {
        self.uploaded.insert((name.to_string(), bytes))
    }

    /// Blocks to map into an arena created now, plus the resulting `act_base`.
    pub fn snapshot(&self) -> RegionSnapshot {
        RegionSnapshot {
            blocks: self
                .blocks
                .iter()
                .map(|(off, b)| (*off, b.handle, b.bytes))
                .collect(),
            committed: self.committed,
        }
    }
}

type Registry = Mutex<HashMap<(i32, u64), SharedParamRegion>>;
static REGIONS: OnceLock<Registry> = OnceLock::new();

/// Lock the region table. Keyed by `(device, scope)` — see [`set_scope`].
pub fn with_region<R>(
    device: cu::CUdevice,
    scope: u64,
    f: impl FnOnce(&mut SharedParamRegion) -> R,
) -> Result<R, cu::CUresult> {
    let reg = REGIONS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard: MutexGuard<'_, HashMap<(i32, u64), SharedParamRegion>> = reg
        .lock()
        .expect("rlx-cuda: shared-param registry poisoned");
    let key = (device, scope);
    // Entry API rather than `contains_key` + `insert` (one lookup, not two).
    // `or_insert_with` is not usable here: constructing a region is fallible
    // and the `?` has to propagate.
    if let std::collections::hash_map::Entry::Vacant(slot) = guard.entry(key) {
        slot.insert(SharedParamRegion::new(device)?);
    }
    Ok(f(guard.get_mut(&key).expect("just inserted")))
}

/// Caller-declared sharing scope. **0 disables sharing entirely** (private
/// arena), and that is the default.
static SCOPE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Declare which compiles may share parameter storage.
///
/// Sharing skips re-uploading a param that is already resident, which is only
/// correct when the same `(name, bytes)` really is the same tensor. **Nothing in
/// the graph can establish that**: parameter *content* arrives later via
/// `set_param`, so two unrelated graphs — or two checkpoints of one architecture
/// — can present identical names and sizes with different weights. Caller
/// identity is therefore load-bearing, not a convenience.
///
/// Set a stable non-zero id per model (e.g. a hash of the weights path). Compiles
/// sharing an id share storage; different ids get independent regions; the
/// default 0 shares nothing.
///
/// This is what `mlx_dequant_matmul_parity` caught: `run_affine` and `run_mxfp`
/// both declare a param `"w"` of exactly 256 bytes with different bytes, and an
/// unscoped region silently served the first one's weights to the second.
pub fn set_scope(scope: u64) {
    SCOPE.store(scope, std::sync::atomic::Ordering::SeqCst);
}

/// Current sharing scope (0 = sharing disabled).
pub fn scope() -> u64 {
    SCOPE.load(std::sync::atomic::Ordering::SeqCst)
}

/// Convenience: derive a stable scope id from any label (a weights path, a model
/// name) and install it.
pub fn set_scope_from_label(label: &str) {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in label.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    set_scope(h | 1); // never 0 — that means "no sharing"
}

/// One executable's address space: `[ shared params | private activations ]`.
///
/// Unmaps and frees the reservation on drop. The shared blocks' pages outlive
/// this because the region still holds their handles.
pub struct VirtualArena {
    va: cu::CUdeviceptr,
    total: usize,
    /// Byte offset at which activations begin (== region `committed` at
    /// creation).
    pub act_base: usize,
}

impl VirtualArena {
    pub fn new(
        device: cu::CUdevice,
        snap: &RegionSnapshot,
        act_bytes: usize,
    ) -> Result<Self, cu::CUresult> {
        let gran = granularity(device)?;
        let act_bytes = round_up(act_bytes.max(1), gran);
        let total = snap.committed + act_bytes;

        let mut va: cu::CUdeviceptr = 0;
        let r = unsafe { cu::cuMemAddressReserve(&mut va, total, gran, 0, 0) };
        if r != cu::CUresult::CUDA_SUCCESS {
            return Err(r);
        }
        let unwind = |e: cu::CUresult, mapped: usize| {
            unsafe {
                if mapped > 0 {
                    cu::cuMemUnmap(va, mapped);
                }
                cu::cuMemAddressFree(va, total);
            }
            e
        };

        // Shared param blocks at the front, in region order.
        let mut mapped = 0usize;
        for (off, handle, bytes) in &snap.blocks {
            let r = unsafe { cu::cuMemMap(va + *off as cu::CUdeviceptr, *bytes, 0, *handle, 0) };
            if r != cu::CUresult::CUDA_SUCCESS {
                return Err(unwind(r, mapped));
            }
            mapped = off + bytes;
        }

        // This executable's private activation pages. Held only by the mapping:
        // released here so the pages die with the unmap in `drop`.
        let act = match PhysMem::new(device, act_bytes, gran) {
            Ok(p) => p,
            Err(e) => return Err(unwind(e, mapped)),
        };
        let r = unsafe {
            cu::cuMemMap(
                va + snap.committed as cu::CUdeviceptr,
                act_bytes,
                0,
                act.handle,
                0,
            )
        };
        if r != cu::CUresult::CUDA_SUCCESS {
            return Err(unwind(r, mapped));
        }
        drop(act);

        let mut access: cu::CUmemAccessDesc = unsafe { std::mem::zeroed() };
        access.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = device;
        access.flags = cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        let r = unsafe { cu::cuMemSetAccess(va, total, &access, 1) };
        if r != cu::CUresult::CUDA_SUCCESS {
            return Err(unwind(r, total));
        }

        Ok(Self {
            va,
            total,
            act_base: snap.committed,
        })
    }

    #[inline]
    pub fn device_ptr(&self) -> cu::CUdeviceptr {
        self.va
    }

    #[inline]
    pub fn total_bytes(&self) -> usize {
        self.total
    }
}

impl Drop for VirtualArena {
    fn drop(&mut self) {
        unsafe {
            cu::cuMemUnmap(self.va, self.total);
            cu::cuMemAddressFree(self.va, self.total);
        }
    }
}

/// Count of distinct slots ever allocated, across every device region. Lets a
/// test assert the shared path was actually taken rather than silently passing
/// on the private-arena fallback.
static SLOTS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Number of distinct `(name, bytes)` slots the shared region has handed out.
pub fn slots_allocated() -> usize {
    SLOTS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Test-only override of [`enabled`]. Tests cannot use `set_var` for this:
/// `std::env::set_var` is `unsafe` and racy against other threads reading the
/// environment, and the harness runs tests in parallel. 0 = defer to the env,
/// 1 = force on, 2 = force off.
static OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Force shared parameters on/off for the rest of the process. Intended for
/// tests; production toggles via `RLX_CUDA_SHARED_PARAMS`.
pub fn set_enabled_for_test(on: bool) {
    OVERRIDE.store(if on { 1 } else { 2 }, std::sync::atomic::Ordering::SeqCst);
}

/// Whether this compile should use the shared parameter region.
///
/// Requires BOTH the opt-in flag and a non-zero [`scope`]: the flag turns the
/// mechanism on, the scope says *which* compiles are the same model. Without a
/// scope there is no sound basis for skipping an upload, so we fall back to a
/// private arena.
pub fn enabled() -> bool {
    if scope() == 0 {
        return false;
    }
    match OVERRIDE.load(std::sync::atomic::Ordering::SeqCst) {
        1 => true,
        2 => false,
        _ => rlx_ir::env::flag("RLX_CUDA_SHARED_PARAMS"),
    }
}
