// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// RLX — versatile ML compiler + runtime.
//
// Pageable or pinned host staging for faster H2D/D2H on the CUDA run hot path.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, DriverError, PinnedHostSlice};

/// Page-locked host buffer allocated **cacheable** (default `cuMemHostAlloc`
/// flags), as opposed to cudarc's [`PinnedHostSlice`] which hardcodes
/// `CU_MEMHOSTALLOC_WRITECOMBINED`.
///
/// Write-combined pinned memory is the right choice for **H2D** staging (the
/// host only ever writes it, and WC gives faster streaming writes + no cache
/// pollution). But for **D2H** output staging the host *reads* the buffer back
/// (`to_vec` / `copy_into`), and reads from WC memory are uncached — measured
/// at ~240 MB/s, which made a 33 MB FFT readback take ~138 ms (vs ~3 ms for the
/// DMA itself). Cacheable pinned memory keeps the fast pinned DMA *and* restores
/// full-bandwidth host reads.
pub struct CacheablePinnedSlice {
    ptr: *mut f32,
    len: usize,
    ctx: Arc<CudaContext>,
}

// SAFETY: mirrors cudarc's own `PinnedHostSlice` — the pointer is a page-locked
// host allocation owned solely by this slot; no aliasing across threads.
unsafe impl Send for CacheablePinnedSlice {}
unsafe impl Sync for CacheablePinnedSlice {}

impl CacheablePinnedSlice {
    fn new(ctx: &Arc<CudaContext>, len: usize) -> Result<Self, DriverError> {
        ctx.bind_to_thread()?;
        let bytes = len * std::mem::size_of::<f32>();
        // flags = 0 → cudaHostAllocDefault: page-locked but cacheable.
        let ptr = unsafe { cudarc::driver::result::malloc_host(bytes, 0)? } as *mut f32;
        assert!(
            !ptr.is_null(),
            "rlx-cuda: cacheable pinned alloc returned null"
        );
        assert!(
            ptr.is_aligned(),
            "rlx-cuda: cacheable pinned alloc misaligned"
        );
        Ok(Self {
            ptr,
            len,
            ctx: ctx.clone(),
        })
    }

    #[inline]
    fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for CacheablePinnedSlice {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        unsafe {
            let _ = cudarc::driver::result::free_host(self.ptr as _);
        }
    }
}

/// Emit a warning once per distinct message — these sit on the run hot path, so
/// a repeating fallback must not turn into a per-step log flood.
fn warn_once(msg: &str) {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    if let Ok(mut g) = seen.lock()
        && g.insert(msg.to_string())
    {
        eprintln!("{msg}");
    }
}

/// Host-side f32 buffer used for input upload / output download.
pub enum F32HostSlot {
    Pageable(Vec<f32>),
    /// Write-combined pinned — for H2D input staging (host writes only).
    ///
    /// `ctx` is retained so the slot can drain in-flight DMA before falling back
    /// (see [`F32HostSlot::copy_from_host`]).
    Pinned(PinnedHostSlice<f32>, Arc<CudaContext>),
    /// Cacheable pinned — for D2H output staging (host reads the result back).
    PinnedCacheable(CacheablePinnedSlice),
    /// A pinned slot that failed and was demoted to pageable staging.
    ///
    /// The original pinned allocation is **retained, not freed**: dropping it
    /// could free host memory that an in-flight DMA is still reading. It is
    /// simply never written again.
    Demoted {
        _retired: PinnedHostSlice<f32>,
        host: Vec<f32>,
    },
}

impl F32HostSlot {
    pub fn new(ctx: &Arc<CudaContext>, len: usize, pinned: bool) -> Self {
        if pinned {
            match unsafe { ctx.alloc_pinned::<f32>(len) } {
                Ok(p) => Self::Pinned(p, ctx.clone()),
                // Pinned staging is a bandwidth optimization, never a
                // correctness requirement — a host-alloc failure (typically
                // pinned-memory exhaustion, which is a global resource) must not
                // take the process down.
                Err(e) => {
                    warn_once(&format!(
                        "rlx-cuda: pinned host alloc failed ({e}); using pageable input staging"
                    ));
                    Self::Pageable(vec![0.0f32; len])
                }
            }
        } else {
            Self::Pageable(vec![0.0f32; len])
        }
    }

    /// Output staging slot. When `pinned`, uses **cacheable** pinned memory so
    /// the host-read side of the D2H readback runs at full bandwidth (see
    /// [`CacheablePinnedSlice`]). Falls back to pageable on alloc failure.
    pub fn new_output(ctx: &Arc<CudaContext>, len: usize, pinned: bool) -> Self {
        if pinned {
            match CacheablePinnedSlice::new(ctx, len) {
                Ok(s) => Self::PinnedCacheable(s),
                Err(_) => Self::Pageable(vec![0.0f32; len]),
            }
        } else {
            Self::Pageable(vec![0.0f32; len])
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Pageable(v) => v.len(),
            Self::Demoted { host, .. } => host.len(),
            Self::Pinned(p, _) => p.len(),
            Self::PinnedCacheable(p) => p.len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Stage `data` for upload.
    ///
    /// `PinnedHostSlice::as_mut_slice` first synchronizes the slice's event —
    /// the guard against overwriting a buffer whose DMA is still in flight — so
    /// it is fallible. It used to be `expect`ed, which turned a driver-level
    /// hiccup in a *bandwidth optimization* into a process abort; that is how the
    /// `CUDA_ERROR_INVALID_VALUE` crash under repeated `set_param` surfaced.
    ///
    /// On failure this now:
    /// 1. drains the context (any in-flight DMA out of this buffer completes),
    /// 2. retries once — a transient stream/event state resolves here,
    /// 3. otherwise demotes the slot to pageable staging for good.
    ///
    /// The drain in step 1 is what makes demotion safe: the retired pinned
    /// allocation is kept alive regardless, but by then nothing is reading it.
    /// Pageable staging is always correct, just slower, so the worst case is a
    /// bandwidth regression on one input with a warning — never a wrong answer
    /// and never a crash.
    pub fn copy_from_host(&mut self, data: &[f32]) {
        match self {
            Self::Pageable(v) => {
                debug_assert!(data.len() <= v.len());
                v[..data.len()].copy_from_slice(data);
            }
            Self::Demoted { host, .. } => {
                debug_assert!(data.len() <= host.len());
                host[..data.len()].copy_from_slice(data);
            }
            Self::PinnedCacheable(p) => {
                debug_assert!(data.len() <= p.len);
                p.as_mut_slice()[..data.len()].copy_from_slice(data);
            }
            Self::Pinned(p, ctx) => {
                debug_assert!(data.len() <= p.len());
                if let Ok(dst) = p.as_mut_slice() {
                    dst[..data.len()].copy_from_slice(data);
                    return;
                }
                // Drain, then retry once.
                let drained = ctx.bind_to_thread().and_then(|_| ctx.synchronize());
                if drained.is_ok()
                    && let Ok(dst) = p.as_mut_slice()
                {
                    dst[..data.len()].copy_from_slice(data);
                    return;
                }
                warn_once(
                    "rlx-cuda: pinned input staging unavailable after a context drain; \
                     falling back to pageable staging for this input (slower H2D, same results)",
                );
                let len = p.len();
                let mut host = vec![0.0f32; len];
                host[..data.len()].copy_from_slice(data);
                // Replace in place, retaining the pinned allocation.
                let old = std::mem::replace(self, Self::Pageable(Vec::new()));
                *self = match old {
                    Self::Pinned(retired, _) => Self::Demoted {
                        _retired: retired,
                        host,
                    },
                    other => other,
                };
            }
        }
    }

    pub fn htod(
        &self,
        stream: &Arc<CudaStream>,
        dst: &mut cudarc::driver::CudaViewMut<f32>,
        len: usize,
    ) -> Result<(), DriverError> {
        debug_assert!(len <= self.len());
        match self {
            Self::Pageable(v) => stream.memcpy_htod(&v[..len], dst),
            Self::Demoted { host, .. } => stream.memcpy_htod(&host[..len], dst),
            Self::Pinned(p, _) => stream.memcpy_htod(p, dst),
            Self::PinnedCacheable(p) => stream.memcpy_htod(&p.as_slice()[..len], dst),
        }
    }

    pub fn dtoh(
        &mut self,
        stream: &Arc<CudaStream>,
        src: &cudarc::driver::CudaView<f32>,
    ) -> Result<(), DriverError> {
        match self {
            Self::Pageable(v) => stream.memcpy_dtoh(src, v.as_mut_slice()),
            Self::Demoted { host, .. } => stream.memcpy_dtoh(src, host.as_mut_slice()),
            Self::Pinned(p, _) => stream.memcpy_dtoh(src, p),
            Self::PinnedCacheable(p) => stream.memcpy_dtoh(src, p.as_mut_slice()),
        }
    }

    /// Read the staged bytes back — the **output** path (`copy_into`, `to_vec`).
    ///
    /// [`Self::Pinned`] is the write-combined *input* staging variant, produced
    /// only by [`Self::new`]; output slots come from [`Self::new_output`], which
    /// yields [`Self::PinnedCacheable`] or [`Self::Pageable`]. So the `Pinned`
    /// arm is unreachable by construction, and reaching it means an input slot
    /// was wired to an output — a wiring bug, not a runtime condition. Unlike
    /// the write path (where the driver can legitimately refuse and we degrade),
    /// there is nothing to degrade *to* here, so it stays a hard error with a
    /// message that names the actual cause.
    pub fn as_slice(&self) -> &[f32] {
        match self {
            Self::Pageable(v) => v.as_slice(),
            Self::Demoted { host, .. } => host.as_slice(),
            Self::PinnedCacheable(p) => p.as_slice(),
            Self::Pinned(p, _) => p.as_slice().expect(
                "rlx-cuda: read-back from a write-combined INPUT staging slot — \
                 output slots must be built with F32HostSlot::new_output",
            ),
        }
    }

    pub fn copy_into(&self, dst: &mut [f32]) {
        let src = self.as_slice();
        debug_assert!(dst.len() <= src.len());
        dst.copy_from_slice(&src[..dst.len()]);
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.as_slice().to_vec()
    }
}
