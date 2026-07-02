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

/// Host-side f32 buffer used for input upload / output download.
pub enum F32HostSlot {
    Pageable(Vec<f32>),
    /// Write-combined pinned — for H2D input staging (host writes only).
    Pinned(PinnedHostSlice<f32>),
    /// Cacheable pinned — for D2H output staging (host reads the result back).
    PinnedCacheable(CacheablePinnedSlice),
}

impl F32HostSlot {
    pub fn new(ctx: &Arc<CudaContext>, len: usize, pinned: bool) -> Self {
        if pinned {
            Self::Pinned(
                unsafe { ctx.alloc_pinned::<f32>(len) }
                    .unwrap_or_else(|e| panic!("rlx-cuda: pinned host alloc failed: {e}")),
            )
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
            Self::Pinned(p) => p.len(),
            Self::PinnedCacheable(p) => p.len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn copy_from_host(&mut self, data: &[f32]) {
        match self {
            Self::Pageable(v) => {
                debug_assert!(data.len() <= v.len());
                v[..data.len()].copy_from_slice(data);
            }
            Self::Pinned(p) => {
                debug_assert!(data.len() <= p.len());
                let dst = p
                    .as_mut_slice()
                    .expect("rlx-cuda: pinned input staging unavailable");
                dst[..data.len()].copy_from_slice(data);
            }
            Self::PinnedCacheable(p) => {
                debug_assert!(data.len() <= p.len);
                p.as_mut_slice()[..data.len()].copy_from_slice(data);
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
            Self::Pinned(p) => stream.memcpy_htod(p, dst),
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
            Self::Pinned(p) => stream.memcpy_dtoh(src, p),
            Self::PinnedCacheable(p) => stream.memcpy_dtoh(src, p.as_mut_slice()),
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        match self {
            Self::Pageable(v) => v.as_slice(),
            Self::Pinned(p) => p.as_slice().expect("rlx-cuda: pinned output read failed"),
            Self::PinnedCacheable(p) => p.as_slice(),
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
