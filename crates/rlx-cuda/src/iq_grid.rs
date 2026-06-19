// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// GPL-3.0-only. See LICENSE.

//! IQ-family grid LUTs staged into a CUDA device buffer.
//!
//! Same layout as `rlx_metal::kernels::iq_grid_buffer`:
//!
//!   offset    bytes    table
//!   0         8        KMASK_IQ2XS (u8 × 8)
//!   8         128      KSIGNS_IQ2XS (u8 × 128)
//!   136       2048     KGRID_IQ2XXS (u64 × 256)
//!   2184      4096     KGRID_IQ2XS (u64 × 512)
//!   6280      8192     KGRID_IQ2S (u64 × 1024)
//!   14472     1024     KGRID_IQ3XXS (u32 × 256)
//!   15496     2048     KGRID_IQ3S (u32 × 512)
//!   17544     16384    KGRID_IQ1S (u64 × 2048)
//!
//! Total ≈ 33 KB. Cached per-context via [`cuda_iq_grid_buffer`].

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use std::sync::{Arc, Mutex, OnceLock};

fn build_bytes() -> Vec<u8> {
    use rlx_gguf::iq_grids::{
        IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS,
        KSIGNS_IQ2XS,
    };
    let mut bytes = Vec::with_capacity(33_944);
    bytes.extend_from_slice(&KMASK_IQ2XS);
    bytes.extend_from_slice(&KSIGNS_IQ2XS);
    for v in IQ2XXS_GRID.iter() {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for v in IQ2XS_GRID.iter() {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for v in IQ2S_GRID.iter() {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for v in IQ3XXS_GRID.iter() {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for v in IQ3S_GRID.iter() {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for v in IQ1S_GRID.iter() {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Per-context cache of the IQ grid LUT device buffer. Built once per
/// CudaContext. We key on the context's ordinal so multi-GPU runs get a
/// buffer per device.
static CACHE: OnceLock<Mutex<Vec<(i32, Arc<CudaSlice<u8>>)>>> = OnceLock::new();

pub fn cuda_iq_grid_buffer(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>) -> Arc<CudaSlice<u8>> {
    let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
    let ord = ctx.ordinal() as i32;
    {
        let guard = cache.lock().expect("iq_grid cache poisoned");
        if let Some((_, buf)) = guard.iter().find(|(o, _)| *o == ord) {
            return Arc::clone(buf);
        }
    }
    let bytes = build_bytes();
    #[allow(deprecated)]
    let slice = stream
        .memcpy_stod(&bytes)
        .expect("rlx-cuda: failed to upload IQ grid LUT");
    let arc = Arc::new(slice);
    let mut guard = cache.lock().expect("iq_grid cache poisoned");
    guard.push((ord, Arc::clone(&arc)));
    arc
}
