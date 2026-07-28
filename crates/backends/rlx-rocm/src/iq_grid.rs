// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! IQ-family grid LUTs staged into a ROCm device buffer.
//!
//! Byte-identical layout to [`rlx_cuda::iq_grid`] / `rlx_metal::kernels::
//! iq_grid_buffer` — the `dequant_gguf` kernel (shared CUDA/ROCm `.cu`) indexes
//! it for the IQ1/IQ2/IQ3 schemes:
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
//! Total ≈ 33 KB, uploaded once and cached for the process (rlx-rocm is a
//! single-device singleton on device 0 — see [`crate::device`]).

use crate::device::RocmContext;
use crate::hip::HipBuffer;
use std::sync::{Arc, OnceLock};

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

static CACHE: OnceLock<Arc<HipBuffer<u8>>> = OnceLock::new();

/// Per-process IQ grid LUT device buffer. Built once from the constant tables
/// and cached; non-IQ GGUF schemes ignore the pointer but the `dequant_gguf`
/// kernel signature requires it, so callers bind it unconditionally (mirrors
/// [`rlx_cuda::iq_grid::cuda_iq_grid_buffer`]).
pub fn rocm_iq_grid_buffer(ctx: &Arc<RocmContext>) -> Arc<HipBuffer<u8>> {
    Arc::clone(CACHE.get_or_init(|| {
        let bytes = build_bytes();
        let mut buf = HipBuffer::<u8>::alloc_zeros(&ctx.runtime, bytes.len())
            .expect("rlx-rocm: failed to alloc IQ grid LUT");
        buf.copy_from_host(&bytes)
            .expect("rlx-rocm: failed to upload IQ grid LUT");
        Arc::new(buf)
    }))
}
