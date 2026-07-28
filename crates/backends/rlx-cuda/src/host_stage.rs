// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CUDA adapter for the shared [`rlx_gpu_host`] host-fallback staging trait.
//!
//! CUDA addresses arena regions as f32-element ranges on a typed
//! `CudaSlice<f32>`; the trait speaks bytes, so this adapter does the `/4`
//! conversion (offsets/lengths are always f32-aligned).

use cudarc::driver::{CudaSlice, CudaStream};
use rlx_gpu_host::DeviceArena;
use std::sync::Arc;

/// Wraps a CUDA stream + f32 arena buffer so the shared `rlx_gpu_host::run_*`
/// host-fallback kernels can stage bytes to/from the device.
pub struct CudaArena<'a> {
    pub stream: &'a Arc<CudaStream>,
    pub buffer: &'a mut CudaSlice<f32>,
    pub size_bytes: usize,
}

impl DeviceArena for CudaArena<'_> {
    fn arena_bytes(&self) -> usize {
        self.size_bytes
    }

    fn sync(&mut self) {
        self.stream
            .synchronize()
            .expect("rlx-cuda: host-fallback pre-sync failed");
    }

    fn dtoh(&mut self, byte_off: usize, dst: &mut [u8]) {
        debug_assert_eq!(
            byte_off % 4,
            0,
            "rlx-cuda host-fallback: unaligned dtoh offset"
        );
        debug_assert_eq!(
            dst.len() % 4,
            0,
            "rlx-cuda host-fallback: unaligned dtoh len"
        );
        let lo = byte_off / 4;
        let hi = lo + dst.len() / 4;
        self.stream
            .memcpy_dtoh(&self.buffer.slice(lo..hi), bytemuck::cast_slice_mut(dst))
            .expect("rlx-cuda: host-fallback dtoh failed");
    }

    fn htod(&mut self, byte_off: usize, src: &[u8]) {
        debug_assert_eq!(
            byte_off % 4,
            0,
            "rlx-cuda host-fallback: unaligned htod offset"
        );
        debug_assert_eq!(
            src.len() % 4,
            0,
            "rlx-cuda host-fallback: unaligned htod len"
        );
        let lo = byte_off / 4;
        let hi = lo + src.len() / 4;
        self.stream
            .memcpy_htod(
                bytemuck::cast_slice(src),
                &mut self.buffer.slice_mut(lo..hi),
            )
            .expect("rlx-cuda: host-fallback htod failed");
    }
}
