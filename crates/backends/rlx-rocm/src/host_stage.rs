// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! ROCm adapter for the shared [`rlx_gpu_host`] host-fallback staging trait.
//!
//! ROCm addresses arena regions as raw byte offsets on a `u64` base pointer,
//! which maps directly onto the trait's byte-oriented `dtoh`/`htod`.

use crate::device::RocmContext;
use crate::hip::HipBuffer;
use rlx_gpu_host::DeviceArena;

/// Wraps a ROCm context + f32 arena buffer so the shared `rlx_gpu_host::run_*`
/// host-fallback kernels can stage bytes to/from the device.
pub struct RocmArena<'a> {
    pub ctx: &'a RocmContext,
    pub buffer: &'a HipBuffer<f32>,
    pub size_bytes: usize,
}

impl DeviceArena for RocmArena<'_> {
    fn arena_bytes(&self) -> usize {
        self.size_bytes
    }

    fn sync(&mut self) {
        unsafe {
            let _ = (self.ctx.runtime.hip_stream_sync)(self.ctx.default_stream);
        }
    }

    fn dtoh(&mut self, byte_off: usize, dst: &mut [u8]) {
        unsafe {
            let _ = (self.ctx.runtime.hip_memcpy_dtoh)(
                dst.as_mut_ptr() as *mut _,
                self.buffer.ptr + byte_off as u64,
                dst.len(),
            );
        }
    }

    fn htod(&mut self, byte_off: usize, src: &[u8]) {
        unsafe {
            let _ = (self.ctx.runtime.hip_memcpy_htod)(
                self.buffer.ptr + byte_off as u64,
                src.as_ptr() as *const _,
                src.len(),
            );
        }
    }
}
