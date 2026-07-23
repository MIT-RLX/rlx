// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Vulkan adapter for shared [`rlx_gpu_host`] host-fallback staging.
//!
//! The Vulkan arena is HOST_VISIBLE + mapped, so `dtoh`/`htod` are plain
//! memcpys — but they must route through [`Arena::read_bytes_at`] /
//! [`Arena::write_bytes_at`] so **sharded** arenas (logical size >
//! `maxStorageBufferRange`) do not SIGSEGV when host ops use a single
//! `mapped_ptr()` that only covers shard 0.

use crate::buffer::Arena;
use rlx_gpu_host::DeviceArena;

/// Wraps a Vulkan arena so shared `rlx_gpu_host::run_*` kernels can stage bytes.
pub struct VulkanArena<'a> {
    pub arena: &'a Arena,
}

impl DeviceArena for VulkanArena<'_> {
    fn arena_bytes(&self) -> usize {
        self.arena.size
    }

    fn sync(&mut self) {
        // HOST_COHERENT; invalidate so GPU writes are visible to the host path.
        self.arena.sync_host_after_gpu();
    }

    fn dtoh(&mut self, byte_off: usize, dst: &mut [u8]) {
        let got = self.arena.read_bytes_at(byte_off, dst.len());
        dst.copy_from_slice(&got);
    }

    fn htod(&mut self, byte_off: usize, src: &[u8]) {
        self.arena.write_bytes_at(byte_off, src);
        self.arena.sync_gpu_after_host();
    }
}
