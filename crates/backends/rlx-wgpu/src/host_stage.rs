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

//! wgpu adapter for the shared [`rlx_gpu_host`] host-fallback staging trait.

use crate::buffer::Arena;
use rlx_gpu_host::DeviceArena;

/// Wraps a wgpu arena + device/queue so shared `rlx_gpu_host::run_*` kernels
/// can stage bytes. Whole-arena ops use `size_bytes`; span ops that only call
/// `dtoh`/`htod` on bounded ranges may pass `size_bytes: 0`.
pub struct WgpuArena<'a> {
    pub arena: &'a Arena,
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub size_bytes: usize,
}

impl DeviceArena for WgpuArena<'_> {
    fn arena_bytes(&self) -> usize {
        self.size_bytes
    }

    fn sync(&mut self) {
        // read_bytes_range already submits + polls; nothing extra required.
    }

    fn dtoh(&mut self, byte_off: usize, dst: &mut [u8]) {
        let got = self
            .arena
            .read_bytes_range(self.device, self.queue, byte_off, dst.len());
        dst.copy_from_slice(&got);
    }

    fn htod(&mut self, byte_off: usize, src: &[u8]) {
        self.arena.write_bytes_range(self.queue, byte_off, src);
    }
}
