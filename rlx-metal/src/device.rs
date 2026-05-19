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

//! Metal device discovery + capabilities.

use metal::{CommandQueue, Device, MTLResourceOptions};
use std::sync::OnceLock;

/// Detected Metal device properties (read once at startup).
pub struct MetalDevice {
    pub device: Device,
    pub queue: CommandQueue,
    pub name: String,
    pub registry_id: u64,
    /// Recommended max working set size (bytes).
    pub max_working_set: u64,
    /// Whether the device has unified memory (true on Apple Silicon).
    pub has_unified_memory: bool,
}

impl MetalDevice {
    fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let queue = device.new_command_queue();
        let name = device.name().to_string();
        let registry_id = device.registry_id();
        let max_working_set = device.recommended_max_working_set_size();
        let has_unified_memory = device.has_unified_memory();
        Some(Self {
            device,
            queue,
            name,
            registry_id,
            max_working_set,
            has_unified_memory,
        })
    }

    /// Allocate a shared (CPU+GPU accessible) buffer. On Apple Silicon
    /// unified memory, this is zero-copy.
    pub fn alloc_shared(&self, bytes: usize) -> metal::Buffer {
        self.device
            .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared)
    }

    /// Allocate a private (GPU-only) buffer. Lower latency for GPU access.
    pub fn alloc_private(&self, bytes: usize) -> metal::Buffer {
        self.device
            .new_buffer(bytes as u64, MTLResourceOptions::StorageModePrivate)
    }
}

// SAFETY: Metal command queues are thread-safe per Apple docs;
// the Device + Queue are Send/Sync via metal-rs's foreign-types wrappers.
unsafe impl Send for MetalDevice {}
unsafe impl Sync for MetalDevice {}

/// Get or initialize the global Metal device singleton.
pub fn metal_device() -> Option<&'static MetalDevice> {
    static DEVICE: OnceLock<Option<MetalDevice>> = OnceLock::new();
    DEVICE.get_or_init(MetalDevice::new).as_ref()
}

/// True if a Metal device is available on this system.
pub fn has_metal_device() -> bool {
    metal_device().is_some()
}
