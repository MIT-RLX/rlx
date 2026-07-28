// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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
        let buf = self
            .device
            .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
        // A Shared buffer is backed by physical RAM at creation. When the request
        // exceeds `maxBufferLength` or free memory, Metal returns a nil-backed
        // buffer (`contents()` == NULL) instead of erroring — and the zero-fill
        // below would then segfault on a NULL write. Fail loudly with the limit.
        if bytes > 0 && buf.contents().is_null() {
            panic!(
                "rlx-metal: failed to allocate a {:.2} GB shared buffer \
                 (maxBufferLength {:.2} GB, likely out of free RAM). \
                 Lower the stage size or shard the arena.",
                bytes as f64 / 1e9,
                self.device.max_buffer_length() as f64 / 1e9,
            );
        }
        // Metal `new_buffer` leaves contents undefined. Ops that read unwritten
        // arena regions (e.g. conv halo padding) would otherwise pick up
        // per-process garbage — a nondeterminism / correctness bug. Shared
        // storage is CPU-visible, so zero it cheaply up front.
        if bytes > 0 {
            unsafe {
                std::ptr::write_bytes(buf.contents() as *mut u8, 0, bytes);
            }
        }
        if std::env::var_os("RLX_METAL_DEBUG").is_some() && bytes > 1 << 30 {
            eprintln!(
                "[rlx-metal] alloc_shared {:.2} GB (maxBufferLength {:.2} GB) → length {:.2} GB",
                bytes as f64 / 1e9,
                self.device.max_buffer_length() as f64 / 1e9,
                buf.length() as f64 / 1e9,
            );
        }
        buf
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

/// Block until the process-global command queue is idle.
///
/// Call after dropping compiled graphs or on GPU fault recovery so later
/// submissions are not rejected with `SubmissionsIgnored`.
#[cfg(all(target_vendor = "apple", not(target_os = "watchos")))]
pub fn drain_command_queue() {
    if let Some(dev) = metal_device() {
        let cb = dev.queue.new_command_buffer();
        cb.commit();
        cb.wait_until_completed();
    }
}

#[cfg(not(all(target_vendor = "apple", not(target_os = "watchos"))))]
pub fn drain_command_queue() {}

/// True if a Metal device is available on this system.
pub fn has_metal_device() -> bool {
    metal_device().is_some()
}
