// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! wgpu device discovery + capabilities.
//!
//! [`wgpu_device`] returns a process-global singleton. [`select_vulkan_backend`]
//! routes subsequent calls to a Vulkan-only instance (for [`Device::Vulkan`]).

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::OnceLock;

const PREF_DEFAULT: u8 = 0;
const PREF_VULKAN: u8 = 1;

static BACKEND_PREF: AtomicU8 = AtomicU8::new(PREF_DEFAULT);

/// Detected wgpu adapter + device. We hold them together because
/// every command submission needs both the device (for encoding) and
/// the queue (for committing).
pub struct WgpuDevice {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub name: String,
    pub backend: wgpu::Backend,
}

impl WgpuDevice {
    fn new_with_backends(backends: wgpu::Backends) -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;

        let info = adapter.get_info();
        let limits = adapter.limits();
        let adapter_feats = adapter.features();
        let mut required_features = wgpu::Features::empty();
        if adapter_feats.contains(wgpu::Features::SHADER_F16) {
            required_features |= wgpu::Features::SHADER_F16;
        }
        if adapter_feats.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
            required_features |= wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX;
        }
        if adapter_feats.contains(wgpu::Features::SUBGROUP) {
            required_features |= wgpu::Features::SUBGROUP;
        }

        let (device, queue) =
            match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("rlx-wgpu device"),
                required_features,
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                trace: wgpu::Trace::Off,
            })) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("rlx-wgpu request_device failed: {e}");
                    return None;
                }
            };

        Some(Self {
            instance,
            adapter,
            device,
            queue,
            name: info.name,
            backend: info.backend,
        })
    }

    fn new_default() -> Option<Self> {
        Self::new_with_backends(wgpu::Backends::from_env().unwrap_or(wgpu::Backends::all()))
    }
}

// SAFETY: wgpu's Device + Queue are documented thread-safe.
unsafe impl Send for WgpuDevice {}
unsafe impl Sync for WgpuDevice {}

fn default_device() -> Option<&'static WgpuDevice> {
    static DEVICE: OnceLock<Option<WgpuDevice>> = OnceLock::new();
    DEVICE.get_or_init(WgpuDevice::new_default).as_ref()
}

fn vulkan_device() -> Option<&'static WgpuDevice> {
    static DEVICE: OnceLock<Option<WgpuDevice>> = OnceLock::new();
    DEVICE
        .get_or_init(|| WgpuDevice::new_with_backends(wgpu::Backends::VULKAN))
        .as_ref()
}

/// Prefer the Vulkan-only wgpu instance for [`Device::Vulkan`] sessions.
/// Call before the first [`wgpu_device`] use in that process (or use
/// `Device::Vulkan` via the runtime registry, which calls this).
pub fn select_vulkan_backend() {
    BACKEND_PREF.store(PREF_VULKAN, Ordering::SeqCst);
}

/// True when a Vulkan adapter is reachable (MoltenVK on macOS, native on Linux/Windows).
pub fn is_vulkan_available() -> bool {
    vulkan_device().is_some()
}

/// Get or initialize the global wgpu device singleton. Returns None
/// on systems with no compatible adapter.
pub fn wgpu_device() -> Option<&'static WgpuDevice> {
    if BACKEND_PREF.load(Ordering::SeqCst) == PREF_VULKAN {
        vulkan_device()
    } else {
        default_device()
    }
}
