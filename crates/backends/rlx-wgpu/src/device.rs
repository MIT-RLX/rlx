// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! wgpu device discovery + capabilities.
//!
//! [`wgpu_device`] returns a process-global singleton. [`select_vulkan_backend`]
//! routes subsequent calls to a Vulkan-only instance (for `Device::Vulkan`).

use std::sync::OnceLock;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicU8, Ordering};

#[cfg(not(target_arch = "wasm32"))]
const PREF_DEFAULT: u8 = 0;
#[cfg(not(target_arch = "wasm32"))]
const PREF_VULKAN: u8 = 1;

#[cfg(not(target_arch = "wasm32"))]
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

/// Pick the subset of adapter features we want to enable on the device.
/// Shared by the native (blocking) and wasm (async) constructors so the
/// capability policy stays in one place.
fn required_features_for(adapter: &wgpu::Adapter) -> wgpu::Features {
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
    required_features |= crate::step_profile::required_feature(adapter);
    required_features
}

fn make_instance(backends: wgpu::Backends) -> wgpu::Instance {
    // Release defaults set `VALIDATION_INDIRECT_CALL`, which clamps
    // `max_buffer_size` to `u32::MAX` (~4 GiB). That forces F5-scale arenas
    // onto sharded stripes. Opt into native Metal-sized buffers with
    // `RLX_WGPU_LARGE_BUFFERS=1` (or `WGPU_VALIDATION_INDIRECT_CALL=0`).
    // Oversized unsharded arenas still stage via the dedicated scratch tail
    // (not bind-window end) so live params are not clobbered.
    let mut flags = wgpu::InstanceFlags::default().with_env();
    if rlx_ir::env::flag("RLX_WGPU_LARGE_BUFFERS") {
        flags.remove(wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL);
    }
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        flags,
        backend_options: wgpu::BackendOptions::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        display: None,
    })
}

const ADAPTER_OPTIONS: wgpu::RequestAdapterOptions = wgpu::RequestAdapterOptions {
    power_preference: wgpu::PowerPreference::HighPerformance,
    compatible_surface: None,
    force_fallback_adapter: false,
    apply_limit_buckets: false,
};

fn device_descriptor(
    required_features: wgpu::Features,
    limits: wgpu::Limits,
) -> wgpu::DeviceDescriptor<'static> {
    wgpu::DeviceDescriptor {
        label: Some("rlx-wgpu device"),
        required_features,
        required_limits: limits,
        memory_hints: wgpu::MemoryHints::Performance,
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        trace: wgpu::Trace::Off,
    }
}

impl WgpuDevice {
    /// Native: block on adapter + device discovery via `pollster`.
    #[cfg(not(target_arch = "wasm32"))]
    fn new_with_backends(backends: wgpu::Backends) -> Option<Self> {
        let instance = make_instance(backends);
        let adapter = pollster::block_on(instance.request_adapter(&ADAPTER_OPTIONS)).ok()?;

        let info = adapter.get_info();
        let mut limits = adapter.limits();
        // When large buffers are enabled, keep storage *bindings* ≤ ~4 GiB so
        // ops window into the single physical buffer (whole-buffer binds of
        // >4 GiB returned all-zero F5 DiT outputs on wgpu-Metal).
        if rlx_ir::env::flag("RLX_WGPU_LARGE_BUFFERS") {
            const BIND_CAP: u64 = (4u64 << 30) - 256;
            if limits.max_buffer_size > BIND_CAP {
                limits.max_storage_buffer_binding_size =
                    limits.max_storage_buffer_binding_size.min(BIND_CAP);
                limits.max_uniform_buffer_binding_size =
                    limits.max_uniform_buffer_binding_size.min(BIND_CAP);
            }
        }
        // `RLX_WGPU_MAX_BIND_MB` caps the storage-binding size the device is
        // created with, to exercise the over-the-binding-limit paths (arena
        // windows, parked weights, staging) on an adapter whose real limit is
        // larger — e.g. NVIDIA's 2 GiB on a Mac. It already capped the GGUF
        // path; capping the device makes every path see the same limit.
        if let Some(mb) =
            rlx_ir::env::var("RLX_WGPU_MAX_BIND_MB").and_then(|v| v.parse::<u64>().ok())
        {
            let cap = (mb << 20) / 256 * 256;
            limits.max_storage_buffer_binding_size =
                limits.max_storage_buffer_binding_size.min(cap);
        }
        if rlx_ir::env::flag("RLX_WGPU_PRINT_LIMITS") {
            eprintln!(
                "[rlx-wgpu] adapter limits: max_buffer_size={:.3}GiB max_storage_binding={:.3}GiB",
                limits.max_buffer_size as f64 / ((1u64 << 30) as f64),
                limits.max_storage_buffer_binding_size as f64 / ((1u64 << 30) as f64),
            );
        }
        let required_features = required_features_for(&adapter);

        let (device, queue) = match pollster::block_on(
            adapter.request_device(&device_descriptor(required_features, limits)),
        ) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("rlx-wgpu request_device failed: {e}");
                return None;
            }
        };
        if rlx_ir::env::flag("RLX_WGPU_PRINT_LIMITS") {
            let dl = device.limits();
            eprintln!(
                "[rlx-wgpu] device limits: max_buffer_size={:.3}GiB max_storage_binding={:.3}GiB",
                dl.max_buffer_size as f64 / ((1u64 << 30) as f64),
                dl.max_storage_buffer_binding_size as f64 / ((1u64 << 30) as f64),
            );
        }

        Some(Self {
            instance,
            adapter,
            device,
            queue,
            name: info.name,
            backend: info.backend,
        })
    }

    /// WASM: the browser cannot block its event loop, so adapter + device
    /// discovery is genuinely async — callers `.await` [`init_wgpu_device`]
    /// once at startup, after which [`wgpu_device`] returns synchronously.
    #[cfg(target_arch = "wasm32")]
    async fn new_with_backends_async(backends: wgpu::Backends) -> Option<Self> {
        let instance = make_instance(backends);
        let adapter = instance.request_adapter(&ADAPTER_OPTIONS).await.ok()?;

        let info = adapter.get_info();
        let limits = adapter.limits();
        let required_features = required_features_for(&adapter);

        let (device, queue) = adapter
            .request_device(&device_descriptor(required_features, limits))
            .await
            .ok()?;

        Some(Self {
            instance,
            adapter,
            device,
            queue,
            name: info.name,
            backend: info.backend,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn new_default() -> Option<Self> {
        Self::new_with_backends(default_backends())
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn default_backends() -> wgpu::Backends {
    if let Some(b) = wgpu::Backends::from_env() {
        return b;
    }
    #[cfg(target_os = "windows")]
    {
        // Native DX12 first on Windows MSVC; Vulkan remains as fallback.
        wgpu::Backends::DX12 | wgpu::Backends::VULKAN
    }
    #[cfg(target_os = "linux")]
    {
        // WSL Ubuntu + native Linux: prefer Vulkan (NVIDIA passthrough).
        wgpu::Backends::VULKAN
    }
    #[cfg(target_vendor = "apple")]
    {
        // Every Apple platform (macOS, iOS, tvOS, visionOS): native Metal,
        // plus MoltenVK-translated Vulkan where the user installed it.
        wgpu::Backends::METAL | wgpu::Backends::VULKAN
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_vendor = "apple")))]
    {
        wgpu::Backends::all()
    }
}

// SAFETY: wgpu's Device + Queue are documented thread-safe.
unsafe impl Send for WgpuDevice {}
unsafe impl Sync for WgpuDevice {}

#[cfg(not(target_arch = "wasm32"))]
fn default_device() -> Option<&'static WgpuDevice> {
    static DEVICE: OnceLock<Option<WgpuDevice>> = OnceLock::new();
    DEVICE.get_or_init(WgpuDevice::new_default).as_ref()
}

#[cfg(not(target_arch = "wasm32"))]
fn vulkan_device() -> Option<&'static WgpuDevice> {
    static DEVICE: OnceLock<Option<WgpuDevice>> = OnceLock::new();
    DEVICE
        .get_or_init(|| WgpuDevice::new_with_backends(wgpu::Backends::VULKAN))
        .as_ref()
}

/// Prefer the Vulkan-only wgpu instance for `Device::Vulkan` sessions.
/// Call before the first [`wgpu_device`] use in that process (or use
/// `Device::Vulkan` via the runtime registry, which calls this).
#[cfg(not(target_arch = "wasm32"))]
pub fn select_vulkan_backend() {
    BACKEND_PREF.store(PREF_VULKAN, Ordering::SeqCst);
}

/// No Vulkan path in the browser — the only wasm adapter is WebGPU.
#[cfg(target_arch = "wasm32")]
pub fn select_vulkan_backend() {}

/// True when a Vulkan adapter is reachable (MoltenVK on macOS, native on Linux/Windows).
#[cfg(not(target_arch = "wasm32"))]
pub fn is_vulkan_available() -> bool {
    vulkan_device().is_some()
}

/// No Vulkan in the browser.
#[cfg(target_arch = "wasm32")]
pub fn is_vulkan_available() -> bool {
    false
}

/// Get or initialize the global wgpu device singleton. Returns None
/// on systems with no compatible adapter.
#[cfg(not(target_arch = "wasm32"))]
pub fn wgpu_device() -> Option<&'static WgpuDevice> {
    if BACKEND_PREF.load(Ordering::SeqCst) == PREF_VULKAN {
        vulkan_device()
    } else {
        default_device()
    }
}

// ── WASM device singleton ──────────────────────────────────────────────
//
// The browser cannot block on adapter/device discovery, so the singleton
// is populated asynchronously by [`init_wgpu_device`] (called once at
// startup from JS via `wasm-bindgen-futures`). After that completes,
// [`wgpu_device`] returns the cached device synchronously, so the rest of
// the (synchronous) backend is unchanged.
#[cfg(target_arch = "wasm32")]
static WASM_DEVICE: OnceLock<Option<WgpuDevice>> = OnceLock::new();

/// Initialize the WebGPU device. Idempotent; returns `true` when a device
/// is available. Must be awaited before any [`wgpu_device`] call on wasm.
#[cfg(target_arch = "wasm32")]
pub async fn init_wgpu_device() -> bool {
    if let Some(slot) = WASM_DEVICE.get() {
        return slot.is_some();
    }
    let dev = WgpuDevice::new_with_backends_async(wgpu::Backends::BROWSER_WEBGPU).await;
    let available = dev.is_some();
    let _ = WASM_DEVICE.set(dev);
    available
}

/// Get the wgpu device. On wasm this returns `None` until
/// [`init_wgpu_device`] has been awaited.
#[cfg(target_arch = "wasm32")]
pub fn wgpu_device() -> Option<&'static WgpuDevice> {
    WASM_DEVICE.get().and_then(|d| d.as_ref())
}

/// Adapter name for calibration cache keys.
pub fn adapter_name() -> Option<String> {
    wgpu_device().map(|d| d.name.clone())
}

/// True on Vulkan or DX12 — discrete GPU cooperative-matrix backends.
static FAST_MATH: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Opt into fast WGSL arithmetic: `/`, `sqrt` and `1/sqrt` use the native
/// WGSL operators instead of rlxsl's correctly-rounded helpers. Native `/` may
/// be 2.5 ULP off and Metal's `sqrt` is often 1 ULP off, so results stop
/// matching the CPU bit for bit. `fmod` stays exact (a fast one is wrong by
/// whole multiples, not ULPs). Off by default.
///
/// Kernels are built once per process, so call this before the first
/// compile; a later call does not rebuild kernels that already exist.
pub fn set_fast_math(on: bool) {
    FAST_MATH.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether [`set_fast_math`] is on.
pub fn fast_math() -> bool {
    FAST_MATH.load(std::sync::atomic::Ordering::Relaxed)
}

/// Whether discrete Vulkan/DX12 should still run elementwise, norm and
/// structural ops on the host. Off by default: that policy was a workaround
/// for a KittenTTS "NSF collapse" whose real causes (arena-stripe addressing
/// past the 2 GiB binding limit) are fixed, and it cost EEGNet 295 s against
/// 6.4 s on an RTX 3080 Ti with identical score bits. `RLX_WGPU_DISCRETE_HOST=1`
/// restores it.
pub fn discrete_hosts_ops() -> bool {
    coop_discrete_backend() && rlx_ir::env::flag("RLX_WGPU_DISCRETE_HOST")
}

pub fn coop_discrete_backend() -> bool {
    wgpu_device()
        .map(|d| matches!(d.backend, wgpu::Backend::Vulkan | wgpu::Backend::Dx12))
        .unwrap_or(false)
}

/// True when the adapter reports 8×8×8 f32 cooperative-matrix support
/// (required for `matmul_coop_f32_portable` on Vulkan/DX12).
pub fn coop_f32_8x8_supported() -> bool {
    let dev = match wgpu_device() {
        Some(d) => d,
        None => return false,
    };
    dev.adapter.cooperative_matrix_properties().iter().any(|p| {
        p.m_size == 8
            && p.n_size == 8
            && p.k_size == 8
            && p.ab_type == wgpu::CooperativeScalarType::F32
            && p.cr_type == wgpu::CooperativeScalarType::F32
    })
}

/// True when the adapter reports 8×8×8 f16 cooperative-matrix support with an
/// f16 accumulator — the one shape `matmul_coop16` uses. Apple reports it
/// (`simdgroup_half8x8`); an RTX 3080 Ti reports only 16×16×16, 16×8×16 and
/// 16×8×8, and there `matmul_coop16` failed its self-check under every
/// load/store convention, because the shape itself is unsupported.
pub fn coop_f16_8x8_supported() -> bool {
    let dev = match wgpu_device() {
        Some(d) => d,
        None => return false,
    };
    dev.adapter.cooperative_matrix_properties().iter().any(|p| {
        p.m_size == 8
            && p.n_size == 8
            && p.k_size == 8
            && p.ab_type == wgpu::CooperativeScalarType::F16
            && p.cr_type == wgpu::CooperativeScalarType::F16
    })
}

/// True when the adapter reports 16×16×16 f16 cooperative-matrix support
/// (NVIDIA / AMD tensor-core path on Vulkan/DX12).
pub fn coop_f16_16x16_supported() -> bool {
    let dev = match wgpu_device() {
        Some(d) => d,
        None => return false,
    };
    dev.adapter.cooperative_matrix_properties().iter().any(|p| {
        p.m_size == 16
            && p.n_size == 16
            && p.k_size == 16
            && p.ab_type == wgpu::CooperativeScalarType::F16
            && (p.cr_type == wgpu::CooperativeScalarType::F16
                || p.cr_type == wgpu::CooperativeScalarType::F32)
    })
}

/// True when f16 operands with f32 accumulator are available (preferred on RTX).
pub fn coop_f16_16x16_f32_acc_supported() -> bool {
    let dev = match wgpu_device() {
        Some(d) => d,
        None => return false,
    };
    dev.adapter.cooperative_matrix_properties().iter().any(|p| {
        p.m_size == 16
            && p.n_size == 16
            && p.k_size == 16
            && p.ab_type == wgpu::CooperativeScalarType::F16
            && p.cr_type == wgpu::CooperativeScalarType::F32
    })
}
