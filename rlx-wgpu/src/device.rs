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

//! wgpu device discovery + capabilities.
//!
//! `wgpu_device()` returns a process-global singleton constructed
//! lazily on first call. The wgpu API is async; we wrap with
//! `pollster::block_on` so the rest of the backend stays synchronous
//! to match the `rlx-cpu` / `rlx-metal` shape.

use std::sync::OnceLock;

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
    fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
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
        // Opt into SHADER_F16 when the adapter supports it. Apple-Metal,
        // Vulkan w/ VK_KHR_shader_float16_int8, and DX12 all expose this.
        // f16 storage / arithmetic halves memory bandwidth on matmul and
        // — on Apple GPUs — also gives 2× ALU throughput vs. f32.
        let mut required_features = wgpu::Features::empty();
        if adapter_feats.contains(wgpu::Features::SHADER_F16) {
            required_features |= wgpu::Features::SHADER_F16;
        }
        // Cooperative matrix (KHR_cooperative_matrix on Vulkan,
        // simdgroup_matrix on Metal). Unlocks hardware GEMM units —
        // 4-8× speedup on matmul vs portable WGSL ALU. Requires
        // Apple7+/Mac2+ on Metal (M-series), or VK_KHR_cooperative_matrix
        // on Vulkan. Landed in wgpu 29 (gfx-rs/wgpu#8251). Native-only.
        if adapter_feats.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
            required_features |= wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX;
        }
        // SUBGROUP unlocks `subgroupAdd` / `subgroupMax` / etc. — used by
        // the LayerNorm and Softmax kernels to do single-instruction
        // reductions across the simdgroup, replacing workgroup-shared
        // reductions that need explicit barriers.
        if adapter_feats.contains(wgpu::Features::SUBGROUP) {
            required_features |= wgpu::Features::SUBGROUP;
        }

        let (device, queue) =
            match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("rlx-wgpu device"),
                required_features,
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                // SAFETY: enabling experimental features acknowledges that
                // they may contain UB-introducing bugs. We only request
                // EXPERIMENTAL_COOPERATIVE_MATRIX, which has been audited
                // by the wgpu team for the Apple-Metal + Vulkan paths
                // (gfx-rs/wgpu#8251). The risk is bounded to matmul
                // dispatches that explicitly select the coop kernel.
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
}

// SAFETY: wgpu's Device + Queue are documented thread-safe. We never
// share mutable state behind the singleton — the only mutation is on
// command encoders, which are short-lived per-call.
unsafe impl Send for WgpuDevice {}
unsafe impl Sync for WgpuDevice {}

/// Get or initialize the global wgpu device singleton. Returns None
/// on systems with no compatible adapter.
pub fn wgpu_device() -> Option<&'static WgpuDevice> {
    static DEVICE: OnceLock<Option<WgpuDevice>> = OnceLock::new();
    DEVICE.get_or_init(WgpuDevice::new).as_ref()
}
