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

//! Device selection — which backend to use.

/// Target device for graph execution.
///
/// Each variant maps to a backend crate gated by a Cargo feature.
/// Use `Device::is_available()` to check if the feature is enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    // ── CPU ─────────────────────────────────────────────────
    /// CPU with SIMD (NEON/AVX) + optional BLAS.
    Cpu,

    // ── Apple ───────────────────────────────────────────────
    /// GPU via Apple Metal (Metal Performance Shaders).
    Metal,
    /// Apple MLX framework (unified memory GPU).
    Mlx,
    /// Apple Neural Engine.
    Ane,

    // ── NVIDIA ──────────────────────────────────────────────
    /// NVIDIA GPU via native CUDA (cuBLAS, cuDNN).
    Cuda,

    // ── AMD ─────────────────────────────────────────────────
    /// AMD GPU via ROCm/HIP.
    Rocm,

    // ── Google ──────────────────────────────────────────────
    /// Google TPU via libtpu's PJRT plugin (no Python).
    Tpu,

    // ── Cross-platform GPU ──────────────────────────────────
    /// Portable GPU via wgpu (Metal/Vulkan/DX12/WebGPU).
    Gpu,
    /// Vulkan compute shaders.
    Vulkan,
    /// OpenGL compute shaders (legacy).
    OpenGl,
    /// DirectX 12 compute (Windows).
    DirectX,
    /// WebGPU (WASM target).
    WebGpu,
}

impl Device {
    /// Human-readable name (no engine-layer info).
    /// `is_available` / `available` live in rlx-runtime since they
    /// consult the engine's backend registry — keeping them out of
    /// the driver layer preserves the one-way dep direction.
    pub fn name(self) -> &'static str {
        match self {
            Device::Cpu => "CPU",
            Device::Metal => "Metal",
            Device::Mlx => "MLX",
            Device::Ane => "ANE",
            Device::Cuda => "CUDA",
            Device::Rocm => "ROCm",
            Device::Tpu => "TPU",
            Device::Gpu => "GPU (wgpu)",
            Device::Vulkan => "Vulkan",
            Device::OpenGl => "OpenGL",
            Device::DirectX => "DirectX 12",
            Device::WebGpu => "WebGPU",
        }
    }

    /// All variant labels — convenience for callers that want to
    /// enumerate without listing every variant manually. Pair
    /// with `rlx_runtime::available_devices()` to filter.
    pub fn all() -> &'static [Device] {
        &[
            Device::Cpu,
            Device::Metal,
            Device::Mlx,
            Device::Ane,
            Device::Cuda,
            Device::Rocm,
            Device::Tpu,
            Device::Gpu,
            Device::Vulkan,
            Device::OpenGl,
            Device::DirectX,
            Device::WebGpu,
        ]
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Error returned by [`Device::from_str`] when the input doesn't match
/// any known device alias.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceFromStrError(pub String);

impl std::fmt::Display for DeviceFromStrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "unknown device '{}' (try: cpu, metal, mlx, ane, cuda, rocm, gpu, vulkan, opengl, directx, webgpu, tpu)",
            self.0
        )
    }
}

impl std::error::Error for DeviceFromStrError {}

impl std::str::FromStr for Device {
    type Err = DeviceFromStrError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let key = s.trim().to_ascii_lowercase();
        Ok(match key.as_str() {
            "cpu" => Device::Cpu,
            "metal" | "mps" | "mtl" => Device::Metal,
            "mlx" => Device::Mlx,
            "ane" | "neural-engine" => Device::Ane,
            "cuda" | "nvidia" => Device::Cuda,
            "rocm" | "hip" | "amd" => Device::Rocm,
            "gpu" | "wgpu" => Device::Gpu,
            "vulkan" | "vk" => Device::Vulkan,
            "opengl" | "gl" => Device::OpenGl,
            "directx" | "dx12" | "d3d12" => Device::DirectX,
            "webgpu" => Device::WebGpu,
            "tpu" => Device::Tpu,
            _ => return Err(DeviceFromStrError(s.to_string())),
        })
    }
}

/// Per-family backend support filter.
///
/// Each model family declares which devices it can execute on (e.g.,
/// SAM adds TPU on top of the standard set; some VLM crates exclude
/// MLX until the vision tower lands). A single shared
/// [`BackendSupport`] impl per family lets [`validate_device`] return
/// uniform error messages instead of every model crate hand-rolling
/// the same `match` ladder.
pub trait BackendSupport {
    /// Short stable family identifier (`"qwen3"`, `"llama32"`, `"sam"`).
    fn family(&self) -> &'static str;

    /// `true` if this family can execute on `device` today.
    fn supports(&self, device: Device) -> bool;
}

/// Workspace-wide standard backend set: CPU, Metal, MLX, CUDA, ROCm, GPU.
///
/// New families default to this set via [`StandardBackends`] until they
/// need to opt in/out. Mirrors `rlx_core::STANDARD_DEVICES`.
pub const STANDARD_DEVICES: &[Device] = &[
    Device::Cpu,
    Device::Metal,
    Device::Mlx,
    Device::Cuda,
    Device::Rocm,
    Device::Gpu,
];

/// Default [`BackendSupport`] for families on the standard backend set.
#[derive(Debug, Clone, Copy)]
pub struct StandardBackends(pub &'static str);

impl BackendSupport for StandardBackends {
    fn family(&self) -> &'static str {
        self.0
    }
    fn supports(&self, device: Device) -> bool {
        STANDARD_DEVICES.contains(&device)
    }
}

/// Validate that `device` is supported by `family`. Returns the same device
/// on success; on failure, formats a uniform error string. Callers that
/// need a typed error should use [`BackendSupport::supports`] directly.
pub fn validate_device<S: BackendSupport>(support: &S, device: Device) -> Result<Device, String> {
    if support.supports(device) {
        Ok(device)
    } else {
        Err(format!(
            "device {} is not supported by family `{}`",
            device.name(),
            support.family()
        ))
    }
}

#[cfg(test)]
mod from_str_tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn parse_basics() {
        assert_eq!(Device::from_str("cpu").unwrap(), Device::Cpu);
        assert_eq!(Device::from_str("CUDA").unwrap(), Device::Cuda);
        assert_eq!(Device::from_str("mps").unwrap(), Device::Metal);
        assert_eq!(Device::from_str("wgpu").unwrap(), Device::Gpu);
        assert!(Device::from_str("nothing").is_err());
    }

    #[test]
    fn standard_backends_set() {
        let s = StandardBackends("qwen3");
        assert!(s.supports(Device::Cpu));
        assert!(s.supports(Device::Metal));
        assert!(!s.supports(Device::Tpu));
        assert!(validate_device(&s, Device::Tpu).is_err());
    }
}
