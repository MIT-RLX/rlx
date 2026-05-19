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
