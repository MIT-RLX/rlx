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

//! RLX Intel oneAPI backend — the dedicated `Device::OneApi` path for Intel
//! Arc / Data Center Max GPUs via the **Level Zero** runtime.
//!
//! Layout mirrors the other native GPU backends (rlx-cuda / rlx-vulkan):
//! - `level_zero` — hand-rolled `ze_*` FFI, dynamic-loaded (`libze_loader`);
//!   builds + links with no oneAPI runtime present (macOS / CI).
//! - `device`     — driver/device/context/compute-queue singleton; gracefully
//!   unavailable when no Level Zero GPU is reachable.
//! - `kernels`    — embedded OpenCL-C→SPIR-V blobs (`build.rs` via `ocloc`) +
//!   their `ze_module`/`ze_kernel` cache.
//! - `arena`      — f32-uniform USM-shared buffer for the native dispatch path.
//! - `host`       — `rlx-cpu` reference eval (whole-graph on the dev box; per-op
//!   fallback on hardware).
//! - `backend`    — `OneApiExecutable`: legalize → run (native dispatch when a
//!   device + kernels exist, else the CPU-reference interpreter).
//!
//! ## Status
//!
//! The Level Zero bring-up, SPIR-V module/kernel wiring, USM arena, and per-op
//! dispatch are implemented but **not yet validated on Intel hardware** — there
//! is none on the dev box. Until then every op is served by the bit-exact
//! `rlx-cpu` reference, so the backend is *correct* everywhere and *native* on
//! Arc / Data Center Max pending bring-up. See `README.md` for the validation
//! plan and the SPIR-V flavor (OpenCL-Kernel vs Vulkan-Shader) rationale.

pub mod arena;
pub mod backend;
pub mod device;
pub mod host;
pub mod kernels;
pub mod level_zero;

/// True if a Level Zero GPU device is reachable on this system. The runtime
/// registry only registers `Device::OneApi` when this returns `true`, so hosts
/// with no oneAPI runtime (macOS, CI) fall through cleanly.
pub fn is_available() -> bool {
    device::oneapi_device().is_some()
}

/// Human-readable name of the selected Intel device, if any.
pub fn device_name() -> Option<String> {
    device::oneapi_device().map(|d| d.name.clone())
}

/// Whether native SPIR-V kernels were embedded for this build (Intel oneAPI
/// build host with `ocloc` + `RLX_ONEAPI_BUILD_KERNELS=1`). When `false`, the
/// native path serves every op through the CPU reference.
pub fn has_native_kernels() -> bool {
    kernels::kernels_built()
}
