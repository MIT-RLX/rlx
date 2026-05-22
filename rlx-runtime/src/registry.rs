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

//! Backend registry — a single registration point for all backends.
//!
//! Adding a new backend (CUDA, ROCm, wgpu, WASM, TPU) is now a self-contained
//! change in its own crate:
//!
//! ```ignore
//! // in rlx-cuda/src/lib.rs
//! #[cfg(feature = "cuda")]
//! pub fn register() {
//!     rlx_runtime::register_backend(Device::Cuda,
//!         || Box::new(CudaBackend) as Box<dyn Backend>);
//! }
//! ```
//!
//! `Session::compile` consults the registry instead of a hardcoded `match`,
//! so the runtime crate has no compile-time knowledge of which backends are
//! available — each enables itself via its Cargo feature.

use crate::backend::Backend;
use rlx_driver::Device;
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

/// Factory closure that constructs a fresh backend instance.
///
/// Called once per `Session::compile`. Implementations are typically
/// stateless (e.g. unit struct `CpuBackend`); the per-graph state lives
/// inside the returned `Box<dyn Backend>`.
pub type BackendFactory = fn() -> Box<dyn Backend>;

struct Registry {
    factories: RwLock<HashMap<Device, BackendFactory>>,
}

fn registry() -> &'static Registry {
    static REGISTRY: OnceLock<Registry> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let r = Registry {
            factories: RwLock::new(HashMap::new()),
        };
        register_builtin(&r);
        r
    })
}

/// Register builtin backends compiled into `rlx-runtime`. External
/// backends (in their own crates) call `register_backend` from their
/// own init path.
#[allow(unused_mut, unused_variables)]
fn register_builtin(r: &Registry) {
    let mut map = r.factories.write().expect("registry poisoned");

    #[cfg(feature = "cpu")]
    map.insert(Device::Cpu, || {
        Box::new(crate::backend::cpu_backend::CpuBackend) as Box<dyn Backend>
    });

    #[cfg(all(feature = "metal", target_os = "macos"))]
    map.insert(Device::Metal, || {
        Box::new(crate::backend::metal_backend::MetalBackend) as Box<dyn Backend>
    });

    #[cfg(all(feature = "mlx", target_os = "macos"))]
    map.insert(Device::Mlx, || {
        Box::new(crate::backend::mlx_backend::MlxBackend) as Box<dyn Backend>
    });

    #[cfg(feature = "gpu")]
    map.insert(Device::Gpu, || {
        Box::new(crate::backend::wgpu_backend::WgpuBackend) as Box<dyn Backend>
    });

    #[cfg(feature = "vulkan")]
    map.insert(Device::Vulkan, || {
        rlx_wgpu::select_vulkan_backend();
        Box::new(crate::backend::wgpu_backend::WgpuBackend) as Box<dyn Backend>
    });

    #[cfg(feature = "cuda")]
    map.insert(Device::Cuda, || {
        Box::new(crate::backend::cuda_backend::CudaBackend) as Box<dyn Backend>
    });

    #[cfg(feature = "rocm")]
    map.insert(Device::Rocm, || {
        Box::new(crate::backend::rocm_backend::RocmBackend) as Box<dyn Backend>
    });

    #[cfg(feature = "tpu")]
    map.insert(Device::Tpu, || {
        Box::new(crate::backend::tpu_backend::TpuBackend) as Box<dyn Backend>
    });
}

/// Register a backend factory for `device`. External backend crates
/// (rlx-cuda, rlx-rocm, rlx-wgpu, rlx-wasm, …) call this once at startup
/// (typically from a `pub fn register()` in their lib.rs that the user
/// invokes — or via a constructor attribute if they use `ctor`/`inventory`).
///
/// Re-registering a device replaces the prior factory, so a custom backend
/// can override a builtin (useful for swap-in alternatives like a tuned
/// CPU backend).
pub fn register_backend(device: Device, factory: BackendFactory) {
    let r = registry();
    let mut map = r.factories.write().expect("registry poisoned");
    map.insert(device, factory);
}

/// Look up a backend factory and instantiate. Returns `None` if no backend
/// is registered for `device`.
pub fn backend_for(device: Device) -> Option<Box<dyn Backend>> {
    let r = registry();
    let map = r.factories.read().expect("registry poisoned");
    map.get(&device).map(|f| f())
}

/// All currently registered devices (deterministic snapshot).
pub fn registered_devices() -> Vec<Device> {
    let r = registry();
    let map = r.factories.read().expect("registry poisoned");
    let mut out: Vec<Device> = map.keys().copied().collect();
    out.sort_by_key(|d| format!("{d:?}"));
    out
}
