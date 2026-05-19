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

//! String <-> Device parsing and availability lookup.
//!
//! Python users pass devices as strings ("cpu", "metal", "cuda", ...);
//! this module is the single point of conversion to `rlx_driver::Device`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rlx_runtime::Device;

/// Map a string identifier to a Device. Accepts the lower-case names
/// used in cargo features (cpu, metal, mlx, ane, cuda, rocm, gpu/wgpu,
/// vulkan, opengl, directx, webgpu) plus a handful of aliases.
pub(crate) fn parse_device(s: &str) -> PyResult<Device> {
    let key = s.trim().to_ascii_lowercase();
    let dev = match key.as_str() {
        "cpu" => Device::Cpu,
        "metal" | "mtl" => Device::Metal,
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
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown device '{other}' (try: cpu, metal, mlx, cuda, rocm, gpu, vulkan)"
            )));
        }
    };
    Ok(dev)
}

pub(crate) fn device_label(d: Device) -> &'static str {
    match d {
        Device::Cpu => "cpu",
        Device::Metal => "metal",
        Device::Mlx => "mlx",
        Device::Ane => "ane",
        Device::Cuda => "cuda",
        Device::Rocm => "rocm",
        Device::Gpu => "gpu",
        Device::Vulkan => "vulkan",
        Device::OpenGl => "opengl",
        Device::DirectX => "directx",
        Device::WebGpu => "webgpu",
        Device::Tpu => "tpu",
    }
}

/// `pyrlx.available_devices()` — list of devices that have a backend
/// registered in this build.
#[pyfunction]
pub(crate) fn available_devices() -> Vec<&'static str> {
    rlx_runtime::available_devices()
        .into_iter()
        .map(device_label)
        .collect()
}

/// `pyrlx.is_available("cuda")`
#[pyfunction]
pub(crate) fn is_available(name: &str) -> PyResult<bool> {
    Ok(rlx_runtime::is_available(parse_device(name)?))
}
