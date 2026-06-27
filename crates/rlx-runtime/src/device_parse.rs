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

//! String identifiers for [`rlx_driver::Device`] — config files, CLI, env vars.

use rlx_driver::Device;

/// Failed to parse a device name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseDeviceError {
    pub input: String,
    pub message: String,
}

impl std::fmt::Display for ParseDeviceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ParseDeviceError {}

/// Lower-case Cargo feature names and common aliases → [`Device`].
pub fn parse_device(s: &str) -> Result<Device, ParseDeviceError> {
    let key = s.trim().to_ascii_lowercase();
    match key.as_str() {
        "cpu" => Ok(Device::Cpu),
        "metal" | "mtl" => Ok(Device::Metal),
        "mlx" => Ok(Device::Mlx),
        "ane" | "neural-engine" => Ok(Device::Ane),
        "cuda" | "nvidia" => Ok(Device::Cuda),
        "rocm" | "hip" | "amd" => Ok(Device::Rocm),
        "oneapi" | "levelzero" | "level-zero" | "l0" | "intel" | "sycl" => Ok(Device::OneApi),
        "gpu" | "wgpu" => Ok(Device::Gpu),
        "vulkan" | "vk" => Ok(Device::Vulkan),
        "opengl" | "gl" => Ok(Device::OpenGl),
        "directx" | "dx12" | "d3d12" => Ok(Device::DirectX),
        "webgpu" => Ok(Device::WebGpu),
        "tpu" => Ok(Device::Tpu),
        "hexagon" | "htp" | "qnn" => Ok(Device::Hexagon),
        "" => Err(ParseDeviceError {
            input: s.to_string(),
            message: "empty device name".into(),
        }),
        other => Err(ParseDeviceError {
            input: s.to_string(),
            message: format!(
                "unknown device '{other}' (try: cpu, metal, mlx, cuda, rocm, gpu, vulkan, tpu)"
            ),
        }),
    }
}

/// Stable lower-case label for `device` (matches Cargo feature names).
pub fn device_label(device: Device) -> &'static str {
    match device {
        Device::Cpu => "cpu",
        Device::Metal => "metal",
        Device::Mlx => "mlx",
        Device::Ane => "ane",
        Device::Cuda => "cuda",
        Device::Rocm => "rocm",
        Device::OneApi => "oneapi",
        Device::Gpu => "gpu",
        Device::Vulkan => "vulkan",
        Device::OpenGl => "opengl",
        Device::DirectX => "directx",
        Device::WebGpu => "webgpu",
        Device::Tpu => "tpu",
        Device::Hexagon => "hexagon",
    }
}

/// Parse comma/semicolon/whitespace-separated device lists (`RLX_DEVICES=cpu,metal`).
pub fn parse_device_list(s: &str) -> Result<Vec<Device>, ParseDeviceError> {
    let mut out = Vec::new();
    for part in s.split([',', ';', ' ']) {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let dev = parse_device(part)?;
        if !out.contains(&dev) {
            out.push(dev);
        }
    }
    if out.is_empty() {
        return Err(ParseDeviceError {
            input: s.to_string(),
            message: "device list is empty".into(),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_aliases() {
        assert_eq!(parse_device("CUDA").unwrap(), Device::Cuda);
        assert_eq!(parse_device("wgpu").unwrap(), Device::Gpu);
        assert_eq!(
            parse_device_list("cpu, metal;mlx").unwrap(),
            vec![Device::Cpu, Device::Metal, Device::Mlx,]
        );
    }
}
