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

//! Explicit host ↔ device buffer abstraction (plan #59).
//!
//! Borrowed from MAX's `max/python/max/driver/buffer.py`. The point
//! is *explicitness*: every host↔device transfer is a method call,
//! not implicit conversion. Misreading "device" data as host (or
//! vice versa) becomes a compile error, not a silent perf bug.
//!
//! Today this wraps a host-side `Vec<u8>` plus a `Device` tag. The
//! `to_host` / `to_device` calls are no-ops on Device::Cpu; on
//! Device::Metal a future commit will route through the existing
//! `rlx-metal::arena` for actual MTLBuffer transfers.

use crate::Device;
use rlx_ir::{DType, Shape};

/// A buffer that knows where its bytes live.
#[derive(Debug, Clone)]
pub struct Buffer {
    bytes: Vec<u8>,
    shape: Shape,
    device: Device,
}

impl Buffer {
    /// Create a buffer holding `data`, tagged as residing on `device`.
    /// On Cpu this is just storage; on Metal callers will use
    /// `to_device` to move it into device memory once the runtime
    /// integrates with the arena.
    pub fn new_host(shape: Shape, data: Vec<u8>) -> Self {
        Self {
            bytes: data,
            shape,
            device: Device::Cpu,
        }
    }

    /// Create a zero-initialized host buffer of `shape`.
    pub fn zeros(shape: Shape) -> Self {
        let n = shape.size_bytes().unwrap_or(0);
        Self {
            bytes: vec![0u8; n],
            shape,
            device: Device::Cpu,
        }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn device(&self) -> Device {
        self.device
    }
    pub fn dtype(&self) -> DType {
        self.shape.dtype()
    }
    pub fn num_elements(&self) -> usize {
        self.shape.num_elements().unwrap_or(0)
    }
    pub fn byte_size(&self) -> usize {
        self.bytes.len()
    }

    /// Read as `&[f32]`. Panics if dtype isn't F32 — explicit type
    /// check (plan #59 spirit: don't let mismatches go silent).
    pub fn as_f32(&self) -> &[f32] {
        assert_eq!(self.dtype(), DType::F32, "as_f32 on non-F32 buffer");
        let n = self.num_elements();
        unsafe { std::slice::from_raw_parts(self.bytes.as_ptr() as *const f32, n) }
    }

    pub fn as_f32_mut(&mut self) -> &mut [f32] {
        assert_eq!(self.dtype(), DType::F32, "as_f32_mut on non-F32 buffer");
        let n = self.num_elements();
        unsafe { std::slice::from_raw_parts_mut(self.bytes.as_mut_ptr() as *mut f32, n) }
    }

    /// "Move to device" — explicit transfer call. CPU is a no-op
    /// (the bytes are already where they need to be). Metal routes
    /// through the backend (TODO: wire to rlx-metal::arena once a
    /// caller needs it).
    pub fn to_device(self, device: Device) -> Self {
        match (self.device, device) {
            (a, b) if a == b => self,
            (Device::Cpu, Device::Metal) => Self {
                device: Device::Metal,
                ..self
            },
            (Device::Metal, Device::Cpu) => Self {
                device: Device::Cpu,
                ..self
            },
            _ => self,
        }
    }

    /// "Read back to host" — explicit transfer. CPU no-op; Metal
    /// blocks on completion + memcpy back from MTLBuffer (TODO).
    pub fn to_host(self) -> Self {
        self.to_device(Device::Cpu)
    }

    /// Raw bytes — host-side access. Panics if the buffer is on a
    /// non-host device (would silently read uninitialized host
    /// memory otherwise).
    pub fn host_bytes(&self) -> &[u8] {
        assert_eq!(
            self.device,
            Device::Cpu,
            "host_bytes on non-host buffer; call .to_host() first"
        );
        &self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_initializes() {
        let b = Buffer::zeros(Shape::new(&[2, 3], DType::F32));
        assert_eq!(b.num_elements(), 6);
        assert_eq!(b.byte_size(), 24);
        for v in b.as_f32() {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn dtype_mismatch_panics() {
        let b = Buffer::zeros(Shape::new(&[4], DType::I32));
        let result = std::panic::catch_unwind(|| b.as_f32());
        assert!(result.is_err());
    }

    #[test]
    fn to_device_round_trip() {
        let b = Buffer::zeros(Shape::new(&[4], DType::F32));
        let on_metal = b.to_device(Device::Metal);
        assert_eq!(on_metal.device(), Device::Metal);
        let back = on_metal.to_host();
        assert_eq!(back.device(), Device::Cpu);
    }
}
