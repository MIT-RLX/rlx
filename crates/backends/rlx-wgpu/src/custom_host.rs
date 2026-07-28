// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Generic host-delegate for `Op::Custom("onnx.*")` on wgpu arenas.
//!
//! Thin adapter over [`rlx_gpu_host::run_custom_host_bytes`]. wgpu's arena is
//! f32-uniform: integer tensors occupy one f32 slot per element.

use crate::buffer::Arena;
use crate::host_stage::WgpuArena;
use rlx_ir::Shape;

pub use rlx_gpu_host::has_host_kernel;

/// Read each input from the f32 arena, re-encode to its declared dtype, run the
/// CPU reference kernel, and write the f32-cast result back.
/// `in_specs` / `out_byte_off` are **byte** offsets.
#[allow(clippy::too_many_arguments)]
pub fn run_custom_host(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    name: &str,
    in_specs: &[(u32, Shape)],
    out_byte_off: usize,
    out_shape: &Shape,
    attrs: &[u8],
) {
    let specs: Vec<(usize, Shape)> = in_specs
        .iter()
        .map(|(off, sh)| (*off as usize, sh.clone()))
        .collect();
    let mut a = WgpuArena {
        arena,
        device,
        queue,
        size_bytes: 0,
    };
    rlx_gpu_host::run_custom_host_bytes(&mut a, name, &specs, out_byte_off, out_shape, attrs);
}
