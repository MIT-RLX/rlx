// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Generic host-delegate for `Op::Custom("onnx.*")` on CUDA f32-uniform arenas.
//!
//! Thin adapter over [`rlx_gpu_host::run_custom_host_f32`].

use crate::host_stage::CudaArena;
use cudarc::driver::{CudaSlice, CudaStream};
use rlx_ir::Shape;
use std::sync::Arc;

pub use rlx_gpu_host::has_host_kernel;

/// Stage inputs → CPU reference → write f32 output slots back.
/// `in_specs` / `out_off` are **f32 element** offsets.
pub fn run_custom_host(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    name: &str,
    in_specs: &[(u32, Shape)],
    out_off: u32,
    out_shape: &Shape,
    attrs: &[u8],
) {
    let specs: Vec<(usize, Shape)> = in_specs
        .iter()
        .map(|(off, sh)| (*off as usize, sh.clone()))
        .collect();
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: 0,
    };
    rlx_gpu_host::run_custom_host_f32(&mut arena, name, &specs, out_off as usize, out_shape, attrs);
}
