// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! IQ-family grid LUTs staged into a wgpu storage buffer.
//!
//! Same byte layout as `rlx_metal::kernels::iq_grid_buffer` /
//! `rlx_cuda::iq_grid::cuda_iq_grid_buffer`.

#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Mutex, OnceLock};

fn build_bytes() -> Vec<u8> {
    use rlx_gguf::iq_grids::{
        IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS,
        KSIGNS_IQ2XS,
    };
    let mut bytes = Vec::with_capacity(33_944);
    bytes.extend_from_slice(&KMASK_IQ2XS);
    bytes.extend_from_slice(&KSIGNS_IQ2XS);
    for v in IQ2XXS_GRID.iter() {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for v in IQ2XS_GRID.iter() {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for v in IQ2S_GRID.iter() {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for v in IQ3XXS_GRID.iter() {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for v in IQ3S_GRID.iter() {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for v in IQ1S_GRID.iter() {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Build the IQ grid LUT storage buffer (uncached). The byte layout matches
/// the Metal / CUDA equivalents.
fn create_grid_buffer(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Buffer {
    let bytes = build_bytes();
    let padded = bytes.len().div_ceil(4) * 4;
    let mut upload = bytes;
    upload.resize(padded, 0);
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rlx-wgpu iq_grid_lut"),
        size: padded as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf, 0, &upload);
    buf
}

/// Cached IQ grid LUT (read-only storage). Built once per process.
///
/// Native uses a `Sync` global; on wasm `wgpu::Buffer` is `!Send + !Sync`
/// (the browser is single-threaded), so the cache is a `thread_local` —
/// equivalent here since there is only ever one thread.
#[cfg(not(target_arch = "wasm32"))]
pub fn wgpu_iq_grid_buffer(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Buffer {
    static CACHE: OnceLock<Mutex<Option<wgpu::Buffer>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    let mut guard = cache.lock().expect("iq_grid cache poisoned");
    if let Some(buf) = guard.as_ref() {
        return buf.clone();
    }
    let buf = create_grid_buffer(device, queue);
    guard.replace(buf.clone());
    buf
}

/// Cached IQ grid LUT (read-only storage). Built once per thread on wasm.
#[cfg(target_arch = "wasm32")]
pub fn wgpu_iq_grid_buffer(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Buffer {
    thread_local! {
        static CACHE: std::cell::RefCell<Option<wgpu::Buffer>> =
            const { std::cell::RefCell::new(None) };
    }
    CACHE.with(|cache| {
        let mut guard = cache.borrow_mut();
        if let Some(buf) = guard.as_ref() {
            return buf.clone();
        }
        let buf = create_grid_buffer(device, queue);
        guard.replace(buf.clone());
        buf
    })
}
