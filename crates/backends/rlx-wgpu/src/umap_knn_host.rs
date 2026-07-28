// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host-side `Op::Custom("umap.knn")` for wgpu arenas (small `n` only).

use crate::buffer::Arena;
use crate::host_stage::WgpuArena;

/// Prefer the in-GPU `umap_knn.wgsl` kernel at or above this point count.
pub const UMAP_KNN_GPU_MIN_N: usize = 256;

pub fn run_umap_knn(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pairwise_byte_off: usize,
    out_byte_off: usize,
    n: usize,
    k: usize,
) {
    debug_assert_eq!(pairwise_byte_off % 4, 0);
    debug_assert_eq!(out_byte_off % 4, 0);
    let mut a = WgpuArena {
        arena,
        device,
        queue,
        size_bytes: 0,
    };
    rlx_gpu_host::run_umap_knn(&mut a, pairwise_byte_off / 4, out_byte_off / 4, n, k);
}
