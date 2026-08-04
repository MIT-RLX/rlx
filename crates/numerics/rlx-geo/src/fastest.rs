// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Backend auto-dispatch: pick the fastest available path for the input.
//!
//! # Why not NPU / ANE / TPU?
//! Delaunay triangulation is **irregular pointer-chasing** over a planar graph
//! with **exact-integer branch decisions** (orientation / in-circle). NPUs, the
//! Apple Neural Engine, and TPUs are fixed-function **dense-tensor** units
//! (matmul / convolution, low precision). There is no tensor/matmul structure to
//! offload and no way to run branchy integer graph code on them. They are
//! excluded on purpose — not a TODO. (The only tensor-shaped sub-step is the
//! Voronoi grid, which is a memory-bound stencil, not a matmul, so even that
//! doesn't fit a systolic array.)
//!
//! # Policy (measured, cross-machine: Apple/Metal, x86+NVIDIA, x86+AMD)
//! * `n < PARALLEL_MIN`  → **CPU serial** — the tuned D&C wins at small n; thread
//!   and dispatch overhead dominate otherwise.
//! * `n >= PARALLEL_MIN` → **CPU parallel** — `std::thread` chunk-build + merge.
//! * **GPU is not auto-selected for host-resident points.** The flip loop is
//!   latency-bound (per-round sync) and loses to CPU D&C on wall-clock once you
//!   count the host→device→host transfer. It wins only when the points already
//!   live in GPU memory — call [`triangulate_on_gpu`] explicitly for that.

use crate::triangulate::{GeoError, parallel_min, triangulate, triangulate_par};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Backend {
    CpuSerial,
    CpuParallel,
    Gpu,
}

impl Backend {
    pub fn name(self) -> &'static str {
        match self {
            Backend::CpuSerial => "cpu-serial",
            Backend::CpuParallel => "cpu-parallel",
            Backend::Gpu => "gpu",
        }
    }
}

/// Triangulate with the fastest CPU backend for the input size. Returns the
/// triangles and which backend was chosen.
///
/// # Errors
/// Propagates [`GeoError`] from the underlying triangulator (oversized span).
pub fn triangulate_fastest(points: &[[i32; 2]]) -> Result<(Vec<[u32; 3]>, Backend), GeoError> {
    if points.len() < parallel_min() {
        Ok((triangulate(points)?, Backend::CpuSerial))
    } else {
        Ok((triangulate_par(points, 0)?, Backend::CpuParallel))
    }
}

/// Data-on-GPU path: seed on the host, run the flip loop on the device. Only
/// beats the CPU backends when the points already reside in GPU memory (no host
/// round-trip). Span ≤ 29 609 and n < 65 536.
#[cfg(feature = "gpu")]
pub fn triangulate_on_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    points: &[[i32; 2]],
) -> (Vec<[u32; 3]>, Backend) {
    let seed = crate::hull_seed(points);
    let tris = crate::flip_gpu::flip_to_delaunay_gpu(device, queue, &seed, points);
    (tris, Backend::Gpu)
}
