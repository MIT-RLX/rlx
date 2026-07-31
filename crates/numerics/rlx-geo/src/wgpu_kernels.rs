// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native wgpu (WGSL) kernel for `geo.voronoi_grid`.
//!
//! One dispatch, one thread per output cell, exact brute-force nearest site —
//! the same result the CPU reference produces. It follows the fixed
//! `WgpuGpuKernel` binding convention (`arena: array<f32>` @0, `params` @1);
//! integer operands ride in the f32 arena via `bitcast`.
//!
//! ## Input layout
//! Because `WgpuGpuKernel` receives no op attrs, the grid dimensions travel in
//! the input buffer: input 0 is `[n+1, 2] I32` where **row 0 is `[width,
//! height]`** and rows `1..=n` are the sites. Output is `[height, width] I32`
//! nearest-site indices (`-1` if there are no sites).
//!
//! Squared distances are accumulated in `i32`, exact for grids up to ~46 340 per
//! axis. For larger grids, tile or switch to the Jump-Flooding variant.
//!
//! NOTE: this kernel is provided as validated *source* following the documented
//! convention; end-to-end dispatch should be exercised on a real wgpu device
//! before relying on it in production.

use std::sync::Arc;

use rlx_wgpu::wgpu_gpu_custom::{WgpuGpuKernel, register_wgpu_gpu_kernel};

use crate::ops::GEO_VORONOI_GRID;

/// WGSL source for the `geo.voronoi_grid` kernel (exposed for on-device tests).
pub const VORONOI_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<storage, read>       params: array<u32>;

// params = [ out_off, out_len, n_inputs, _pad, in0_off, in0_len, ... ]
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let out_off = params[0];
    let out_len = params[1];
    if (i >= out_len) { return; }

    let in0     = params[4];
    let in0_len = params[5];

    // Row 0 of the input packs [width, height]; sites follow.
    let width  = bitcast<i32>(arena[in0]);
    let w      = u32(width);
    let n      = (in0_len - 2u) / 2u;

    let x = i32(i % w);
    let y = i32(i / w);

    var best_idx: i32 = -1;
    var best_d:   i32 = 2147483647;
    for (var k: u32 = 0u; k < n; k = k + 1u) {
        let base = in0 + 2u + 2u * k;
        let sx = bitcast<i32>(arena[base]);
        let sy = bitcast<i32>(arena[base + 1u]);
        let dx = sx - x;
        let dy = sy - y;
        let d  = dx * dx + dy * dy;
        if (d < best_d) { best_d = d; best_idx = i32(k); }
    }
    arena[out_off + i] = bitcast<f32>(best_idx);
}
"#;

#[derive(Debug)]
struct VoronoiGridWgpu;

impl WgpuGpuKernel for VoronoiGridWgpu {
    fn name(&self) -> &str {
        GEO_VORONOI_GRID
    }
    fn wgsl(&self) -> &str {
        VORONOI_WGSL
    }
    // (VORONOI_WGSL is `pub` so the on-device example validates the same source.)
    fn entry_point(&self) -> &str {
        "main"
    }
    fn workgroups(&self, out_elems: u32) -> (u32, u32, u32) {
        (out_elems.div_ceil(64).max(1), 1, 1)
    }
}

/// Register rlx-geo's native wgpu kernels. Called from `register_geo_ops`.
pub fn register_wgpu_geo_kernels() {
    register_wgpu_gpu_kernel(Arc::new(VoronoiGridWgpu));
}
