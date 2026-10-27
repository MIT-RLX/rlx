// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! GPU kernels for ops that used to run as host steps: `ArgMax` / `ArgMin` and
//! `Reverse`.
//!
//! A host step makes the run wait on the GPU mid-graph, which the browser
//! cannot do (`run_async` refuses such a schedule), so a WebGPU model with one
//! argmax or flip fell back to the CPU there. Both ops are pure indexing, so
//! they run here as [`crate::wgpu_gpu_custom`] kernels against an arena window:
//! `params = [out_off, out_len, n_inputs=1, 0, in_off, in_len, extra…]`, all
//! offsets in f32 elements relative to the window.
//!
//! Semantics are rlx-cpu's exactly. Argreduce is a sequential scan per output
//! with a strict comparison, so ties keep the first index and a NaN never wins
//! (a NaN in the first slot keeps index 0); the index is written as `f32`.

use std::sync::{Arc, Once};

use crate::wgpu_gpu_custom::{WgpuGpuKernel, register_wgpu_gpu_kernel};

pub const ARGREDUCE: &str = "rlx.argreduce";
pub const REVERSE: &str = "rlx.reverse";

/// Up to this many dims in a reversed tensor (rlx graphs stay well below).
pub const REVERSE_MAX_RANK: usize = 8;

/// 2-D grid for `n` threads at 64 per workgroup (≤ 65535 groups per axis).
fn grid(n: u32) -> (u32, u32, u32) {
    let groups = n.div_ceil(64).max(1);
    let gx = groups.min(65535);
    (gx, groups.div_ceil(gx), 1)
}

#[derive(Debug)]
struct ArgReduce;

impl WgpuGpuKernel for ArgReduce {
    fn name(&self) -> &str {
        ARGREDUCE
    }
    fn wgsl(&self) -> &str {
        ARGREDUCE_WGSL
    }
    fn workgroups(&self, out_elems: u32) -> (u32, u32, u32) {
        grid(out_elems)
    }
}

#[derive(Debug)]
struct Reverse;

impl WgpuGpuKernel for Reverse {
    fn name(&self) -> &str {
        REVERSE
    }
    fn wgsl(&self) -> &str {
        REVERSE_WGSL
    }
    fn workgroups(&self, out_elems: u32) -> (u32, u32, u32) {
        grid(out_elems)
    }
}

/// Register both kernels (idempotent).
pub fn ensure_registered() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        register_wgpu_gpu_kernel(Arc::new(ArgReduce));
        register_wgpu_gpu_kernel(Arc::new(Reverse));
    });
}

// params: [out_off, out_len, 1, 0, in_off, in_len, outer, reduced, inner, is_max]
const ARGREDUCE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let idx = gid.x + gid.y * ngs.x * 64u;
    let out_off = params[0];
    let in_off = params[4];
    let reduced = params[7];
    let inner = params[8];
    let is_max = params[9] != 0u;
    if (idx >= params[1]) { return; }
    let o = idx / inner;
    let i = idx % inner;
    let base = in_off + o * reduced * inner + i;
    var best = arena[base];
    var best_idx = 0u;
    for (var r = 1u; r < reduced; r = r + 1u) {
        let v = arena[base + r * inner];
        let better = select((v < best), (v > best), is_max);
        if (better) {
            best = v;
            best_idx = r;
        }
    }
    arena[out_off + idx] = f32(best_idx);
}
"#;

// params: [out_off, out_len, 1, 0, in_off, in_len, rank, flip_mask, dims[0..8]]
const REVERSE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let idx = gid.x + gid.y * ngs.x * 64u;
    if (idx >= params[1]) { return; }
    let rank = params[6];
    let mask = params[7];
    // Row-major: walk the axes from the last, flipping the masked ones.
    var rem = idx;
    var src = 0u;
    var stride = 1u;
    for (var k = 0u; k < rank; k = k + 1u) {
        let ax = rank - 1u - k;
        let d = params[8u + ax];
        let c = rem % d;
        rem = rem / d;
        let s = select(c, d - 1u - c, ((mask >> ax) & 1u) != 0u);
        src = src + s * stride;
        stride = stride * d;
    }
    arena[params[0] + idx] = arena[params[4] + src];
}
"#;
