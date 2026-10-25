// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// NCHW nearest 2× upsample: `[n, c, h, w]` → `[n, c, 2h, 2w]`.
// One thread per output element.

struct Params {
    n_total: u32, // planes * 4 * h * w
    planes: u32,  // n * c
    h: u32,
    w: u32,
    in_off: u32,
    out_off: u32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn resize_nearest2x(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= params.n_total) { return; }
    let w2 = 2u * params.w;
    let plane_out = 4u * params.h * params.w;
    let plane = i / plane_out;
    let r = i - plane * plane_out;
    let oy = r / w2;
    let ox = r - oy * w2;
    let src = params.in_off + plane * params.h * params.w + (oy / 2u) * params.w + ox / 2u;
    arena[params.out_off + i] = arena[src];
}
