// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Element-wise select: y[i] = cond[i] != 0 ? x[i] : z[i].

struct Params {
    n: u32,
    cond_off: u32,
    x_off: u32,
    y_off: u32,
    out_off: u32,
    _p0: u32, _p1: u32, _p2: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn where_select(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= params.n) { return; }
    let c = arena[params.cond_off + i];
    let x = arena[params.x_off + i];
    let y = arena[params.y_off + i];
    arena[params.out_off + i] = select(y, x, c != 0.0);
}
