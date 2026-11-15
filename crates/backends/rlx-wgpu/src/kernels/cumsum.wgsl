// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Cumulative sum along the last axis. Sequential per-row scan in chunks;
// `exclusive` shifts the result so out[..., 0] = 0.

struct Params {
    outer: u32,
    inner: u32,
    in_off: u32,
    out_off: u32,
    exclusive: u32,
    _p0: u32, _p1: u32, _p2: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn cumsum(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let row = gid.x + gid.y * ngs.x * 64u;
    if (row >= params.outer) { return; }
    let in_base  = params.in_off  + row * params.inner;
    let out_base = params.out_off + row * params.inner;
    // Chunks of 256 scanned from zero, each output = chunk offset + local
    // sum. One running f32 sum over a long row (a vocoder's phase, 80k
    // samples) adds tiny steps to a large total and drifts by ~1e-2; this
    // rounds against the large offset once per chunk instead of per element.
    var offset: f32 = 0.0;
    var start: u32 = 0u;
    loop {
        if (start >= params.inner) { break; }
        let end = min(start + 256u, params.inner);
        var local: f32 = 0.0;
        for (var i: u32 = start; i < end; i = i + 1u) {
            if (params.exclusive != 0u) {
                arena[out_base + i] = offset + local;
                local = local + arena[in_base + i];
            } else {
                local = local + arena[in_base + i];
                arena[out_base + i] = offset + local;
            }
        }
        offset = offset + local;
        start = end;
    }
}
