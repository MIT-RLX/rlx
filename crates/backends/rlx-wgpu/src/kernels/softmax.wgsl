// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

// Numerically-stable softmax along the last axis. Three sequential
// passes per row by one thread (find max → sum exp → normalize).
// Slow but correct; future work: workgroup tree reduction.

struct Params {
    outer: u32,
    inner: u32,
    in_off: u32,
    out_off: u32,
    _p0: u32, _p1: u32, _p2: u32, _p3: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn softmax(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let row = gid.x + gid.y * ngs.x * 64u;
    if (row >= params.outer || params.inner == 0u) { return; }
    let in_base  = params.in_off  + row * params.inner;
    let out_base = params.out_off + row * params.inner;

    // Pass 1: row max for numerical stability.
    var m: f32 = arena[in_base];
    for (var i: u32 = 1u; i < params.inner; i = i + 1u) {
        m = max(m, arena[in_base + i]);
    }

    // Pass 2: sum of exp(x - max). Kahan compensated sum — F5 DiT has 22
    // Softmax ops per step; plain f32 sum drift compounds across the ODE
    // (wgpu fox 0/6 at NFE=32; Metal stays cos≈1). WGSL has no f64.
    var s: f32 = 0.0;
    var c: f32 = 0.0;
    for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
        let e = exp(arena[in_base + i] - m);
        let y = e - c;
        let t = s + y;
        c = (t - s) - y;
        s = t;
    }

    // Pass 3: write normalized (recompute exp; keeps peak memory low).
    let inv_s = 1.0 / s;
    for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
        arena[out_base + i] = exp(arena[in_base + i] - m) * inv_s;
    }
}
