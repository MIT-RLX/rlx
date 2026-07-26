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

// Cumulative product / maximum along the last axis. Sequential per-row scan;
// `is_max` selects a running max (identity -inf) over a running product
// (identity 1). `exclusive` shifts the result so out[..., 0] = identity.

struct Params {
    outer: u32,
    inner: u32,
    in_off: u32,
    out_off: u32,
    exclusive: u32,
    is_max: u32,
    _p0: u32, _p1: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn cum_scan(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let row = gid.x + gid.y * ngs.x * 64u;
    if (row >= params.outer) { return; }
    let in_base  = params.in_off  + row * params.inner;
    let out_base = params.out_off + row * params.inner;
    var acc: f32 = select(1.0, -3.4028235e38, params.is_max != 0u);
    for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
        let v = arena[in_base + i];
        if (params.exclusive != 0u) {
            arena[out_base + i] = acc;
            acc = select(acc * v, max(acc, v), params.is_max != 0u);
        } else {
            acc = select(acc * v, max(acc, v), params.is_max != 0u);
            arena[out_base + i] = acc;
        }
    }
}
