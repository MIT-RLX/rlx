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

// Argmax along the last axis. Returns f32-encoded indices (rlx
// convention). One thread per output row, sequential scan to find
// the position of the max. Ties broken by the smaller index.

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
fn argmax(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let row = gid.x + gid.y * ngs.x * 64u;
    if (row >= params.outer || params.inner == 0u) { return; }
    let base = params.in_off + row * params.inner;
    var best_idx: u32 = 0u;
    var best_val: f32 = arena[base];
    for (var i: u32 = 1u; i < params.inner; i = i + 1u) {
        let v = arena[base + i];
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }
    arena[params.out_off + row] = f32(best_idx);
}
