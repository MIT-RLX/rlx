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

// Per-row reduction along the LAST axis. Output shape: input shape
// minus its last dim (or with the last dim collapsed to 1 when
// keep_dim is set; the dispatcher handles the shape arithmetic).
//
// One thread per output row — sequential read across the inner axis.
// Slow by GPU standards but functional; tree-reduction with shared
// memory is the obvious optimization once the op set is broad enough
// to run real models.

struct Params {
    outer: u32,    // total rows (product of leading dims)
    inner: u32,    // size of the reduced axis
    in_off: u32,
    out_off: u32,
    op: u32,       // 0=sum, 1=mean, 2=max, 3=min, 4=prod
    _p0: u32, _p1: u32, _p2: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn reduce(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let row = gid.x + gid.y * ngs.x * 64u;
    if (row >= params.outer || params.inner == 0u) { return; }
    let base = params.in_off + row * params.inner;

    // Initialize acc from element 0 to avoid having to spell out
    // identity values per op (especially +/-inf in WGSL).
    var acc: f32 = arena[base];
    for (var i: u32 = 1u; i < params.inner; i = i + 1u) {
        let v = arena[base + i];
        switch (params.op) {
            case 0u, 1u: { acc = acc + v; }
            case 2u:     { acc = max(acc, v); }
            case 3u:     { acc = min(acc, v); }
            case 4u:     { acc = acc * v; }
            default:     {}
        }
    }
    if (params.op == 1u) {
        acc = acc / f32(params.inner);
    }
    arena[params.out_off + row] = acc;
}
