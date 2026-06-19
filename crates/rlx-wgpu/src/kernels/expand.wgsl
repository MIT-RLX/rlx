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

// Broadcast input → output by replicating along axes where the
// input dim is 1 but the output dim is larger. The host pre-computes
// per-output-axis input strides: 0 where the input axis is broadcast,
// the actual cumulative input stride otherwise. Same 3-binding
// pattern as transpose (per-axis arrays in a STORAGE buffer).

struct Params {
    rank: u32,
    out_total: u32,
    in_off: u32,
    out_off: u32,
    _p0: u32, _p1: u32, _p2: u32, _p3: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;
@group(0) @binding(2) var<storage, read>        axis_meta: array<u32>;

@compute @workgroup_size(64)
fn expand(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= params.out_total) { return; }
    var rem = i;
    var src_idx: u32 = 0u;
    for (var ax: u32 = params.rank; ax > 0u; ax = ax - 1u) {
        let a = ax - 1u;
        let dim = axis_meta[a];                        // out_dims[a]
        let stride = axis_meta[params.rank + a];       // in_strides[a] (0 if broadcast)
        let c = rem % dim;
        rem = rem / dim;
        src_idx = src_idx + c * stride;
    }
    arena[params.out_off + i] = arena[params.in_off + src_idx];
}
