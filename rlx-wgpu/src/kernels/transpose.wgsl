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

// Generic permute. Handles arbitrary rank up to MAX_RANK=8.
//
// Per-axis arrays live in a STORAGE buffer (binding 2) — uniform
// arrays of u32 in WGSL pad each element to 16 bytes which gets
// awkward fast. The axis_meta buffer holds:
//   dims[0..rank]     — output dim sizes
//   strides[0..rank]  — input strides indexed by *output* axis
//                       (host pre-computes strides[i] = input's
//                        cumulative stride for the perm[i] axis).

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
fn transpose(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= params.out_total) { return; }
    var rem = i;
    var src_idx: u32 = 0u;
    // Walk axes from right to left, peeling off the contribution of
    // each output axis to the flat output index, then accumulating
    // src_idx using the input stride for that output axis.
    for (var ax: u32 = params.rank; ax > 0u; ax = ax - 1u) {
        let a = ax - 1u;
        let dim = axis_meta[a];
        let stride = axis_meta[params.rank + a];
        let c = rem % dim;
        rem = rem / dim;
        src_idx = src_idx + c * stride;
    }
    arena[params.out_off + i] = arena[params.in_off + src_idx];
}
