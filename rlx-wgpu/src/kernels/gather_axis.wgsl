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
// Gather along an arbitrary axis. Output layout [outer, num_idx, trailing];
// source layout [outer, axis_dim, trailing]. Indices are f32-encoded.

struct Params {
    total: u32,
    outer: u32,
    axis_dim: u32,
    num_idx: u32,
    trailing: u32,
    table_off: u32,
    idx_off: u32,
    out_off: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn gather_axis(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let o = gid.x + gid.y * ngs.x * 64u;
    if (o >= params.total) { return; }
    let t = o % params.trailing;
    let tmp = o / params.trailing;
    let k = tmp % params.num_idx;
    let outer_o = tmp / params.num_idx;
    var row = u32(max(arena[params.idx_off + k], 0.0));
    if (row >= params.axis_dim) { row = params.axis_dim - 1u; }
    let src = (outer_o * params.axis_dim + row) * params.trailing + t;
    let dst = (outer_o * params.num_idx + k) * params.trailing + t;
    arena[params.out_off + dst] = arena[params.table_off + src];
}
