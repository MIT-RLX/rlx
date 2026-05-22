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
struct Params {
    outer: u32,
    axis_dim: u32,
    num_idx: u32,
    trailing: u32,
    dy_off: u32,
    idx_off: u32,
    dst_off: u32,
    _p0: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(256)
fn gather_bwd_zero(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = params.outer * params.axis_dim * params.trailing;
    if (i < n) {
        arena[params.dst_off + i] = 0.0;
    }
}

@compute @workgroup_size(1)
fn gather_bwd_acc(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    if (o >= params.outer) { return; }
    for (var k: u32 = 0u; k < params.num_idx; k = k + 1u) {
        let row = u32(arena[params.idx_off + k]);
        if (row >= params.axis_dim) { continue; }
        for (var j: u32 = 0u; j < params.trailing; j = j + 1u) {
            let v = arena[params.dy_off + (o * params.num_idx + k) * params.trailing + j];
            arena[params.dst_off + (o * params.axis_dim + row) * params.trailing + j] =
                arena[params.dst_off + (o * params.axis_dim + row) * params.trailing + j] + v;
        }
    }
}
