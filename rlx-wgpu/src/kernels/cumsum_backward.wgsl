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
    inner: u32,
    dy_off: u32,
    dx_off: u32,
    exclusive: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn cumsum_bwd(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let row = gid.x + gid.y * ngs.x * 64u;
    if (row >= params.outer) { return; }
    let dy_base = params.dy_off + row * params.inner;
    let dx_base = params.dx_off + row * params.inner;
    var suffix: f32 = 0.0;
    for (var i: i32 = i32(params.inner) - 1; i >= 0; i = i - 1) {
        let ui = u32(i);
        if (params.exclusive != 0u) {
            arena[dx_base + ui] = suffix;
            suffix = suffix + arena[dy_base + ui];
        } else {
            suffix = suffix + arena[dy_base + ui];
            arena[dx_base + ui] = suffix;
        }
    }
}
