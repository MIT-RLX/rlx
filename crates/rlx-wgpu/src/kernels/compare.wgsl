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

// Element-wise compare. Output is f32 (1.0 / 0.0) — rlx-ir treats
// Bool as 1-byte but the wgpu backend keeps everything as f32 for
// arena uniformity. Bool consumers (Where) treat any nonzero as true.

struct Params {
    n: u32,
    a_off: u32,
    b_off: u32,
    c_off: u32,
    op: u32,        // 0=eq, 1=ne, 2=lt, 3=le, 4=gt, 5=ge
    _p0: u32, _p1: u32, _p2: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn compare(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= params.n) { return; }
    let a = arena[params.a_off + i];
    let b = arena[params.b_off + i];
    var r: bool = false;
    switch (params.op) {
        case 0u: { r = a == b; }
        case 1u: { r = a != b; }
        case 2u: { r = a <  b; }
        case 3u: { r = a <= b; }
        case 4u: { r = a >  b; }
        case 5u: { r = a >= b; }
        default: { r = false; }
    }
    arena[params.c_off + i] = select(0.0, 1.0, r);
}
