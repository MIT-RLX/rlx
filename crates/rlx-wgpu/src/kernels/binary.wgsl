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

// Element-wise binary op. Equal-shape inputs, output same shape.
// Op kind dispatched via params.op (0..6).

struct Params {
    n: u32,         // total elements
    a_off: u32,     // f32-element offset
    b_off: u32,
    c_off: u32,
    op: u32,        // 0=add, 1=sub, 2=mul, 3=div, 4=max, 5=min, 6=pow
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn binary(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= params.n) { return; }
    let a = arena[params.a_off + i];
    let b = arena[params.b_off + i];
    var c: f32 = 0.0;
    switch (params.op) {
        case 0u: { c = a + b; }
        case 1u: { c = a - b; }
        case 2u: { c = a * b; }
        case 3u: { c = a / b; }
        case 4u: { c = max(a, b); }
        case 5u: { c = min(a, b); }
        case 6u: { c = pow(a, b); }
        default: { c = 0.0; }
    }
    arena[params.c_off + i] = c;
}
