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

// 1D NCL pool. Op-kind selector matches Pool2D.

struct Params {
    n: u32, c: u32, l: u32,
    l_out: u32,
    kl: u32,
    sl: u32,
    pl: u32,
    op: u32,
    in_off: u32, out_off: u32,
    _p0: u32, _p1: u32, _p2: u32, _p3: u32, _p4: u32, _p5: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

fn pad_value(op: u32) -> f32 {
    switch (op) {
        case 2u: { return -3.4e38; }
        case 3u: { return  3.4e38; }
        case 4u: { return  1.0; }
        default: { return  0.0; }
    }
}

@compute @workgroup_size(64)
fn pool1d(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let total = params.n * params.c * params.l_out;
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= total) { return; }
    let lo = i % params.l_out;
    let q1 = i / params.l_out;
    let cc = q1 % params.c;
    let nn = q1 / params.c;

    var acc: f32 = pad_value(params.op);
    var have_init: bool = false;
    for (var k: u32 = 0u; k < params.kl; k = k + 1u) {
        let in_l_signed = i32(lo * params.sl + k) - i32(params.pl);
        var v: f32;
        if (in_l_signed < 0 || in_l_signed >= i32(params.l)) {
            v = pad_value(params.op);
        } else {
            let il = u32(in_l_signed);
            let idx = (nn * params.c + cc) * params.l + il;
            v = arena[params.in_off + idx];
        }
        if (!have_init) { acc = v; have_init = true; }
        else {
            switch (params.op) {
                case 0u, 1u: { acc = acc + v; }
                case 2u:     { acc = max(acc, v); }
                case 3u:     { acc = min(acc, v); }
                case 4u:     { acc = acc * v; }
                default:     {}
            }
        }
    }
    if (params.op == 1u) { acc = acc / f32(params.kl); }
    arena[params.out_off + i] = acc;
}
