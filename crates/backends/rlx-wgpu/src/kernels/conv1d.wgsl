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

// 1D NCL conv. Weight: [c_out, c_in/groups, kl].

struct Params {
    n: u32, c_in: u32, c_out: u32,
    l: u32, l_out: u32,
    kl: u32, sl: u32, pl: u32, dl: u32,
    groups: u32,
    in_off: u32, w_off: u32, out_off: u32,
    _p0: u32, _p1: u32, _p2: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn conv1d(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let total = params.n * params.c_out * params.l_out;
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= total) { return; }
    let lo = i % params.l_out;
    let q1 = i / params.l_out;
    let co = q1 % params.c_out;
    let nn = q1 / params.c_out;

    let c_in_per_g = params.c_in / params.groups;
    let c_out_per_g = params.c_out / params.groups;
    let g = co / c_out_per_g;
    let ci_start = g * c_in_per_g;

    var acc: f32 = 0.0;
    for (var ci_off: u32 = 0u; ci_off < c_in_per_g; ci_off = ci_off + 1u) {
        let ci = ci_start + ci_off;
        for (var k: u32 = 0u; k < params.kl; k = k + 1u) {
            let in_l_s = i32(lo * params.sl + k * params.dl) - i32(params.pl);
            if (in_l_s < 0 || in_l_s >= i32(params.l)) { continue; }
            let il = u32(in_l_s);
            let in_idx = (nn * params.c_in + ci) * params.l + il;
            let w_idx  = (co * c_in_per_g + ci_off) * params.kl + k;
            acc = acc + arena[params.in_off + in_idx] * arena[params.w_off + w_idx];
        }
    }
    arena[params.out_off + i] = acc;
}
