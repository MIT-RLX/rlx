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

// 2D NCHW pool with all five rlx ReduceOp kinds. One thread per
// output element (n, c, h_out, w_out); the thread iterates the
// kernel window and accumulates per the op kind.
//
// Padding values (matching the rlx-mlx pool):
//   Max  → -3.4e38
//   Min  → +3.4e38
//   Prod → 1.0
//   Sum/Mean → 0.0

struct Params {
    n: u32, c: u32, h: u32, w: u32,
    h_out: u32, w_out: u32,
    kh: u32, kw: u32,
    sh: u32, sw: u32,
    ph: u32, pw: u32,
    op: u32,           // 0=Sum, 1=Mean, 2=Max, 3=Min, 4=Prod
    in_off: u32,
    out_off: u32,
    _p0: u32, _p1: u32, _p2: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

fn pad_value(op: u32) -> f32 {
    switch (op) {
        case 2u:  { return -3.4e38; }
        case 3u:  { return  3.4e38; }
        case 4u:  { return  1.0; }
        default:  { return  0.0; }
    }
}

@compute @workgroup_size(64)
fn pool2d(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let total = params.n * params.c * params.h_out * params.w_out;
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= total) { return; }
    let wo = i % params.w_out;
    let q1 = i / params.w_out;
    let ho = q1 % params.h_out;
    let q2 = q1 / params.h_out;
    let cc = q2 % params.c;
    let nn = q2 / params.c;

    var acc: f32 = pad_value(params.op);
    var have_init: bool = false;

    for (var kr: u32 = 0u; kr < params.kh; kr = kr + 1u) {
        for (var kc: u32 = 0u; kc < params.kw; kc = kc + 1u) {
            let in_r_signed = i32(ho * params.sh + kr) - i32(params.ph);
            let in_c_signed = i32(wo * params.sw + kc) - i32(params.pw);
            var v: f32;
            if (in_r_signed < 0 || in_c_signed < 0
                || in_r_signed >= i32(params.h)
                || in_c_signed >= i32(params.w)) {
                v = pad_value(params.op);
            } else {
                let in_r = u32(in_r_signed);
                let in_c = u32(in_c_signed);
                let idx = ((nn * params.c + cc) * params.h + in_r) * params.w + in_c;
                v = arena[params.in_off + idx];
            }
            if (!have_init) {
                acc = v;
                have_init = true;
            } else {
                switch (params.op) {
                    case 0u, 1u: { acc = acc + v; }
                    case 2u:     { acc = max(acc, v); }
                    case 3u:     { acc = min(acc, v); }
                    case 4u:     { acc = acc * v; }
                    default:     {}
                }
            }
        }
    }
    if (params.op == 1u) {
        acc = acc / f32(params.kh * params.kw);
    }
    arena[params.out_off + i] = acc;
}
