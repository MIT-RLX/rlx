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

//
// Unary / activation with an f16 shadow write for CoopF16Vk operands.
// Same op table as `unary.wgsl`; additionally mirrors each f32 output
// element into `arena_f16[out_off + i]`.

enable f16;

struct Params {
    n: u32,
    in_off: u32,
    out_off: u32,
    op: u32,
    _p0: u32, _p1: u32, _p2: u32, _p3: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;
@group(0) @binding(2) var<storage, read_write> arena_f16: array<f16>;

@compute @workgroup_size(64)
fn unary_f16_mirror(@builtin(global_invocation_id) gid: vec3<u32>,
                    @builtin(num_workgroups) ngs: vec3<u32>) {
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= params.n) { return; }
    let x = arena[params.in_off + i];
    var y: f32 = 0.0;
    switch (params.op) {
        case 0u:  { y = max(x, 0.0); }
        case 1u:  { y = 1.0 / (1.0 + exp(-x)); }
        case 2u:  { y = tanh(clamp(x, -15.0, 15.0)); } // clamp: tanh NaNs on large |x|
        case 3u:  { y = exp(x); }
        case 4u:  { y = log(x); }
        case 5u:  { y = sqrt(x); }
        case 6u:  { y = inverseSqrt(x); }
        case 7u:  { y = -x; }
        case 8u:  { y = abs(x); }
        case 9u:  {
            let c = 0.7978845608028654;
            let x3 = x * x * x;
            let inner = clamp(c * (x + 0.044715 * x3), -15.0, 15.0);
            y = 0.5 * x * (1.0 + tanh(inner));
        }
        case 10u: {
            let nx = clamp(-x, -88.0, 88.0);
            y = x / (1.0 + exp(nx));
        }
        case 11u: {
            let c = 0.7978845608028654;
            let x3 = x * x * x;
            let inner = clamp(c * (x + 0.044715 * x3), -15.0, 15.0);
            y = 0.5 * x * (1.0 + tanh(inner));
        }
        case 13u: { y = sin(x); }
        case 14u: { y = cos(x); }
        case 15u: { y = tan(x); }
        case 16u: { y = atan(x); }
        case 17u: { y = 1.0 / x; }
        default: { y = x; }
    }
    arena[params.out_off + i] = y;
    arena_f16[params.out_off + i] = f16(y);
}
