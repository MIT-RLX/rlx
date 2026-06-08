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

// Element-wise unary / activation. One kernel covers every per-element
// f32 → f32 transform via an op-kind selector.

struct Params {
    n: u32,
    in_off: u32,
    out_off: u32,
    op: u32,
    // 0=relu, 1=sigmoid, 2=tanh, 3=exp, 4=log, 5=sqrt, 6=rsqrt,
    // 7=neg, 8=abs, 9=gelu, 10=silu, 11=gelu_approx
    _p0: u32, _p1: u32, _p2: u32, _p3: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

// Abramowitz & Stegun 7.1.26 — matches rlx-cpu `scalar_gelu` / matmul gelu_erf.
fn gelu_erf(x: f32) -> f32 {
    let arg = x * 0.70710678118654752;
    let s = select(-1.0, 1.0, arg >= 0.0);
    let xa = abs(arg);
    let t = 1.0 / (1.0 + 0.3275911 * xa);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
                + t * (-1.453152027 + t * 1.061405429))));
    let e = s * (1.0 - poly * exp(-xa * xa));
    return 0.5 * x * (1.0 + e);
}

fn gelu_approx(x: f32) -> f32 {
    let c = 0.7978845608028654;
    let x3 = x * x * x;
    let inner = clamp(c * (x + 0.044715 * x3), -15.0, 15.0);
    return 0.5 * x * (1.0 + tanh(inner));
}

@compute @workgroup_size(64)
fn unary(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= params.n) { return; }
    let x = arena[params.in_off + i];
    var y: f32 = 0.0;
    switch (params.op) {
        case 0u:  { y = max(x, 0.0); }
        case 1u:  { y = 1.0 / (1.0 + exp(-x)); }
        case 2u:  { y = tanh(x); }
        case 3u:  { y = exp(x); }
        case 4u:  { y = log(x); }
        case 5u:  { y = sqrt(x); }
        case 6u:  { y = inverseSqrt(x); }
        case 7u:  { y = -x; }
        case 8u:  { y = abs(x); }
        case 9u:  { y = gelu_erf(x); }
        case 10u: {
            // silu(x) = x * sigmoid(x); clamp -x to avoid exp overflow.
            let nx = clamp(-x, -88.0, 88.0);
            y = x / (1.0 + exp(nx));
        }
        case 11u: { y = gelu_approx(x); }
        case 13u: { y = sin(x); }
        case 14u: { y = cos(x); }
        case 15u: { y = tan(x); }
        case 16u: { y = atan(x); }
        default: { y = x; }
    }
    arena[params.out_off + i] = y;
}
