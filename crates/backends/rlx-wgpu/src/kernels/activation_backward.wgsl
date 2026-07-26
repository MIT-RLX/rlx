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

// Element-wise activation / ReLU backward. `op` selector matches
// `activation_op_id` / unary.wgsl forward ids:
//   0=relu 1=sigmoid 2=tanh 3=exp 4=log 5=sqrt 6=rsqrt
//   7=neg  8=abs     9=gelu 10=silu 11=gelu_approx
//   12=round 13=sin 14=cos 15=tan 16=atan
// Formulas mirror rlx-cpu `activation_backward_kernel`.

struct Params {
    n: u32,
    x_off: u32,
    dy_off: u32,
    dx_off: u32,
    op: u32,
    _p0: u32, _p1: u32, _p2: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

// Abramowitz & Stegun 7.1.26 — matches rlx-cpu `erf_f32`.
fn erf_approx(x: f32) -> f32 {
    let s = select(-1.0, 1.0, x >= 0.0);
    let xa = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * xa);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
                + t * (-1.453152027 + t * 1.061405429))));
    return s * (1.0 - poly * exp(-xa * xa));
}

@compute @workgroup_size(64)
fn activation_backward(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= params.n) { return; }
    let x = arena[params.x_off + i];
    let dy = arena[params.dy_off + i];
    var dx: f32 = dy;
    switch (params.op) {
        case 0u: { // relu
            dx = select(0.0, dy, x > 0.0);
        }
        case 1u: { // sigmoid: σ(1-σ)
            let xc = clamp(x, -88.0, 88.0);
            let s = 1.0 / (1.0 + exp(-xc));
            dx = s * (1.0 - s) * dy;
        }
        case 2u: { // tanh: 1 - t²
            let t = tanh(clamp(x, -15.0, 15.0));
            dx = (1.0 - t * t) * dy;
        }
        case 3u: { // exp
            dx = exp(x) * dy;
        }
        case 4u: { // log
            dx = dy / x;
        }
        case 5u: { // sqrt
            let s = sqrt(x);
            dx = select(0.0, 0.5 * dy / s, s > 0.0);
        }
        case 6u: { // rsqrt
            let s = sqrt(x);
            dx = select(0.0, -0.5 * dy / (x * s), s > 0.0);
        }
        case 7u: { // neg
            dx = -dy;
        }
        case 8u: { // abs: sign(x), 0 at 0
            dx = select(select(0.0, -dy, x < 0.0), dy, x > 0.0);
        }
        case 9u: { // gelu (erf)
            let INV_SQRT2 = 0.7071067811865475;
            let INV_SQRT_2PI = 0.3989422804014327;
            let phi = 0.5 * (1.0 + erf_approx(x * INV_SQRT2));
            let pdf = INV_SQRT_2PI * exp(-0.5 * x * x);
            dx = (phi + x * pdf) * dy;
        }
        case 10u: { // silu: σ · (1 + x · (1 - σ))
            let xc = clamp(x, -88.0, 88.0);
            let s = 1.0 / (1.0 + exp(-xc));
            dx = s * (1.0 + x * (1.0 - s)) * dy;
        }
        case 11u: { // gelu_approx (tanh)
            let C = 0.7978845608028654;
            let A = 0.044715;
            let inner = clamp(C * (x + A * x * x * x), -15.0, 15.0);
            let t = tanh(inner);
            let dinner = C * (1.0 + 3.0 * A * x * x);
            let d = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dinner;
            dx = d * dy;
        }
        case 12u: { // round STE: identity
            dx = dy;
        }
        case 13u: { // sin
            dx = cos(x) * dy;
        }
        case 14u: { // cos
            dx = -sin(x) * dy;
        }
        case 15u: { // tan: 1 + tan²
            let t = tan(x);
            dx = (1.0 + t * t) * dy;
        }
        case 16u: { // atan: 1 / (1 + x²)
            dx = dy / (1.0 + x * x);
        }
        case 17u: { // recip: -1/x²
            dx = -dy / (x * x);
        }
        case 18u: { dx = 0.0; } // floor
        case 19u: { dx = 0.0; } // ceil
        case 20u: { dx = 0.0; } // sign
        case 21u: { dx = dy / (1.0 + exp(-x)); } // softplus': sigmoid(x)
        case 22u: { dx = select(dy * exp(x), dy, x > 0.0); } // elu'
        default: { dx = dy; }
    }
    arena[params.dx_off + i] = dx;
}
