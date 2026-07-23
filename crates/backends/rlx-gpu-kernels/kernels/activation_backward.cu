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
// `activation_op_id` / unary.cu forward ids:
//   0=relu 1=sigmoid 2=tanh 3=exp 4=log 5=sqrt 6=rsqrt
//   7=neg  8=abs     9=gelu 10=silu 11=gelu_approx
//   12=round 13=sin 14=cos 15=tan 16=atan 17=recip (1/x)
// Formulas mirror rlx-cpu `activation_backward_kernel`.

extern "C" __global__ void relu_backward(
    float* arena,
    unsigned int n,
    unsigned int x_off,
    unsigned int dy_off,
    unsigned int dx_off
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = arena[x_off + i];
    float dy = arena[dy_off + i];
    arena[dx_off + i] = (x > 0.0f) ? dy : 0.0f;
}

extern "C" __global__ void activation_backward(
    float* arena,
    unsigned int n,
    unsigned int x_off,
    unsigned int dy_off,
    unsigned int dx_off,
    unsigned int op
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = arena[x_off + i];
    float dy = arena[dy_off + i];
    float dx;
    switch (op) {
        case 0: // relu
            dx = (x > 0.0f) ? dy : 0.0f;
            break;
        case 1: { // sigmoid: σ(1-σ)
            float xc = fminf(fmaxf(x, -88.0f), 88.0f);
            float s = 1.0f / (1.0f + expf(-xc));
            dx = s * (1.0f - s) * dy;
        } break;
        case 2: { // tanh: 1 - t²
            float t = tanhf(fminf(fmaxf(x, -15.0f), 15.0f));
            dx = (1.0f - t * t) * dy;
        } break;
        case 3: // exp
            dx = expf(x) * dy;
            break;
        case 4: // log
            dx = dy / x;
            break;
        case 5: { // sqrt
            float s = sqrtf(x);
            dx = (s > 0.0f) ? (0.5f * dy / s) : 0.0f;
        } break;
        case 6: { // rsqrt
            float s = sqrtf(x);
            dx = (s > 0.0f) ? (-0.5f * dy / (x * s)) : 0.0f;
        } break;
        case 7: // neg
            dx = -dy;
            break;
        case 8: // abs: sign(x), 0 at 0
            dx = (x > 0.0f) ? dy : ((x < 0.0f) ? -dy : 0.0f);
            break;
        case 9: { // gelu (erf)
            // dy/dx = 0.5 (1 + erf(x/√2)) + (x / √(2π)) · exp(-x²/2)
            const float INV_SQRT2 = 0.7071067811865475f;
            const float INV_SQRT_2PI = 0.3989422804014327f;
            float phi = 0.5f * (1.0f + erff(x * INV_SQRT2));
            float pdf = INV_SQRT_2PI * expf(-0.5f * x * x);
            dx = (phi + x * pdf) * dy;
        } break;
        case 10: { // silu: σ · (1 + x · (1 - σ))
            float xc = fminf(fmaxf(x, -88.0f), 88.0f);
            float s = 1.0f / (1.0f + expf(-xc));
            dx = s * (1.0f + x * (1.0f - s)) * dy;
        } break;
        case 11: { // gelu_approx (tanh)
            const float C = 0.7978845608028654f;
            const float A = 0.044715f;
            float inner = C * (x + A * x * x * x);
            inner = fminf(fmaxf(inner, -15.0f), 15.0f);
            float t = tanhf(inner);
            float dinner = C * (1.0f + 3.0f * A * x * x);
            float d = 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * dinner;
            dx = d * dy;
        } break;
        case 12: // round STE: identity
            dx = dy;
            break;
        case 13: // sin
            dx = cosf(x) * dy;
            break;
        case 14: // cos
            dx = -sinf(x) * dy;
            break;
        case 15: { // tan: 1 + tan²
            float t = tanf(x);
            dx = (1.0f + t * t) * dy;
        } break;
        case 16: // atan: 1 / (1 + x²)
            dx = dy / (1.0f + x * x);
            break;
        case 17: // recip: d(1/x)/dx = -1/x²
            dx = -dy / (x * x);
            break;
        default:
            dx = dy;
            break;
    }
    arena[dx_off + i] = dx;
}
