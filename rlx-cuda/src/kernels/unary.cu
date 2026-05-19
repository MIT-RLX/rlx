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

// Element-wise unary / activation. Selector in `op`:
//   0=relu 1=sigmoid 2=tanh 3=exp 4=log 5=sqrt 6=rsqrt
//   7=neg  8=abs     9=gelu 10=silu 11=gelu_approx

extern "C" __global__ void unary(
    float* arena,
    unsigned int n,
    unsigned int in_off,
    unsigned int out_off,
    unsigned int op
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = arena[in_off + i];
    float y;
    switch (op) {
        case 0: y = fmaxf(x, 0.0f); break;
        case 1: y = 1.0f / (1.0f + expf(-fminf(fmaxf(x, -88.0f), 88.0f))); break;
        case 2: y = tanhf(fminf(fmaxf(x, -15.0f), 15.0f)); break;
        case 3: y = expf(x); break;
        case 4: y = logf(x); break;
        case 5: y = sqrtf(x); break;
        case 6: y = rsqrtf(x); break;
        case 7: y = -x; break;
        case 8: y = fabsf(x); break;
        case 9: case 11: {
            // GELU (tanh approximation), clamped to keep tanh stable.
            const float c = 0.7978845608028654f;
            float x3 = x * x * x;
            float inner = c * (x + 0.044715f * x3);
            inner = fminf(fmaxf(inner, -15.0f), 15.0f);
            y = 0.5f * x * (1.0f + tanhf(inner));
        } break;
        case 10: {
            // SiLU = x · sigmoid(x), with exp clamp.
            float nx = fminf(fmaxf(-x, -88.0f), 88.0f);
            y = x / (1.0f + expf(nx));
        } break;
        default: y = x;
    }
    arena[out_off + i] = y;
}
