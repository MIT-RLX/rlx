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
//   12=round 13=sin 14=cos 15=tan 16=atan
// Keep in sync with `activation_op_id` in the CUDA/ROCm backends.
//
// Cast selectors (f32-uniform arena — inputs/outputs are f32 lanes; the
// dst dtype's *value* is written back as f32). Keep in sync with
// `classify_cast` in the CUDA/ROCm backends and unary.comp / unary.cl:
//   100=f32->i8  101=f32->i16 102=f32->i32 103=f32->i64
//   104=f32->u8  105=f32->u32 106=(x!=0)->bool
// float->int truncates toward zero and saturates to the dst range (matches
// Rust `as`, i.e. rlx-cpu); NaN maps to 0.

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
        case 9:  { y = gelu_erf(x); } break;
        case 11: { y = gelu_approx(x); } break;
        case 10: {
            // SiLU = x · sigmoid(x), with exp clamp.
            float nx = fminf(fmaxf(-x, -88.0f), 88.0f);
            y = x / (1.0f + expf(nx));
        } break;
        case 12: y = rintf(x); break;   // round half-to-even
        case 13: y = sinf(x); break;
        case 14: y = cosf(x); break;
        case 15: y = tanf(x); break;
        case 16: y = atanf(x); break;
        // f32 -> int: truncate toward zero, saturate to dst range, NaN -> 0.
        case 100: y = isnan(x) ? 0.0f : fminf(fmaxf(truncf(x), -128.0f), 127.0f); break;
        case 101: y = isnan(x) ? 0.0f : fminf(fmaxf(truncf(x), -32768.0f), 32767.0f); break;
        case 102: y = isnan(x) ? 0.0f : fminf(fmaxf(truncf(x), -2147483648.0f), 2147483647.0f); break;
        case 103: y = isnan(x) ? 0.0f : fminf(fmaxf(truncf(x), -9223372036854775808.0f), 9223372036854775807.0f); break;
        case 104: y = isnan(x) ? 0.0f : fminf(fmaxf(truncf(x), 0.0f), 255.0f); break;
        case 105: y = isnan(x) ? 0.0f : fminf(fmaxf(truncf(x), 0.0f), 4294967295.0f); break;
        // -> Bool: x != 0 ? 1 : 0 (NaN is non-zero -> 1, matching Rust).
        case 106: y = (x != 0.0f) ? 1.0f : 0.0f; break;
        default: y = x;
    }
    arena[out_off + i] = y;
}
