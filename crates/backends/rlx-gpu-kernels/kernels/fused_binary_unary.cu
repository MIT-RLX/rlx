// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Fused element-wise binary + unary. Computes `out[i] = unary(binary(a[i], b[i]))`
// in one kernel; saves one kernel launch + the round-trip to global
// memory for the intermediate.
//
// `bin_op` matches binary.cu's table:
//   0=add 1=sub 2=mul 3=div 4=max 5=min 6=pow
// `un_op` matches unary.cu's table:
//   0=relu 1=sigmoid 2=tanh 3=exp 4=log 5=sqrt 6=rsqrt
//   7=neg  8=abs     9=gelu 10=silu 11=gelu_approx
//   0xFFFF = identity (skip — caller would just emit a Binary in this case)

extern "C" __global__ void fused_binary_unary(
    float* arena,
    unsigned int n,
    unsigned int a_off,
    unsigned int b_off,
    unsigned int out_off,
    unsigned int bin_op,
    unsigned int un_op
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float a = arena[a_off + i];
    float b = arena[b_off + i];
    float v;
    switch (bin_op) {
        case 0: v = a + b; break;
        case 1: v = a - b; break;
        case 2: v = a * b; break;
        case 3: v = a / b; break;
        case 4: v = fmaxf(a, b); break;
        case 5: v = fminf(a, b); break;
        case 6: v = powf(a, b); break;
        default: v = 0.0f;
    }

    if (un_op != 0xFFFFu) {
        switch (un_op) {
            case 0: v = fmaxf(v, 0.0f); break;
            case 1: v = 1.0f / (1.0f + expf(-fminf(fmaxf(v, -88.0f), 88.0f))); break;
            case 2: v = tanhf(fminf(fmaxf(v, -15.0f), 15.0f)); break;
            case 3: v = expf(v); break;
            case 4: v = logf(v); break;
            case 5: v = sqrtf(v); break;
            case 6: v = rsqrtf(v); break;
            case 7: v = -v; break;
            case 8: v = fabsf(v); break;
            case 9:  { v = gelu_erf(v); } break;
            case 11: { v = gelu_approx(v); } break;
            case 10: {
                float nx = fminf(fmaxf(-v, -88.0f), 88.0f);
                v = v / (1.0f + expf(nx));
            } break;
        }
    }

    arena[out_off + i] = v;
}
