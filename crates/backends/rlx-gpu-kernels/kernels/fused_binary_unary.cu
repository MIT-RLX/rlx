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
            // 12..28 (relu-first, matches unary.cu). Previously absent + no
            // default → un_op>=12 left `v` unchanged (identity), silently
            // breaking any fused Binary→Sin/Cos/Round/… — e.g. the StyleTTS2 /
            // Kokoro sine source `sin(2*pi*phase)` fusing Mul→Sin.
            case 12: v = rintf(v); break;                                    // Round
            case 13: v = sinf(v); break;                                     // Sin
            case 14: v = cosf(v); break;                                     // Cos
            case 15: v = tanf(v); break;                                     // Tan
            case 16: v = atanf(v); break;                                    // Atan
            case 17: v = 1.0f / v; break;                                    // Recip
            case 18: v = floorf(v); break;                                   // Floor
            case 19: v = ceilf(v); break;                                    // Ceil
            case 20: v = (float)(v > 0.0f) - (float)(v < 0.0f); break;       // Sign
            case 21: v = fmaxf(v, 0.0f) + logf(1.0f + expf(-fabsf(v))); break; // Softplus
            case 22: v = (v > 0.0f) ? v : (expf(v) - 1.0f); break;           // Elu
            case 23: v = erff(v); break;                                     // Erf
            case 24: v = (v * fminf(fmaxf(v + 3.0f, 0.0f), 6.0f)) / 6.0f; break; // HardSwish
            case 25: v = fminf(fmaxf(v / 6.0f + 0.5f, 0.0f), 1.0f); break;   // HardSigmoid
            case 26: { float sp = fmaxf(v, 0.0f) + logf(1.0f + expf(-fabsf(v))); v = v * tanhf(sp); } break; // Mish
            case 27: v = v / (1.0f + fabsf(v)); break;                       // Softsign
            case 28: v = fminf(v, 0.0f) - logf(1.0f + expf(-fabsf(v))); break; // LogSigmoid
        }
    }

    arena[out_off + i] = v;
}
