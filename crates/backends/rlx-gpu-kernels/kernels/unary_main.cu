// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Element-wise unary/activation + cast on the f32-uniform arena.
//
// The activation dispatch `rlx_activation_apply(op, x)` (op 0..28) is @generated
// from the shared rlxsl manifest and prepended to this file at build time (see
// build.rs). The math is defined ONCE in `rlxsl::activation_expr` and shared
// with the other backends' emitters; op ids follow
// `rlx_ir::opcodes::Activation::opcode_relu_first`. `erf` lowers to the native
// hardware `erff` here.
//
// Cast selectors stay here (they are `Op::Cast` on the f32-uniform arena — the
// dst dtype's *value* is written back as f32). Keep in sync with `classify_cast`
// in the CUDA/ROCm backends and unary.comp / unary.cl:
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
    if (op < 100u) {
        y = rlx_activation_apply(op, x);
    } else {
        switch (op) {
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
    }
    arena[out_off + i] = y;
}
