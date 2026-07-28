// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Element-wise activation backward: dx = act'(x) · dy.
//
// The derivative dispatch `rlx_activation_backward(op, x, dy)` (op 0..17,
// relu-first ids) is @generated from the shared rlxsl manifest — the derivative
// is auto-differentiated from the forward `activation_expr`, so it is exactly
// the gradient of the forward we ship — and prepended to this file at build
// time (see build.rs). `relu_backward` is the standalone fast path.

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
    arena[dx_off + i] = rlx_activation_backward(op, x, dy);
}
