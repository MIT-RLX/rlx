// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// 3-input select: y[i] = cond[i] ? x[i] : y[i]
// `cond` is the f32-encoded Bool (≠0 → true).
extern "C" __global__ void where_select(
    float* arena,
    unsigned int n,
    unsigned int cond_off,
    unsigned int x_off,
    unsigned int y_off,
    unsigned int out_off
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float c = arena[cond_off + i];
    arena[out_off + i] = (c != 0.0f) ? arena[x_off + i] : arena[y_off + i];
}
