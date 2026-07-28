// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Argmax along the last axis. Output is f32-encoded index per row.

extern "C" __global__ void argmax(
    float* arena,
    unsigned int outer,
    unsigned int inner,
    unsigned int in_off,
    unsigned int out_off
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= outer || inner == 0) return;
    unsigned int base = in_off + row * inner;
    float best_v = arena[base];
    unsigned int best_i = 0;
    for (unsigned int i = 1; i < inner; ++i) {
        float v = arena[base + i];
        if (v > best_v) { best_v = v; best_i = i; }
    }
    arena[out_off + row] = (float)best_i;
}
