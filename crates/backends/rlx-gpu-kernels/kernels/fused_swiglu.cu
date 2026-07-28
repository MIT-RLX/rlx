// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Fused SwiGLU: y = up * silu(gate) with concatenated [..., 2*n_half] input.
// Matches Metal `fused_swiglu` (f32 arena path).

extern "C" __global__ void fused_swiglu(
    float* arena,
    unsigned int n_half,
    unsigned int total,
    unsigned int gate_first,
    unsigned int in_off,
    unsigned int out_off
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int row = i / n_half;
    unsigned int col = i % n_half;
    unsigned int base = row * (2u * n_half);
    float up;
    float gate;
    if (gate_first != 0u) {
        gate = arena[in_off + base + col];
        up = arena[in_off + base + n_half + col];
    } else {
        up = arena[in_off + base + col];
        gate = arena[in_off + base + n_half + col];
    }
    arena[out_off + i] = up * (gate / (1.0f + expf(-gate)));
}
