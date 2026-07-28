// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Flat copy / generic memcpy in arena element units.
extern "C" __global__ void copy(
    float* arena,
    unsigned int n,
    unsigned int in_off,
    unsigned int out_off
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    arena[out_off + i] = arena[in_off + i];
}
