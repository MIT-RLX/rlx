// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// Cumsum backward along last axis (one thread per row).

extern "C" __global__ void rlx_cumsum_bwd(
    float* arena,
    unsigned int outer,
    unsigned int inner,
    unsigned int dy_off,
    unsigned int dx_off,
    unsigned int exclusive
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= outer) return;
    unsigned int dy_base = dy_off + row * inner;
    unsigned int dx_base = dx_off + row * inner;
    float suffix = 0.0f;
    for (int i = (int)inner - 1; i >= 0; --i) {
        if (exclusive != 0u) {
            arena[dx_base + (unsigned int)i] = suffix;
            suffix += arena[dy_base + (unsigned int)i];
        } else {
            suffix += arena[dy_base + (unsigned int)i];
            arena[dx_base + (unsigned int)i] = suffix;
        }
    }
}
