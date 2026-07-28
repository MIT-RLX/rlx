// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// DiT gated residual: out = x + gate * y
// gate broadcasts over leading dims via lead_pack meta (same layout as
// ada_layer_norm: [lead_rank, x_lead[8], mod_lead[8]]).
// Launch: 1-D grid over outer*inner elements.

__device__ __forceinline__ unsigned int gate_mod_base(
        unsigned int row, unsigned int inner, const unsigned int* lead) {
    unsigned int lead_rank = lead[0];
    unsigned int rem = row;
    unsigned int mod_base = 0;
    unsigned int mod_stride = inner;
    for (int j = (int)lead_rank - 1; j >= 0; --j) {
        unsigned int xd = lead[1 + j];
        if (xd == 0u) xd = 1u;
        unsigned int xi = rem % xd;
        rem /= xd;
        unsigned int md = lead[9 + j];
        if (md == 0u) md = 1u;
        if (md != 1u) {
            mod_base += xi * mod_stride;
        }
        mod_stride *= md;
    }
    return mod_base;
}

extern "C" __global__ void gated_residual(
    float* arena,
    unsigned int total,
    unsigned int inner,
    unsigned int x_off,
    unsigned int y_off,
    unsigned int gate_off,
    unsigned int out_off,
    const unsigned int* lead
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int row = i / inner;
    unsigned int col = i % inner;
    unsigned int mod_base = gate_mod_base(row, inner, lead);
    arena[out_off + i] =
        arena[x_off + i] + arena[gate_off + mod_base + col] * arena[y_off + i];
}
