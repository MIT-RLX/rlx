// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Nearest-neighbor NCDHW resample to an explicit output size.
// One thread per output element. Mapping: src = min(floor(dst * in / out), in-1).

extern "C" __global__ void interpolate3d(
    float* arena,
    unsigned int src_off,
    unsigned int dst_off,
    unsigned int n,
    unsigned int c,
    unsigned int d_in,
    unsigned int h_in,
    unsigned int w_in,
    unsigned int d_out,
    unsigned int h_out,
    unsigned int w_out
) {
    unsigned int total = n * c * d_out * h_out * w_out;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int wo = i % w_out;
    unsigned int q1 = i / w_out;
    unsigned int ho = q1 % h_out;
    unsigned int q2 = q1 / h_out;
    unsigned int do_ = q2 % d_out;
    unsigned int q3 = q2 / d_out;
    unsigned int co = q3 % c;
    unsigned int bn = q3 / c;

    unsigned int di = (d_out == 0) ? 0u : (do_ * d_in) / d_out;
    unsigned int hi = (h_out == 0) ? 0u : (ho * h_in) / h_out;
    unsigned int wi = (w_out == 0) ? 0u : (wo * w_in) / w_out;
    if (di >= d_in) di = d_in - 1u;
    if (hi >= h_in) hi = h_in - 1u;
    if (wi >= w_in) wi = w_in - 1u;

    float v = arena[src_off + ((((bn * c + co) * d_in + di) * h_in + hi) * w_in + wi)];
    arena[dst_off + i] = v;
}
