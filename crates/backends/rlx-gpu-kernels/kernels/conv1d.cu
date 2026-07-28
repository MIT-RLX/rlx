// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// 1D conv NCL. Weight: [c_out, c_in/groups, kl].
extern "C" __global__ void conv1d(
    float* arena,
    unsigned int n, unsigned int c_in, unsigned int c_out,
    unsigned int l, unsigned int l_out,
    unsigned int kl, unsigned int sl, unsigned int pl, unsigned int dl,
    unsigned int groups,
    unsigned int in_off,
    unsigned int w_off,
    unsigned int out_off
) {
    unsigned int total = n * c_out * l_out;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int lo = i % l_out;
    unsigned int q1 = i / l_out;
    unsigned int co = q1 % c_out;
    unsigned int nn = q1 / c_out;
    unsigned int c_in_per_g = c_in / groups;
    unsigned int c_out_per_g = c_out / groups;
    unsigned int g = co / c_out_per_g;
    unsigned int ci_start = g * c_in_per_g;
    float acc = 0.0f;
    for (unsigned int ci_off = 0; ci_off < c_in_per_g; ++ci_off) {
        unsigned int ci = ci_start + ci_off;
        for (unsigned int ki = 0; ki < kl; ++ki) {
            int il = (int)(lo * sl + ki * dl) - (int)pl;
            if (il < 0 || il >= (int)l) continue;
            float xv = arena[in_off + (nn * c_in + ci) * l + il];
            float wv = arena[w_off + (co * c_in_per_g + ci_off) * kl + ki];
            acc += xv * wv;
        }
    }
    arena[out_off + i] = acc;
}
