// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Single-rounded fused multiply-add: out[i] = fma(a[i], b[i], c[i]).
// Matches Metal `elem_fma` / wgpu `fma.wgsl` (one rounding).

extern "C" __global__ void fma_elem(
    float* arena,
    unsigned int n,
    unsigned int a_off,
    unsigned int b_off,
    unsigned int c_off,
    unsigned int out_off
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = arena[a_off + i];
    float b = arena[b_off + i];
    float c = arena[c_off + i];
    arena[out_off + i] = __fmaf_rn(a, b, c);
}
