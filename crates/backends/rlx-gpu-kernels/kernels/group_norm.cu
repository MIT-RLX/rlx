// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//
// NCHW group norm: normalize each (C/G)×H×W block. One block per
// (batch, group); 256-thread reduction.

#define GN_BLOCK 256

extern "C" __global__ void group_norm(
    float* arena,
    unsigned int src_off,
    unsigned int g_off,
    unsigned int b_off,
    unsigned int dst_off,
    unsigned int n,
    unsigned int c,
    unsigned int h,
    unsigned int w,
    unsigned int num_groups,
    unsigned int eps_bits
) {
    unsigned int ng = blockIdx.x;
    if (ng >= n * num_groups) return;
    unsigned int bn = ng / num_groups;
    unsigned int g = ng % num_groups;
    unsigned int cpg = c / num_groups;
    unsigned int c0 = g * cpg;
    unsigned int plane = h * w;
    unsigned int count = cpg * plane;

    unsigned int tid = threadIdx.x;
    __shared__ float partial[GN_BLOCK];

    // ── Pass 1: mean ──
    float local_sum = 0.0f;
    for (unsigned int i = tid; i < count; i += GN_BLOCK) {
        unsigned int c_off = i / plane;
        unsigned int s = i % plane;
        unsigned int ch = c0 + c_off;
        local_sum += arena[src_off + ((bn * c + ch) * plane) + s];
    }
    partial[tid] = local_sum;
    __syncthreads();
    for (unsigned int stride = GN_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    float mean = partial[0] / (float)count;
    __syncthreads();

    // ── Pass 2: variance about the mean. TWO-PASS mean((x-mean)^2) — NOT the
    // one-pass E[x^2]-E[x]^2, which catastrophically cancels in f32 when the
    // group has a DC offset (E[x^2] ~ mean^2), corrupting the normalization.
    // For deep generative nets (e.g. bfm2's SplitUNet) that error compounds
    // per-block; the stable form matches CPU/other GPU backends. Same fix as
    // the wgpu two-pass LayerNorm.
    float local_var = 0.0f;
    for (unsigned int i = tid; i < count; i += GN_BLOCK) {
        unsigned int c_off = i / plane;
        unsigned int s = i % plane;
        unsigned int ch = c0 + c_off;
        float d = arena[src_off + ((bn * c + ch) * plane) + s] - mean;
        local_var += d * d;
    }
    partial[tid] = local_var;
    __syncthreads();
    for (unsigned int stride = GN_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    float var = partial[0] / (float)count;
    float eps = __int_as_float((int)eps_bits);
    float inv = 1.0f / sqrtf(var + eps);

    for (unsigned int i = tid; i < count; i += GN_BLOCK) {
        unsigned int c_off = i / plane;
        unsigned int s = i % plane;
        unsigned int ch = c0 + c_off;
        unsigned int idx = ((bn * c + ch) * plane) + s;
        float v = (arena[src_off + idx] - mean) * inv;
        arena[dst_off + idx] = v * arena[g_off + ch] + arena[b_off + ch];
    }
}
