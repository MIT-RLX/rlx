// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
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
    float local_sum = 0.0f;
    float local_sumsq = 0.0f;
    for (unsigned int i = tid; i < count; i += GN_BLOCK) {
        unsigned int c_off = i / plane;
        unsigned int s = i % plane;
        unsigned int ch = c0 + c_off;
        float v = arena[src_off + ((bn * c + ch) * plane) + s];
        local_sum += v;
        local_sumsq += v * v;
    }

    __shared__ float partial_sum[GN_BLOCK];
    __shared__ float partial_sumsq[GN_BLOCK];
    partial_sum[tid] = local_sum;
    partial_sumsq[tid] = local_sumsq;
    __syncthreads();
    for (unsigned int stride = GN_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
            partial_sumsq[tid] += partial_sumsq[tid + stride];
        }
        __syncthreads();
    }

    float mean = partial_sum[0] / (float)count;
    float var = partial_sumsq[0] / (float)count - mean * mean;
    float eps = __int_as_float((int)eps_bits);
    float inv = rsqrtf(var + eps);

    for (unsigned int i = tid; i < count; i += GN_BLOCK) {
        unsigned int c_off = i / plane;
        unsigned int s = i % plane;
        unsigned int ch = c0 + c_off;
        unsigned int idx = ((bn * c + ch) * plane) + s;
        float v = (arena[src_off + idx] - mean) * inv;
        arena[dst_off + idx] = v * arena[g_off + ch] + arena[b_off + ch];
    }
}
