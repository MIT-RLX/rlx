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

// ── Split group norm ──
// `group_norm` gives each (batch, group) one block, so SegResNet's 8 groups
// ran on 8 blocks of the whole GPU. These cut a group into GN_CHUNK-element
// chunks, `blockIdx.x` the chunk and `blockIdx.y` the (batch, group):
//   group_norm_stats   a chunk's mean and centred sum of squares (two passes,
//                      for the cancellation reason above), into `stats_off`
//                      as (mean, M2) per (batch, group, chunk);
//   group_norm_apply   merges the group's chunks (Chan et al.'s pairwise
//                      update), normalises its own chunk and applies the
//                      activation folded into it (`act`, `alpha_bits`).
// Launched back to back on one stream; `stats_off` is the arena tail.

#define GN_CHUNK 8192u

extern "C" __global__ void group_norm_stats(
    float* arena,
    unsigned int src_off,
    unsigned int stats_off,
    unsigned int c,
    unsigned int plane,
    unsigned int num_groups
) {
    unsigned int cpg = c / num_groups, count = cpg * plane;
    unsigned int chunks = (count + GN_CHUNK - 1u) / GN_CHUNK;
    unsigned int k = blockIdx.x, ng = blockIdx.y;
    if (k >= chunks) return;
    unsigned int bn = ng / num_groups, g = ng % num_groups;
    unsigned int base = src_off + (bn * c + g * cpg) * plane;
    unsigned int start = k * GN_CHUNK;
    unsigned int end = min(count, start + GN_CHUNK);
    unsigned int tid = threadIdx.x;
    __shared__ float partial[GN_BLOCK];
    float s = 0.0f;
    for (unsigned int i = start + tid; i < end; i += GN_BLOCK) s += arena[base + i];
    partial[tid] = s;
    __syncthreads();
    for (unsigned int st = GN_BLOCK / 2; st > 0; st >>= 1) {
        if (tid < st) partial[tid] += partial[tid + st];
        __syncthreads();
    }
    float mean = partial[0] / (float)(end - start);
    __syncthreads();
    float q = 0.0f;
    for (unsigned int i = start + tid; i < end; i += GN_BLOCK) {
        float d = arena[base + i] - mean;
        q += d * d;
    }
    partial[tid] = q;
    __syncthreads();
    for (unsigned int st = GN_BLOCK / 2; st > 0; st >>= 1) {
        if (tid < st) partial[tid] += partial[tid + st];
        __syncthreads();
    }
    if (tid == 0) {
        arena[stats_off + (ng * chunks + k) * 2u] = mean;
        arena[stats_off + (ng * chunks + k) * 2u + 1u] = partial[0];
    }
}

__device__ __forceinline__ void gn_merge(float& na, float& ma, float& qa, float nb, float mb, float qb) {
    if (nb == 0.0f) return;
    float nt = na + nb, d = mb - ma;
    ma += d * (nb / nt);
    qa += qb + d * d * (na * nb / nt);
    na = nt;
}

extern "C" __global__ void group_norm_apply(
    float* arena,
    unsigned int src_off,
    unsigned int g_off,
    unsigned int b_off,
    unsigned int dst_off,
    unsigned int stats_off,
    unsigned int c,
    unsigned int plane,
    unsigned int num_groups,
    unsigned int eps_bits,
    unsigned int act,
    unsigned int alpha_bits
) {
    unsigned int cpg = c / num_groups, count = cpg * plane;
    unsigned int chunks = (count + GN_CHUNK - 1u) / GN_CHUNK;
    unsigned int k = blockIdx.x, ng = blockIdx.y;
    if (k >= chunks) return;
    unsigned int bn = ng / num_groups, g = ng % num_groups;
    unsigned int tid = threadIdx.x;
    float alpha = __int_as_float((int)alpha_bits);
    __shared__ float sn[GN_BLOCK], sm[GN_BLOCK], sq[GN_BLOCK];
    float na = 0.0f, ma = 0.0f, qa = 0.0f;
    for (unsigned int j = tid; j < chunks; j += GN_BLOCK) {
        float nb = (float)min(count - j * GN_CHUNK, GN_CHUNK);
        gn_merge(na, ma, qa, nb, arena[stats_off + (ng * chunks + j) * 2u],
                 arena[stats_off + (ng * chunks + j) * 2u + 1u]);
    }
    sn[tid] = na; sm[tid] = ma; sq[tid] = qa;
    __syncthreads();
    for (unsigned int st = GN_BLOCK / 2; st > 0; st >>= 1) {
        if (tid < st) {
            float a0 = sn[tid], a1 = sm[tid], a2 = sq[tid];
            gn_merge(a0, a1, a2, sn[tid + st], sm[tid + st], sq[tid + st]);
            sn[tid] = a0; sm[tid] = a1; sq[tid] = a2;
        }
        __syncthreads();
    }
    float mean = sm[0];
    float eps = __int_as_float((int)eps_bits);
    float inv = 1.0f / sqrtf(sq[0] / (float)count + eps);
    unsigned int c0 = g * cpg;
    unsigned int base = (bn * c + c0) * plane;
    unsigned int start = k * GN_CHUNK;
    unsigned int end = min(count, start + GN_CHUNK);
    for (unsigned int i = start + tid; i < end; i += GN_BLOCK) {
        unsigned int ch = c0 + i / plane;
        float v = (arena[src_off + base + i] - mean) * inv * arena[g_off + ch] + arena[b_off + ch];
        // The activation the graph had next, folded in: 1 ReLU, 2 LeakyReLU,
        // 3 SiLU (`rlx_opt::memory::adjacent_group_norm_activations`).
        if (act == 1u) v = fmaxf(v, 0.0f);
        else if (act == 2u) v = fmaxf(v, alpha * v);
        else if (act == 3u) v = v / (1.0f + __expf(-v));
        arena[dst_off + base + i] = v;
    }
}
