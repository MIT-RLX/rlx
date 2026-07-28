// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Softmax cross-entropy along the last axis. One block per row.
// Launch: grid=(outer,1,1), block=(256,1,1).
//
// Dense / soft-label (`softmax_cross_entropy`):
//   loss[n] = logsumexp(logits[n]) - Σ_c targets[n,c]·logits[n,c]
// Integer labels (`softmax_cross_entropy_with_logits`):
//   loss[n] = logsumexp(logits[n]) - logits[n, label]
// Backward (`softmax_cross_entropy_backward`):
//   dlogits[n,c] = (softmax(logits[n])[c] - [c==label]) * d_loss[n]

#define SCE_BLOCK 256

extern "C" __global__ void softmax_cross_entropy(
    float* arena,
    unsigned int outer,
    unsigned int inner,
    unsigned int logits_off,
    unsigned int targets_off,
    unsigned int out_off
) {
    unsigned int row = blockIdx.x;
    if (row >= outer || inner == 0u) return;
    unsigned int tid = threadIdx.x;
    unsigned int bsz = blockDim.x;
    unsigned int lbase = logits_off + row * inner;
    unsigned int tbase = targets_off + row * inner;

    __shared__ float s[SCE_BLOCK];

    float local_max = -3.4e38f;
    for (unsigned int i = tid; i < inner; i += bsz) {
        local_max = fmaxf(local_max, arena[lbase + i]);
    }
    s[tid] = local_max;
    __syncthreads();
    for (unsigned int off = bsz / 2; off > 0; off >>= 1) {
        if (tid < off) s[tid] = fmaxf(s[tid], s[tid + off]);
        __syncthreads();
    }
    float row_max = s[0];
    __syncthreads();

    float local_sum = 0.0f;
    float local_dot = 0.0f;
    for (unsigned int i = tid; i < inner; i += bsz) {
        float v = arena[lbase + i];
        local_sum += expf(v - row_max);
        local_dot += arena[tbase + i] * v;
    }
    s[tid] = local_sum;
    __syncthreads();
    for (unsigned int off = bsz / 2; off > 0; off >>= 1) {
        if (tid < off) s[tid] += s[tid + off];
        __syncthreads();
    }
    float sum_exp = s[0];
    __syncthreads();

    s[tid] = local_dot;
    __syncthreads();
    for (unsigned int off = bsz / 2; off > 0; off >>= 1) {
        if (tid < off) s[tid] += s[tid + off];
        __syncthreads();
    }
    if (tid == 0) {
        arena[out_off + row] = (row_max + logf(sum_exp)) - s[0];
    }
}

extern "C" __global__ void softmax_cross_entropy_with_logits(
    float* arena,
    unsigned int outer,
    unsigned int inner,
    unsigned int logits_off,
    unsigned int labels_off,
    unsigned int out_off
) {
    unsigned int row = blockIdx.x;
    if (row >= outer || inner == 0u) return;
    unsigned int tid = threadIdx.x;
    unsigned int bsz = blockDim.x;
    unsigned int lbase = logits_off + row * inner;

    __shared__ float s[SCE_BLOCK];

    float local_max = -3.4e38f;
    for (unsigned int i = tid; i < inner; i += bsz) {
        local_max = fmaxf(local_max, arena[lbase + i]);
    }
    s[tid] = local_max;
    __syncthreads();
    for (unsigned int off = bsz / 2; off > 0; off >>= 1) {
        if (tid < off) s[tid] = fmaxf(s[tid], s[tid + off]);
        __syncthreads();
    }
    float row_max = s[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (unsigned int i = tid; i < inner; i += bsz) {
        local_sum += expf(arena[lbase + i] - row_max);
    }
    s[tid] = local_sum;
    __syncthreads();
    for (unsigned int off = bsz / 2; off > 0; off >>= 1) {
        if (tid < off) s[tid] += s[tid + off];
        __syncthreads();
    }
    if (tid == 0) {
        unsigned int label = (unsigned int)arena[labels_off + row];
        if (label >= inner) label = inner - 1u;
        arena[out_off + row] = (row_max + logf(s[0])) - arena[lbase + label];
    }
}

extern "C" __global__ void softmax_cross_entropy_backward(
    float* arena,
    unsigned int outer,
    unsigned int inner,
    unsigned int logits_off,
    unsigned int labels_off,
    unsigned int d_loss_off,
    unsigned int out_off
) {
    unsigned int row = blockIdx.x;
    if (row >= outer || inner == 0u) return;
    unsigned int tid = threadIdx.x;
    unsigned int bsz = blockDim.x;
    unsigned int lbase = logits_off + row * inner;
    unsigned int obase = out_off + row * inner;

    __shared__ float s[SCE_BLOCK];

    float local_max = -3.4e38f;
    for (unsigned int i = tid; i < inner; i += bsz) {
        local_max = fmaxf(local_max, arena[lbase + i]);
    }
    s[tid] = local_max;
    __syncthreads();
    for (unsigned int off = bsz / 2; off > 0; off >>= 1) {
        if (tid < off) s[tid] = fmaxf(s[tid], s[tid + off]);
        __syncthreads();
    }
    float row_max = s[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (unsigned int i = tid; i < inner; i += bsz) {
        local_sum += expf(arena[lbase + i] - row_max);
    }
    s[tid] = local_sum;
    __syncthreads();
    for (unsigned int off = bsz / 2; off > 0; off >>= 1) {
        if (tid < off) s[tid] += s[tid + off];
        __syncthreads();
    }
    float inv_sum = 1.0f / s[0];
    float scale = arena[d_loss_off + row];
    unsigned int label = (unsigned int)arena[labels_off + row];
    if (label >= inner) label = inner - 1u;

    for (unsigned int k = tid; k < inner; k += bsz) {
        float p = expf(arena[lbase + k] - row_max) * inv_sum;
        float oh = (k == label) ? 1.0f : 0.0f;
        arena[obase + k] = (p - oh) * scale;
    }
}
