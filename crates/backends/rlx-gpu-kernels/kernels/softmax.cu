// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Softmax along an ARBITRARY axis. Block-per-vector with shared-memory tree
// reductions for max and sum_exp. Each softmax vector has `axis_len` elements
// with `stride` between consecutive elements (stride==1 → the fast contiguous
// last-axis case). There are `num_rows` (= outer * stride) vectors:
//   row r → outer index o = r / stride, inner index s = r % stride,
//   base   = o * axis_len * stride + s,  element j at base + j*stride.
//
// Launch shape: grid=(num_rows,1,1), block=(256,1,1).

#define SM_BLOCK 256

extern "C" __global__ void softmax(
    float* arena,
    unsigned int num_rows,
    unsigned int axis_len,
    unsigned int stride,
    unsigned int in_off,
    unsigned int out_off
) {
    unsigned int row = blockIdx.x;
    if (row >= num_rows) return;
    unsigned int tid = threadIdx.x;
    unsigned int bsz = blockDim.x;
    unsigned int o = row / stride;
    unsigned int sidx = row % stride;
    unsigned int base = o * axis_len * stride + sidx;
    unsigned int in_base  = in_off  + base;
    unsigned int out_base = out_off + base;

    __shared__ float s[SM_BLOCK];

    // Phase 1: row max.
    float local_max = -3.4e38f;
    for (unsigned int j = tid; j < axis_len; j += bsz) {
        local_max = fmaxf(local_max, arena[in_base + j * stride]);
    }
    s[tid] = local_max;
    __syncthreads();
    for (unsigned int s_off = bsz / 2; s_off > 0; s_off >>= 1) {
        if (tid < s_off) s[tid] = fmaxf(s[tid], s[tid + s_off]);
        __syncthreads();
    }
    float row_max = s[0];
    __syncthreads();

    // Phase 2: stash exp(x - max), accumulate sum in double for ODE-stable
    // attention (F5 DiT has 22 Softmax ops; float sum drift compounds).
    double local_sum = 0.0;
    for (unsigned int j = tid; j < axis_len; j += bsz) {
        float e = expf(arena[in_base + j * stride] - row_max);
        arena[out_base + j * stride] = e;
        local_sum += (double)e;
    }
    // Tree-reduce via float shared mem (sum fits f32 for typical attn rows).
    s[tid] = (float)local_sum;
    __syncthreads();
    for (unsigned int s_off = bsz / 2; s_off > 0; s_off >>= 1) {
        if (tid < s_off) s[tid] += s[tid + s_off];
        __syncthreads();
    }
    float inv_sum = 1.0f / s[0];
    __syncthreads();

    // Phase 3: normalize.
    for (unsigned int j = tid; j < axis_len; j += bsz) {
        arena[out_base + j * stride] = arena[out_base + j * stride] * inv_sum;
    }
}
