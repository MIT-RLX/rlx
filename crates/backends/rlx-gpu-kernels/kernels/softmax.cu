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

// Thread-per-vector softmax for short axes (the hosts check axis_len <=
// SOFTMAX_ROWS_MAX). A block per vector leaves 256 threads to a few dozen
// elements: SynthSeg's 33-label softmax over 160^3 voxels took 50 ms on an
// RTX 3080 Ti. With stride > 1, neighbouring threads read neighbouring
// voxels, so the loads coalesce. Same arithmetic as `softmax`.
// Launch shape: grid=(ceil(num_rows/256),1,1), block=(256,1,1).
extern "C" __global__ void softmax_rows(
    float* arena,
    unsigned int num_rows,
    unsigned int axis_len,
    unsigned int stride,
    unsigned int in_off,
    unsigned int out_off
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    unsigned int base = (row / stride) * axis_len * stride + row % stride;
    unsigned int in_base  = in_off  + base;
    unsigned int out_base = out_off + base;
    float row_max = -3.4e38f;
    for (unsigned int j = 0; j < axis_len; ++j) {
        row_max = fmaxf(row_max, arena[in_base + j * stride]);
    }
    double sum = 0.0;
    for (unsigned int j = 0; j < axis_len; ++j) {
        float e = expf(arena[in_base + j * stride] - row_max);
        arena[out_base + j * stride] = e;
        sum += (double)e;
    }
    float inv_sum = 1.0f / (float)sum;
    for (unsigned int j = 0; j < axis_len; ++j) {
        arena[out_base + j * stride] = arena[out_base + j * stride] * inv_sum;
    }
}

// Warp-per-vector softmax for short contiguous axes (stride == 1, the hosts
// check axis_len <= SOFTMAX_ROWS_MAX = 64): `softmax_rows` would have adjacent
// threads read axis_len*4 bytes apart; here the 32 lanes of a warp read a
// vector's adjacent elements and reduce by shuffles. On HIP the mask-free
// shuffles with width 32 keep each 32-lane group apart on wave64 too.
// Launch shape: grid=(ceil(num_rows*32/256),1,1), block=(256,1,1).
#if defined(__HIP_DEVICE_COMPILE__) || defined(__HIP_PLATFORM_AMD__)
#define RLX_SM_SHFL_XOR(v, m) __shfl_xor((v), (m), 32)
#else
#define RLX_SM_SHFL_XOR(v, m) __shfl_xor_sync(0xffffffffu, (v), (m), 32)
#endif
extern "C" __global__ void softmax_warp(
    float* arena,
    unsigned int num_rows,
    unsigned int axis_len,
    unsigned int stride,
    unsigned int in_off,
    unsigned int out_off
) {
    unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = g >> 5;
    unsigned int lane = g & 31u;
    // Whole warps leave together (num_rows*32 rounds up to whole warps).
    if (row >= num_rows) return;
    (void)stride;
    unsigned int in_base  = in_off  + row * axis_len;
    unsigned int out_base = out_off + row * axis_len;
    // axis_len <= 64: a lane's two elements stay in registers, so the
    // vector is read once and written once.
    unsigned int j0 = lane, j1 = lane + 32u;
    float x0 = j0 < axis_len ? arena[in_base + j0] : -3.4e38f;
    float x1 = j1 < axis_len ? arena[in_base + j1] : -3.4e38f;
    float m = fmaxf(x0, x1);
    for (unsigned int o = 16u; o > 0u; o >>= 1) m = fmaxf(m, RLX_SM_SHFL_XOR(m, o));
    float e0 = j0 < axis_len ? expf(x0 - m) : 0.0f;
    float e1 = j1 < axis_len ? expf(x1 - m) : 0.0f;
    double sum = (double)e0 + (double)e1;
    for (unsigned int o = 16u; o > 0u; o >>= 1) sum += RLX_SM_SHFL_XOR(sum, o);
    float inv_sum = 1.0f / (float)sum;
    if (j0 < axis_len) arena[out_base + j0] = e0 * inv_sum;
    if (j1 < axis_len) arena[out_base + j1] = e1 * inv_sum;
}
