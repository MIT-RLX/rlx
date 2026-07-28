// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// DiT adaLN-Zero: out = norm(x) * (1 + scale) + shift
// scale/shift broadcast over leading dims via lead_pack meta:
//   [lead_rank, x_lead[0..8], mod_lead[0..8]]  (17 uints)
// Launch: grid=(outer,1,1), block=(256,1,1)
// layer_norm != 0 → mean-subtract; else RMS only.

#define ADA_BLOCK 256

__device__ __forceinline__ float ada_block_sum(float v, float* s,
        unsigned int tid, unsigned int bsz) {
    s[tid] = v;
    __syncthreads();
    for (unsigned int s_off = bsz / 2; s_off > 0; s_off >>= 1) {
        if (tid < s_off) s[tid] += s[tid + s_off];
        __syncthreads();
    }
    float r = s[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ unsigned int ada_mod_base(
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

extern "C" __global__ void ada_layer_norm(
    float* arena,
    unsigned int outer,
    unsigned int inner,
    unsigned int in_off,
    unsigned int scale_off,
    unsigned int shift_off,
    unsigned int out_off,
    unsigned int eps_bits,
    unsigned int layer_norm,
    const unsigned int* lead
) {
    unsigned int row = blockIdx.x;
    if (row >= outer) return;
    unsigned int tid = threadIdx.x;
    unsigned int bsz = blockDim.x;
    unsigned int in_base = in_off + row * inner;
    unsigned int out_base = out_off + row * inner;
    unsigned int mod_base = ada_mod_base(row, inner, lead);
    float n_inv = 1.0f / (float)inner;
    float eps = __int_as_float((int)eps_bits);

    __shared__ float s[ADA_BLOCK];

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (unsigned int i = tid; i < inner; i += bsz) {
        float v = arena[in_base + i];
        local_sum += v;
        local_sum_sq += v * v;
    }
    float sum_x = ada_block_sum(local_sum, s, tid, bsz);
    __syncthreads();
    float sum_x2 = ada_block_sum(local_sum_sq, s, tid, bsz);

    float mean = 0.0f;
    float inv;
    // Precise 1/sqrt (not rsqrtf): F5 DiT ODE is sensitive to LN scale bias;
    // fast rsqrt compounded across ~44 adaLN + Softmax/MatMul left the CUDA
    // trajectory (fox 0/6) while CPU `1/sqrt` stayed in-basin.
    if (layer_norm != 0u) {
        mean = sum_x * n_inv;
        float var = fmaxf(sum_x2 * n_inv - mean * mean, 0.0f);
        inv = 1.0f / sqrtf(var + eps);
    } else {
        inv = 1.0f / sqrtf(sum_x2 * n_inv + eps);
    }

    for (unsigned int i = tid; i < inner; i += bsz) {
        float n = (arena[in_base + i] - mean) * inv;
        arena[out_base + i] =
            n * (1.0f + arena[scale_off + mod_base + i]) + arena[shift_off + mod_base + i];
    }
}
