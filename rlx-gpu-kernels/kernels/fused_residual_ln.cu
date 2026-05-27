// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

// Fused (residual add + optional bias add) + LayerNorm.
// Block-per-row with shared-memory tree reductions.
//   y = layer_norm(x + residual + [bias])
//
// Launch shape: grid=(outer,1,1), block=(256,1,1). Phase 1 folds
// residual + bias into out and accumulates the row sum; phases 2 and 3
// run variance and the normalize/scale/shift pass.

#define FRL_BLOCK 256

__device__ __forceinline__ float frl_block_sum(float v, float* s,
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

extern "C" __global__ void fused_residual_ln(
    float* arena,
    unsigned int outer,
    unsigned int inner,
    unsigned int in_off,
    unsigned int residual_off,
    unsigned int bias_off,
    unsigned int gamma_off,
    unsigned int beta_off,
    unsigned int out_off,
    unsigned int eps_bits,
    unsigned int has_bias
) {
    unsigned int row = blockIdx.x;
    if (row >= outer) return;
    unsigned int tid = threadIdx.x;
    unsigned int bsz = blockDim.x;
    unsigned int in_base  = in_off       + row * inner;
    unsigned int res_base = residual_off + row * inner;
    unsigned int out_base = out_off      + row * inner;
    float n_inv = 1.0f / (float)inner;
    float eps = __int_as_float((int)eps_bits);
    bool with_bias = has_bias != 0;

    __shared__ float s[FRL_BLOCK];

    // Phase 1: fold residual + bias into out, accumulate row sum.
    float local_sum = 0.0f;
    for (unsigned int i = tid; i < inner; i += bsz) {
        float v = arena[in_base + i] + arena[res_base + i];
        if (with_bias) v += arena[bias_off + i];
        arena[out_base + i] = v;
        local_sum += v;
    }
    float mean = frl_block_sum(local_sum, s, tid, bsz) * n_inv;

    // Phase 2: variance.
    float local_var = 0.0f;
    for (unsigned int i = tid; i < inner; i += bsz) {
        float d = arena[out_base + i] - mean;
        local_var += d * d;
    }
    float var = frl_block_sum(local_var, s, tid, bsz);
    float inv_std = rsqrtf(var * n_inv + eps);

    // Phase 3: normalize, scale, shift.
    for (unsigned int i = tid; i < inner; i += bsz) {
        float g = arena[gamma_off + i];
        float b = arena[beta_off + i];
        arena[out_base + i] = (arena[out_base + i] - mean) * inv_std * g + b;
    }
}
