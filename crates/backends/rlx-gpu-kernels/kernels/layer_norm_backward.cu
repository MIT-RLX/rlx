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

// LayerNorm backward (last-axis rows). Matches CPU `LayerNormBackward*` /
// Metal `layer_norm_bwd` / wgpu `layer_norm_bwd_*`.
//   Input: one block per row, shared-memory tree reductions.
//   Gamma: serial single-thread (same shape as `group_norm_bwd_gamma`).

#define LNB_BLOCK 256

__device__ __forceinline__ float lnb_block_sum(float v, float* s,
        unsigned int tid, unsigned int bsz) {
    s[tid] = v;
    __syncthreads();
    for (unsigned int off = bsz / 2; off > 0; off >>= 1) {
        if (tid < off) s[tid] += s[tid + off];
        __syncthreads();
    }
    float r = s[0];
    __syncthreads();
    return r;
}

extern "C" __global__ void layer_norm_bwd_input(
    float* arena,
    unsigned int outer,
    unsigned int inner,
    unsigned int x_off,
    unsigned int gamma_off,
    unsigned int dy_off,
    unsigned int out_off,
    unsigned int eps_bits
) {
    unsigned int row = blockIdx.x;
    if (row >= outer || inner == 0u) return;
    unsigned int tid = threadIdx.x;
    unsigned int bsz = blockDim.x;
    unsigned int x_base = x_off + row * inner;
    unsigned int dy_base = dy_off + row * inner;
    unsigned int out_base = out_off + row * inner;
    float n_inv = 1.0f / (float)inner;
    float eps = __int_as_float((int)eps_bits);

    __shared__ float s_a[LNB_BLOCK];
    __shared__ float s_b[LNB_BLOCK];

    float local_sum = 0.0f;
    for (unsigned int i = tid; i < inner; i += bsz) {
        local_sum += arena[x_base + i];
    }
    float mean = lnb_block_sum(local_sum, s_a, tid, bsz) * n_inv;

    float local_var = 0.0f;
    for (unsigned int i = tid; i < inner; i += bsz) {
        float d = arena[x_base + i] - mean;
        local_var += d * d;
    }
    // Precise 1/sqrt — matches CPU `1.0 / (var*n_inv + eps).sqrt()`.
    float inv_std = 1.0f / sqrtf(lnb_block_sum(local_var, s_a, tid, bsz) * n_inv + eps);

    float local_sy = 0.0f;
    float local_sxh = 0.0f;
    for (unsigned int i = tid; i < inner; i += bsz) {
        float xh = (arena[x_base + i] - mean) * inv_std;
        float sy = arena[dy_base + i] * arena[gamma_off + i];
        local_sy += sy;
        local_sxh += sy * xh;
    }
    s_a[tid] = local_sy;
    s_b[tid] = local_sxh;
    __syncthreads();
    for (unsigned int off = bsz / 2; off > 0; off >>= 1) {
        if (tid < off) {
            s_a[tid] += s_a[tid + off];
            s_b[tid] += s_b[tid + off];
        }
        __syncthreads();
    }
    float m_sy = s_a[0] * n_inv;
    float m_sxh = s_b[0] * n_inv;

    for (unsigned int i = tid; i < inner; i += bsz) {
        float xh = (arena[x_base + i] - mean) * inv_std;
        float sy = arena[dy_base + i] * arena[gamma_off + i];
        arena[out_base + i] = inv_std * (sy - m_sy - xh * m_sxh);
    }
}

extern "C" __global__ void layer_norm_bwd_gamma(
    float* arena,
    unsigned int outer,
    unsigned int inner,
    unsigned int x_off,
    unsigned int dy_off,
    unsigned int out_off,
    unsigned int eps_bits
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (inner == 0u) return;
    float n_inv = 1.0f / (float)inner;
    float eps = __int_as_float((int)eps_bits);

    for (unsigned int i = 0; i < inner; ++i) {
        arena[out_off + i] = 0.0f;
    }

    for (unsigned int row = 0; row < outer; ++row) {
        unsigned int x_base = x_off + row * inner;
        unsigned int dy_base = dy_off + row * inner;
        float sum = 0.0f;
        for (unsigned int i = 0; i < inner; ++i) sum += arena[x_base + i];
        float mean = sum * n_inv;
        float var = 0.0f;
        for (unsigned int i = 0; i < inner; ++i) {
            float d = arena[x_base + i] - mean;
            var += d * d;
        }
        float inv_std = 1.0f / sqrtf(var * n_inv + eps);
        for (unsigned int i = 0; i < inner; ++i) {
            float xh = (arena[x_base + i] - mean) * inv_std;
            arena[out_off + i] += arena[dy_base + i] * xh;
        }
    }
}
